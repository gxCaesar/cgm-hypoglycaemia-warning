from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

import glycoshift.revision_training as callbacks_module
from glycoshift.calibration import apply_temperature, fit_temperature
from glycoshift.fixed_frame_evaluation import (
    FixedFrameSpec,
    selection_fixed_frame_curve,
)
from glycoshift.revision_inputs import ReplayInputs
from glycoshift.revision_training import RevisionTrainingContext
from glycoshift.sequences import REVISION_CONTRACT, TASKS


def _fixture():
    replays, predictions = {}, {}
    origin = pd.Timestamp("2026-01-01", tz="UTC")
    for partition in ("selection", "calibration"):
        rows, event_rows, exposure_rows = [], [], []
        for source in ("A", "B"):
            participant = f"p-{partition}"
            event_rows.append(
                {
                    "source_id": source,
                    "participant_id": participant,
                    "event_id": "e",
                    "event_onset": origin + pd.Timedelta(minutes=70),
                }
            )
            exposure_rows.append(
                {
                    "source_id": source,
                    "participant_id": participant,
                    "patient_days": 1,
                    "observation_minutes": 20,
                }
            )
            for minute, glucose, base, _logit in (
                (0, 65, False, 3.0),
                (20, 50, True, -1.0),
                (31, 90, True, 2.0),
                (60, 90, True, 1.0),
                (100, 90, True, -2.0),
                (140, 90, False, 3.0),
            ):
                row = {
                    "source_id": source,
                    "participant_id": participant,
                    "partition": partition,
                    "context_end": origin + pd.Timedelta(minutes=minute),
                    "patient_day": origin,
                    "cgm_last": float(glucose),
                    "anchor_cgm_observed": True,
                    "anchor_cgm_interpolated": False,
                    "anchor_cgm_censored": False,
                    "base_evaluable": base,
                    "hypo_currently_active": minute in (0, 31, 60),
                    "severe_hypo_currently_active": minute in (0, 31, 60),
                    "hyper_currently_active": minute in (20, 31),
                }
                for horizon in (30, 60):
                    severe = int(0 < 70 - minute <= horizon)
                    hyper = int(minute in (20, 31, 60))
                    row.update(
                        {
                            f"base_evaluable_{horizon}": base,
                            f"hypo_{horizon}_label": 1,
                            f"severe_hypo_{horizon}_label": severe,
                            f"hyper_{horizon}_label": hyper,
                            f"hypo_{horizon}_training_target": severe,
                            f"hypo_{horizon}_training_mask": base and glucose >= 54,
                            f"hyper_{horizon}_training_target": hyper,
                            f"hyper_{horizon}_training_mask": base
                            and minute not in (20, 31),
                            f"cgm_{horizon}_target": 100.0,
                            f"cgm_{horizon}_target_available": base,
                        }
                    )
                rows.append(row)
        timeline = pd.DataFrame(rows)
        events = {h: pd.DataFrame(event_rows) for h in (30, 60)}
        exposure = {h: pd.DataFrame(exposure_rows) for h in (30, 60)}
        replays[partition] = ReplayInputs(timeline, events, exposure, {})
        scored = timeline.copy(deep=True)
        logits = np.tile([3.0, -1, 2, 1, -2, 3], 2)
        for task in TASKS:
            scored[f"{task}_logit"] = logits
            scored[f"{task}_probability"] = apply_temperature(logits, 1)
        for horizon in (30, 60):
            scored[f"severe_hypo_{horizon}_raw_logit"] = logits
            scored[f"severe_hypo_{horizon}_raw_probability"] = apply_temperature(
                logits, 1
            )
            scored[f"cgm_{horizon}_prediction"] = 115.0
        predictions[partition] = scored
    config = {
        "project": {"synthetic": True},
        "protocol": {"horizons_minutes": [30, 60]},
        "training": {"revision_contract": REVISION_CONTRACT},
        "evaluation": {
            "threshold_candidate_count": 7,
            "false_alarms_per_patient_day": [0.5, 1.0],
            "ece_bins": 3,
        },
    }
    return replays, predictions, config


def _context(replays, **kwargs):
    return RevisionTrainingContext(replays, source_order=("A", "B"), **kwargs)


def _calibrate(context, predictions, config):
    return context.calibrate(
        predictions["selection"],
        predictions["calibration"],
        config=config,
        source_order=("A", "B"),
    )


def test_checkpoint_matches_raw_fixed_frontier_and_legacy_triple_without_ledgers(
    monkeypatch,
):
    replays, predictions, _ = _fixture()
    context = _context(replays)
    calls = []
    monkeypatch.setattr(
        callbacks_module,
        "evaluate_fixed_frame",
        lambda *args, **kwargs: calls.append("unexpected full ledger"),
    )
    score, tie, proxy = context.checkpoint_proxy(
        predictions["selection"],
        source_order=("A", "B"),
        candidate_count=None,
        selection_rule="area",
        budget=0.5,
    )
    curves = []
    for horizon in (30, 60):
        frame = predictions["selection"].assign(
            base_evaluable=predictions["selection"][f"base_evaluable_{horizon}"]
        )
        curves.append(
            selection_fixed_frame_curve(
                frame,
                replays["selection"].events[horizon],
                replays["selection"].exposure[horizon],
                spec=FixedFrameSpec(
                    f"severe_hypo_{horizon}_raw_logit", horizon, ("A", "B")
                ),
            )
        )
    assert score == np.mean([curve["raw_auc"] for curve in curves])
    assert tie == np.mean([curve["fapd_at_best_sensitivity"] for curve in curves])
    assert proxy["evaluator"] == "fixed_frame"
    assert not calls
    # First unevaluable alarm at minute 0 still controls the chronological history.
    assert curves[0]["raw_auc"] == 0
    assert curves[1]["raw_auc"] == 2


def test_temperature_uses_calibration_actual_severe_target_mask_and_relinks_heads():
    replays, predictions, config = _fixture()
    context = _context(replays)
    task_summary, operational = _calibrate(context, predictions, config)
    calibration = predictions["calibration"]
    for horizon in (30, 60):
        task = f"hypo_{horizon}"
        mask = calibration[f"{task}_training_mask"].to_numpy()
        target = calibration.loc[mask, f"severe_hypo_{horizon}_label"].to_numpy()
        logits = calibration.loc[mask, f"{task}_logit"].to_numpy()
        expected = fit_temperature(target, logits)
        assert task_summary[task]["temperature"] == expected
        assert task_summary[task]["target_column"] == f"severe_hypo_{horizon}_label"
        assert task_summary[task]["calibration_retained_count"] == 6
        assert task_summary[task]["temperature_fit_partition"] == "calibration"
        assert set(task_summary[task]["fixed_frame_policies"]) == {"0.5", "1.0"}
        assert "conformal_nonconformity_threshold" not in task_summary[task]
        for frame in predictions.values():
            np.testing.assert_array_equal(
                frame[f"{task}_probability"],
                frame[f"severe_hypo_{horizon}_probability"],
            )
            np.testing.assert_allclose(
                frame[f"{task}_probability"],
                apply_temperature(frame[f"{task}_logit"], expected),
            )
    assert task_summary["hypo_30"]["temperature"] != 1
    assert "fixed_frame_policies" not in task_summary["hyper_30"]
    assert task_summary["hyper_30"]["calibration_retained_count"] == 4
    assert operational["threshold_partition"] == "calibration"
    assert (
        operational["horizons"]["30"]["partitions"]["selection"]["0.5"][
            "base_event_count"
        ]
        == 2
    )


def test_selection_never_fits_temperature_and_offline_hypo_active_flags_are_ignored():
    replays, predictions, config = _fixture()
    baseline, _ = _calibrate(_context(replays), predictions, config)
    replays, altered, config = _fixture()
    for horizon in (30, 60):
        altered["selection"][f"hypo_{horizon}_logit"] *= -5
        altered["selection"][f"severe_hypo_{horizon}_raw_logit"] *= -5
    for frame in altered.values():
        frame["hypo_currently_active"] = False
        frame["severe_hypo_currently_active"] = False
    changed, _ = _calibrate(_context(replays), altered, config)
    for task in TASKS:
        assert changed[task]["temperature"] == baseline[task]["temperature"]
    for task in ("hypo_30", "hypo_60"):
        assert (
            changed[task]["fixed_frame_policies"]
            == baseline[task]["fixed_frame_policies"]
        )


def test_calibrator_runs_one_threshold_sweep_per_horizon(monkeypatch):
    replays, predictions, config = _fixture()
    context = _context(replays)
    original = callbacks_module.selection_fixed_frame_curve
    calls = []

    def wrapped(*args, **kwargs):
        calls.append((kwargs["spec"].horizon_minutes, kwargs["candidate_count"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(callbacks_module, "selection_fixed_frame_curve", wrapped)
    _calibrate(context, predictions, config)
    assert calls == [(30, 7), (60, 7)]


def test_reporter_returns_fixed_counts_correct_targets_continuous_metrics_and_ledgers(
    tmp_path,
    monkeypatch,
):
    replays, predictions, config = _fixture()
    context = _context(replays, output_dir=tmp_path / "synthetic-ledgers")
    tasks, operational = _calibrate(context, predictions, config)
    summary = {"tasks": tasks, "revision_contract": REVISION_CONTRACT}
    combined = pd.concat(predictions.values(), ignore_index=True)

    def unexpected(*args, **kwargs):
        raise AssertionError(
            "unchanged final report must reuse validated fitted ledgers"
        )

    monkeypatch.setattr(callbacks_module, "evaluate_fixed_frame", unexpected)
    report = context.report(combined, config=config, model_summary=summary)
    assert report["operational_fixed_frame"] == operational
    assert not report["legacy_pooled_evaluator_reportable"]
    for partition in ("selection", "calibration"):
        part = report["partitions"][partition]
        assert "severe_hypo_30" not in part  # No old pooled severe evaluator.
        assert part["probability_diagnostics"]["hypo_30"]["target_column"] == (
            "severe_hypo_30_label"
        )
        assert part["probability_diagnostics"]["hypo_30"]["n"] == 6
        for horizon in ("30", "60"):
            assert part["continuous_forecast"][horizon] == {
                "rmse": 15,
                "mae": 15,
                "n": 8,
            }
            for budget in ("0.5", "1.0"):
                assert part["fixed_frame"][horizon][budget]["base_event_count"] == 2
                root = context.output_dir / partition / horizon / budget
                assert (
                    pd.read_parquet(root / "participants.parquet")["patient_days"].sum()
                    == 2
                )
                assert (root / "alarms.parquet").is_file()
                assert (root / "events.parquet").is_file()
    assert (context.output_dir / "fixed_frame_metrics.json").is_file()


def test_base_truth_is_an_immutable_snapshot_and_predictions_can_reorder():
    replays, predictions, _ = _fixture()
    context = _context(replays)
    for replay in replays.values():
        replay.events[30].loc[0, "event_id"] = "changed-after-context"
        replay.exposure[30].loc[0, "patient_days"] = 999
    original = context.checkpoint_proxy(
        predictions["selection"], source_order=("A", "B"), candidate_count=None
    )
    reordered = context.checkpoint_proxy(
        predictions["selection"].iloc[::-1],
        source_order=("A", "B"),
        candidate_count=None,
    )
    assert original == reordered


def test_context_rejects_participant_overlap_and_labels_inconsistent_with_catalogue():
    replays, _, _ = _fixture()
    selection = replays["selection"]
    selection.timeline["participant_id"] = "p-calibration"
    for table in [*selection.events.values(), *selection.exposure.values()]:
        table["participant_id"] = "p-calibration"
    with pytest.raises(ValueError, match="participants overlap"):
        _context(replays)
    replays, _, _ = _fixture()
    timeline = replays["calibration"].timeline
    timeline.loc[3, ["severe_hypo_30_label", "hypo_30_training_target"]] = 0
    with pytest.raises(ValueError, match="fixed base catalogue"):
        _context(replays)


@pytest.mark.parametrize(
    "kind",
    [
        "missing_row",
        "extra_row",
        "mixed_partition",
        "selection_as_calibration",
        "wrong_target",
        "active_derived_mask",
        "base_mask",
        "observed",
        "censored",
        "missing_provenance",
        "raw_head_mismatch",
        "source_order",
        "nonfinite_logit",
        "continuous_target",
    ],
)
def test_callback_known_bad_inputs_fail_closed(kind):
    replays, predictions, config = _fixture()
    context = _context(replays)
    selection, calibration = predictions["selection"], predictions["calibration"]
    source_order = ("A", "B")
    if kind == "missing_row":
        selection = selection.iloc[1:]
    elif kind == "extra_row":
        selection = pd.concat([selection, selection.iloc[:1]], ignore_index=True)
    elif kind == "mixed_partition":
        selection.loc[0, "partition"] = "calibration"
    elif kind == "selection_as_calibration":
        calibration = selection
    elif kind == "wrong_target":
        calibration.loc[3, "hypo_30_training_target"] = 0
    elif kind == "active_derived_mask":
        calibration["hypo_30_training_mask"] &= ~calibration[
            "severe_hypo_currently_active"
        ]
    elif kind == "base_mask":
        calibration.loc[2, "base_evaluable_30"] = False
    elif kind == "observed":
        selection.loc[0, "anchor_cgm_observed"] = False
    elif kind == "censored":
        selection.loc[0, "anchor_cgm_censored"] = True
    elif kind == "missing_provenance":
        selection = selection.drop(columns="anchor_cgm_interpolated")
    elif kind == "raw_head_mismatch":
        selection.loc[0, "severe_hypo_30_raw_logit"] += 1
    elif kind == "source_order":
        source_order = ("B", "A")
    elif kind == "nonfinite_logit":
        calibration.loc[0, "hypo_30_logit"] = np.nan
    elif kind == "continuous_target":
        calibration.loc[3, "cgm_30_target"] += 1
    with pytest.raises(ValueError):
        context.calibrate(
            selection, calibration, config=config, source_order=source_order
        )


@pytest.mark.parametrize("kind", ["contract", "budgets", "candidates", "horizons"])
def test_config_cannot_silently_change_revision_contract(kind):
    replays, predictions, config = _fixture()
    context = _context(replays)
    if kind == "contract":
        config["training"]["revision_contract"] = None
    elif kind == "budgets":
        config["evaluation"]["false_alarms_per_patient_day"] = [0.5]
    elif kind == "candidates":
        config["project"]["synthetic"] = False
    else:
        config["protocol"]["horizons_minutes"] = [60, 30]
    with pytest.raises(ValueError):
        _calibrate(context, predictions, config)


def test_reporter_rejects_stale_or_mislinked_probabilities_and_noncalibration_policy():
    replays, predictions, config = _fixture()
    context = _context(replays)
    tasks, _ = _calibrate(context, predictions, config)
    summary = {"tasks": tasks, "revision_contract": REVISION_CONTRACT}
    combined = pd.concat(predictions.values(), ignore_index=True)
    bad = combined.copy()
    bad.loc[0, "severe_hypo_30_probability"] = 0.123
    with pytest.raises(ValueError, match="severe probability differs"):
        context.report(bad, config=config, model_summary=summary)
    altered = deepcopy(summary)
    altered["tasks"]["hypo_30"]["fixed_frame_policies"]["0.5"]["partition"] = (
        "selection"
    )
    with pytest.raises(ValueError, match="calibration contract"):
        context.report(combined, config=config, model_summary=altered)
    altered = deepcopy(summary)
    altered["tasks"]["hypo_30"]["target_column"] = "hypo_30_label"
    with pytest.raises(ValueError, match="target/mask/temperature"):
        context.report(combined, config=config, model_summary=altered)
