"""Known-answer gates for the prospective corrected-run cut-grid consumer."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from glycoshift.fixed_frame_evaluation import (
    FixedFrameSpec,
    evaluate_fixed_frame,
    fit_fixed_frame_alarm_policy,
)
from glycoshift.revision_analysis import (
    HORIZONS,
    MODEL_SEEDS,
    AnalysisKey,
    _entries,
    analyze_arm,
    joint_primary_inference,
    run_revision_analysis,
)
from glycoshift.sequences import REVISION_CONTRACT

SOURCES = ("dev",)


def _timeline(partition):
    times = pd.date_range("2026-01-01", periods=3, freq="5min", tz="UTC")
    frame = pd.DataFrame(
        {
            "source_id": "dev",
            "participant_id": partition,
            "partition": partition,
            "context_end": times,
            "patient_day": times.normalize(),
            "cgm_last": [60.0, 80.0, 80.0],
            "anchor_cgm_observed": True,
            "anchor_cgm_interpolated": False,
            "anchor_cgm_censored": False,
            "base_evaluable_30": True,
            "base_evaluable_60": True,
            "severe_hypo_30_label": 1,
            "severe_hypo_60_label": 1,
        }
    )
    return frame


def _endpoints():
    result = {}
    for partition in ("calibration", "selection"):
        for horizon in HORIZONS:
            events = pd.DataFrame(
                {
                    "source_id": ["dev"],
                    "participant_id": [partition],
                    "event_id": [f"{partition}-{horizon}"],
                    "event_onset": [pd.Timestamp("2026-01-01T00:15:00Z")],
                    "partition": partition,
                    "horizon_minutes": horizon,
                }
            )
            exposure = pd.DataFrame(
                {
                    "source_id": ["dev"],
                    "participant_id": [partition],
                    "patient_days": [1],
                    "observation_minutes": [15.0],
                    "partition": partition,
                    "horizon_minutes": horizon,
                }
            )
            result[partition, horizon] = events, exposure
    return result


def _scores(frame, arm):
    frame = frame.copy()
    for horizon in HORIZONS:
        frame[f"severe_hypo_{horizon}_probability"] = (
            [0.9, 0.1, 0.1] if arm == "persistence" else [0.1, 0.9, 0.1]
        )
    return frame


def _tasks(calibration, endpoints):
    tasks = {}
    for horizon in HORIZONS:
        timeline = calibration.copy()
        timeline["base_evaluable"] = timeline[f"base_evaluable_{horizon}"]
        policy = fit_fixed_frame_alarm_policy(
            timeline,
            *endpoints["calibration", horizon],
            spec=FixedFrameSpec(f"severe_hypo_{horizon}_probability", horizon, SOURCES),
            partition="calibration",
            candidate_count=2001,
        )
        tasks[f"hypo_{horizon}"] = {
            "target_column": f"severe_hypo_{horizon}_label",
            "temperature_fit_partition": "calibration",
            "threshold_partition": "calibration",
            "fixed_frame_policies": {"0.5": policy.as_dict()},
        }
    return tasks


def _manual_primary():
    evaluations = {}
    endpoints = _endpoints()
    for arm, seeds in (("lstm", MODEL_SEEDS), ("persistence", (None,))):
        timeline = _scores(_timeline("selection"), arm)
        timeline["base_evaluable"] = True
        for seed in seeds:
            for horizon in HORIZONS:
                for cut in (54, 70):
                    evaluations[AnalysisKey(arm, seed, cut, horizon)] = (
                        evaluate_fixed_frame(
                            timeline,
                            *endpoints["selection", horizon],
                            spec=FixedFrameSpec(
                                f"severe_hypo_{horizon}_probability", horizon, SOURCES
                            ),
                            threshold=0.5,
                            glucose_cut=cut,
                        )
                    )
    return evaluations


def test_exact_joint_contrast_and_shared_control_oracle():
    result = joint_primary_inference(_manual_primary(), source_order=SOURCES)
    contrasts = result["sensitivity"]["contrasts"]
    assert contrasts["lstm_minus_persistence_at_54"]["estimate"] == 0
    assert contrasts["lstm_minus_persistence_at_70"]["estimate"] == 1
    interaction = contrasts["interaction_70_minus_54"]
    assert (
        interaction["estimate"] == interaction["ci_low"] == interaction["ci_high"] == 1
    )
    assert result["shared_control"]["unique_fit_count"] == 1
    assert result["shared_control"]["aliases_are_independent_fits"] is False
    assert result["shared_control"]["bootstrap_alias_seeds"] == list(MODEL_SEEDS)
    assert result["sensitivity"]["replicates"] == 1000
    assert result["sensitivity"]["bootstrap_seed"] == 20260810
    assert result["unmatched_alarm_burden"]["population"] == "all_base_participants"


def test_incomplete_seed_set_and_same_count_different_event_identity_rejected():
    evaluations = _manual_primary()
    missing = dict(evaluations)
    del missing[AnalysisKey("lstm", 101, 70, 60)]
    with pytest.raises(ValueError, match="exactly five LSTM seeds"):
        joint_primary_inference(missing, source_order=SOURCES)
    changed = dict(evaluations)
    key = AnalysisKey("lstm", 23, 54, 30)
    changed[key] = replace(
        changed[key], events=changed[key].events.assign(event_id="wrong-event")
    )
    with pytest.raises(ValueError, match="same-horizon event/exposure truth differs"):
        joint_primary_inference(changed, source_order=SOURCES)


def test_gate_only_uses_original_threshold_while_refit_changes_it():
    endpoints = _endpoints()
    cal, selected = (
        _scores(_timeline(p), "persistence") for p in ("calibration", "selection")
    )
    results = list(
        analyze_arm(
            cal, selected, endpoints, tasks=_tasks(cal, endpoints), source_order=SOURCES
        )
    )
    indexed = {
        (h, c, m): (policy, evaluation) for h, c, m, policy, evaluation in results
    }
    assert len(results) == 24
    base = indexed[30, 54, "refitted"][1]
    refit = indexed[30, 70, "refitted"][1]
    gate = indexed[30, 70, "gate_only"][1]
    assert base.threshold == gate.threshold == 0.9
    assert refit.threshold == 0.1
    assert refit.summary["detected_event_count"] == 1
    assert gate.summary["detected_event_count"] == 0
    assert gate.summary["base_event_count"] == 1
    assert gate.summary["participant_count"] == 1
    assert gate.summary["patient_days"] == 1
    assert gate.summary["observation_minutes"] == 15
    assert (
        indexed[30, 90, "refitted"][1].summary[
            "equal_source_participant_macro_fixed_sensitivity"
        ]
        == 0
    )
    assert indexed[30, 70, "gate_only"][0]["policy"]["glucose_cut"] == 54


def test_frozen_policy_parity_and_partition_roles_fail_closed():
    endpoints = _endpoints()
    cal, selected = (
        _scores(_timeline(p), "lstm") for p in ("calibration", "selection")
    )
    tasks = _tasks(cal, endpoints)
    tasks["hypo_30"]["fixed_frame_policies"]["0.5"]["threshold"] += 1e-6
    with pytest.raises(ValueError, match="base54 policy parity failed: threshold"):
        list(analyze_arm(cal, selected, endpoints, tasks=tasks, source_order=SOURCES))
    with pytest.raises(ValueError, match="only the calibration partition"):
        list(
            analyze_arm(
                selected,
                cal,
                endpoints,
                tasks=_tasks(cal, endpoints),
                source_order=SOURCES,
            )
        )
    bad_endpoints = dict(endpoints)
    bad_endpoints["calibration", 30] = tuple(
        table.assign(partition="selection") for table in endpoints["calibration", 30]
    )
    with pytest.raises(ValueError, match="endpoint partition differs"):
        list(
            analyze_arm(
                cal,
                selected,
                bad_endpoints,
                tasks=_tasks(cal, endpoints),
                source_order=SOURCES,
            )
        )


def test_complete_timeline_keeps_unevaluable_alarm_in_refractory_history():
    endpoints = _endpoints()
    cal = _scores(_timeline("calibration"), "lstm")
    selected = _scores(_timeline("selection"), "lstm")
    for horizon in HORIZONS:
        selected.loc[0, f"base_evaluable_{horizon}"] = False
        selected.loc[0, f"severe_hypo_{horizon}_probability"] = 0.95
        events, exposure = endpoints["selection", horizon]
        endpoints["selection", horizon] = (
            events,
            exposure.assign(observation_minutes=10.0),
        )
    results = list(
        analyze_arm(
            cal, selected, endpoints, tasks=_tasks(cal, endpoints), source_order=SOURCES
        )
    )
    evaluation = next(
        e for h, cut, mode, _, e in results if (h, cut, mode) == (30, 54, "refitted")
    )
    assert evaluation.summary["emitted_alarm_count"] == 1
    assert evaluation.summary["unevaluable_alarm_count"] == 1
    assert evaluation.summary["detected_event_count"] == 0


def _bundle(tmp_path):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    endpoints = _endpoints()
    manifest, paths = [], {}
    for partition in ("calibration", "selection"):
        timeline = _timeline(partition)
        shard = prepared / f"{partition}.parquet"
        timeline.to_parquet(shard, index=False)
        manifest.append(
            {
                "source_id": "dev",
                "participant_id": partition,
                "partition": partition,
                "timeline_path": str(shard),
                "stored_rows": len(timeline),
            }
        )
        for horizon in HORIZONS:
            directory = prepared / "endpoints" / partition
            directory.mkdir(parents=True, exist_ok=True)
            paths[f"{partition}/{horizon}"] = {}
            for category, table in zip(
                ("events", "exposure"), endpoints[partition, horizon], strict=True
            ):
                path = directory / f"{category}-{horizon}.parquet"
                table.to_parquet(path, index=False)
                paths[f"{partition}/{horizon}"][category] = str(path)
    # A nonexistent training shard proves this analysis does not open it.
    manifest.append(
        {
            "source_id": "dev",
            "participant_id": "train",
            "partition": "train",
            "timeline_path": str(prepared / "not-readable-train.parquet"),
            "stored_rows": 999,
        }
    )
    (prepared / "participant-manifest.private.json").write_text(
        json.dumps({"participants": manifest})
    )
    (prepared / "input-audit.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "revision_contract": REVISION_CONTRACT,
                "source_order": list(SOURCES),
                "endpoint_paths": paths,
            }
        )
    )
    arms = []
    for arm, seeds in (("persistence", (None,)), ("lstm", MODEL_SEEDS)):
        cal, selected = (
            _scores(_timeline(p), arm) for p in ("calibration", "selection")
        )
        for seed in seeds:
            directory = tmp_path / f"{arm}-{seed}"
            directory.mkdir()
            pd.concat([cal, selected], ignore_index=True).to_parquet(
                directory / "predictions.parquet", index=False
            )
            tasks = _tasks(cal, endpoints)
            summary = {
                "revision_contract": REVISION_CONTRACT,
                "training_seed": seed,
                "architecture": arm,
                "tasks": tasks,
            }
            (directory / "model_summary.json").write_text(json.dumps(summary))
            for partition, timeline in (("calibration", cal), ("selection", selected)):
                for horizon in HORIZONS:
                    current = timeline.copy()
                    current["base_evaluable"] = True
                    evaluated = evaluate_fixed_frame(
                        current,
                        *endpoints[partition, horizon],
                        spec=FixedFrameSpec(
                            f"severe_hypo_{horizon}_probability", horizon, SOURCES
                        ),
                        threshold=tasks[f"hypo_{horizon}"]["fixed_frame_policies"][
                            "0.5"
                        ]["threshold"],
                    )
                    target = (
                        directory / "fixed-frame" / partition / str(horizon) / "0.5"
                    )
                    target.mkdir(parents=True)
                    evaluated.events.to_parquet(target / "events.parquet", index=False)
                    evaluated.participants.to_parquet(
                        target / "participants.parquet", index=False
                    )
            arms.append({"arm": arm, "model_seed": seed, "run_dir": str(directory)})
    return {
        "revision_contract": REVISION_CONTRACT,
        "prepared_dir": str(prepared),
        "output_dir": str(tmp_path / "analysis"),
        "arms": arms,
    }


def test_end_to_end_aggregate_only_exports_and_private_ledgers(tmp_path):
    config = _bundle(tmp_path)
    output = run_revision_analysis(config)
    summary = json.loads((output / "exportable/cut-grid-summary.json").read_text())
    assert summary["unique_fit_count"] == 6
    assert summary["primary_control_unique_fit_count"] == 1
    assert summary["primary_lstm_unique_fit_count"] == 5
    assert len(summary["analyses"]) == 144
    assert all(
        "ppv" in row and "coverage_15_event_fraction" in row
        for row in summary["analyses"]
    )
    assert {path.suffix for path in (output / "exportable").iterdir()} == {
        ".json",
        ".csv",
    }
    assert (
        "participant_id"
        not in (output / "exportable/cut-grid-summary.json").read_text()
    )
    ledgers = output / "private/persistence/shared/gate_only/70/30"
    assert {path.name for path in ledgers.iterdir()} == {
        "policy.json",
        "summary.json",
        "participants.parquet",
        "sources.parquet",
        "events.parquet",
        "alarms.parquet",
    }
    assert (
        json.loads((ledgers / "policy.json").read_text())["policy"]["threshold"] == 0.9
    )
    assert (
        json.loads((output / "analysis-status.json").read_text())["status"]
        == "complete"
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_revision_analysis(config)


def test_input_manifest_missing_seed_duplicate_fit_and_replay_mismatch(tmp_path):
    config = _bundle(tmp_path)
    missing = {**config, "arms": config["arms"][:-1]}
    with pytest.raises(ValueError, match="complete five-seed"):
        _entries(missing)
    with pytest.raises(ValueError, match="duplicate fit/alias"):
        _entries({**config, "arms": config["arms"] + [config["arms"][0]]})
    run = Path(config["arms"][0]["run_dir"])
    frame = pd.read_parquet(run / "predictions.parquet")
    frame.loc[0, "cgm_last"] += 1
    frame.to_parquet(run / "predictions.parquet", index=False)
    with pytest.raises(ValueError, match="metadata/truth differs"):
        run_revision_analysis(config)
    assert (
        json.loads((Path(config["output_dir"]) / "analysis-status.json").read_text())[
            "status"
        ]
        == "failed"
    )


def test_changed_prepared_event_id_is_rejected_even_with_same_onset_and_counts(
    tmp_path,
):
    config = _bundle(tmp_path)
    path = Path(config["prepared_dir"]) / "endpoints/selection/events-30.parquet"
    events = pd.read_parquet(path)
    events["event_id"] = "changed-id-same-event-count"
    events.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="frozen run event truth differs"):
        run_revision_analysis(config)
