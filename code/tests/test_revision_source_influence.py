"""Hand-calculated fixed-ledger estimands, source effects and joint coverage."""

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from glycoshift import revision_source_influence as source_module
from glycoshift.fixed_frame_evaluation import FixedFrameSpec, evaluate_fixed_frame
from glycoshift.revision_source_influence import (
    MODEL_SEEDS,
    SourceInfluenceKey,
    source_influence_report,
    summarize_fixed_frame_estimands,
    summarize_fixed_match_coverage,
)

_ORIGIN = pd.Timestamp("2026-01-01", tz="UTC")


def _evaluate(rows, events, *, horizon=30, cut=54, threshold=0.5):
    timeline = pd.DataFrame([
        {
            "source_id": source, "participant_id": participant,
            "context_end": _ORIGIN + pd.Timedelta(days=day, minutes=minute),
            "patient_day": _ORIGIN + pd.Timedelta(days=day),
            "cgm_last": glucose, "score": score, "base_evaluable": True,
            "anchor_cgm_observed": True, "anchor_cgm_interpolated": False,
            "anchor_cgm_censored": False,
        }
        for source, participant, day, minute, glucose, score in rows
    ])
    event_table = pd.DataFrame([
        {"source_id": source, "participant_id": participant, "event_id": event,
         "event_onset": _ORIGIN + pd.Timedelta(days=day, minutes=minute)}
        for source, participant, event, day, minute in events
    ], columns=["source_id", "participant_id", "event_id", "event_onset"])
    exposure = timeline.groupby(
        ["source_id", "participant_id"], observed=True, as_index=False
    ).agg(patient_days=("patient_day", "nunique"), observation_minutes=(
        "context_end", lambda times: len(times) * 5,
    ))
    return evaluate_fixed_frame(
        timeline, event_table, exposure,
        spec=FixedFrameSpec(
            "score", horizon, tuple(sorted(timeline.source_id.unique()))
        ),
        threshold=threshold, glucose_cut=cut,
    )


def _unequal_fixture(*, zero_source=False):
    rows = [
        ("A", "p0", 0, 0, 70, 1), ("A", "p0", 1, 0, 69, 1),
        ("A", "p0", 2, 0, 90, 0), ("A", "p1", 0, 0, 90, 0),
        ("A", "p1", 0, 60, 90, 1), ("A", "zero", 0, 0, 90, 1),
        ("B", "p0", 0, 0, 90, 1), ("B", "p0", 1, 0, 90, 1),
    ]
    if zero_source:
        rows.append(("C", "zero", 0, 0, 90, 1))
    return _evaluate(rows, [
        ("A", "p0", "e0", 0, 20), ("A", "p0", "e1", 1, 10),
        ("A", "p0", "e2", 2, 20), ("A", "p1", "e0", 0, 15),
        ("B", "p0", "e0", 0, 15),
    ])


def test_distinct_estimands_have_hand_calculated_values_and_denominators():
    report = summarize_fixed_frame_estimands(_unequal_fixture())
    sensitivity = report["sensitivity"]
    assert sensitivity["equal_source_participant_macro"]["estimate"] == pytest.approx(
        2 / 3
    )
    assert sensitivity["equal_source_participant_macro"]["denominator"] == 2
    assert sensitivity["participant_pooled"]["estimate"] == pytest.approx(5 / 9)
    assert sensitivity["participant_pooled"]["denominator"] == 3
    assert sensitivity["event_pooled"] == {
        "estimate": 3 / 5, "numerator": 3, "denominator": 5,
    }
    unmatched = report["burden"]["unmatched"]
    assert unmatched["equal_source_participant_macro_per_patient_day"][
        "estimate"
    ] == pytest.approx(7 / 12)
    assert unmatched["participant_pooled_per_patient_day"]["estimate"] == 5 / 8
    assert unmatched["participant_pooled_per_patient_day"]["denominator"] == 4
    assert unmatched["exposure_pooled_per_patient_day"] == {
        "estimate": 3 / 7, "numerator": 3, "denominator": 7,
    }
    assert report["burden"]["delivered"]["exposure_pooled_per_patient_day"][
        "estimate"
    ] == 6 / 7
    assert report["participant_count"] == 4
    assert report["base_event_count"] == 5


def test_joint_coverage_uses_same_match_and_all_base_event_denominators():
    evaluation = _unequal_fixture()
    original_events = evaluation.events.copy()
    original_alarms = evaluation.alarms.copy()
    report = summarize_fixed_match_coverage(evaluation)
    for row in report["coverage"][:2]:
        assert row["qualifying_event_count"] == 2
        assert row["equal_source_participant_macro"]["estimate"] == pytest.approx(
            7 / 12
        )
        assert row["participant_pooled"]["estimate"] == pytest.approx(4 / 9)
        assert row["event_pooled"] == {
            "estimate": 2 / 5, "numerator": 2, "denominator": 5,
        }
    last = report["coverage"][2]
    assert last["qualifying_event_count"] == 1
    assert last["equal_source_participant_macro"]["estimate"] == pytest.approx(1 / 12)
    assert last["participant_pooled"]["estimate"] == pytest.approx(1 / 9)
    assert last["event_pooled"]["estimate"] == 1 / 5
    pd.testing.assert_frame_equal(evaluation.events, original_events)
    pd.testing.assert_frame_equal(evaluation.alarms, original_alarms)


def test_joint_coverage_glucose_and_lead_boundaries_are_inclusive():
    evaluation = _evaluate(
        [("A", "p", day, 0, glucose, 1)
         for day, glucose in enumerate((70, 70, 70, 69))],
        [("A", "p", f"e{day}", day, lead)
         for day, lead in enumerate((10, 15, 20, 20))],
    )
    rows = summarize_fixed_match_coverage(evaluation)["coverage"]
    assert [row["event_pooled"]["estimate"] for row in rows] == [3 / 4, 1 / 2, 1 / 4]


def test_declared_empty_source_is_preserved_and_full_macro_is_undefined():
    evaluation = _unequal_fixture()
    evaluation = replace(evaluation, spec=replace(
        evaluation.spec, source_order=("A", "B", "empty")
    ))
    report = summarize_fixed_frame_estimands(evaluation)
    assert report["source_order"] == ["A", "B", "empty"]
    assert report["undefined_sensitivity_sources"] == ["empty"]
    assert report["undefined_burden_sources"] == ["empty"]
    assert report["per_source"][-1]["participant_count"] == 0
    assert np.isnan(report["sensitivity"]["equal_source_participant_macro"]["estimate"])
    assert report["sensitivity"]["equal_source_participant_macro"]["denominator"] == 3
    assert report["sensitivity"]["event_pooled"]["estimate"] == 3 / 5
    assert np.isnan(report["burden"]["unmatched"][
        "equal_source_participant_macro_per_patient_day"
    ]["estimate"])
    coverage = summarize_fixed_match_coverage(evaluation)
    assert coverage["undefined_sensitivity_sources"] == ["empty"]
    assert np.isnan(coverage["coverage"][0][
        "equal_source_participant_macro"
    ]["estimate"])


def test_exposed_event_free_source_keeps_its_burden_and_undefined_sensitivity_term():
    evaluation = _unequal_fixture(zero_source=True)
    report = summarize_fixed_frame_estimands(evaluation)
    assert report["undefined_sensitivity_sources"] == ["C"]
    assert report["undefined_burden_sources"] == []
    assert report["per_source"][-1]["participant_count"] == 1
    assert report["per_source"][-1]["base_event_count"] == 0
    assert np.isnan(report["sensitivity"]["equal_source_participant_macro"]["estimate"])
    assert report["burden"]["unmatched"][
        "equal_source_participant_macro_per_patient_day"
    ]["estimate"] == pytest.approx(13 / 18)
    assert report["burden"]["unmatched"][
        "exposure_pooled_per_patient_day"
    ]["estimate"] == 1 / 2


@pytest.mark.parametrize("positive", [True, False])
def test_empty_alarm_ledger_retains_zero_burden_and_full_event_denominator(positive):
    evaluation = _evaluate(
        [("A", "p", 0, 0, 90, 0)], [("A", "p", "e", 0, 20)] if positive else [],
    )
    report = summarize_fixed_frame_estimands(evaluation)
    assert report["burden"]["unmatched"][
        "exposure_pooled_per_patient_day"
    ]["estimate"] == 0
    coverage = summarize_fixed_match_coverage(evaluation)["coverage"]
    assert [row["qualifying_event_count"] for row in coverage] == [0, 0, 0]
    assert coverage[0]["event_pooled"]["denominator"] == int(positive)
    if positive:
        assert coverage[0]["event_pooled"]["estimate"] == 0
    else:
        assert np.isnan(coverage[0]["event_pooled"]["estimate"])


def _primary_fixture(*, control_seed=None, empty_source=False):
    result = {}
    for horizon in (30, 60):
        days = (0, 1) if horizon == 30 else (0,)
        events = [(source, "p", f"e{day}", day, 20)
                  for source in ("A", "B") for day in days]
        for arm, seeds in (("lstm", MODEL_SEEDS), ("persistence", (control_seed,))):
            for seed in seeds:
                for cut in (54, 70):
                    rows = []
                    for source in ("A", "B"):
                        first = (
                            0.6 if arm == "persistence" or (
                                (source == "A") == (seed == 17)
                            ) else 0.9
                        )
                        for day in days:
                            rows.append((
                                source, "p", day, 0, 90, first if day == 0 else 0.4
                            ))
                    threshold = (0.75 if cut == 54 else 0.5) if arm == "lstm" else (
                        0.5 if cut == 54 else 0.3
                    )
                    # A's control remains unchanged at 70; B's second event
                    # becomes detected. Both use one threshold, with A's second
                    # score below both control thresholds.
                    if arm == "persistence":
                        rows = [(s, p, d, m, g, 0.1 if s == "A" and d == 1 else score)
                                for s, p, d, m, g, score in rows]
                    evaluation = _evaluate(
                        rows, events, horizon=horizon, cut=cut, threshold=threshold
                    )
                    if empty_source:
                        evaluation = replace(evaluation, spec=replace(
                            evaluation.spec, source_order=("A", "B", "empty")
                        ))
                    result[SourceInfluenceKey(arm, seed, cut, horizon)] = evaluation
    return result


def test_five_seed_two_horizon_source_and_loo_effects_match_hand_calculations():
    report = source_influence_report(_primary_fixture(), source_order=("A", "B"))
    assert len(report["conditions"]) == 24
    assert report["control"]["unique_fit_count"] == 1
    assert report["full_contrast"]["estimate"] == pytest.approx(1 / 4)
    sources = {row["source_id"]: row for row in report["per_source"]}
    assert sources["A"]["estimate"] == pytest.approx(3 / 20)
    assert sources["B"]["estimate"] == pytest.approx(7 / 20)
    assert sources["A"]["contribution_to_full"] == pytest.approx(3 / 40)
    assert sources["B"]["contribution_to_full"] == pytest.approx(7 / 40)
    loo = {row["excluded_source"]: row for row in report["leave_one_source_out"]}
    assert loo["A"]["estimate"] == pytest.approx(7 / 20)
    assert loo["A"]["difference_from_full"] == pytest.approx(1 / 10)
    assert loo["B"]["estimate"] == pytest.approx(3 / 20)
    assert all(row["estimand_changed"] for row in loo.values())
    assert "ci_low" not in report["full_contrast"]


def test_loo_removes_exactly_one_declared_source_and_propagates_undefined():
    report = source_influence_report(
        _primary_fixture(empty_source=True), source_order=("A", "B", "empty")
    )
    assert np.isnan(report["full_contrast"]["estimate"])
    assert report["full_contrast"]["undefined_sources"] == ["empty"]
    loo = {row["excluded_source"]: row for row in report["leave_one_source_out"]}
    assert loo["empty"]["estimate"] == pytest.approx(1 / 4)
    assert np.isnan(loo["empty"]["difference_from_full"])
    assert np.isnan(loo["A"]["estimate"])
    assert loo["A"]["retained_source_order"] == ["B", "empty"]


def test_explicit_seed17_control_is_one_fit_and_aliases_are_rejected():
    values = _primary_fixture(control_seed=17)
    report = source_influence_report(values, source_order=("A", "B"), control_seed=17)
    assert report["full_contrast"]["estimate"] == pytest.approx(1 / 4)
    key = SourceInfluenceKey("persistence", 17, 54, 30)
    values[replace(key, model_seed=23)] = values[key]
    with pytest.raises(ValueError, match="one shared control"):
        source_influence_report(values, source_order=("A", "B"), control_seed=17)


@pytest.mark.parametrize("defect", [
    "event_count", "detected", "lead", "match_id", "alarm_time", "burden", "duplicate",
])
def test_known_bad_load_bearing_fixed_ledger_contracts_are_rejected(defect):
    evaluation = _unequal_fixture()
    participants, events, alarms = (
        evaluation.participants.copy(), evaluation.events.copy(),
        evaluation.alarms.copy(),
    )
    if defect == "event_count":
        participants.loc[0, "base_event_count"] += 1
    elif defect == "detected":
        events.loc[0, "detected"] = False
    elif defect == "lead":
        alarms.loc[alarms.matched, "lead_minutes"] += 1
    elif defect == "match_id":
        alarms.loc[alarms.matched, "matched_event_id"] = "wrong-event"
    elif defect == "alarm_time":
        events.loc[events.detected, "alarm_context_end"] += pd.Timedelta(minutes=1)
    elif defect == "burden":
        participants.loc[0, "unmatched_alarms_per_patient_day"] = 2
    else:
        events = pd.concat([events, events.iloc[:1]], ignore_index=True)
    bad = replace(evaluation, participants=participants, events=events, alarms=alarms)
    with pytest.raises(ValueError):
        summarize_fixed_match_coverage(bad)


def test_same_horizon_exposure_drift_is_not_hidden_by_valid_condition_summaries():
    values = _primary_fixture()
    key = SourceInfluenceKey("lstm", 23, 70, 60)
    table = values[key].participants.copy()
    table["observation_minutes"] += 1
    values[key] = replace(values[key], participants=table)
    with pytest.raises(ValueError, match="same-horizon event/exposure truth"):
        source_influence_report(values, source_order=("A", "B"))


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    source_module._json(path, value)


def _disk_bundle(tmp_path):
    """Synthetic completed fourteen-fit inventory; never a training result."""
    prepared, analysis = tmp_path / "prepared", tmp_path / "analysis"
    values = _primary_fixture()
    for seed in MODEL_SEEDS:
        for cut in (54, 70):
            for horizon in (30, 60):
                original = values[SourceInfluenceKey("lstm", seed, cut, horizon)]
                values[SourceInfluenceKey("tcn", seed, cut, horizon)] = original
    for arm in ("linear", "summary_logistic", "xgboost"):
        for cut in (54, 70):
            for horizon in (30, 60):
                values[SourceInfluenceKey(arm, None, cut, horizon)] = values[
                    SourceInfluenceKey("persistence", None, cut, horizon)
                ]
    endpoint_paths = {}
    for horizon in (30, 60):
        evaluation = values[SourceInfluenceKey("lstm", 17, 54, horizon)]
        endpoint_paths[f"selection/{horizon}"] = {}
        for category, table in (
            ("events", evaluation.events[[
                "source_id", "participant_id", "event_id", "event_onset"
            ]]),
            ("exposure", evaluation.participants[[
                "source_id", "participant_id", "patient_days", "observation_minutes"
            ]]),
        ):
            path = prepared / "endpoints/selection" / f"{category}-{horizon}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            table.assign(partition="selection").to_parquet(path, index=False)
            endpoint_paths[f"selection/{horizon}"][category] = str(path)
    audit = {
        "status": "passed", "revision_contract": source_module.REVISION_CONTRACT,
        "ascertainment_contract": source_module.CONFIRMATION_CONTRACT,
        "timeline_storage": "original_partitions", "source_order": ["A", "B"],
        "endpoint_paths": endpoint_paths,
        "normalized_path": str(tmp_path / "do-not-open-raw-cgm.parquet"),
        "legacy_windows": str(tmp_path / "do-not-open-windows"),
    }
    _write_json(prepared / "input-audit.json", audit)
    arms, fit_rows, index_rows = [], [], []
    for arm in source_module.METHODS:
        for seed in (MODEL_SEEDS if arm in ("lstm", "tcn") else (None,)):
            directory = tmp_path / "producers" / f"{arm}-{seed}"
            actual_seed = 17 if seed is None else seed

            def policy(evaluation, horizon):
                return {
                    "partition": "calibration", "glucose_cut": evaluation.glucose_cut,
                    "nominal_fapd": 0.5, "threshold": evaluation.threshold,
                    "endpoint": {
                        "score_column": f"severe_hypo_{horizon}_probability",
                        "horizon_minutes": horizon, "source_order": ["A", "B"],
                    },
                    "threshold_count": 3,
                    "candidate_rule": (
                        "2001_online_eligible_quantiles_plus_no_alarm_sentinel"
                    ),
                    "tie_rule": "maximum_fixed_sensitivity_then_highest_threshold",
                    "achieved_equal_source_participant_macro_fapd": 0,
                    "achieved_equal_source_participant_macro_fixed_sensitivity": (
                        evaluation.summary["equal_source_participant_macro_fixed_sensitivity"]
                    ),
                    **{name: evaluation.summary[name] for name in (
                        "participant_count", "base_event_count", "patient_days",
                        "observation_minutes",
                    )},
                }

            summary = {
                "revision_contract": source_module.REVISION_CONTRACT,
                "training_seed": actual_seed, "input_feature_mode": "cgm_mask_clock",
                "architecture" if arm in ("lstm", "tcn") else "baseline": arm,
                "tasks": {
                    f"hypo_{h}": {
                        "threshold_partition": "calibration",
                        "temperature_fit_partition": "calibration",
                        "target_column": f"severe_hypo_{h}_label",
                        "fixed_frame_policies": {"0.5": policy(values[
                            SourceInfluenceKey(arm, seed, 54, h)
                        ], h)},
                    } for h in (30, 60)
                },
            }
            effective = {
                "project": {"synthetic": True},
                "protocol": {
                    "ascertainment_contract": source_module.CONFIRMATION_CONTRACT
                },
                "training": {"seed": actual_seed,
                             "revision_contract": source_module.REVISION_CONTRACT},
                "model": {"input_feature_mode": "cgm_mask_clock"},
                "data": {"development_sources": ["A", "B"]},
                "output": {"run_dir": str(directory)},
            }
            _write_json(directory / "model_summary.json", summary)
            _write_json(directory / "config_resolved.yaml", effective)
            _write_json(directory / "cohort_audit.json", audit)
            fit = {"arm": arm, "model_seed": seed, "run_dir": str(directory)}
            arms.append(fit)
            fit_rows.append(fit)
            for cut in source_module.ORIGINAL_CUTS:
                for horizon in (30, 60):
                    for mode in ("refitted", "gate_only"):
                        row = {"arm": arm, "model_seed": seed, "glucose_cut": cut,
                               "horizon_minutes": horizon, "mode": mode}
                        if cut in (54, 70):
                            evaluation = values[SourceInfluenceKey(
                                arm, seed, cut, horizon
                            )]
                            row.update(
                                threshold=evaluation.threshold, **evaluation.summary
                            )
                        index_rows.append(row)
                    if cut not in (54, 70):
                        continue
                    evaluation = values[SourceInfluenceKey(arm, seed, cut, horizon)]
                    root = analysis / "private" / arm / (
                        "shared" if seed is None else str(seed)
                    ) / "refitted" / str(cut) / str(horizon)
                    _write_json(root / "policy.json", {
                        "policy": policy(evaluation, horizon),
                        "applied_glucose_cut": cut,
                        "policy_origin": "verified_frozen_base54" if cut == 54 else (
                            "calibration_refit_at_cut"
                        ),
                        "base54_policy_parity_checked": True,
                    })
                    _write_json(root / "summary.json", evaluation.summary)
                    for category in ("participants", "events", "alarms", "sources"):
                        getattr(evaluation, category).to_parquet(
                            root / f"{category}.parquet", index=False
                        )
    _write_json(analysis / "private/unique-fit-ledger.json", {"fits": fit_rows})
    _write_json(analysis / "analysis-status.json", {"status": "complete"})
    _write_json(analysis / "exportable/cut-grid-summary.json", {
        "status": "complete", "revision_contract": source_module.REVISION_CONTRACT,
        "source_order": ["A", "B"], "cuts": source_module.ORIGINAL_CUTS,
        "horizons": (30, 60), "model_seeds": MODEL_SEEDS, "nominal_fapd": 0.5,
        "evaluation_partition": "selection", "threshold_partition": "calibration",
        "unique_fit_count": 14, "primary_control_unique_fit_count": 1,
        "primary_lstm_unique_fit_count": 5, "analyses": index_rows,
    })
    analysis_config = tmp_path / "original-analysis.json"
    _write_json(analysis_config, {
        "revision_contract": source_module.REVISION_CONTRACT,
        "prepared_dir": str(prepared), "output_dir": str(analysis), "arms": arms,
    })
    return analysis_config


def test_bound_disk_cli_exports_all_methods_and_no_private_identifiers(
    tmp_path, monkeypatch,
):
    original = _disk_bundle(tmp_path)
    config = source_module.prepare_source_influence_config(
        original, tmp_path / "result"
    )
    config_path = tmp_path / "consumer-config.json"
    _write_json(config_path, config)
    parquet_file = source_module.pq.ParquetFile

    def guard(path, *args, **kwargs):
        path = Path(path)
        assert "predictions" not in path.name and "raw-cgm" not in path.name
        assert path.suffix not in (".pt", ".joblib")
        return parquet_file(path, *args, **kwargs)

    monkeypatch.setattr(source_module.pq, "ParquetFile", guard)
    monkeypatch.setattr(sys, "argv", [
        "revision_source_influence", "--config", str(config_path)
    ])
    source_module.main()
    output = Path(config["output_dir"])
    report = json.loads((output / "exportable/source-influence.json").read_text())
    assert report["unique_fit_count"] == 14 and report["condition_count"] == 56
    assert set(report["methods"]) == set(source_module.METHODS)
    assert report["full_contrast"]["estimate"] == pytest.approx(1 / 4)
    estimands = pd.read_csv(output / "exportable/estimands.csv")
    assert set(estimands.arm) == set(source_module.METHODS)
    assert len(estimands) == 56 * 10
    assert len(pd.read_csv(output / "exportable/joint-coverage.csv")) == 56 * 3 * 3
    for path in (output / "exportable").iterdir():
        text = path.read_text()
        assert all(token not in text for token in (
            "participant_id", '"p"', "run_dir", "input_bindings", "size_bytes",
        ))
    receipt = json.loads((output / "private/receipt.json").read_text())
    assert receipt["status"] == "complete"
    assert receipt["raw_cgm_or_weights_read"] is receipt["policies_refit"] is False
    with pytest.raises(FileExistsError, match="non-empty"):
        source_module.run_source_influence(config)


@pytest.mark.parametrize("defect", [
    "binding", "missing_condition", "duplicate_condition", "duplicate_index",
    "historical_prepared", "historical_producer", "wrong_actual_seed", "gate_policy",
])
def test_disk_runner_rejects_binding_and_scientific_identity_defects(tmp_path, defect):
    original_path = _disk_bundle(tmp_path)
    original = json.loads(original_path.read_text())
    analysis = Path(original["output_dir"])
    if defect in ("historical_prepared", "historical_producer", "wrong_actual_seed"):
        if defect == "historical_prepared":
            path = Path(original["prepared_dir"]) / "input-audit.json"
        else:
            path = Path(original["arms"][0]["run_dir"]) / (
                "config_resolved.yaml" if defect == "historical_producer"
                else "model_summary.json"
            )
        value = json.loads(path.read_text())
        if defect == "historical_prepared":
            value["ascertainment_contract"] = None
        elif defect == "historical_producer":
            value["protocol"]["ascertainment_contract"] = None
        else:
            value["training_seed"] = 999
        _write_json(path, value)
    elif defect == "duplicate_index":
        path = analysis / "exportable/cut-grid-summary.json"
        value = json.loads(path.read_text())
        value["analyses"].append(value["analyses"][0])
        _write_json(path, value)
    elif defect == "gate_policy":
        path = analysis / "private/lstm/17/refitted/70/30/policy.json"
        value = json.loads(path.read_text())
        value["policy_origin"] = "frozen_base54_gate_only"
        _write_json(path, value)
    config = source_module.prepare_source_influence_config(
        original_path, tmp_path / "bad"
    )
    if defect == "binding":
        path = Path(config["conditions"][0]["files"]["participants"]["path"])
        path.write_bytes(path.read_bytes() + b"changed-size")
    elif defect == "missing_condition":
        config["conditions"].pop()
    elif defect == "duplicate_condition":
        config["conditions"][-1] = config["conditions"][0]
    with pytest.raises(ValueError):
        source_module.run_source_influence(config)
    assert not Path(config["output_dir"]).exists()


def test_disk_runner_rechecks_native_bindings_after_ledger_reads(tmp_path, monkeypatch):
    original = _disk_bundle(tmp_path)
    config = source_module.prepare_source_influence_config(
        original, tmp_path / "changed"
    )
    audit = Path(config["input_bindings"]["prepared_audit"]["path"])
    parquet_file = source_module.pq.ParquetFile
    changed = False

    def change_after_metadata_validation(path, *args, **kwargs):
        nonlocal changed
        if not changed:
            audit.write_text(audit.read_text() + " ")
            changed = True
        return parquet_file(path, *args, **kwargs)

    monkeypatch.setattr(
        source_module.pq, "ParquetFile", change_after_metadata_validation
    )
    with pytest.raises(ValueError, match="binding changed"):
        source_module.run_source_influence(config)
    assert not Path(config["output_dir"]).exists()
