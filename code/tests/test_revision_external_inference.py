"""Synthetic ledgers and accepted-fixture receipts, never clinical records."""

from __future__ import annotations

import copy
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from test_revision_external import _binding, _fixture

from glycoshift import revision_external as external
from glycoshift import revision_external_inference as inference
from glycoshift.config import save_config
from glycoshift.fixed_frame_evaluation import (
    FixedFrameKey,
    FixedFrameSpec,
    _aggregate,
    evaluate_fixed_frame,
    joint_fixed_frame_bootstrap,
)
from glycoshift.revision_composition import (
    FrozenRevisionComposition,
    FrozenStepECDF,
    save_frozen_composition,
)
from glycoshift.sequences import REVISION_CONTRACT

SOURCE = "Loop Observational Study"
SEEDS = inference.MODEL_SEEDS
IDS = ["source_id", "participant_id"]


def _evaluation(horizon, detected, *, zero_alarm=False, cut=54, positive=("p0", "p1")):
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    timeline = pd.DataFrame(
        {
            "source_id": SOURCE,
            "participant_id": ["p0", "p1", "zero"],
            "context_end": start,
            "patient_day": start,
            "cgm_last": 120.0,
            "anchor_cgm_observed": True,
            "anchor_cgm_interpolated": False,
            "anchor_cgm_censored": False,
            "base_evaluable": True,
            "score": [
                float(p in detected or (p == "zero" and zero_alarm))
                for p in ("p0", "p1", "zero")
            ],
        }
    )
    events = pd.DataFrame(
        {
            "source_id": [SOURCE] * len(positive),
            "participant_id": list(positive),
            "event_id": [f"event-{p}-{horizon}" for p in positive],
            "event_onset": [start + pd.Timedelta(minutes=20)] * len(positive),
        }
    )
    exposure = pd.DataFrame(
        {
            "source_id": SOURCE,
            "participant_id": ["p0", "p1", "zero"],
            "patient_days": 1,
            "observation_minutes": 5.0,
        }
    )
    return evaluate_fixed_frame(
        timeline,
        events,
        exposure,
        spec=FixedFrameSpec("score", horizon, (SOURCE,)),
        threshold=0.6,
        glucose_cut=cut,
    )


def _joint_fixture(
    *, seeds=(17, 23), positivity=None, cuts=(54,), with_composition=True,
):
    result = {}
    for horizon in (30, 60):
        positive = (positivity or {}).get(horizon, ("p0", "p1"))
        for cut in cuts:
            control = _evaluation(horizon, (), positive=positive, cut=cut)
            for seed in seeds:
                # Each horizon/seed has complementary patients; averaging must
                # happen after the same participant draw.
                selected = "p0" if (seed == seeds[0]) == (horizon == 30) else "p1"
                values = {
                    "persistence": control,
                    "lstm": _evaluation(
                        horizon, (selected,), positive=positive, cut=cut
                    ),
                }
                if with_composition:
                    values["composition"] = _evaluation(
                        horizon, positive, zero_alarm=True, positive=positive, cut=cut
                    )
                for arm, evaluation in values.items():
                    result[FixedFrameKey(arm, cut, horizon, seed)] = evaluation
    return result


@pytest.mark.parametrize("column", ["patient_days", "observation_minutes"])
def test_participant_exposure_accepts_equal_integer_and_float_storage(column):
    evaluation = _evaluation(30, ("p0",))
    table = evaluation.participants
    exposure = table[inference._EXPOSURE].copy()
    exposure[column] = exposure[column].astype(
        "float64" if column == "patient_days" else "int64"
    )
    assert not table[inference._EXPOSURE].equals(exposure)
    accepted = inference._participant_ledger(table, evaluation.events, exposure, SOURCE)
    pd.testing.assert_frame_equal(accepted, table)


@pytest.mark.parametrize(
    "defect", ["changed_days", "changed_minutes", "tiny_change", "bool", "text", "nan"]
)
def test_participant_exposure_storage_fix_still_rejects_changed_or_invalid_values(
    defect,
):
    evaluation = _evaluation(30, ("p0",))
    exposure = evaluation.participants[inference._EXPOSURE].copy()
    if defect == "changed_days":
        exposure.loc[0, "patient_days"] += 1
    elif defect == "changed_minutes":
        exposure.loc[0, "observation_minutes"] += 1
    elif defect == "tiny_change":
        exposure.loc[0, "observation_minutes"] += 1e-13
    elif defect == "bool":
        exposure["patient_days"] = True
    elif defect == "text":
        exposure["observation_minutes"] = exposure.observation_minutes.astype(str)
    else:
        exposure.loc[0, "observation_minutes"] = np.nan
    with pytest.raises(ValueError, match="identity/exposure"):
        inference._participant_ledger(
            evaluation.participants, evaluation.events, exposure, SOURCE
        )


def test_two_horizon_two_seed_shared_draw_hand_oracle_and_all_patient_burden():
    evaluations = _joint_fixture()
    result = inference.paired_external_inference(
        evaluations,
        source_order=(SOURCE,),
        model_seed_order=(17, 23),
        replicates=127,
        include_replicates=True,
    )
    sensitivity = result["metrics"]["fixed_sensitivity"]
    contrast = sensitivity["contrasts"]["lstm_minus_persistence_at_54"]
    assert contrast["estimate"] == contrast["ci_low"] == contrast["ci_high"] == 0.5
    assert set(contrast["replicate_estimates"]) == {0.5}
    assert contrast["per_horizon"][30]["estimate"] == 0.5
    assert contrast["per_horizon"][60]["estimate"] == 0.5
    assert sensitivity["stratum_counts"] == [
        {
            "source_id": SOURCE,
            "positivity_pattern": [True, True],
            "participant_count": 2,
        }
    ]
    burden = result["metrics"]["unmatched_alarms_per_patient_day"]
    assert burden["stratum_counts"] == [{"source_id": SOURCE, "participant_count": 3}]
    assert burden["contrasts"]["composition_at_54"]["estimate"] == pytest.approx(1 / 3)
    assert (
        result["metrics"]["delivered_alarms_per_patient_day"]["contrasts"][
            "composition_at_54"
        ]["estimate"]
        == 1.0
    )
    # Direct parity with the authoritative joint oracle includes every draw.
    oracle = joint_fixed_frame_bootstrap(
        evaluations,
        contrasts=inference._contrasts((54,)),
        source_order=(SOURCE,),
        model_seed_order=(17, 23),
        replicates=127,
        include_replicates=True,
    )
    assert sensitivity["contrasts"] == oracle["contrasts"]


def test_horizon_positivity_patterns_are_preserved_without_intersection():
    values = _joint_fixture(positivity={30: ("p0",), 60: ("p1",)})
    result = inference.paired_external_inference(
        values,
        source_order=(SOURCE,),
        model_seed_order=(17, 23),
        replicates=11,
    )
    sensitivity = result["metrics"]["fixed_sensitivity"]
    assert sensitivity["stratum_counts"] == [
        {
            "source_id": SOURCE,
            "positivity_pattern": [False, True],
            "participant_count": 1,
        },
        {
            "source_id": SOURCE,
            "positivity_pattern": [True, False],
            "participant_count": 1,
        },
    ]
    assert sensitivity["contrasts"]["lstm_at_54"]["estimate"] == 0.5
    assert (
        sensitivity["interval_status"]
        == "descriptive_sparse_event_positive_participants"
    )
    assert sensitivity["contrasts"]["lstm_at_54"]["ci_low"] is None


def test_zero_event_horizon_is_explicitly_undefined_but_burden_retains_every_patient():
    values = _joint_fixture(positivity={30: (), 60: ("p0", "p1")})
    result = inference.paired_external_inference(
        values,
        source_order=(SOURCE,),
        model_seed_order=(17, 23),
        replicates=11,
    )
    sensitivity = result["metrics"]["fixed_sensitivity"]
    assert sensitivity["interval_status"] == "undefined_base_sensitivity"
    assert np.isnan(sensitivity["contrasts"]["composition_at_54"]["estimate"])
    assert (
        sensitivity["contrasts"]["composition_at_54"]["per_horizon"][60]["estimate"]
        == 1
    )
    assert sensitivity["contrasts"]["composition_at_54"]["ci_low"] is None
    burden = result["metrics"]["unmatched_alarms_per_patient_day"]
    assert burden["stratum_counts"][0]["participant_count"] == 3
    assert burden["contrasts"]["composition_at_54"]["estimate"] == pytest.approx(1 / 3)


@pytest.mark.parametrize(
    "defect",
    ["seed", "patient", "source", "onset", "exposure", "sensitivity", "control_alias"],
)
def test_joint_consumer_rejects_missing_and_drifting_truth(defect):
    values = _joint_fixture()
    key = FixedFrameKey("lstm", 54, 60, 23)
    value = values[key]
    if defect == "seed":
        del values[key]
    elif defect == "patient":
        values[key] = replace(value, participants=value.participants.iloc[:-1].copy())
    elif defect == "source":
        values[key] = replace(value, spec=replace(value.spec, source_order=("wrong",)))
    elif defect == "onset":
        events = value.events.copy()
        events.loc[0, "event_onset"] += pd.Timedelta(minutes=1)
        values[key] = replace(value, events=events)
    elif defect == "exposure":
        table = value.participants.copy()
        table.loc[0, "observation_minutes"] += 1
        values[key] = replace(value, participants=table)
    elif defect == "sensitivity":
        table = value.participants.copy()
        table.loc[0, "fixed_sensitivity"] = 0.123
        values[key] = replace(value, participants=table)
    else:
        key = FixedFrameKey("persistence", 54, 60, 23)
        values[key] = _evaluation(60, ("p0",))
    with pytest.raises(ValueError):
        inference.paired_external_inference(
            values,
            source_order=(SOURCE,),
            model_seed_order=(17, 23),
            replicates=3,
        )


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    external._json(path, value)


def _write_parquet(path, table):
    path.parent.mkdir(parents=True, exist_ok=True)
    external._parquet(path, table)


def _accepted_fixture(
    tmp_path, *, cut70=False, positivity=None, horizon_members=None,
    with_composition=True,
):
    """Synthetic native producer outputs without model or source replay."""
    freeze_path, entries, normalized = _fixture(tmp_path / "fixture", cut70=cut70)
    freeze = json.loads(freeze_path.read_text())
    persistence = entries[0]
    persistence_config = persistence["config"]["snapshot"]
    persistence_config["training"]["seed"] = 17
    save_config(
        persistence_config, Path(persistence["run_dir"]) / "config_resolved.yaml"
    )
    persistence["config"]["binding"] = _binding(
        Path(persistence["run_dir"]) / "config_resolved.yaml"
    )
    persistence["model_summary"]["snapshot"]["training_seed"] = 17
    _write_json(
        Path(persistence["run_dir"]) / "model_summary.json",
        persistence["model_summary"]["snapshot"],
    )
    persistence["model_summary"]["binding"] = _binding(
        Path(persistence["run_dir"]) / "model_summary.json"
    )
    runs = [persistence]
    template = entries[1]
    for seed in SEEDS:
        entry = copy.deepcopy(template)
        run = tmp_path / f"lstm-fit-{seed}"
        shutil.copytree(template["run_dir"], run)
        config = entry["config"]["snapshot"]
        config["training"]["seed"] = seed
        config["output"]["run_dir"] = str(run)
        save_config(config, run / "config_resolved.yaml")
        entry["run_dir"] = str(run)
        entry["config"]["binding"] = _binding(run / "config_resolved.yaml")
        entry["model_summary"]["snapshot"]["training_seed"] = seed
        _write_json(run / "model_summary.json", entry["model_summary"]["snapshot"])
        entry["model_summary"]["binding"] = _binding(run / "model_summary.json")
        entry["model_artifacts"] = [
            _binding(run / "ckpts/best.pt"),
            _binding(run / "scaler.json"),
        ]
        if cut70:
            for h in (30, 60):
                path = run / f"synthetic-dev-policy-70-{h}.json"
                entry["cut_policies"]["70"][str(h)]["0.5"]["evidence"] = _binding(path)
        runs.append(entry)
    freeze["runs"] = runs
    source = freeze["partitions"]["external_loop"]
    # The synthetic accepted source matches the three private participant groups.
    original = pd.read_parquet(normalized)
    blocks = [
        original.iloc[:72].assign(participant_id=participant)
        for participant in ("p0", "p1", "zero")
    ]
    pd.concat(blocks, ignore_index=True).to_parquet(normalized, index=False)
    source["normalized_inputs"] = [_binding(normalized)]
    source_order = tuple(sorted(persistence_config["data"]["development_sources"]))
    recipes = {}
    if with_composition:
        freeze["compositions"] = []
    else:
        freeze.pop("compositions", None)
    for seed, lstm in (
        zip(SEEDS, runs[1:], strict=True) if with_composition else ()
    ):
        metadata = {
            arm: {
                "run_dir": entry["run_dir"],
                "training_seed": actual_seed,
                "comparison_lstm_seed": seed,
                "shared_control_fit": arm == "persistence",
                "revision_contract": REVISION_CONTRACT,
                "input_feature_mode": "cgm_mask_clock",
                "score_columns": {
                    str(h): f"severe_hypo_{h}_probability" for h in (30, 60)
                },
            }
            for arm, actual_seed, entry in (
                ("lstm", seed, lstm),
                ("persistence", 17, persistence),
            )
        }
        policies = {}
        for h in (30, 60):
            policy = copy.deepcopy(
                persistence["cut_policies"]["54"][str(h)]["0.5"]["policy"]
            )
            policy["endpoint"]["score_column"] = f"composition_{h}_score"
            policies[h] = policy
        frozen = FrozenRevisionComposition(
            seed,
            source_order,
            {
                (arm, h): FrozenStepECDF((0.2, 0.8), (1, 3), 3)
                for arm in ("lstm", "persistence")
                for h in (30, 60)
            },
            policies,
            metadata,
        )
        recipe = save_frozen_composition(frozen, tmp_path / f"recipe-{seed}")
        recipe_entry = {
            "recipe": _binding(recipe),
            "accepted_development_evidence": freeze["accepted_development_evidence"],
        }
        if cut70:
            recipe_entry["cut70_policies"] = {}
            for h in (30, 60):
                policy = copy.deepcopy(policies[h])
                policy["glucose_cut"] = 70
                evidence = recipe.parent / f"synthetic-cut70-{h}.json"
                _write_json(
                    evidence,
                    {
                        "policy": policy,
                        "policy_origin": "calibration_refit_at_cut",
                        "base54_policy_parity_checked": True,
                    },
                )
                recipe_entry["cut70_policies"][str(h)] = {
                    "policy": policy,
                    "evidence": _binding(evidence),
                }
        recipes[seed] = recipe
        freeze["compositions"].append(recipe_entry)
    _write_json(freeze_path, freeze)
    counts = {
        "normalized_participants": 3,
        "predicted_participants": 3,
        "zero_base_exposure_participants": 0,
        "decision_rows": 3,
        "tail_rows": 0,
        "raw_rows": len(original) // 2 * 3,
    }
    horizon_counts = None if horizon_members is None else {
        str(h): {
            "evaluated_participants": len(horizon_members[h]),
            "zero_base_exposure_participants": 3 - len(horizon_members[h]),
        } for h in (30, 60)
    }
    horizon_metadata = {} if horizon_counts is None else {
        "horizon_participant_counts": horizon_counts
    }
    producers = []
    for entry in runs:
        summary = entry["model_summary"]["snapshot"]
        arm = "lstm" if summary.get("architecture") == "lstm" else "persistence"
        seed = summary["training_seed"]
        output = tmp_path / f"producer-{arm}-{seed}"
        specification = inference.ExternalProducer(
            arm, seed, output, run_dir=Path(entry["run_dir"])
        )
        _producer_ledgers(
            specification, freeze, counts, positivity=positivity, cut70=cut70,
            horizon_members=horizon_members,
        )
        external._write_run_receipt(
            output,
            run_dir=Path(entry["run_dir"]),
            freeze_id=freeze["freeze_id"],
            partition="external_loop",
            source=source,
            summary=summary,
            counts=counts,
            horizon_participant_counts=horizon_counts,
        )
        producers.append(specification)
    for seed in SEEDS if with_composition else ():
        output = tmp_path / f"producer-composition-{seed}"
        specification = inference.ExternalProducer(
            "composition", seed, output, recipe=recipes[seed]
        )
        _producer_ledgers(
            specification, freeze, counts, positivity=positivity, cut70=cut70,
            horizon_members=horizon_members,
        )
        members = {
            arm: _binding(
                tmp_path
                / f"producer-{arm}-{seed if arm == 'lstm' else 17}"
                / "run-receipt.json"
            )
            for arm in ("lstm", "persistence")
        }
        audit = {
            "revision_contract": REVISION_CONTRACT,
            "freeze_id": freeze["freeze_id"],
            "recipe": _binding(recipes[seed]),
            "member_receipts": members,
            "scored_rows": 3,
            "tail_rows": 0,
            "source_normalization_contract": source["source_normalization_contract"],
            "global_cohort_audit": (
                "reused original accepted producer evidence; no new global audit"
            ),
            **horizon_metadata,
        }
        _write_json(output / "private/composition-input-receipt.json", audit)
        receipt = {
            "schema_version": "1.0",
            "producer_kind": "frozen_step_ecdf_composition",
            "status": "complete",
            "partition": "external_loop",
            "source_id": SOURCE,
            "comparison_lstm_seed": seed,
            "counts": counts,
            **audit,
        }
        receipt["files"] = [
            {
                "relative_path": str(path.relative_to(output)),
                **{k: v for k, v in _binding(path).items() if k != "path"},
            }
            for path in [
                *sorted((output / "private").rglob("*.parquet")),
                output / "private/composition-input-receipt.json",
                *sorted((output / "exportable").glob("*")),
                output / "status.json",
            ]
        ]
        external._publish_run_receipt(output, receipt)
        producers.append(specification)
    return freeze_path, producers


def _producer_ledgers(
    specification, freeze, counts, *, positivity, cut70, horizon_members=None,
):
    output = specification.output_dir
    rows = []
    for horizon in (30, 60):
        positive = (positivity or {}).get(horizon, ("p0", "p1"))
        selected = ("p0",) if specification.model_seed in (17, 42, 101) else ("p1",)
        detected = (
            ()
            if specification.arm == "persistence"
            else positive
            if specification.arm == "composition"
            else selected
        )
        for cut in (54, 70) if cut70 else (54,):
            evaluation = _evaluation(
                horizon,
                detected,
                zero_alarm=specification.arm == "composition",
                cut=cut,
                positive=positive,
            )
            if horizon_members is not None:
                def selected(frame, members=horizon_members[horizon]):
                    return frame.loc[
                        frame.participant_id.isin(members)
                    ].copy()

                evaluation = replace(
                    evaluation, participants=selected(evaluation.participants),
                    events=selected(evaluation.events),
                    alarms=selected(evaluation.alarms),
                )
            if cut == 54:
                for ordinal, participant in enumerate(("p0", "p1", "zero"), start=1):
                    _write_parquet(
                        output
                        / f"private/endpoints/events-{horizon}"
                        / f"part-{ordinal:05d}.parquet",
                        evaluation.events.loc[
                            evaluation.events.participant_id.eq(participant),
                            [*IDS, "event_id", "event_onset"],
                        ],
                    )
                    _write_parquet(
                        output
                        / f"private/endpoints/exposure-{horizon}"
                        / f"part-{ordinal:05d}.parquet",
                        evaluation.participants.loc[
                            evaluation.participants.participant_id.eq(participant),
                            [*IDS, "patient_days", "observation_minutes"],
                        ],
                    )
            for budget in (
                ("0.5", "1.0")
                if specification.arm != "composition" and cut == 54
                else ("0.5",)
            ):
                for category in ("participants", "sources", "events", "alarms"):
                    _write_parquet(
                        output
                        / f"private/evaluations/{cut}/{horizon}/{budget}/{category}"
                        / "part-00001.parquet",
                        getattr(evaluation, category),
                    )
                rows.append(
                    {
                        **_aggregate(evaluation.participants),
                        "source_id": SOURCE,
                        "source_count": 1,
                        "glucose_cut": cut,
                        "horizon_minutes": horizon,
                        "nominal_fapd": float(budget),
                        "threshold": 0.6,
                    }
                )
    # Deliberately unreadable prediction payloads prove the CI consumer uses only
    # their native file bindings, not Parquet decoding or an accidental refit.
    for ordinal in range(1, 4):
        path = output / f"private/predictions/part-{ordinal:05d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic-never-read-prediction-payload")
    source = freeze["partitions"]["external_loop"]
    report = {
        "status": "complete",
        "revision_contract": REVISION_CONTRACT,
        "freeze_id": freeze["freeze_id"],
        "partition": "external_loop",
        "external_already_exposed": True,
        "policies": rows,
    }
    horizon_metadata = {} if horizon_members is None else {
        "horizon_participant_counts": {
            str(h): {"evaluated_participants": len(horizon_members[h]),
                     "zero_base_exposure_participants": 3 - len(horizon_members[h])}
            for h in (30, 60)
        }
    }
    report.update(horizon_metadata)
    if specification.arm == "composition":
        report.update(
            comparison_lstm_seed=specification.model_seed,
            fitted_on_external=False,
            scored_rows=3,
            tail_rows=0,
            score_interpretation="rank_mean_not_calibrated_event_probability",
            inherited_probability_or_regression_diagnostics=False,
        )
    else:
        report.update(counts=counts, thresholds_fitted_on_external=False)
        _write_json(
            output / "private/input-audit.json",
            {
                **counts,
                "source_normalization_contract": source[
                    "source_normalization_contract"
                ],
                "accepted_global_cohort_audit": source["accepted_global_cohort_audit"],
                "global_cohort_audit": (
                    "reused frozen accepted evidence; not recomputed per participant"
                ),
                **horizon_metadata,
            },
        )
        _write_json(output / "private/recipe-freeze-used.json", freeze)
    _write_json(output / "exportable/fixed-frame-summary.json", report)
    pd.DataFrame(rows).to_csv(
        output / "exportable/fixed-frame-summary.csv", index=False
    )
    _write_json(
        output / "exportable/alarm-glucose-summary.json",
        {"synthetic_fixture_only": True},
    )
    (output / "exportable/alarm-glucose-summary.csv").write_text(
        "synthetic_fixture_only\ntrue\n"
    )
    _write_json(
        output / "status.json", {"status": "complete", "freeze_id": freeze["freeze_id"]}
    )


def _accepted(
    freeze_path, producers, *, cuts=(54,),
    comparison_scope=inference.LEGACY_COMPARISON_SCOPE,
):
    return inference._accept_producers(
        freeze_path,
        inference._producer_specifications(
            producers, comparison_scope=comparison_scope
        ),
        "external_loop",
        cuts,
        comparison_scope=comparison_scope,
    )


def _rebind_native_file(producer, relative):
    """Explicit synthetic bad-producer mutation; never modifies old real receipts."""
    path = producer.output_dir / "run-receipt.json"
    receipt = json.loads(path.read_text())
    binding = _binding(producer.output_dir / relative)
    for item in receipt["files"]:
        if item["relative_path"] == relative:
            item.update(size_bytes=binding["size_bytes"], mtime_ns=binding["mtime_ns"])
    _write_json(path, receipt)


def test_native_end_to_end_exports_only_aggregates_and_never_reads_predictions(
    tmp_path, monkeypatch
):
    freeze, producers = _accepted_fixture(tmp_path)
    original = pq.ParquetFile

    def guard(path, *args, **kwargs):
        name = str(path)
        assert "private/predictions" not in name
        assert "/alarms/" not in name
        assert "normalized.parquet" not in name
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pq, "ParquetFile", guard)
    output = inference.run_external_paired_inference(
        freeze,
        partition="external_loop",
        producers=producers,
        output_dir=tmp_path / "aggregate",
    )
    report = json.loads((output / "external-paired-inference.json").read_text())
    assert (
        report["bootstrap_replicates"] == 1000 and report["bootstrap_seed"] == 20260810
    )
    assert report["actual_fit_counts"] == {"persistence": 1, "lstm": 5}
    assert report["derived_composition_recipe_count"] == 5
    assert report["native_completed_producer_count"] == 11
    assert report["prediction_or_glucose_records_read"] is False
    assert report["counts"][0]["participant_count"] == 3
    sensitivity = report["metrics"]["fixed_sensitivity"]["contrasts"]
    assert sensitivity["lstm_minus_persistence_at_54"]["estimate"] == 0.5
    assert sensitivity["composition_minus_persistence_at_54"]["estimate"] == 1.0
    assert sensitivity["composition_minus_lstm_at_54"]["estimate"] == 0.5
    assert len(report["descriptive_policies"]) == 22
    text = (output / "external-paired-inference.json").read_text()
    assert all(
        word not in text
        for word in (
            "participant_id",
            "event_onset",
            "cgm_last",
            "run_dir",
            "p0",
            'zero"',
        )
    )
    assert len(pd.read_csv(output / "external-paired-inference.csv")) == 54
    assert not (output / "private").exists()
    with pytest.raises(FileExistsError, match="overwrite"):
        inference.run_external_paired_inference(
            freeze, partition="external_loop", producers=producers, output_dir=output
        )


@pytest.mark.parametrize(
    "defect",
    [
        "missing_seed",
        "extra_seed",
        "wrong_fit_seed",
        "missing_acceptance",
        "wrong_role",
        "unbound_ledger",
        "changed_file",
        "member_receipt",
        "missing_member",
        "missing_receipt",
    ],
)
def test_native_receipt_and_frozen_lineage_defects_fail_closed(tmp_path, defect):
    freeze, producers = _accepted_fixture(tmp_path)
    if defect == "missing_seed":
        producers.pop()
    elif defect == "extra_seed":
        producers.append(producers[-1])
    elif defect == "wrong_fit_seed":
        producers[1] = replace(producers[1], model_seed=23)
    elif defect == "missing_acceptance":
        payload = json.loads(freeze.read_text())
        payload["accepted_development_evidence"] = []
        _write_json(freeze, payload)
    elif defect == "wrong_role":
        path = producers[1].output_dir / "run-receipt.json"
        payload = json.loads(path.read_text())
        payload["partition"] = "confirmatory_ohio"
        _write_json(path, payload)
    elif defect == "unbound_ledger":
        path = producers[1].output_dir / "run-receipt.json"
        payload = json.loads(path.read_text())
        payload["files"] = [
            item
            for item in payload["files"]
            if "/evaluations/" not in item["relative_path"]
        ]
        _write_json(path, payload)
    elif defect == "changed_file":
        path = (
            producers[1].output_dir
            / "private/evaluations/54/30/0.5/participants/part-00001.parquet"
        )
        path.write_bytes(path.read_bytes() + b"changed")
    elif defect in ("member_receipt", "missing_member"):
        path = producers[-1].output_dir / "run-receipt.json"
        payload = json.loads(path.read_text())
        if defect == "member_receipt":
            payload["member_receipts"]["lstm"] = _binding(
                producers[1].output_dir / "run-receipt.json"
            )
        else:
            payload["member_receipts"].pop("persistence")
        _write_json(path, payload)
    else:
        # Renaming our synthetic receipt retains evidence, without deletion.
        (producers[1].output_dir / "run-receipt.json").rename(
            producers[1].output_dir / "incomplete-receipt.json"
        )
    with pytest.raises(ValueError):
        _accepted(freeze, producers)


@pytest.mark.parametrize(
    "defect",
    [
        "missing_patient",
        "duplicate_patient",
        "event_onset",
        "event_id",
        "exposure",
        "burden_rate",
        "detected_count",
    ],
)
def test_bound_native_ledgers_with_exact_rebinding_still_reject_truth_drift(
    tmp_path, defect
):
    freeze, producers = _accepted_fixture(tmp_path)
    producer = producers[-1]  # Composition receipt has no later dependent member.
    if defect in ("event_onset", "event_id"):
        relative = "private/endpoints/events-30/part-00001.parquet"
    elif defect == "exposure":
        relative = "private/endpoints/exposure-30/part-00001.parquet"
    else:
        relative = "private/evaluations/54/30/0.5/participants/part-00001.parquet"
    path = producer.output_dir / relative
    frame = pd.read_parquet(path)
    if defect == "missing_patient":
        frame = frame.iloc[:-1].copy()
    elif defect == "duplicate_patient":
        frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    elif defect == "event_onset":
        frame.loc[0, "event_onset"] += pd.Timedelta(minutes=1)
    elif defect == "event_id":
        frame.loc[0, "event_id"] = "wrong-event-same-count"
    elif defect == "exposure":
        frame.loc[0, "observation_minutes"] += 1
    elif defect == "burden_rate":
        frame.loc[0, "unmatched_alarms_per_patient_day"] = 999.0
    else:
        frame.loc[0, "detected_event_count"] = 0
    _write_parquet(path, frame)
    _rebind_native_file(producer, relative)
    _, source, accepted = _accepted(freeze, producers)
    with pytest.raises(ValueError):
        inference._load_evaluations(accepted, source["source_id"], (54,))


def test_optional_cut70_requires_all_frozen_model_and_composition_recipes(tmp_path):
    freeze, producers = _accepted_fixture(tmp_path / "missing")
    with pytest.raises(ValueError, match="explicitly frozen"):
        _accepted(freeze, producers, cuts=(54, 70))
    freeze, producers = _accepted_fixture(tmp_path / "complete", cut70=True)
    _, source, accepted = _accepted(freeze, producers, cuts=(54, 70))
    values, points = inference._load_evaluations(
        accepted, source["source_id"], (54, 70)
    )
    assert len(values) == 3 * 2 * 2 * 5
    assert len(points) == 11 * 2 * 2
    payload = json.loads(freeze.read_text())
    payload["compositions"][-1].pop("cut70_policies")
    # Isolate development recipe acceptance before raw receipt snapshot checking.
    last = producers[-1]
    with pytest.raises(ValueError, match="cut70 composition"):
        inference._composition_policies(
            last,
            payload,
            accepted,
            tuple(
                sorted(
                    payload["runs"][0]["config"]["snapshot"]["data"][
                        "development_sources"
                    ]
                )
            ),
            (54, 70),
        )


def test_no_confirmatory_partition_and_no_arbitrary_cut_or_budget(tmp_path):
    freeze, producers = _accepted_fixture(tmp_path)
    with pytest.raises(ValueError, match="already exposed"):
        inference._accept_producers(freeze, producers, "confirmatory_ohio", (54,))
    for cuts in ((70,), (54, 65), (54, 54), (70, 54)):
        with pytest.raises(ValueError, match="main cut"):
            inference._cuts(cuts)


def _lstm_control_records():
    return [
        {
            "arm": arm, "model_seed": seed,
            "output_dir": f"/synthetic/output-{arm}-{seed}",
            "run_dir": f"/synthetic/run-{arm}-{seed}",
        }
        for arm, seeds in (("persistence", (17,)), ("lstm", SEEDS))
        for seed in seeds
    ]


def test_explicit_lstm_control_manifest_accepts_exact_six_producers():
    records = _lstm_control_records()
    specifications = inference._manifest_producers(
        {"comparison_scope": "lstm-control", "producers": records},
        comparison_scope="lstm-control",
    )
    assert [(p.arm, p.model_seed) for p in specifications] == [
        (record["arm"], record["model_seed"]) for record in records
    ]
    assert all(p.recipe is None for p in specifications)
    # The historical default still requires all five real composition producers.
    with pytest.raises(ValueError, match="LSTM/composition"):
        inference._manifest_producers({"producers": records})


@pytest.mark.parametrize("defect", [
    "missing_seed", "duplicate_seed", "wrong_seed", "bool_seed",
    "wrong_control_seed", "composition_placeholder", "extra_producer_field",
    "missing_scope", "wrong_scope", "extra_manifest_field", "nonlist_producers",
])
def test_reduced_manifest_known_defects_fail_closed(defect):
    records = _lstm_control_records()
    manifest = {"comparison_scope": "lstm-control", "producers": records}
    if defect == "missing_seed":
        records.pop()
    elif defect == "duplicate_seed":
        records.append(copy.deepcopy(records[-1]))
    elif defect == "wrong_seed":
        records[-1]["model_seed"] = 102
    elif defect == "bool_seed":
        records[-1]["model_seed"] = True
    elif defect == "wrong_control_seed":
        records[0]["model_seed"] = 23
    elif defect == "composition_placeholder":
        records.append({
            "arm": "composition", "model_seed": 17,
            "output_dir": "/synthetic/unused-composition", "recipe": "/missing",
        })
    elif defect == "extra_producer_field":
        records[0]["recipe"] = None
    elif defect == "missing_scope":
        manifest.pop("comparison_scope")
    elif defect == "wrong_scope":
        manifest["comparison_scope"] = "legacy-with-composition"
    elif defect == "extra_manifest_field":
        manifest["ignored"] = True
    else:
        manifest["producers"] = {"producers": records}
    with pytest.raises(ValueError):
        inference._manifest_producers(manifest, comparison_scope="lstm-control")


def test_two_method_arithmetic_matches_historical_joint_draws_exactly():
    legacy = _joint_fixture(seeds=SEEDS, cuts=(54, 70))
    values = {key: value for key, value in legacy.items() if key.arm != "composition"}
    kwargs = {
        "source_order": (SOURCE,), "glucose_cuts": (54, 70),
        "replicates": 37, "include_replicates": True,
        "participant_groups": {
            (SOURCE, participant): participant for participant in ("p0", "p1", "zero")
        },
    }
    historical = inference.paired_external_inference(legacy, **kwargs)
    reduced = inference.paired_external_inference(
        values, **kwargs, comparison_scope="lstm-control"
    )
    assert reduced["comparison_arms"] == ["persistence", "lstm"]
    assert reduced["counts"] == historical["counts"]
    for metric, payload in reduced["metrics"].items():
        assert len(payload["contrasts"]) == 7
        for name, contrast in payload["contrasts"].items():
            assert "composition" not in name
            if name == "lstm_minus_persistence_contrast_change_70_minus_54":
                assert set(contrast["replicate_estimates"]) == {0.0}
            else:
                assert contrast == historical["metrics"][metric]["contrasts"][name]


def test_cross_cut_change_uses_paired_draws_instead_of_marginal_interval_endpoints(
    monkeypatch,
):
    values = {}
    for horizon in (30, 60):
        for cut in (54, 70):
            for seed in SEEDS:
                for arm in ("persistence", "lstm"):
                    detected = () if arm == "persistence" else (
                        ("p0",) if cut == 54 else ("p0", "p1")
                    )
                    values[FixedFrameKey(arm, cut, horizon, seed)] = _evaluation(
                        horizon, detected, cut=cut, positive=("p0", "p1", "zero"),
                    )

    class OracleDraw:
        def __init__(self):
            self.draws = iter(([0, 0, 0], [1, 1, 1], [0, 1, 2]))

        def choice(self, options, *, size, replace):
            assert size == len(options) == 3 and replace
            return options[next(self.draws)]

    monkeypatch.setattr(np.random, "default_rng", lambda _: OracleDraw())
    result = inference.paired_external_inference(
        values, source_order=(SOURCE,), glucose_cuts=(54, 70),
        comparison_scope="lstm-control", replicates=3, include_replicates=True,
    )
    contrasts = result["metrics"]["fixed_sensitivity"]["contrasts"]
    change = contrasts["lstm_minus_persistence_contrast_change_70_minus_54"]
    expected = [0.0, 1.0, 1 / 3]
    np.testing.assert_allclose(change["replicate_estimates"], expected)
    assert change["estimate"] == pytest.approx(1 / 3)
    np.testing.assert_allclose(
        [change["ci_low"], change["ci_high"]], np.percentile(expected, [2.5, 97.5])
    )
    for horizon in (30, 60):
        np.testing.assert_allclose(
            [point[0 if horizon == 30 else 1] for point in
             change["replicate_horizon_seed_estimates"]],
            np.tile(np.array(expected)[:, None], (1, 5)),
        )
        assert change["per_horizon"][horizon]["estimate"] == pytest.approx(1 / 3)
    np.testing.assert_allclose(list(change["per_model_seed"].values()), 1 / 3)
    at54 = contrasts["lstm_minus_persistence_at_54"]
    at70 = contrasts["lstm_minus_persistence_at_70"]
    assert change["ci_low"] != pytest.approx(at70["ci_low"] - at54["ci_high"])
    assert change["ci_high"] != pytest.approx(at70["ci_high"] - at54["ci_low"])


@pytest.mark.parametrize("defect", [
    "missing_seed", "missing_cut", "missing_horizon", "extra_composition",
    "control_alias", "partial_seed_order", "cut54_only",
])
def test_two_method_arithmetic_preserves_complete_matrix_and_control_checks(defect):
    values = _joint_fixture(
        seeds=SEEDS, cuts=(54, 70), with_composition=False,
    )
    cuts, seeds = (54, 70), SEEDS
    if defect == "missing_seed":
        values.pop(FixedFrameKey("lstm", 54, 30, 101))
    elif defect == "missing_cut":
        values = {key: value for key, value in values.items() if key.glucose_cut != 70}
    elif defect == "missing_horizon":
        values = {
            key: value for key, value in values.items() if key.horizon_minutes != 60
        }
    elif defect == "extra_composition":
        values[FixedFrameKey("composition", 54, 30, 17)] = _evaluation(30, ())
    elif defect == "control_alias":
        values[FixedFrameKey("persistence", 70, 60, 101)] = _evaluation(
            60, ("p0",), cut=70
        )
    elif defect == "partial_seed_order":
        seeds = (17, 23)
    else:
        cuts = (54,)
    with pytest.raises(ValueError):
        inference.paired_external_inference(
            values, source_order=(SOURCE,), glucose_cuts=cuts,
            model_seed_order=seeds, comparison_scope="lstm-control", replicates=3,
        )


@pytest.mark.parametrize("grouped", [False, True])
def test_native_two_method_export_needs_no_composition_and_keeps_group_sensitivity(
    tmp_path, monkeypatch, grouped,
):
    freeze_path, producers = _accepted_fixture(
        tmp_path / "native", cut70=True, with_composition=False,
        horizon_members=(
            {30: ("p0", "p1", "zero"), 60: ("p0", "zero")} if grouped else None
        ),
    )
    freeze = json.loads(freeze_path.read_text())
    assert "compositions" not in freeze
    assert len(producers) == 6

    def unused_composition(*args, **kwargs):
        pytest.fail("two-method inference must not require composition artifacts")

    original = pq.ParquetFile

    def ledger_only(path, *args, **kwargs):
        assert "private/predictions" not in str(path)
        assert "normalized.parquet" not in str(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(inference, "_composition_policies", unused_composition)
    monkeypatch.setattr(pq, "ParquetFile", ledger_only)
    group_manifest = None
    if grouped:
        from test_revision_group_inference import _group_files

        group_manifest, _ = _group_files(
            tmp_path / "private-inputs", freeze_id=freeze["freeze_id"],
            source=freeze["partitions"]["external_loop"],
        )
    output = inference.run_external_paired_inference(
        freeze_path, partition="external_loop", producers=producers,
        output_dir=tmp_path / "aggregate", glucose_cuts=(54, 70),
        group_manifest_path=group_manifest, comparison_scope="lstm-control",
    )
    report = json.loads((output / "external-paired-inference.json").read_text())
    assert report["comparison_scope"] == "lstm-control"
    assert report["comparison_arms"] == ["persistence", "lstm"]
    assert report["glucose_cuts"] == [54, 70]
    assert report["model_seeds"] == list(SEEDS)
    assert report["bootstrap_replicates"] == 1000
    assert report["bootstrap_seed"] == 20260810
    assert report["nominal_fapd"] == 0.5
    assert report["actual_fit_counts"] == {"persistence": 1, "lstm": 5}
    assert report["native_completed_producer_count"] == 6
    assert report["derived_composition_recipe_count"] == 0
    assert len(report["descriptive_policies"]) == 24
    analyses = [report]
    if grouped:
        assert report["correlation_handling"]["full_group_count"] == 2
        assert [row["participant_count"] for row in report["counts"]] == [3, 2]
        excluded = report["named_sensitivities"]["exclude_verified_pair_records"]
        analyses.append(excluded)
        assert excluded["comparison_scope"] == "lstm-control"
        assert excluded["thresholds_refit"] is excluded["fitted_on_external"] is False
        assert {row["participant_count"] for row in excluded["counts"]} == {1}
        assert len(excluded["descriptive_policies"]) == 24
    for analysis in analyses:
        assert {row["arm"] for row in analysis["descriptive_policies"]} == {
            "persistence", "lstm"
        }
        for payload in analysis["metrics"].values():
            assert len(payload["contrasts"]) == 7
            assert all("composition" not in name for name in payload["contrasts"])
            assert payload["bootstrap_seed"] == 20260810
        assert all(
            row["nominal_fapd"] == 0.5 for row in analysis["descriptive_policies"]
        )
    csv = pd.read_csv(output / "external-paired-inference.csv")
    assert len(csv) == (126 if grouped else 63)
    assert not (output / "private").exists()


def test_reduced_production_requires_both_frozen_cuts_before_reading_inputs(tmp_path):
    producers = inference._producer_specifications(
        _lstm_control_records(), comparison_scope="lstm-control"
    )
    with pytest.raises(ValueError, match="cuts54/70"):
        inference.run_external_paired_inference(
            tmp_path / "not-read.json", partition="external_loop", producers=producers,
            output_dir=tmp_path / "not-created", comparison_scope="lstm-control",
        )
    assert not (tmp_path / "not-created").exists()


@pytest.mark.parametrize(
    "comparison_scope", ["legacy-with-composition", "lstm-control"]
)
def test_cli_selects_scope_and_manifest_contract(
    tmp_path, monkeypatch, comparison_scope,
):
    records = _lstm_control_records()
    if comparison_scope == "legacy-with-composition":
        records.extend({
            "arm": "composition", "model_seed": seed,
            "output_dir": f"/synthetic/composition-{seed}",
            "recipe": f"/synthetic/recipe-{seed}",
        } for seed in SEEDS)
    manifest = {"producers": records}
    arguments = [
        "inference", "--freeze", str(tmp_path / "freeze.json"),
        "--manifest", str(tmp_path / "manifest.json"),
        "--partition", "external_loop", "--output", str(tmp_path / "aggregate"),
    ]
    if comparison_scope == "lstm-control":
        manifest["comparison_scope"] = comparison_scope
        arguments += [
            "--comparison-scope", comparison_scope,
            "--glucose-cut", "54", "--glucose-cut", "70",
        ]
    _write_json(tmp_path / "manifest.json", manifest)
    calls = []
    monkeypatch.setattr(sys, "argv", arguments)
    monkeypatch.setattr(
        inference, "run_external_paired_inference",
        lambda *args, **kwargs: calls.append(kwargs),
    )
    inference.main()
    assert len(calls) == 1
    assert calls[0]["comparison_scope"] == comparison_scope
    reduced = comparison_scope == "lstm-control"
    assert len(calls[0]["producers"]) == (6 if reduced else 11)
    assert calls[0]["glucose_cuts"] == ((54, 70) if reduced else (54,))
