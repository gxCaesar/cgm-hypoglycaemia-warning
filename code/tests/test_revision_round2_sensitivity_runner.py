"""Accepted synthetic cut-grid consumers; no fitting or external participant data."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml
from test_revision_event_sensitivity import _accepted_fixture, _binding
from test_revision_inputs import fixture

from glycoshift.fixed_frame_evaluation import FixedFrameSpec, evaluate_fixed_frame
from glycoshift.revision_event_sensitivity import validate_event_sensitivity_inputs
from glycoshift.revision_external import _parquet
from glycoshift.revision_inputs import CONFIRMATION_CONTRACT, build_participant_replay
from glycoshift.revision_round2_sensitivity_runner import (
    METHODS,
    MODEL_SEEDS,
    run_round2_sensitivity,
)
from glycoshift.sequences import REVISION_CONTRACT


def _accepted_round2(tmp_path):
    normalized, prepared, analysis = (
        tmp_path / name for name in ("normalized", "prepared", "analysis")
    )
    for directory in (normalized, prepared, analysis / "exportable"):
        directory.mkdir(parents=True)
    frame, replay_config = fixture()
    replay_config.update(
        project={"synthetic": True},
        split={"validation_fraction": 0.3},
        model={},
        evaluation={},
        output={},
    )
    replay_config["protocol"]["ascertainment_contract"] = CONFIRMATION_CONTRACT
    replay_config["data"].update(
        development_sources=["A", "B"], forbidden_development_sources=["OhioT1DM"]
    )
    manifest, normalized_files, prepared_files, replay_inputs = [], [], [], []
    for index, source in enumerate(("A", "B")):
        part = frame.copy()
        part["source_id"], part["participant_id"] = source, f"private-person-{index}"
        part.loc[20:23, "cgm_mg_dl"] = 50
        if index == 0:
            part.loc[20:23, ["cgm_observed", "cgm_interpolated"]] = [False, True]
        path = normalized / f"person-{index}.parquet"
        part.to_parquet(path, index=False)
        normalized_files.append(_binding(path))
        replay = build_participant_replay(part, replay_config, partition="selection")
        timeline = prepared / f"timeline-{index}.parquet"
        replay.timeline.to_parquet(timeline, index=False)
        prepared_files.append(timeline)
        manifest.append(
            {
                "source_id": source,
                "participant_id": f"private-person-{index}",
                "partition": "selection",
                "timeline_path": str(timeline),
                "stored_rows": len(replay.timeline),
                "normalized_paths": [str(path)],
            }
        )
        replay_inputs.append(replay)
    endpoints = {}
    for horizon in (30, 60):
        endpoints[f"selection/{horizon}"] = {}
        for category in ("events", "exposure"):
            path = prepared / f"{category}-{horizon}.parquet"
            _parquet(
                path,
                pd.concat(
                    [getattr(r, category)[horizon] for r in replay_inputs],
                    ignore_index=True,
                ),
            )
            endpoints[f"selection/{horizon}"][category] = str(path)
            prepared_files.append(path)
    # The consumer must not open training values while discovering metadata.
    manifest.append(
        {
            "source_id": "A",
            "participant_id": "unopened-train",
            "partition": "train",
            "timeline_path": str(prepared / "absent.parquet"),
        }
    )
    manifest_path = prepared / "participant-manifest.private.json"
    manifest_path.write_text(json.dumps({"participants": manifest}))
    audit = {
        "status": "passed",
        "revision_contract": REVISION_CONTRACT,
        "ascertainment_contract": CONFIRMATION_CONTRACT,
        "source_order": ["A", "B"],
        "normalized_path": str(normalized),
        "legacy_windows": str(tmp_path / "retained-legacy"),
        "timeline_storage": "original_partitions",
        "original_participants": 3,
        "endpoint_paths": endpoints,
    }
    audit_path = prepared / "input-audit.json"
    audit_path.write_text(json.dumps(audit))
    prepared_files.extend([manifest_path, audit_path])
    config_path = tmp_path / "replay.yaml"
    config_path.write_text(yaml.safe_dump(replay_config))
    evidence = tmp_path / "accepted-evidence.json"
    evidence.write_text(json.dumps({"synthetic_only": True}))
    runs, analyses = [], []
    for arm in METHODS:
        for seed in MODEL_SEEDS if arm in {"lstm", "tcn"} else (None,):
            run = tmp_path / "fits" / arm / ("shared" if seed is None else str(seed))
            run.mkdir(parents=True)
            summary_path = run / "model_summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "revision_contract": REVISION_CONTRACT,
                        "architecture" if arm in {"lstm", "tcn"} else "baseline": arm,
                        "training_seed": 17 if seed is None else seed,
                    }
                )
            )
            cohort_path = run / "cohort_audit.json"
            cohort_path.write_text(json.dumps(audit))
            conditions = []
            for horizon in (30, 60):
                for cut in (54, 70):
                    directory = (
                        analysis
                        / "private"
                        / arm
                        / ("shared" if seed is None else str(seed))
                        / "refitted"
                        / str(cut)
                        / str(horizon)
                    )
                    directory.mkdir(parents=True)
                    outputs = {
                        category: []
                        for category in ("alarms", "events", "participants")
                    }
                    for replay in replay_inputs:
                        scored = replay.timeline.copy()
                        scored["base_evaluable"] = scored[f"base_evaluable_{horizon}"]
                        scored["frozen_score"] = scored.context_end.isin(
                            frame.timestamp.iloc[[16, 70]]
                        ).astype(float)
                        evaluation = evaluate_fixed_frame(
                            scored,
                            replay.events[horizon],
                            replay.exposure[horizon],
                            spec=FixedFrameSpec(
                                "frozen_score",
                                horizon,
                                (str(replay.timeline.source_id.iloc[0]),),
                            ),
                            threshold=0.5,
                            glucose_cut=cut,
                        )
                        for category in outputs:
                            outputs[category].append(getattr(evaluation, category))
                    condition = {
                        "horizon_minutes": horizon,
                        "glucose_cut": cut,
                        "nominal_fapd": 0.5,
                        "mode": "refitted",
                    }
                    for category, tables in outputs.items():
                        path = directory / f"{category}.parquet"
                        _parquet(path, pd.concat(tables, ignore_index=True))
                        condition[category] = _binding(path)
                    conditions.append(condition)
                    analyses.append(
                        {
                            "arm": arm,
                            "model_seed": seed,
                            "horizon_minutes": horizon,
                            "glucose_cut": cut,
                            "mode": "refitted",
                        }
                    )
            runs.append(
                {
                    "arm": arm,
                    "model_seed": seed,
                    "run_dir": str(run),
                    "model_summary": _binding(summary_path),
                    "cohort_audit": _binding(cohort_path),
                    "accepted_evidence": [_binding(evidence)],
                    "ledger_root": str(analysis / "private"),
                    "conditions": conditions,
                }
            )
    summary = analysis / "exportable/cut-grid-summary.json"
    summary.write_text(
        json.dumps(
            {
                "status": "complete",
                "revision_contract": REVISION_CONTRACT,
                "source_order": ["A", "B"],
                "model_seeds": MODEL_SEEDS,
                "nominal_fapd": 0.5,
                "evaluation_partition": "selection",
                "threshold_partition": "calibration",
                "unique_fit_count": 14,
                "analyses": analyses,
            }
        )
    )
    started = max(binding["mtime_ns"] for binding in normalized_files) + 1
    return {
        "revision_contract": REVISION_CONTRACT,
        "partition": "selection",
        "prepared_dir": str(prepared),
        "replay_config": _binding(config_path),
        "preparation_started_at": pd.Timestamp(started, tz="UTC").isoformat(),
        "output_dir": str(tmp_path / "result"),
        "analysis_summary": _binding(summary),
        "input_acceptance": {
            "status": "accepted",
            "binding_time": "current_sensitivity_acceptance",
            "evidence": [_binding(evidence)],
            "normalized_files": normalized_files,
            "prepared_files": [_binding(path) for path in prepared_files],
        },
        "runs": runs,
    }


def test_extracted_input_context_preserves_old_acceptance_without_output(tmp_path):
    config = _accepted_fixture(tmp_path)
    context = validate_event_sensitivity_inputs(config)
    assert context.sources == ("s",)
    assert len(context.selected) == 2
    assert context.output == Path(config["output_dir"])
    assert not context.output.exists()


@pytest.mark.parametrize("defect", ["missing_normalized", "changed_binding"])
def test_extracted_context_rejects_known_binding_defects_without_output(
    tmp_path, defect
):
    config = _accepted_fixture(tmp_path)
    if defect == "missing_normalized":
        config["input_acceptance"]["normalized_files"].pop()
        expected = "normalized bindings differ"
    else:
        Path(config["replay_config"]["path"]).touch()
        expected = "binding changed"
    with pytest.raises(ValueError, match=expected):
        validate_event_sensitivity_inputs(config)
    assert not Path(config["output_dir"]).exists()


def test_round2_runner_end_to_end_uses_fixed_ledgers_and_exports_no_participant_data(
    tmp_path, monkeypatch
):
    config = _accepted_round2(tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("the accepted-ledger consumer must not fit or regenerate emissions")

    monkeypatch.setattr(
        "glycoshift.fixed_frame_evaluation.evaluate_fixed_frame", forbidden
    )
    monkeypatch.setattr(
        "glycoshift.fixed_frame_evaluation.fit_fixed_frame_alarm_policy", forbidden
    )
    output = run_round2_sensitivity(config)
    summary = json.loads((output / "exportable/summary.json").read_text())
    assert summary["fit_count"] == 14 and summary["condition_count"] == 56
    assert not summary["fitted_or_reemitted"]
    pair = output / "exportable/primary-vs-observed_only"
    audit = json.loads((pair / "support-audit.json").read_text())
    assert audit["common_source_order"] == ["B"]
    points = json.loads((pair / "points.json").read_text())
    full_observed = [
        row
        for row in points
        if row["scope"] == "full_sources" and row["catalogue"] == "observed_only"
    ]
    assert all(row["fixed_sensitivity"] is None for row in full_observed)
    aggregate = json.loads((pair / "all-method-aggregates.json").read_text())
    assert {row["arm"] for row in aggregate} == set(METHODS)
    elapsed = output / "exportable/primary-vs-elapsed_span"
    assert json.loads((elapsed / "support-audit.json").read_text())[
        "common_source_order"
    ] == ["A", "B"]
    private = output / "private/evaluations/elapsed_span/elapsed_span/run-000/54/60"
    events = pd.read_parquet(private / "events.parquet")
    assert "right_censored" in events and not events.right_censored.any()
    alarms = pd.read_parquet(private / "alarms.parquet")
    assert len(alarms) == 4
    assert alarms.base_evaluable.tolist() == [True, False, True, False]
    assert not alarms.loc[~alarms.base_evaluable, "unmatched"].any()
    for path in (output / "exportable").rglob("*.json"):
        assert "private-person" not in path.read_text()
        assert str(tmp_path) not in path.read_text()
    with pytest.raises(FileExistsError):
        run_round2_sensitivity(config)


@pytest.mark.parametrize(
    "defect", ["binding", "mask", "missing_fit", "missing_condition", "mode", "cohort"]
)
def test_round2_runner_fails_closed_on_unaccepted_or_incomplete_inputs(
    tmp_path, defect
):
    config = _accepted_round2(tmp_path)
    if defect == "binding":
        path = Path(config["input_acceptance"]["normalized_files"][0]["path"])
        path.touch()
    elif defect == "mask":
        binding = next(
            binding for binding in config["input_acceptance"]["prepared_files"]
            if "timeline-0" in binding["path"]
        )
        path = Path(binding["path"])
        table = pd.read_parquet(path)
        table.loc[0, "hypo_60_training_mask"] = not bool(
            table.loc[0, "hypo_60_training_mask"]
        )
        table.to_parquet(path, index=False)
        binding.update(_binding(path))
    elif defect == "missing_fit":
        config["runs"].pop()
    elif defect == "missing_condition":
        config["runs"][0]["conditions"].pop()
    elif defect == "mode":
        config["runs"][0]["conditions"][0]["mode"] = "frozen_base54"
    else:
        path = Path(config["runs"][0]["cohort_audit"]["path"])
        payload = json.loads(path.read_text())
        payload["crossfit_contract"] = "unexpected_oof_fit"
        path.write_text(json.dumps(payload))
        config["runs"][0]["cohort_audit"] = _binding(path)
    expected = {
        "binding": "binding changed",
        "mask": "confirmation/training mask mismatch",
        "missing_fit": "fourteen actual fits",
        "missing_condition": "both original horizons",
        "mode": "already-refitted",
        "cohort": "corrected-original pool",
    }
    with pytest.raises(ValueError, match=expected[defect]):
        run_round2_sensitivity(config)
    status = Path(config["output_dir"]) / "status.json"
    assert not status.exists() or json.loads(status.read_text())["status"] == "failed"
