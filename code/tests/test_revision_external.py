"""Synthetic frozen-recipe contracts; no real freeze, source data or fit is made."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch import nn

from glycoshift import calibration, fixed_frame_evaluation, pipeline, windows
from glycoshift import revision_external as external
from glycoshift.config import load_config, save_config
from glycoshift.schema import ACTION_FIELDS
from glycoshift.sequences import REVISION_CONTRACT, TASKS, SequenceScaler


class FrozenClassicalFixture:
    def predict_proba(self, features):
        return np.tile([0.25, 0.75], (len(features), 1))


class FrozenDeepFixture(nn.Module):
    supports_memory = False

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(d_model=4, memory_k=0)

    def encode_sequence(self, sequence):
        return sequence[:, :, :4]

    def forward_from_representation(self, representation, **kwargs):
        value = torch.full(
            (len(representation),), np.log(3), device=representation.device
        )
        return {
            "logits": {task: value for task in TASKS},
            "regression": {str(h): torch.zeros_like(value) for h in (30, 60)},
        }


def _binding(path):
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _normalized(*, lower_censor=True, observed=True):
    blocks = []
    for participant in ("synthetic-B", "synthetic-A"):
        times = pd.date_range("2026-01-01", periods=72, freq="5min", tz="UTC")
        values = np.full(72, 120.0)
        if participant == "synthetic-A":
            values[20:23] = 50.0
        frame = pd.DataFrame(
            {
                "source_id": "Loop Observational Study",
                "participant_id": participant,
                "timestamp": times,
                "cgm_mg_dl": values,
                "cgm_interpolated": False,
            }
        )
        if lower_censor:
            frame["cgm_lower_censored"] = False
        if observed:
            frame["cgm_observed"] = True
        for field in ACTION_FIELDS:
            frame[field] = np.nan
            frame[f"{field}_available"] = False
            frame[f"{field}_observed"] = False
            frame[f"{field}_zero_ambiguous"] = False
        blocks.append(frame)
    return pd.concat(blocks, ignore_index=True)


def _policy(config, cut, horizon, budget):
    # A declared synthetic training policy; no calibration fit is performed here.
    return {
        "partition": "calibration",
        "endpoint": {
            "score_column": f"severe_hypo_{horizon}_probability",
            "horizon_minutes": horizon,
            "source_order": sorted(config["data"]["development_sources"]),
        },
        "glucose_cut": cut,
        "nominal_fapd": float(budget),
        "threshold": 0.6,
        "threshold_count": 5,
        "candidate_rule": "2001_online_eligible_quantiles_plus_no_alarm_sentinel",
        "tie_rule": "maximum_fixed_sensitivity_then_highest_threshold",
        "achieved_equal_source_participant_macro_fapd": 0.25,
        "achieved_equal_source_participant_macro_fixed_sensitivity": 0.5,
        "participant_count": 2,
        "base_event_count": 2,
        "patient_days": 2,
        "observation_minutes": 600.0,
    }


def _fixture(
    tmp_path,
    *,
    families=("persistence", "lstm"),
    cut70=True,
    lower_censor=True,
    observed=True,
    absent_contract=False,
    partition="external_loop",
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    normalized = tmp_path / "normalized.parquet"
    source_id = external._ALLOWED[partition]
    origin = "metabonet-public" if partition == "external_loop" else "mendeley-direct"
    _normalized(lower_censor=lower_censor, observed=observed).assign(
        source_id=source_id
    ).to_parquet(normalized, index=False)
    manifest = tmp_path / "source.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "release_id": "synthetic-fixture",
                "input_file": "normalized.parquet",
                "sha256": "0" * 64,
                "source_map": "synthetic-map",
                "sources": {source_id: {}},
                "role": partition,
                "origin": origin,
            }
        )
    )
    evidence = tmp_path / "synthetic-accepted-evidence.json"
    evidence.write_text(json.dumps({"synthetic_fixture_only": True}))
    runs = []
    for family in families:
        run = tmp_path / family
        run.mkdir()
        config = load_config(Path(__file__).parents[1] / "configs/smoke.yaml")
        config["protocol"]["context_hours"] = 0.25
        config["data"].update(minimum_days=1, minimum_day_coverage=0.1)
        config["data"]["evaluation_inputs"] = {
            partition: {
                "input_path": str(normalized),
                "source_manifest": str(manifest),
                "expected_source": source_id,
                "required_origin": origin,
            }
        }
        config["model"]["input_feature_mode"] = "cgm_mask_clock"
        config["training"].update(
            device="cpu",
            batch_size=16,
            num_workers=0,
            revision_contract=REVISION_CONTRACT,
        )
        config["output"]["run_dir"] = str(run)
        save_config(config, run / "config_resolved.yaml")
        tasks = {}
        cuts = {"54": {}}
        if cut70:
            cuts["70"] = {}
        for task in TASKS:
            kind, horizon_text = task.split("_")
            tasks[task] = {
                "temperature": 2.0,
                "temperature_fit_partition": "calibration",
                "threshold_partition": "calibration",
                "features": ["risk_score"],
                "target_column": f"severe_hypo_{horizon_text}_label"
                if kind == "hypo"
                else f"{task}_label",
            }
        for horizon in (30, 60):
            policies = {
                budget: _policy(config, 54, horizon, budget)
                for budget in ("0.5", "1.0")
            }
            tasks[f"hypo_{horizon}"]["fixed_frame_policies"] = policies
            cuts["54"][str(horizon)] = {
                budget: {"policy": value} for budget, value in policies.items()
            }
            if cut70:
                policy = _policy(config, 70, horizon, "0.5")
                policy_file = run / f"synthetic-dev-policy-70-{horizon}.json"
                policy_file.write_text(
                    json.dumps(
                        {
                            "policy": policy,
                            "policy_origin": "calibration_refit_at_cut",
                            "base54_policy_parity_checked": True,
                        }
                    )
                )
                cuts["70"][str(horizon)] = {
                    "0.5": {"policy": policy, "evidence": _binding(policy_file)}
                }
        summary = {
            "revision_contract": REVISION_CONTRACT,
            "input_feature_mode": "cgm_mask_clock",
            "tasks": tasks,
            "continuous_prediction_available": family == "persistence",
        }
        if family == "lstm":
            summary["architecture"] = "lstm"
            (run / "ckpts").mkdir()
            (run / "ckpts/best.pt").write_bytes(b"synthetic-loader-is-patched")
            (run / "scaler.json").write_text(
                json.dumps(
                    {
                        "cgm_mg_dl": {"mean": 100.0, "std": 10.0},
                        **{field: {"mean": 0.0, "std": 1.0} for field in ACTION_FIELDS},
                    }
                )
            )
            artifacts = [_binding(run / "ckpts/best.pt"), _binding(run / "scaler.json")]
        else:
            summary["baseline"] = "persistence"
            joblib.dump(
                {
                    task: {
                        "model": FrozenClassicalFixture(),
                        "features": ["risk_score"],
                        "baseline": "persistence",
                    }
                    for task in TASKS
                },
                run / "baseline_models.joblib",
            )
            artifacts = [_binding(run / "baseline_models.joblib")]
        (run / "model_summary.json").write_text(json.dumps(summary))
        frozen_config = {
            key: value for key, value in config.items() if key != "_config_path"
        }
        runs.append(
            {
                "run_dir": str(run),
                "config": {
                    "binding": _binding(run / "config_resolved.yaml"),
                    "snapshot": frozen_config,
                },
                "model_summary": {
                    "binding": _binding(run / "model_summary.json"),
                    "snapshot": summary,
                },
                "model_artifacts": artifacts,
                "cut_policies": cuts,
            }
        )
    source = {
        "source_id": source_id,
        "input_path": str(normalized),
        "source_manifest": _binding(manifest),
        "required_origin": origin,
        "normalized_inputs": [_binding(normalized)],
        "accepted_normalization_evidence": [_binding(evidence)],
        "accepted_global_cohort_audit": [_binding(evidence)],
        "source_normalization_contract": {
            "lower_censoring_absent_means_false": absent_contract,
            "source_preprocessing_limits": (
                "Synthetic fixture only; no global/clinical audit claimed."
            ),
        },
    }
    freeze = {
        "revision_contract": REVISION_CONTRACT,
        "status": "development_recipe_frozen",
        "freeze_id": "synthetic-fixture-not-a-real-freeze",
        "accepted_development_evidence": [_binding(evidence)],
        "runs": runs,
        "partitions": {partition: source},
    }
    path = tmp_path / "synthetic-freeze.json"
    path.write_text(json.dumps(freeze))
    return path, runs, normalized


def _forbid_fit(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("external fitting/legacy window rebuilding is forbidden")

    for obj, name in (
        (SequenceScaler, "fit"),
        (StandardScaler, "fit"),
        (LogisticRegression, "fit"),
        (calibration, "fit_temperature"),
        (fixed_frame_evaluation, "fit_fixed_frame_alarm_policy"),
        (pipeline, "prepare_evaluation_dataset"),
        (windows, "build_windows"),
    ):
        monkeypatch.setattr(obj, name, forbidden)


def _deep_loader(run_dir, *, device):
    return FrozenDeepFixture().to(device).eval(), json.loads(
        (run_dir / "model_summary.json").read_text()
    )


def test_frozen_control_and_deep_share_independent_full_timeline_without_fit(
    tmp_path, monkeypatch
):
    freeze, runs, _ = _fixture(tmp_path)
    _forbid_fit(monkeypatch)
    monkeypatch.setattr(external, "_load_frozen_model", _deep_loader)
    tables = []
    for entry in runs:
        run = Path(entry["run_dir"])
        output = tmp_path / f"external-{run.name}"
        external.run_revision_external(
            freeze,
            run_dir=run,
            partition="external_loop",
            output_dir=output,
            batch_rows=17,
        )
        prediction = pd.read_parquet(output / "private/predictions")
        events = pd.read_parquet(output / "private/endpoints/events-30")
        assert len(events) == 1
        assert events.participant_id.iloc[0] == "synthetic-A"
        participant_metrics = pd.read_parquet(
            output / "private/evaluations/54/30/0.5/participants"
        )
        assert len(participant_metrics) == 2
        tables.append(prediction)
        assert len(prediction) == 140
        assert prediction.context_end.max() == pd.Timestamp("2026-01-01T05:55:00Z")
        assert (~prediction.base_evaluable).sum() == 24
        assert (
            prediction.loc[
                prediction.participant_id.eq("synthetic-B"), "hypo_30_event_id"
            ]
            .isna()
            .all()
        )
        for horizon in (30, 60):
            assert np.array_equal(
                prediction[f"severe_hypo_{horizon}_probability"],
                prediction[f"hypo_{horizon}_probability"],
            )
            assert np.array_equal(
                prediction[f"severe_hypo_{horizon}_raw_logit"],
                prediction[f"hypo_{horizon}_logit"],
            )
        assert prediction.hypo_30_raw_probability.iloc[0] == pytest.approx(
            0.75, abs=1e-7
        )
        assert prediction.hypo_30_probability.iloc[0] == pytest.approx(
            1 / (1 + np.exp(-np.log(3) / 2)), abs=1e-7
        )
        report = json.loads(
            (output / "exportable/fixed-frame-summary.json").read_text()
        )
        assert len(report["policies"]) == 6
        assert not any(
            item["glucose_cut"] == 70 and item["nominal_fapd"] == 1.0
            for item in report["policies"]
        )
        assert report["external_already_exposed"]
        assert report["thresholds_fitted_on_external"] is False
        assert {item["threshold"] for item in report["policies"]} == {0.6}
        assert {item["participant_count"] for item in report["policies"]} == {2}
        assert {
            item["event_positive_participant_count"] for item in report["policies"]
        } == {1}
        assert all(
            item["delivered_alarm_count"] > item["matched_alarm_count"]
            for item in report["policies"]
        )
        assert (
            "participant_id"
            not in (output / "exportable/fixed-frame-summary.json").read_text()
        )
        audit = json.loads((output / "private/input-audit.json").read_text())
        assert "not recomputed" in audit["global_cohort_audit"]
        assert audit["prediction_loader_workers"] == 0
        glucose = json.loads(
            (output / "exportable/alarm-glucose-summary.json").read_text()
        )
        assert glucose["participant_quantiles_averaged"] is False
        emitted = next(
            row
            for row in glucose["rows"]
            if row["scope"] == "pooled_alarms"
            and row["alarm_category"] == "emitted"
            and row["glucose_cut"] == 54
            and row["horizon_minutes"] == 30
            and row["nominal_fapd"] == 0.5
        )
        endpoint = next(
            row
            for row in report["policies"]
            if row["glucose_cut"] == 54
            and row["horizon_minutes"] == 30
            and row["nominal_fapd"] == 0.5
        )
        assert emitted["alarm_count"] == endpoint["emitted_alarm_count"]
        assert emitted["median"] == emitted["mean"] == 120.0
        receipt = json.loads((output / "run-receipt.json").read_text())
        assert receipt["status"] == "complete"
        assert receipt["run_dir"] == str(run.resolve())
        assert receipt["model_summary_snapshot"] == entry["model_summary"]["snapshot"]
        assert receipt["source_id"] == "Loop Observational Study"
        assert receipt["counts"]["decision_rows"] == len(prediction)
        paths = {item["relative_path"] for item in receipt["files"]}
        assert "private/input-audit.json" in paths
        assert "exportable/alarm-glucose-summary.json" in paths
        assert any(path.startswith("private/predictions/") for path in paths)
        assert any(path.startswith("private/endpoints/events-30/") for path in paths)
        assert any("/participants/" in path for path in paths)
        assert any("/alarms/" in path for path in paths)
        assert any("/events/" in path for path in paths)
        actual_ledgers = {
            str(path.relative_to(output))
            for path in (output / "private/evaluations").rglob("*.parquet")
        }
        assert actual_ledgers.issubset(paths)
        assert not (output / "run-receipt.pending.json").exists()
    keys = ["source_id", "participant_id", "context_end"]
    pd.testing.assert_frame_equal(tables[0][keys], tables[1][keys])


def test_freeze_required_changed_policy_and_ohio_rejected(tmp_path):
    freeze, runs, _ = _fixture(tmp_path, families=("persistence",))
    run = Path(runs[0]["run_dir"])
    with pytest.raises(ValueError, match="isolated from normalized source"):
        external.run_revision_external(
            freeze, run_dir=run, partition="external_loop", output_dir=tmp_path
        )
    with pytest.raises(ValueError, match="already exposed Loop/HUPA"):
        external.load_development_freeze(freeze, run, "confirmatory_ohio")
    original = json.loads(freeze.read_text())
    changed = json.loads(freeze.read_text())
    changed["status"] = "draft"
    freeze.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="explicit accepted development"):
        external.load_development_freeze(freeze, run, "external_loop")
    changed = json.loads(json.dumps(original))
    changed["runs"][0]["cut_policies"]["54"]["30"]["0.5"]["policy"]["threshold"] = 0.61
    freeze.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="base54 policy differs"):
        external.load_development_freeze(freeze, run, "external_loop")
    changed = json.loads(json.dumps(original))
    del changed["runs"][0]["cut_policies"]["70"]["30"]["0.5"]["evidence"]
    freeze.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="cut70 lacks frozen development"):
        external.load_development_freeze(freeze, run, "external_loop")


def test_unknown_source_fails_before_inference(tmp_path, monkeypatch):
    freeze, runs, normalized = _fixture(tmp_path, families=("persistence",))
    frame = pd.read_parquet(normalized)
    frame.loc[0, "source_id"] = "OhioT1DM"
    frame.to_parquet(normalized, index=False)
    payload = json.loads(freeze.read_text())
    payload["partitions"]["external_loop"]["normalized_inputs"] = [_binding(normalized)]
    freeze.write_text(json.dumps(payload))

    def forbidden(*args, **kwargs):
        raise AssertionError("unknown source reached inference")

    monkeypatch.setattr(external.FrozenPredictor, "predict", forbidden)
    with pytest.raises(ValueError, match="unknown or confirmatory sources"):
        external.run_revision_external(
            freeze,
            run_dir=Path(runs[0]["run_dir"]),
            partition="external_loop",
            output_dir=tmp_path / "out",
        )
    assert not (tmp_path / "out/run-receipt.json").exists()


def test_no_complete_receipt_when_freeze_changes_during_final_exports(
    tmp_path, monkeypatch
):
    freeze, runs, _ = _fixture(tmp_path, families=("persistence",))
    export = external._export_alarm_glucose

    def change_freeze_after_export(*args, **kwargs):
        export(*args, **kwargs)
        freeze.write_text(freeze.read_text() + "\n")

    monkeypatch.setattr(external, "_export_alarm_glucose", change_freeze_after_export)
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="development freeze changed"):
        external.run_revision_external(
            freeze,
            run_dir=Path(runs[0]["run_dir"]),
            partition="external_loop",
            output_dir=output,
        )
    assert json.loads((output / "status.json").read_text())["status"] == "failed"
    assert not (output / "run-receipt.json").exists()


def test_missing_censor_needs_explicit_source_contract_and_observed_never_inferred(
    tmp_path,
):
    freeze, runs, _ = _fixture(
        tmp_path / "missing", families=("persistence",), lower_censor=False
    )
    with pytest.raises(ValueError, match="lower-censor provenance is missing"):
        external.run_revision_external(
            freeze,
            run_dir=Path(runs[0]["run_dir"]),
            partition="external_loop",
            output_dir=tmp_path / "no-contract",
        )
    freeze, runs, _ = _fixture(
        tmp_path / "allowed",
        families=("persistence",),
        lower_censor=False,
        absent_contract=True,
    )
    external.run_revision_external(
        freeze,
        run_dir=Path(runs[0]["run_dir"]),
        partition="external_loop",
        output_dir=tmp_path / "with-contract",
    )
    freeze, runs, _ = _fixture(
        tmp_path / "no-mask",
        families=("persistence",),
        observed=False,
        absent_contract=True,
    )
    with pytest.raises(ValueError, match="missing observed-anchor provenance"):
        external.run_revision_external(
            freeze,
            run_dir=Path(runs[0]["run_dir"]),
            partition="external_loop",
            output_dir=tmp_path / "no-observation-mask",
        )


def test_normalization_global_audit_and_cross_run_timeline_contract_required(tmp_path):
    freeze, runs, _ = _fixture(tmp_path)
    payload = json.loads(freeze.read_text())
    payload["partitions"]["external_loop"]["accepted_global_cohort_audit"] = []
    freeze.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="actual accepted evidence"):
        external.load_development_freeze(
            freeze, Path(runs[0]["run_dir"]), "external_loop"
        )
    freeze, runs, _ = _fixture(tmp_path / "different")
    payload = json.loads(freeze.read_text())
    payload["runs"][1]["config"]["snapshot"]["protocol"]["context_hours"] = 0.5
    freeze.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="source/protocol/QC/cut timeline contract"):
        external.load_development_freeze(
            freeze, Path(runs[0]["run_dir"]), "external_loop"
        )


def test_hupa_base54_only_uses_its_frozen_direct_role(tmp_path):
    freeze, runs, _ = _fixture(
        tmp_path,
        families=("persistence",),
        partition="stress_hupa_ucm",
        lower_censor=False,
        absent_contract=True,
        cut70=False,
    )
    output = tmp_path / "hupa-output"
    external.run_revision_external(
        freeze,
        run_dir=Path(runs[0]["run_dir"]),
        partition="stress_hupa_ucm",
        output_dir=output,
    )
    report = json.loads((output / "exportable/fixed-frame-summary.json").read_text())
    assert len(report["policies"]) == 4
    assert {item["glucose_cut"] for item in report["policies"]} == {54}
    assert {item["source_id"] for item in report["policies"]} == {"HUPA-UCM"}


@pytest.mark.parametrize("kind", ["noncontiguous", "row_bound"])
def test_participant_stream_order_and_memory_bounds_fail_closed(tmp_path, kind):
    freeze, runs, normalized = _fixture(tmp_path, families=("persistence",))
    if kind == "noncontiguous":
        frame = pd.read_parquet(normalized)
        frame = pd.concat([frame, frame.iloc[:5]], ignore_index=True)
        frame.to_parquet(normalized, index=False)
        payload = json.loads(freeze.read_text())
        payload["partitions"]["external_loop"]["normalized_inputs"] = [
            _binding(normalized)
        ]
        freeze.write_text(json.dumps(payload))
    with pytest.raises(
        ValueError, match="contiguity" if kind == "noncontiguous" else "row bound"
    ):
        external.run_revision_external(
            freeze,
            run_dir=Path(runs[0]["run_dir"]),
            partition="external_loop",
            output_dir=tmp_path / "failed-output",
            max_participant_rows=3 if kind == "row_bound" else 1000,
        )
    assert (
        json.loads((tmp_path / "failed-output/status.json").read_text())["status"]
        == "failed"
    )
