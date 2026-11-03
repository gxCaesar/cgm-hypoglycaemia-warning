from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from glycoshift.config import load_config
from glycoshift.forecast_training import _checkpoint_proxy, run_forecast_training
from glycoshift.operational_metrics import OperationalEndpointSpec


def test_patchmlp_glimmer_training_smoke_preserves_split_roles(
    tmp_path: Path,
) -> None:
    config = load_config(Path(__file__).parents[1] / "configs" / "smoke.yaml")
    # Eleven participants gives both synthetic development sources at least
    # one severe event in selection and calibration under split seed 42.
    config["data"]["participants_per_source"] = 11
    config["data"]["days_per_participant"] = 1
    config["protocol"]["context_hours"] = 0.5
    config["model"]["taskaware_forecast"] = {
        "d_model": 96,
        "encoder_layers": 1,
        "moving_average_kernel": 5,
        "dropout": 0.0,
        "patch_lengths": [6, 4, 2],
    }
    config["training"]["batch_size"] = 512
    config["training"]["effective_batch_size"] = 512
    config["training"]["maximum_epochs"] = 1
    config["training"]["early_stopping_patience"] = 1
    config["training"]["num_workers"] = 0
    config["training"]["feature_group_dropout_max"] = 0.0
    config["evaluation"]["threshold_candidate_count"] = 3
    config["evaluation"]["event_ap_candidate_count"] = 3
    config["evaluation"]["selection_operational_candidate_count"] = 3
    config["output"]["run_dir"] = str(tmp_path / "patchmlp-glimmer")

    run_dir = run_forecast_training(config, loss_name="glimmer")

    assert (run_dir / "ckpts" / "best.pt").exists()
    assert (run_dir / "ckpts" / "last.pt").exists()
    assert (run_dir / "operational_metrics.json").exists()
    first_log = json.loads(
        (run_dir / "logs" / "train.jsonl").read_text().splitlines()[0]
    )
    assert first_log["event"] == "first_finite_loss"
    assert first_log["loss_name"] == "glimmer"
    assert first_log["loss"] > 0

    predictions = pd.read_parquet(run_dir / "predictions.parquet")
    assert set(predictions["partition"]) == {"selection", "calibration"}
    for horizon in (30, 60):
        assert predictions[f"severe_hypo_{horizon}_probability"].notna().all()
        assert predictions[f"cgm_{horizon}_prediction"].notna().all()
    operational = json.loads((run_dir / "operational_metrics.json").read_text())
    assert operational["threshold_partition"] == "calibration"
    assert operational["evaluation_partition"] == "selection"
    assert set(operational["horizons"]) == {"30", "60"}
    assert all(
        horizon["policy"]["partition"] == "calibration"
        for horizon in operational["horizons"].values()
    )
    summary = json.loads((run_dir / "model_summary.json").read_text())
    assert summary["compatibility_status"] == (
        "information_matched_adapted_comparator"
    )
    assert summary["unit_contract"]["loss_and_alarm"].startswith("mg/dL")


def test_checkpoint_proxy_uses_every_unique_raw_logit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_columns: list[str] = []

    def fake_curve(
        predictions: pd.DataFrame,
        *,
        spec: OperationalEndpointSpec,
        candidate_count: int | None,
    ) -> dict[str, float]:
        del predictions
        assert candidate_count is None
        observed_columns.append(spec.score_column)
        return {"raw_auc": 1.0, "fapd_at_best_sensitivity": 0.25}

    monkeypatch.setattr(
        "glycoshift.forecast_training.selection_operational_curve",
        fake_curve,
    )
    _checkpoint_proxy(
        pd.DataFrame(),
        source_order=("source",),
        candidate_count=None,
    )

    assert observed_columns == [
        "severe_hypo_30_raw_logit",
        "severe_hypo_60_raw_logit",
    ]
