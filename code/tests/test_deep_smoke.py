import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from glycoshift.config import load_config
from glycoshift.deep_training import (
    _load_development_training_data,
    _memory_bank,
    _selection_operational_candidate_count,
    run_deep_training,
    run_frozen_evaluation,
)


def test_glycoshift_tiny_training_smoke(tmp_path: Path) -> None:
    config = load_config(Path(__file__).parents[1] / "configs" / "smoke.yaml")
    # Eleven participants gives every development source severe-event support
    # in both selection and calibration under the frozen split seed.
    config["data"]["participants_per_source"] = 11
    config["data"]["days_per_participant"] = 1
    config["protocol"]["context_hours"] = 0.5
    config["training"]["batch_size"] = 64
    config["training"]["effective_batch_size"] = 64
    config["training"]["maximum_epochs"] = 1
    config["training"]["warmup_memory_epochs"] = 1
    config["training"]["num_workers"] = 0
    config["evaluation"]["threshold_candidate_count"] = 5
    config["evaluation"]["event_ap_candidate_count"] = 5
    config["evaluation"]["selection_operational_candidate_count"] = 3
    config["output"]["run_dir"] = str(tmp_path / "deep-run")
    run_dir = run_deep_training(config, architecture="glycoshift")
    assert (run_dir / "ckpts" / "best.pt").exists()
    assert (run_dir / "ckpts" / "last.pt").exists()
    assert (run_dir / "training_summary.json").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "operational_metrics.json").exists()
    assert (run_dir / "selection_operational_proxy.json").exists()
    assert (run_dir / "feature_group_dropout.json").exists()
    first_log = json.loads(
        (run_dir / "logs" / "train.jsonl").read_text().splitlines()[0]
    )
    assert first_log["event"] == "first_finite_loss"
    assert first_log["loss"] > 0
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert "continuous_forecast" in metrics["partitions"]["selection"]
    assert "severe_hypo_30" in metrics["partitions"]["selection"]
    metrics_path = run_frozen_evaluation(
        config,
        run_dir=run_dir,
        partition="stress_hupa_ucm",
    )
    assert metrics_path.exists()
    with pytest.raises(FileExistsError, match="refusing to repeat"):
        run_frozen_evaluation(
            config,
            run_dir=run_dir,
            partition="stress_hupa_ucm",
        )


def test_lstm_tls_training_smoke_preserves_hard_evaluation(tmp_path: Path) -> None:
    config = load_config(Path(__file__).parents[1] / "configs" / "smoke.yaml")
    config["data"]["participants_per_source"] = 11
    config["data"]["days_per_participant"] = 1
    config["protocol"]["context_hours"] = 0.5
    config["training"]["batch_size"] = 64
    config["training"]["effective_batch_size"] = 64
    config["training"]["maximum_epochs"] = 1
    config["training"]["num_workers"] = 0
    config["training"]["event_objective"] = "tls_exp"
    config["training"]["tls_gamma_per_hour"] = 0.2
    config["training"]["data_sampling_seed"] = 42
    config["evaluation"]["threshold_candidate_count"] = 5
    config["evaluation"]["event_ap_candidate_count"] = 5
    config["evaluation"]["selection_operational_candidate_count"] = 3
    config["output"]["run_dir"] = str(tmp_path / "lstm-tls-run")

    run_dir = run_deep_training(config, architecture="lstm")
    audit = json.loads((run_dir / "tls_target_audit.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert audit["objective"] == "tls_exp"
    assert audit["gamma_per_hour"] == 0.2
    assert all(
        task["soft_positive_count"] >= task["hard_positive_count"]
        for task in audit["tasks"].values()
    )
    assert "hypo_30" in metrics["partitions"]["selection"]
    assert "operational_v1_3" in metrics


def test_real_deep_training_data_uses_audited_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(Path(__file__).parents[1] / "configs" / "smoke.yaml")
    config["project"]["synthetic"] = False
    frame = pd.DataFrame({"frame": [1]})
    windows = pd.DataFrame({"windows": [1]})
    audit = {"status": "passed"}

    monkeypatch.setattr(
        "glycoshift.deep_training.load_frame",
        lambda *_args, **_kwargs: frame,
    )
    monkeypatch.setattr(
        "glycoshift.deep_training.load_audited_development_windows",
        lambda _config: (windows, audit),
    )

    observed_frame, observed_windows, observed_audit = (
        _load_development_training_data(config)
    )
    assert observed_frame is frame
    assert observed_windows is windows
    assert observed_audit is audit


def test_real_deep_checkpoint_proxy_rejects_threshold_downsampling() -> None:
    config = load_config(Path(__file__).parents[1] / "configs" / "smoke.yaml")
    config["project"]["synthetic"] = False
    config["evaluation"]["selection_operational_candidate_count"] = 3

    with pytest.raises(ValueError, match="all unique selection scores"):
        _selection_operational_candidate_count(config)


def test_memory_bank_preserves_full_window_encoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    windows = pd.DataFrame(
        {
            "source_id": ["s", "s", "s"],
            "participant_id": ["p", "p", "p"],
            "context_end": pd.to_datetime(
                [
                    "2026-01-01T00:05:00Z",
                    "2026-01-01T00:25:00Z",
                    "2026-01-01T00:35:00Z",
                ],
                utc=True,
            ),
        }
    )
    dataset = SimpleNamespace(windows=windows)
    model = SimpleNamespace(
        config=SimpleNamespace(memory_k=2, d_model=4)
    )
    observed: dict[str, object] = {}

    def fake_encode(
        _model: object,
        _dataset: object,
        *,
        device: torch.device,
        batch_size: int,
        num_workers: int,
        dataset_indices: object = None,
    ) -> torch.Tensor:
        observed["dataset_indices"] = dataset_indices
        return torch.arange(12, dtype=torch.float32).reshape(3, 4)

    monkeypatch.setattr(
        "glycoshift.deep_training._encode_dataset",
        fake_encode,
    )
    bank = _memory_bank(
        model,
        dataset,
        device=torch.device("cpu"),
        batch_size=2,
        num_workers=0,
        stride_minutes=30,
    )

    assert observed["dataset_indices"] is None
    assert bank.record_count == 2
