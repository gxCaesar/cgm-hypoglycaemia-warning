"""The participant-memory ablation must be a controlled ablation.

Pillar A of the manuscript is that strict-past participant memory is where the
gain comes from. That claim is only supportable if disabling retrieval changes
exactly one thing. These tests pin the three ways it could quietly change more:

  1. the ablated model must have identical architecture and parameter count,
  2. the setting must apply to training AND prediction, not one of them,
  3. an ablated checkpoint must not regain retrieval at evaluation time.

The third is the dangerous one: training without memory and evaluating with it
would produce a number that corresponds to no model that was ever trained.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from glycoshift.config import load_config
from glycoshift.deep_training import _load_frozen_model, run_deep_training
from glycoshift.models import GlycoShift, GlycoShiftModelConfig, count_parameters


def _smoke_config(tmp_path: Path, *, use_memory: bool) -> dict:
    config = load_config(Path(__file__).parents[1] / "configs" / "smoke.yaml")
    # Eleven participants gives every source severe-event support in the
    # frozen selection split used by the operational checkpoint proxy.
    config["data"]["participants_per_source"] = 11
    config["data"]["days_per_participant"] = 1
    config["protocol"]["context_hours"] = 0.5
    config["training"]["batch_size"] = 64
    config["training"]["effective_batch_size"] = 64
    config["training"]["maximum_epochs"] = 1
    config["training"]["warmup_memory_epochs"] = 0
    config["training"]["num_workers"] = 0
    config["evaluation"]["threshold_candidate_count"] = 5
    config["evaluation"]["event_ap_candidate_count"] = 5
    config["evaluation"]["selection_operational_candidate_count"] = 3
    config["model"]["use_participant_memory"] = use_memory
    config["output"]["run_dir"] = str(
        tmp_path / ("with-memory" if use_memory else "without-memory")
    )
    return config


def test_ablation_changes_no_parameters() -> None:
    """Disabling retrieval must not change the network at all.

    `_fuse_memory` falls back to the learned population_memory, so the memory
    attention, the gate and population_memory itself all remain present and
    trained. If this count ever differs, the ablation has become a different
    architecture and the comparison is confounded.
    """
    model_config = GlycoShiftModelConfig.variant(
        "tiny", context_samples=6, memory_k=8, dropout=0.1
    )
    retrieving = GlycoShift(model_config)
    ablated = GlycoShift(model_config)
    ablated.supports_memory = False

    assert count_parameters(retrieving) == count_parameters(ablated)
    assert {name for name, _ in retrieving.named_parameters()} == {
        name for name, _ in ablated.named_parameters()
    }
    # population_memory must still exist and be trainable in the ablated model,
    # because it is what the ablated model actually attends over.
    assert ablated.population_memory.requires_grad


def test_ablated_run_records_and_restores_the_setting(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path, use_memory=False)
    run_dir = run_deep_training(config, architecture="glycoshift")

    for name in ("best.pt", "last.pt"):
        checkpoint = torch.load(
            run_dir / "ckpts" / name, map_location="cpu", weights_only=False
        )
        assert checkpoint["uses_participant_memory"] is False, name

    # Reloading must not silently re-enable retrieval.
    model, _ = _load_frozen_model(run_dir, device=torch.device("cpu"))
    assert model.supports_memory is False


def test_default_config_keeps_retrieval_enabled(tmp_path: Path) -> None:
    """Backward compatibility: omitting the key must behave as before."""
    config = _smoke_config(tmp_path, use_memory=True)
    del config["model"]["use_participant_memory"]
    run_dir = run_deep_training(config, architecture="glycoshift")

    checkpoint = torch.load(
        run_dir / "ckpts" / "best.pt", map_location="cpu", weights_only=False
    )
    assert checkpoint["uses_participant_memory"] is True
    model, _ = _load_frozen_model(run_dir, device=torch.device("cpu"))
    assert model.supports_memory is True


def test_ablated_and_retrieving_runs_both_complete(tmp_path: Path) -> None:
    """Both arms must produce the same artifact set, or they are not
    comparable as arms of one experiment."""
    expected = {
        "metrics.json",
        "training_summary.json",
        "model_summary.json",
        "predictions.parquet",
        "config_resolved.yaml",
    }
    produced = {}
    for use_memory in (True, False):
        run_dir = run_deep_training(
            _smoke_config(tmp_path, use_memory=use_memory),
            architecture="glycoshift",
        )
        produced[use_memory] = {
            name for name in expected if (run_dir / name).exists()
        }
        metrics = json.loads((run_dir / "metrics.json").read_text())
        assert "selection" in metrics["partitions"]
    assert produced[True] == expected
    assert produced[False] == expected
