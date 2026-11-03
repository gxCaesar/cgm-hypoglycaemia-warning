from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from glycoshift.config import load_config
from glycoshift.deep_training import _multitask_loss
from glycoshift.sequences import TASKS
from glycoshift.sota_objectives import (
    FrozenSourceRiskReference,
    MinimaxExcessRisk,
    attach_tls_exp_targets,
    tls_exp_target,
)


def test_tls_exp_target_boundaries_and_monotonicity() -> None:
    distances = np.array([np.inf, 65, 60, 55, 30, 5, 0, -5], dtype=float)
    targets = tls_exp_target(
        distances,
        horizon_minutes=30,
        gamma_per_hour=0.2,
    )
    assert targets[:3].tolist() == [0.0, 0.0, 0.0]
    assert targets[-2:].tolist() == [1.0, 1.0]
    assert np.all(np.diff(targets) >= 0)
    assert 0 < targets[4] < 1


def _tls_windows() -> pd.DataFrame:
    times = pd.date_range("2026-01-01", periods=13, freq="5min", tz="UTC")
    onset = times[-1]
    rows: list[dict[str, object]] = []
    for participant, participant_onset in (("p1", onset), ("p2", None)):
        for time in times:
            delta = (
                (participant_onset - time).total_seconds() / 60
                if participant_onset is not None
                else np.inf
            )
            row: dict[str, object] = {
                "source_id": "source",
                "participant_id": participant,
                "context_end": time,
            }
            for kind in ("hypo", "hyper"):
                for horizon in (30, 60):
                    positive = 0 < delta <= horizon
                    row[f"{kind}_{horizon}_label"] = int(positive)
                    row[f"{kind}_{horizon}_event_onset"] = (
                        participant_onset if positive else pd.NaT
                    )
            rows.append(row)
    return pd.DataFrame(rows)


def test_tls_targets_use_unsampled_future_events_without_crossing_people() -> None:
    windows = _tls_windows()
    hard_before = windows.filter(regex="_label$").copy()
    output = attach_tls_exp_targets(
        windows,
        sample_minutes=5,
        gamma_per_hour=0.2,
    )
    pd.testing.assert_frame_equal(output[hard_before.columns], hard_before)

    p1 = output.loc[output["participant_id"] == "p1"].reset_index(drop=True)
    p2 = output.loc[output["participant_id"] == "p2"].reset_index(drop=True)
    target = p1["hypo_30_training_target"].to_numpy()
    assert target[0] == 0.0
    assert target[1] > 0.0
    # The event-onset row is excluded from EEP by the currently-active mask;
    # the official TLS construction selects strictly future onsets.
    assert np.all(np.diff(target[:-1]) >= 0)
    assert not p1.loc[1, "hypo_30_label"]
    assert p1.loc[1, "hypo_30_training_target"] > 0
    assert (p2.filter(regex="_training_target$") == 0).all().all()


def test_tls_disabled_path_is_identical_and_active_tasks_stay_masked() -> None:
    outputs = {
        "logits": {task: torch.tensor([0.3]) for task in TASKS},
        "regression": {
            "30": torch.tensor([0.0]),
            "60": torch.tensor([0.0]),
        },
    }
    batch = {
        "labels": torch.tensor([[1.0, 0.0, 1.0, 0.0]]),
        "label_mask": torch.tensor([[True, False, True, True]]),
        "regression_targets": torch.zeros((1, 2)),
        "regression_mask": torch.ones((1, 2), dtype=torch.bool),
        "sampling_weight": torch.ones(1),
    }
    control = _multitask_loss(
        outputs,
        batch,
        positive_weights=torch.ones(len(TASKS)),
        auxiliary_weight=0.1,
    )
    same = _multitask_loss(
        outputs,
        {**batch, "training_labels": batch["labels"].clone()},
        positive_weights=torch.ones(len(TASKS)),
        auxiliary_weight=0.1,
    )
    masked_change = batch["labels"].clone()
    masked_change[0, 1] = 1.0
    masked = _multitask_loss(
        outputs,
        {**batch, "training_labels": masked_change},
        positive_weights=torch.ones(len(TASKS)),
        auxiliary_weight=0.1,
    )
    torch.testing.assert_close(control, same)
    torch.testing.assert_close(control, masked)


def test_minimax_excess_risk_uses_regret_not_raw_difficulty() -> None:
    aggregator = MinimaxExcessRisk(
        torch.tensor([1.0, 9.0, 0.0]),
        source_order=("a", "b", "c"),
    )
    objective, raw, excess, weights = aggregator.aggregate(
        torch.tensor([2.0, 10.0, 3.0], requires_grad=True),
        torch.tensor([0, 1, 2], dtype=torch.long),
    )
    assert raw == pytest.approx({0: 2.0, 1: 10.0, 2: 3.0})
    assert excess == pytest.approx({0: 1.0, 1: 1.0, 2: 3.0})
    assert objective.item() == pytest.approx(3.0)
    assert weights.tolist() == [0.0, 0.0, 1.0]


def test_minimax_excess_risk_zero_translation_and_gradient() -> None:
    parameter = torch.tensor(2.0, requires_grad=True)
    losses = torch.stack(
        ((parameter - 0.0).square(), (parameter - 1.0).square())
    )
    groups = torch.tensor([0, 1], dtype=torch.long)
    references = torch.tensor([4.0, 0.0], requires_grad=True)
    aggregator = MinimaxExcessRisk(
        references,
        source_order=("a", "b"),
    )
    objective, _, excess, _ = aggregator.aggregate(losses, groups)
    assert objective.item() == pytest.approx(1.0)
    assert excess == pytest.approx({0: 0.0, 1: 1.0})
    objective.backward()
    assert parameter.grad is not None
    assert torch.isfinite(parameter.grad)
    assert parameter.grad.item() == pytest.approx(2.0)
    assert references.grad is None

    epsilon = 1e-3
    plus = (2.0 + epsilon - 1.0) ** 2
    minus = (2.0 - epsilon - 1.0) ** 2
    finite_difference = (plus - minus) / (2 * epsilon)
    assert parameter.grad.item() == pytest.approx(finite_difference, rel=1e-4)

    zero = MinimaxExcessRisk(
        torch.tensor([2.0, 10.0, 3.0]),
        source_order=("a", "b", "c"),
    )
    zero_objective, _, zero_excess, _ = zero.aggregate(
        torch.tensor([2.0, 10.0, 3.0]),
        torch.tensor([0, 1, 2], dtype=torch.long),
    )
    assert zero_objective.item() == pytest.approx(0.0)
    assert zero_excess == pytest.approx({0: 0.0, 1: 0.0, 2: 0.0})

    shifted = MinimaxExcessRisk(
        torch.tensor([14.0, 20.0]),
        source_order=("a", "b"),
    )
    translated, _, translated_excess, _ = shifted.aggregate(
        torch.tensor([14.0, 21.0]),
        groups,
    )
    assert translated.item() == pytest.approx(1.0)
    assert translated_excess == pytest.approx({0: 0.0, 1: 1.0})


def test_minimax_excess_risk_rejects_missing_group() -> None:
    aggregator = MinimaxExcessRisk(
        torch.zeros(3),
        source_order=("a", "b", "c"),
    )
    with pytest.raises(ValueError, match="requires every source group"):
        aggregator.aggregate(
            torch.tensor([1.0, 2.0]),
            torch.tensor([0, 2], dtype=torch.long),
        )


def test_frozen_source_reference_round_trip(tmp_path: Path) -> None:
    digest = "a" * 64
    reference = FrozenSourceRiskReference(
        source_order=("a", "b"),
        reference_risks=(0.25, 0.5),
        fold_count=5,
        reference_seed=42,
        data_sampling_seed=42,
        input_manifest_sha256=digest,
        window_manifest_sha256="b" * 64,
        training_config_sha256="c" * 64,
    )
    path = tmp_path / "reference.json"
    reference.write(path)
    restored = FrozenSourceRiskReference.read(path)
    assert restored == reference
    assert restored.sha256 == reference.sha256



























