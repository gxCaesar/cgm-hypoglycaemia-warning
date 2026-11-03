"""Weighting the event loss by lead time must change emphasis and nothing else.

The published temporal-label-smoothing scheme shapes by time to onset and is
monotone *decreasing* in lead time: at the frozen `gamma_per_hour = 0.2` a
five-minute lead carries target 0.908818 against 0.475021 at thirty minutes.
This arm inverts that. The tests below hold three lines: the unweighted path
stays bitwise what it was, the weights average one over positives so the arm is
a single-variable change from its control, and the direction really is inverted
rather than merely intended.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from glycoshift.config import load_config
from glycoshift.deep_training import _unweighted_multitask_loss
from glycoshift.fast_sequences import FastWindowSequenceDataset
from glycoshift.lead_time_weight import (
    attach_lead_time_loss_weights,
    summarize_lead_time_weights,
)
from glycoshift.sequences import (
    LOSS_WEIGHT_SUFFIX,
    REGRESSION_HORIZONS,
    TASKS,
    SequenceScaler,
    WindowSequenceDataset,
    validate_optional_loss_weights,
)
from glycoshift.sota_objectives import tls_exp_target
from glycoshift.synthetic import make_synthetic_cgm
from glycoshift.windows import build_windows

CONFIG = Path(__file__).parents[1] / "configs" / "smoke.yaml"


def _cohort(*, participants: int = 3, days: int = 3, context_hours: float = 0.5):
    config = load_config(CONFIG)
    config["data"]["participants_per_source"] = participants
    config["data"]["days_per_participant"] = days
    config["protocol"]["context_hours"] = context_hours
    frame = make_synthetic_cgm(config)
    blocks = []
    for _key, group in frame.groupby(["source_id", "participant_id"], sort=True):
        block = group.sort_values("timestamp").reset_index(drop=True)
        blocks.append(build_windows(block, config))
    windows = pd.concat(blocks, ignore_index=True)
    keys = set(
        zip(  # noqa: B905
            frame["source_id"].astype(str), frame["participant_id"].astype(str)
        )
    )
    scaler = SequenceScaler.fit(frame, keys)
    samples = int(context_hours * 60 / int(config["protocol"]["sample_minutes"]))
    return frame, windows, scaler, samples


def _random_batch(rng: np.random.Generator, size: int):
    tasks, horizons = len(TASKS), len(REGRESSION_HORIZONS)
    outputs = {
        "logits": {
            task: torch.from_numpy(rng.normal(0, 3, size).astype(np.float32))
            for task in TASKS
        },
        "regression": {
            str(h): torch.from_numpy(rng.normal(0, 1, size).astype(np.float32))
            for h in REGRESSION_HORIZONS
        },
    }
    batch = {
        "labels": torch.from_numpy(
            (rng.random((size, tasks)) < 0.3).astype(np.float32)
        ),
        "label_mask": torch.from_numpy(
            (rng.random((size, tasks)) < 0.8).astype(np.float32)
        ),
        "regression_targets": torch.from_numpy(
            rng.normal(0, 1, (size, horizons)).astype(np.float32)
        ),
        "sampling_weight": torch.from_numpy(
            rng.random(size).astype(np.float32) + 0.5
        ),
    }
    return outputs, batch


def test_all_ones_weights_leave_the_loss_bitwise_unchanged() -> None:
    """G1. Multiplying a loss by one changes nothing, over 256 batches.

    The premise is arithmetic, which is the point: the gate exists to catch a
    plumbing change, not a numerical one. Compared as raw bits rather than with
    a tolerance, because a tolerance would hide exactly what this looks for.
    """

    rng = np.random.default_rng(20260821)
    positive_weights = torch.tensor([2.0, 3.0, 1.5, 4.0])
    for _ in range(256):
        outputs, batch = _random_batch(rng, 16)
        plain = _unweighted_multitask_loss(
            outputs, batch, positive_weights=positive_weights, auxiliary_weight=0.3
        )
        weighted_batch = dict(batch)
        weighted_batch["loss_weight"] = torch.ones(16, len(TASKS))
        weighted = _unweighted_multitask_loss(
            outputs,
            weighted_batch,
            positive_weights=positive_weights,
            auxiliary_weight=0.3,
        )
        assert torch.equal(plain, weighted)


def test_absent_weights_emit_no_key_in_either_dataset_path() -> None:
    """Absence, not a tensor of ones, is how the default path stays untouched."""

    frame, windows, scaler, samples = _cohort()
    assert not any(f"{task}{LOSS_WEIGHT_SUFFIX}" in windows.columns for task in TASKS)
    slow = WindowSequenceDataset(
        frame, windows, scaler=scaler, context_samples=samples
    )
    fast = FastWindowSequenceDataset(
        frame, windows, scaler=scaler, context_samples=samples
    )
    assert "loss_weight" not in slow[0]
    assert "loss_weight" not in fast[0]


def test_the_weight_reaches_both_dataset_paths_identically() -> None:
    frame, windows, scaler, samples = _cohort()
    weighted = attach_lead_time_loss_weights(windows)
    slow = WindowSequenceDataset(
        frame, weighted, scaler=scaler, context_samples=samples
    )
    fast = FastWindowSequenceDataset(
        frame, weighted, scaler=scaler, context_samples=samples
    )
    expected = np.stack(
        [
            weighted[f"{task}{LOSS_WEIGHT_SUFFIX}"].to_numpy(np.float32)
            for task in TASKS
        ],
        axis=1,
    )
    slow_weight = np.stack(
        [slow[i]["loss_weight"].numpy() for i in range(len(slow))]
    )
    fast_weight = np.stack(
        [fast[i]["loss_weight"].numpy() for i in range(len(fast))]
    )
    np.testing.assert_array_equal(slow_weight, expected)
    np.testing.assert_array_equal(fast_weight, expected)


def test_weights_average_one_over_positives_and_are_exactly_one_on_negatives() -> None:
    """The property that makes this a single-variable change from the control.

    The weights redistribute emphasis among positive windows without altering
    the aggregate weight positives carry against negatives, so the arm differs
    from its control in the distribution of the event loss and in nothing else.
    """

    _frame, windows, _scaler, _samples = _cohort()
    weighted = attach_lead_time_loss_weights(windows)
    for task in TASKS:
        positive = windows[f"{task}_label"].fillna(0).to_numpy(float) > 0.5
        assert positive.sum() > 0, f"{task} has no positives; test would be vacuous"
        values = weighted[f"{task}{LOSS_WEIGHT_SUFFIX}"].to_numpy(float)
        np.testing.assert_array_equal(values[~positive], 1.0)
        assert abs(values[positive].mean() - 1.0) < 1e-6
        assert abs(values[positive].sum() - positive.sum()) < 1e-3


def test_the_direction_is_inverted_relative_to_published_label_smoothing() -> None:
    """The substantive gate, and the reason the arm exists.

    A sign error here would present as a null result, so the direction is
    asserted against the published scheme it inverts rather than merely against
    zero. `tls_exp_target` is monotone decreasing in lead time; this weight is
    monotone increasing, and the two ratios must straddle one.
    """

    _frame, windows, _scaler, _samples = _cohort()
    weighted = attach_lead_time_loss_weights(windows)

    for task in TASKS:
        horizon = int(task.rsplit("_", 1)[1])
        positive = windows[f"{task}_label"].fillna(0).to_numpy(float) > 0.5
        onset = pd.to_datetime(windows[f"{task}_event_onset"], utc=True)
        query = pd.to_datetime(windows["context_end"], utc=True)
        lead = ((onset - query).dt.total_seconds() / 60).to_numpy(float)[positive]
        values = weighted[f"{task}{LOSS_WEIGHT_SUFFIX}"].to_numpy(float)[positive]

        order = np.argsort(lead, kind="stable")
        assert np.all(np.diff(values[order]) >= -1e-9), "weight must rise with lead"
        assert values.max() / values.min() > 1.0

        near, far = 5.0, float(horizon)
        tls = tls_exp_target(
            np.array([near, far]), horizon_minutes=horizon, gamma_per_hour=0.2
        )
        assert tls[0] / tls[1] > 1.0, "TLS values the near warning more"
        assert (far / near) > 1.0, "this arm values the far warning more"


def test_a_lead_time_outside_the_label_interval_is_refused() -> None:
    """`next_event_within` accepts an onset in (query, query+h], so this catches
    a timezone slip or a sign error before it can look like a null."""

    _frame, windows, _scaler, _samples = _cohort()
    broken = windows.copy()
    first = broken.index[broken["hypo_30_label"].fillna(0) > 0.5][0]
    broken.loc[first, "hypo_30_event_onset"] = broken.loc[first, "context_end"]
    with pytest.raises(ValueError, match=r"outside \(0, 30\] minutes"):
        attach_lead_time_loss_weights(broken)


def test_a_positive_window_without_an_onset_is_refused() -> None:
    _frame, windows, _scaler, _samples = _cohort()
    broken = windows.copy()
    first = broken.index[broken["hypo_30_label"].fillna(0) > 0.5][0]
    broken.loc[first, "hypo_30_event_onset"] = None
    with pytest.raises(ValueError, match="positive window without an onset"):
        attach_lead_time_loss_weights(broken)


def test_a_task_without_positives_keeps_a_uniform_weight() -> None:
    """The synthetic cohort never reaches severe hypoglycaemia, and a real task
    can be empty in one fold, so this must not divide by zero."""

    _frame, windows, _scaler, _samples = _cohort()
    empty = windows.copy()
    empty["hypo_30_label"] = 0.0
    empty["hypo_30_event_onset"] = pd.NaT
    weighted = attach_lead_time_loss_weights(empty)
    values = weighted[f"hypo_30{LOSS_WEIGHT_SUFFIX}"].to_numpy(float)
    np.testing.assert_array_equal(values, 1.0)
    summary = summarize_lead_time_weights(weighted)
    assert summary["tasks"]["hypo_30"]["positive_window_count"] == 0
    assert "mean_lead_minutes" not in summary["tasks"]["hypo_30"]


def test_negative_and_non_finite_weights_are_refused() -> None:
    _frame, windows, _scaler, _samples = _cohort()
    for value, message in ((-1.0, "negative weights"), (np.inf, "non-finite")):
        broken = windows.copy()
        broken[f"hypo_30{LOSS_WEIGHT_SUFFIX}"] = value
        with pytest.raises(ValueError, match=message):
            validate_optional_loss_weights(broken)


def test_the_objective_enum_admits_the_arm_and_keeps_the_others_exclusive() -> None:
    """One enum, so lead-time weighting and severe retargeting cannot both run.

    They compose in principle and the pre-registration takes them one at a time,
    so the composed arm needs its own value and its own document rather than
    arriving by accident.
    """

    from glycoshift.config import validate_config

    config = load_config(CONFIG)
    for value in ("bce", "tls_exp", "severe_retarget", "lead_time_weighted"):
        config["training"]["event_objective"] = value
        if value == "tls_exp":
            config["training"]["tls_gamma_per_hour"] = 0.2
        validate_config(config)

    config["training"]["event_objective"] = "severe_retarget+lead_time_weighted"
    with pytest.raises(ValueError, match="lead_time_weighted"):
        validate_config(config)


def test_training_under_the_weighted_objective_runs_and_records_the_direction(
    tmp_path: Path,
) -> None:
    from glycoshift.deep_training import run_deep_training

    config = load_config(CONFIG)
    config["data"]["participants_per_source"] = 11
    config["data"]["days_per_participant"] = 1
    config["protocol"]["context_hours"] = 0.5
    config["training"]["batch_size"] = 64
    config["training"]["effective_batch_size"] = 64
    config["training"]["maximum_epochs"] = 1
    config["training"]["warmup_memory_epochs"] = 1
    config["training"]["num_workers"] = 0
    config["training"]["event_objective"] = "lead_time_weighted"
    config["evaluation"]["threshold_candidate_count"] = 5
    config["evaluation"]["event_ap_candidate_count"] = 5
    config["evaluation"]["selection_operational_candidate_count"] = 3
    config["output"]["run_dir"] = str(tmp_path / "leadweight-run")

    run_dir = run_deep_training(config, architecture="glycoshift")

    audit = json.loads((run_dir / "lead_time_weight_audit.json").read_text())
    assert audit["objective"] == "lead_time_weighted"
    assert audit["weight_form"] == "v(d) = d / mean(d over positives)"
    assert not (run_dir / "severe_target_audit.json").exists()
    assert not (run_dir / "tls_target_audit.json").exists()

    for task in ("hypo_30", "hypo_60"):
        entry = audit["tasks"][task]
        assert entry["positive_window_count"] > 0, "fixture has no positives"
        assert abs(entry["positive_weight_mean"] - 1.0) < 1e-6
        assert entry["weight_ratio_far_over_near"] > 1.0, "direction not inverted"
