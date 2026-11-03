"""Retargeting the scored head must change the target and nothing else.

The endpoint scores severe hypoglycaemia and no head is trained on it; the
severe score is the 70 mg/dL head's logit bound verbatim. `severe_target`
repoints that head by writing the two optional per-window columns the sequence
datasets already read, deliberately avoiding an extension of `TASKS`, which
would also move the negative-subsampling mask and so confound a target change
with a data change. These tests hold that line: the default path must be
untouched, the override must reach both dataset implementations identically,
and the sampled training set must not move.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from glycoshift.config import load_config
from glycoshift.fast_sequences import FastWindowSequenceDataset
from glycoshift.sequences import (
    TASKS,
    SequenceScaler,
    WindowSequenceDataset,
    validate_optional_training_masks,
)
from glycoshift.severe_target import (
    SCORED_HORIZONS,
    attach_severe_training_targets,
    summarize_severe_targets,
)
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


def _default_mask(windows: pd.DataFrame) -> np.ndarray:
    """The mask formula as it stood before the override existed."""

    return np.stack(
        [
            (~windows[f"{task.split('_')[0]}_currently_active"].to_numpy(bool))
            & windows[f"{task}_label"].notna().to_numpy()
            for task in TASKS
        ],
        axis=1,
    )


def _masks(frame, windows, scaler, samples):
    slow = WindowSequenceDataset(
        frame, windows, scaler=scaler, context_samples=samples
    )
    fast = FastWindowSequenceDataset(
        frame, windows, scaler=scaler, context_samples=samples
    )
    slow_mask = np.stack(
        [slow[index]["label_mask"].numpy() for index in range(len(slow))]
    )
    fast_mask = np.stack(
        [fast[index]["label_mask"].numpy() for index in range(len(fast))]
    )
    return slow_mask, fast_mask


def test_absent_override_reproduces_the_pre_change_mask_in_both_paths() -> None:
    """G1. With no override column, nothing moved.

    Compared against the formula as it stood before, not against the other
    implementation, so the two cannot agree on the wrong thing together.
    """

    frame, windows, scaler, samples = _cohort()
    assert not any(f"{task}_training_mask" in windows.columns for task in TASKS)

    expected = _default_mask(windows)
    slow_mask, fast_mask = _masks(frame, windows, scaler, samples)

    np.testing.assert_array_equal(slow_mask, expected)
    np.testing.assert_array_equal(fast_mask, expected)


def test_override_reaches_both_dataset_paths_identically() -> None:
    """An override must not be honoured by one implementation and not the other."""

    frame, windows, scaler, samples = _cohort()
    windows = windows.copy()
    rng = np.random.default_rng(20260821)
    forced = {}
    for task in TASKS:
        values = rng.random(len(windows)) < 0.5
        windows[f"{task}_training_mask"] = values
        forced[task] = values
    expected = np.stack([forced[task] for task in TASKS], axis=1)

    slow_mask, fast_mask = _masks(frame, windows, scaler, samples)
    np.testing.assert_array_equal(slow_mask, expected)
    np.testing.assert_array_equal(fast_mask, expected)


def test_a_null_or_non_boolean_mask_is_refused_rather_than_coerced() -> None:
    """`bool(nan)` is true, so a null mask would silently train a dropped window."""

    _frame, windows, _scaler, _samples = _cohort()

    nulled = windows.copy()
    nulled["hypo_30_training_mask"] = pd.Series(
        [True] * (len(nulled) - 1) + [None], dtype="object"
    )
    with pytest.raises(ValueError, match="contains nulls"):
        validate_optional_training_masks(nulled)

    floated = windows.copy()
    floated["hypo_30_training_mask"] = 1.0
    with pytest.raises(ValueError, match="must be boolean"):
        validate_optional_training_masks(floated)


def test_retargeting_leaves_every_evaluation_label_and_the_sample_set_alone() -> None:
    """The confound guard, and the reason `TASKS` was not extended.

    `sequences.py` builds the negative-subsampling positive mask from
    `<task>_label` over `TASKS`. If retargeting moved any of those columns, the
    sampled training set would move with it and the arm would differ from its
    control in two ways at once.
    """

    _frame, windows, _scaler, _samples = _cohort()
    label_columns = [f"{task}_label" for task in TASKS]
    before = windows[label_columns].copy()
    before_positive = windows[label_columns].fillna(0).any(axis=1).to_numpy()

    retargeted = attach_severe_training_targets(windows)

    pd.testing.assert_frame_equal(retargeted[label_columns], before)
    after_positive = (
        retargeted[label_columns].fillna(0).any(axis=1).to_numpy()
    )
    np.testing.assert_array_equal(after_positive, before_positive)
    # and the source frame was not mutated in place
    pd.testing.assert_frame_equal(windows[label_columns], before)


def test_retargeting_unmasks_the_band_where_the_benchmark_is_decided() -> None:
    """The substantive half: the default mask hides the 54-to-70 mg/dL band.

    The default mask drops every window in which a hypo event is already
    running. Those are the decisions the endpoint still scores -- it excludes
    only an active *severe* event -- and they are where the leading trivial arm,
    which alarms at 61.0 mg/dL, earns its entire advantage. A retargeting that
    kept the default mask would train on the severe label while hiding that
    band, so this is the assertion that the change does what it claims.
    """

    _frame, windows, _scaler, _samples = _cohort()
    hypo_active = windows["hypo_currently_active"].to_numpy(bool)
    severe_active = windows["severe_hypo_currently_active"].to_numpy(bool)
    band = hypo_active & ~severe_active
    assert band.sum() > 0, "fixture does not exercise the band; test would be vacuous"

    summary = summarize_severe_targets(attach_severe_training_targets(windows))
    for horizon in SCORED_HORIZONS:
        task = f"hypo_{horizon}"
        entry = summary["tasks"][task]
        assert entry["newly_unmasked"] == int(band.sum())
        assert entry["newly_masked"] == 0
        assert (
            entry["trained_window_count"]
            == entry["default_trained_window_count"] + int(band.sum())
        )


def test_the_training_target_becomes_the_severe_label_and_the_mask_its_eligibility(
) -> None:
    """The mapping itself, exercised on injected severe events.

    The synthetic cohort never falls below 54 mg/dL, so its severe labels are
    all zero and would make this vacuous. The labels are injected directly:
    this checks the column mapping, not the synthesiser.
    """

    frame, windows, scaler, samples = _cohort()
    windows = windows.copy()
    rng = np.random.default_rng(17)
    injected_active = rng.random(len(windows)) < 0.05
    windows["severe_hypo_currently_active"] = injected_active
    for horizon in SCORED_HORIZONS:
        column = f"severe_hypo_{horizon}_label"
        windows[column] = (rng.random(len(windows)) < 0.10).astype(float)

    retargeted = attach_severe_training_targets(windows)
    dataset = WindowSequenceDataset(
        frame, retargeted, scaler=scaler, context_samples=samples
    )
    fast = FastWindowSequenceDataset(
        frame, retargeted, scaler=scaler, context_samples=samples
    )

    for horizon in SCORED_HORIZONS:
        task = f"hypo_{horizon}"
        position = TASKS.index(task)
        expected_target = retargeted[f"severe_hypo_{horizon}_label"].to_numpy(float)
        expected_mask = ~injected_active & retargeted[
            f"severe_hypo_{horizon}_label"
        ].notna().to_numpy()

        slow_target = np.array(
            [
                float(dataset[i]["training_labels"][position])
                for i in range(len(dataset))
            ]
        )
        slow_mask = np.array(
            [bool(dataset[i]["label_mask"][position]) for i in range(len(dataset))]
        )
        fast_target = np.array(
            [float(fast[i]["training_labels"][position]) for i in range(len(fast))]
        )
        np.testing.assert_array_equal(slow_target, expected_target)
        np.testing.assert_array_equal(slow_mask, expected_mask)
        np.testing.assert_array_equal(fast_target, expected_target)

        # the evaluation label is untouched, which is what keeps the endpoint honest
        slow_label = np.array(
            [float(dataset[i]["labels"][position]) for i in range(len(dataset))]
        )
        np.testing.assert_array_equal(
            slow_label, retargeted[f"{task}_label"].to_numpy(float)
        )


def test_retargeting_refuses_a_frame_that_lacks_the_severe_columns() -> None:
    _frame, windows, _scaler, _samples = _cohort()
    stripped = windows.drop(columns=["severe_hypo_30_label"])
    with pytest.raises(ValueError, match="missing columns"):
        attach_severe_training_targets(stripped)


def test_the_objective_enum_admits_the_new_arm_and_still_rejects_nonsense() -> None:
    from glycoshift.config import validate_config

    config = load_config(CONFIG)
    for value in ("bce", "tls_exp", "severe_retarget"):
        config["training"]["event_objective"] = value
        if value == "tls_exp":
            config["training"]["tls_gamma_per_hour"] = 0.2
        validate_config(config)

    config["training"]["event_objective"] = "not_an_objective"
    with pytest.raises(ValueError, match="training.event_objective must be"):
        validate_config(config)


def test_training_under_the_retargeted_objective_runs_and_records_what_moved(
    tmp_path: Path,
) -> None:
    """End to end, on a fixture that really does contain severe events.

    Eleven participants per source is the count the deep smoke test uses to give
    every development source severe-event support, so the audit below is not
    vacuous the way a three-participant cohort would make it.
    """

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
    config["training"]["event_objective"] = "severe_retarget"
    config["evaluation"]["threshold_candidate_count"] = 5
    config["evaluation"]["event_ap_candidate_count"] = 5
    config["evaluation"]["selection_operational_candidate_count"] = 3
    config["output"]["run_dir"] = str(tmp_path / "severe-run")

    run_dir = run_deep_training(config, architecture="glycoshift")

    audit_path = run_dir / "severe_target_audit.json"
    assert audit_path.exists(), "the retargeting must leave a record of what it did"
    audit = json.loads(audit_path.read_text())
    assert audit["objective"] == "severe_retarget"
    assert not (run_dir / "tls_target_audit.json").exists()

    for horizon in SCORED_HORIZONS:
        entry = audit["tasks"][f"hypo_{horizon}"]
        assert entry["severe_positive_count"] > 0, "fixture has no severe events"
        assert entry["newly_unmasked"] > 0, "the mask did not widen"
        assert entry["newly_masked"] == 0
    # the trained head is still the one the severe score is bound to
    summary = json.loads((run_dir / "model_summary.json").read_text())
    assert set(summary["tasks"]) == set(TASKS)
