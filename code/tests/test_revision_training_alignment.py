"""Known-answer gates for the explicitly opted-in final-review training path."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import average_precision_score

from glycoshift.config import load_config
from glycoshift.deep_training import (
    _drop_action_groups,
    _positive_weights,
    _selection_score,
    _training_target_audit,
    run_deep_training,
)
from glycoshift.fast_sequences import FastWindowSequenceDataset
from glycoshift.schema import ACTION_FIELDS
from glycoshift.sequences import (
    REVISION_CONTRACT,
    SEQUENCE_FEATURES,
    TASKS,
    ParticipantSequenceStore,
    SequenceScaler,
    WindowSequenceDataset,
    revision_anchor_mask,
    sample_training_windows,
    training_target_arrays,
    training_window_sampling_masks,
)
from glycoshift.severe_target import attach_severe_training_targets


def _fixture():
    times = pd.date_range("2026-01-01", periods=12, freq="5min", tz="UTC")
    frame = pd.DataFrame(
        {
            "source_id": "s",
            "participant_id": "p",
            "timestamp": times,
            "cgm_mg_dl": [100.0, 110, 120, 130, 60, 140, 150, 160, 170, 180, 190, 200],
            "cgm_observed": True,
            "cgm_interpolated": False,
            "cgm_lower_censored": False,
        }
    )
    for field in ACTION_FIELDS:
        frame[field] = np.arange(len(frame), dtype=float)
        frame[f"{field}_observed"] = True
        frame[f"{field}_available"] = True
        frame[f"{field}_zero_ambiguous"] = False
    ends = np.arange(3, 9)
    windows = pd.DataFrame(
        {
            "source_id": "s",
            "participant_id": "p",
            "participant_end_index": ends,
            "context_start": times[ends - 2],
            "context_end": times[ends],
            "partition": "train",
            "hypo_currently_active": [False, True, False, False, False, False],
            "hyper_currently_active": False,
            "severe_hypo_currently_active": False,
        }
    )
    for task in TASKS:
        windows[f"{task}_label"] = [1, 1, 1, 0, 0, 0]
    windows["severe_hypo_30_label"] = [0, 1, 0, 0, 0, 1]
    windows["severe_hypo_60_label"] = [0, 1, 0, 0, 0, 0]
    for horizon in (30, 60):
        windows[f"cgm_{horizon}_target"] = 100.0
    windows = attach_severe_training_targets(windows)
    scaler = SequenceScaler.fit(frame, {("s", "p")})
    return frame, windows, scaler


def _dataset(cls, frame, windows, scaler, **kwargs):
    return cls(frame, windows, scaler=scaler, context_samples=3, **kwargs)


def test_revision_weights_count_the_actual_severe_target_and_retained_mask():
    _, windows, _ = _fixture()
    ordinary = _positive_weights(windows)
    revised = _positive_weights(windows, use_training_targets=True)
    torch.testing.assert_close(ordinary, torch.tensor([1.5, 1.5, 1.0, 1.0]))
    torch.testing.assert_close(revised, torch.tensor([2.0, 5.0, 1.0, 1.0]))
    # Removing a severe-positive row changes only the actual target/mask count.
    windows.loc[5, "hypo_30_training_mask"] = False
    revised = _positive_weights(windows, use_training_targets=True)
    assert revised[0].item() == 4.0
    audit = _training_target_audit(windows, revised)["tasks"]["hypo_30"]
    assert audit["retained_count"] == 5
    assert audit["positive_count"] == 1
    assert audit["negative_count"] == 4
    assert audit["positive_weight"] == 4
    torch.testing.assert_close(_positive_weights(windows), ordinary)
    rare = pd.concat([windows] * 100, ignore_index=True)
    rare["hypo_30_training_target"] = 0.0
    rare.loc[0, "hypo_30_training_target"] = 1.0
    assert _positive_weights(rare, use_training_targets=True)[0].item() == 100.0


def test_revision_retains_the_54_to_70_band_and_uses_target_for_selection():
    _, windows, _ = _fixture()
    target, mask = training_target_arrays(windows)
    assert windows.loc[1, "hypo_currently_active"]
    assert mask[1, 0] and target[1, 0] == 1
    for task in TASKS:
        windows[f"{task}_probability"] = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]
    expected = np.mean(
        [
            average_precision_score(
                target[mask[:, i], i], windows.loc[mask[:, i], f"{task}_probability"]
            )
            for i, task in enumerate(TASKS)
        ]
    )
    assert _selection_score(windows, use_training_targets=True) == pytest.approx(
        expected
    )
    assert _selection_score(windows) != pytest.approx(expected)


def test_revision_sampling_uses_masked_targets_and_preserves_auxiliary_rows():
    _, windows, _ = _fixture()
    windows["row_id"] = np.arange(len(windows))
    for task in TASKS:
        windows[f"{task}_label"] = [1, 0, 0, 0, 0, 0]
        windows[f"{task}_training_target"] = 0.0
        windows[f"{task}_training_mask"] = True
        windows.loc[[2, 3, 4], f"{task}_training_mask"] = False
    windows.loc[[1, 2], "hypo_30_training_target"] = 1.0
    for horizon in (30, 60):
        windows[f"cgm_{horizon}_target"] = np.nan
    windows.loc[3, "cgm_30_target"] = 100.0
    positive, usable = training_window_sampling_masks(windows)
    np.testing.assert_array_equal(positive, [False, True, False, False, False, False])
    np.testing.assert_array_equal(usable, [True, True, False, True, False, True])
    kwargs = {"negative_ratio": 0.0, "seed": 42}
    legacy = sample_training_windows(windows, **kwargs)
    explicit_legacy = sample_training_windows(
        windows, use_training_targets=False, **kwargs
    )
    pd.testing.assert_frame_equal(legacy, explicit_legacy)
    assert legacy["row_id"].tolist() == [0]
    revised = sample_training_windows(windows, use_training_targets=True, **kwargs)
    assert revised["row_id"].tolist() == [1]
    retained = sample_training_windows(
        windows, use_training_targets=True, negative_ratio=20, seed=42
    )
    assert retained["row_id"].tolist() == [0, 1, 3, 5]
    sampled = sample_training_windows(
        windows, use_training_targets=True, negative_ratio=1, seed=42
    )
    assert len(sampled) == 2
    assert sampled.loc[sampled["row_id"] == 1, "sampling_weight"].item() == 1
    assert sampled.loc[sampled["row_id"] != 1, "sampling_weight"].item() == 3


@pytest.mark.parametrize("cls", [WindowSequenceDataset, FastWindowSequenceDataset])
def test_revision_missing_targets_require_an_explicit_false_mask(cls):
    frame, windows, scaler = _fixture()
    windows.loc[0, "hypo_30_training_target"] = np.nan
    with pytest.raises(ValueError, match="unmasked training targets"):
        _dataset(cls, frame, windows, scaler, revision_contract=REVISION_CONTRACT)
    windows.loc[0, "hypo_30_training_mask"] = False
    dataset = _dataset(cls, frame, windows, scaler, revision_contract=REVISION_CONTRACT)
    assert dataset[0]["training_labels"][0] == 0
    assert not dataset[0]["label_mask"][0]
    assert pd.isna(dataset.windows.loc[0, "hypo_30_training_target"])
    windows = windows.drop(columns="hypo_30_training_mask")
    with pytest.raises(ValueError, match="requires hypo_30_training_mask"):
        _dataset(cls, frame, windows, scaler, revision_contract=REVISION_CONTRACT)


@pytest.mark.parametrize(
    "column", ["cgm_observed", "cgm_interpolated", "cgm_lower_censored"]
)
def test_revision_anchor_provenance_fails_closed(column):
    frame, windows, scaler = _fixture()
    for cls in (WindowSequenceDataset, FastWindowSequenceDataset):
        with pytest.raises(ValueError, match="require provenance column"):
            _dataset(
                cls,
                frame.drop(columns=column),
                windows,
                scaler,
                revision_contract=REVISION_CONTRACT,
            )


@pytest.mark.parametrize(
    "column,value",
    [
        ("cgm_observed", False),
        ("cgm_interpolated", True),
        ("cgm_lower_censored", True),
        ("cgm_mg_dl", np.nan),
    ],
)
def test_revision_rejects_unobserved_interpolated_censored_or_nonfinite_anchor(
    column, value
):
    frame, windows, scaler = _fixture()
    frame.loc[4, column] = value
    store = ParticipantSequenceStore.from_frame(frame)
    expected = np.ones(len(windows), dtype=bool)
    expected[1] = False
    np.testing.assert_array_equal(revision_anchor_mask(windows, store), expected)
    for cls in (WindowSequenceDataset, FastWindowSequenceDataset):
        with pytest.raises(
            ValueError, match="require observed noninterpolated uncensored"
        ):
            _dataset(cls, frame, windows, scaler, revision_contract=REVISION_CONTRACT)


def test_anchor_binding_is_source_scoped_and_checks_normalized_time_against_index():
    frame, windows, _ = _fixture()
    store = ParticipantSequenceStore.from_frame(frame)
    shifted = windows.copy()
    shifted["context_end"] = shifted["context_end"].dt.tz_convert("Asia/Shanghai")
    assert revision_anchor_mask(shifted, store).all()
    shifted.loc[0, "participant_end_index"] += 1
    with pytest.raises(ValueError, match="conflicts with context_end"):
        revision_anchor_mask(shifted, store)
    shifted = windows.copy()
    shifted.loc[0, "context_end"] += pd.Timedelta(seconds=1)
    with pytest.raises(ValueError, match="no exact participant timestamp"):
        revision_anchor_mask(shifted, store)
    shifted = windows.copy()
    shifted.loc[0, "source_id"] = "different-source"
    with pytest.raises(ValueError, match="participant is missing"):
        revision_anchor_mask(shifted, store)


def test_cgm_only_fast_slow_parity_action_invariance_and_legacy_default():
    frame, windows, scaler = _fixture()
    options = {
        "input_feature_mode": "cgm_mask_clock",
        "revision_contract": REVISION_CONTRACT,
    }
    slow = _dataset(WindowSequenceDataset, frame, windows, scaler, **options)
    fast = _dataset(FastWindowSequenceDataset, frame, windows, scaler, **options)
    for index in range(len(windows)):
        for key, value in slow[index].items():
            assert torch.equal(value, fast[index][key]), key
        assert slow[index]["sequence"].shape == (3, len(SEQUENCE_FEATURES))
        assert torch.count_nonzero(slow[index]["sequence"][:, 2:-2]) == 0
    changed = frame.copy()
    for field in ACTION_FIELDS:
        changed[field] = np.nan
        changed[f"{field}_available"] = False
        changed[f"{field}_observed"] = False
        changed[f"{field}_zero_ambiguous"] = True
    cgm_scaler = SequenceScaler.fit(
        frame, {("s", "p")}, input_feature_mode="cgm_mask_clock"
    )
    changed_scaler = SequenceScaler.fit(
        changed, {("s", "p")}, input_feature_mode="cgm_mask_clock"
    )
    assert cgm_scaler.as_dict() == changed_scaler.as_dict()
    # No action columns are needed at all in this explicitly declared mode.
    minimal = changed.drop(
        columns=[c for c in changed if any(c.startswith(f) for f in ACTION_FIELDS)]
    )
    other = _dataset(
        FastWindowSequenceDataset, minimal, windows, changed_scaler, **options
    )
    for index in range(len(windows)):
        assert torch.equal(slow[index]["sequence"], other[index]["sequence"])
    original = _dataset(WindowSequenceDataset, frame, windows, scaler)
    explicit = _dataset(
        WindowSequenceDataset, frame, windows, scaler, input_feature_mode="multimodal"
    )
    assert torch.equal(original[1]["sequence"], explicit[1]["sequence"])
    assert not torch.equal(original[1]["sequence"], slow[1]["sequence"])
    legacy_without_provenance = _dataset(
        WindowSequenceDataset,
        frame.drop(columns=["cgm_observed", "cgm_interpolated"]),
        windows,
        scaler,
    )
    assert torch.equal(
        original[1]["sequence"], legacy_without_provenance[1]["sequence"]
    )
    batch = slow[0]["sequence"].unsqueeze(0)
    assert torch.equal(
        batch, _drop_action_groups(batch, probabilities=(0.0,) * len(ACTION_FIELDS))
    )


def test_revision_runner_uses_callbacks_and_records_actual_contract(tmp_path):
    frame, windows, _ = _fixture()
    frames, blocks = [], []
    for partition in ("train", "selection", "calibration"):
        frames.append(frame.assign(participant_id=partition))
        blocks.append(windows.assign(participant_id=partition, partition=partition))
    config = load_config(Path(__file__).parents[1] / "configs" / "smoke.yaml")
    config["data"]["development_sources"] = ["s"]
    config["protocol"]["context_hours"] = 0.25
    config["training"].update(
        {
            "revision_contract": REVISION_CONTRACT,
            "event_objective": "severe_retarget",
            "maximum_epochs": 1,
            "batch_size": 4,
            "effective_batch_size": 4,
            "num_workers": 0,
            "device": "cpu",
        }
    )
    config["model"]["input_feature_mode"] = "cgm_mask_clock"
    config["output"]["run_dir"] = str(tmp_path / "revision")
    with pytest.raises(
        ValueError, match="requires prepared_data and fixed-frame callbacks"
    ):
        run_deep_training(config, architecture="lstm")
    calls = []

    def checkpoint(predictions, **kwargs):
        calls.append(("checkpoint", len(predictions)))
        return 0.25, 0.0, {"fixed_frame": True}

    def calibrator(selection, calibration, **kwargs):
        calls.append(("calibration", len(calibration)))
        return {task: {"temperature": 1.0} for task in TASKS}, {"fixed_frame": True}

    def reporter(predictions, **kwargs):
        calls.append(("reporter", len(predictions)))
        return {"fixed_frame": True}

    prepared_frame = pd.concat(frames, ignore_index=True)
    prepared_windows = pd.concat(blocks, ignore_index=True)
    bad_target = prepared_windows.copy()
    bad_target.loc[0, "hypo_30_training_target"] = 1.0
    with pytest.raises(ValueError, match="target differs from severe label"):
        run_deep_training(
            config,
            architecture="lstm",
            prepared_data=(prepared_frame, bad_target, {}),
            revision_checkpoint_proxy=checkpoint,
            revision_calibrator=calibrator,
            revision_reporter=reporter,
        )
    assert calls == []
    directory = run_deep_training(
        config,
        architecture="lstm",
        prepared_data=(
            prepared_frame,
            prepared_windows,
            {},
        ),
        revision_checkpoint_proxy=checkpoint,
        revision_calibrator=calibrator,
        revision_reporter=reporter,
    )
    assert calls == [("checkpoint", 6), ("calibration", 6), ("reporter", 12)]
    summary = json.loads((directory / "training_summary.json").read_text())
    assert summary["revision_contract"] == REVISION_CONTRACT
    assert summary["input_feature_mode"] == "cgm_mask_clock"
    assert set(summary["feature_group_dropout_probabilities"].values()) == {0.0}
    audit = json.loads((directory / "training_target_audit.json").read_text())
    assert audit["tasks"]["hypo_30"]["positive_weight"] == 2.0
    assert audit["tasks"]["hypo_60"]["positive_weight"] == 5.0
    assert audit["positive_weight_mode"] == "prevalence"
    assert audit["positive_weight_basis"] == "retained_training_targets_and_masks"
    metrics = json.loads((directory / "metrics.json").read_text())
    assert metrics["operational_fixed_frame"] == {"fixed_frame": True}
    assert "operational_v1_3" not in metrics
