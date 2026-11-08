"""Exact numerical and RNG gates for scoped prediction bookkeeping changes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import Dataset

from glycoshift import deep_training
from glycoshift.baselines import LSTMBaseline, TCNBaseline
from glycoshift.memory import ContiguousStrictPastMemoryBank
from glycoshift.sequences import (
    REGRESSION_HORIZONS,
    SEQUENCE_FEATURES,
    TASKS,
    training_target_arrays,
)


class _NoFrameRowSelection(pd.DataFrame):
    @property
    def loc(self):
        raise AssertionError("selection scoring must not copy whole frame rows")


def _score_frame(revision):
    frame = pd.DataFrame(index=pd.Index([31, -5, 31, 90, 4, 12, 88, 0], name="row"))
    frame["hypo_currently_active"] = [False] * 4 + [True] * 4
    frame["hyper_currently_active"] = [True] * 4 + [False] * 4
    positions = ([0, 1, 2, 3], [1, 2, 4, 5], [0, 2, 4, 6], [3, 5, 6, 7])
    for position, task in enumerate(TASKS):
        mask = np.zeros(len(frame), dtype=bool)
        if revision:
            mask[positions[position]] = True
        else:
            mask = ~frame[f"{task.split('_')[0]}_currently_active"].to_numpy()
        labels = np.full(len(frame), np.nan)
        labels[mask] = [0, 1, 0, 1]
        frame[f"{task}_label"] = labels
        if revision:
            frame[f"{task}_training_target"] = labels
            frame[f"{task}_training_mask"] = mask
        probability = np.full(len(frame), np.nan)
        probability[mask] = [0.5, 0.5, 0.2, 0.8]
        dtype = np.float32 if position % 2 else np.float64
        frame[f"{task}_probability"] = probability.astype(dtype)
    return frame


def _legacy_score_inputs(frame, revision):
    resolved = training_target_arrays(frame) if revision else None
    inputs = []
    for position, task in enumerate(TASKS):
        if resolved is None:
            kind = task.split("_")[0]
            eligible = frame.loc[~frame[f"{kind}_currently_active"]]
            labels = eligible[f"{task}_label"].to_numpy(dtype=int)
        else:
            targets, masks = resolved
            eligible = frame.loc[masks[:, position]]
            labels = targets[masks[:, position], position]
        if len(np.unique(labels)) >= 2:
            inputs.append((labels, eligible[f"{task}_probability"]))
    return inputs


@pytest.mark.parametrize("revision", [False, True])
@pytest.mark.parametrize("single_class", [False, True])
def test_score_matches_legacy_inputs_and_known_tied_ap_without_frame_copy(
    monkeypatch, revision, single_class
):
    frame = _score_frame(revision)
    if single_class:
        task = TASKS[-1]
        column = f"{task}_training_target" if revision else f"{task}_label"
        frame[column] = frame[column].fillna(0) * 0
        frame = frame.drop(columns=f"{task}_probability")
    original = frame.copy(deep=True)
    expected = _legacy_score_inputs(frame, revision)
    observed = []

    def record(labels, probability):
        observed.append((labels.copy(), probability.copy()))
        return average_precision_score(labels, probability)

    monkeypatch.setattr(deep_training, "average_precision_score", record)
    actual = deep_training._selection_score(
        _NoFrameRowSelection(frame), use_training_targets=revision
    )
    assert actual == np.mean([average_precision_score(*values) for values in expected])
    assert actual == pytest.approx(5 / 6)
    assert len(observed) == len(expected) == (3 if single_class else 4)
    for (labels, probability), (old_labels, old_probability) in zip(
        observed, expected, strict=True
    ):
        assert labels.dtype == old_labels.dtype
        np.testing.assert_array_equal(labels, old_labels)
        pd.testing.assert_series_equal(probability, old_probability, check_exact=True)
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("revision", [False, True])
def test_all_single_class_tasks_keep_the_original_error(revision):
    frame = _score_frame(revision)
    for task in TASKS:
        column = f"{task}_training_target" if revision else f"{task}_label"
        frame[column] = 0.0
        frame = frame.drop(columns=f"{task}_probability")
    with pytest.raises(
        ValueError, match="selection split has no task with both classes"
    ):
        deep_training._selection_score(
            _NoFrameRowSelection(frame), use_training_targets=revision
        )


class _TinyPredictionDataset(Dataset):
    def __init__(self):
        times = pd.date_range("2026-01-01", periods=5, freq="15min", tz="UTC")
        self.windows = pd.DataFrame(
            {
                "source_id": "s",
                "participant_id": "p",
                "context_start": times - pd.Timedelta(minutes=10),
                "context_end": times,
            },
            index=[8, 4, 99, -1, 6],
        )
        self.sequence = torch.arange(
            5 * 3 * len(SEQUENCE_FEATURES), dtype=torch.float32
        ).reshape(5, 3, len(SEQUENCE_FEATURES)) / 100
        self.metadata_calls = []

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        return {"sequence": self.sequence[index], "dataset_index": index}

    def metadata(self, index):
        self.metadata_calls.append(index)
        row = self.windows.iloc[index]
        return {
            "source_id": str(row["source_id"]),
            "participant_id": str(row["participant_id"]),
            "context_start": pd.Timestamp(row["context_start"]),
            "context_end": pd.Timestamp(row["context_end"]),
        }


def _legacy_predict(model, dataset):
    """Retain the previous unconditional metadata path as the CPU parity oracle."""
    device = torch.device("cpu")
    embeddings = deep_training._encode_dataset(
        model, dataset, device=device, batch_size=2, num_workers=0
    )
    bank = None
    if model.supports_memory:
        bank = ContiguousStrictPastMemoryBank(
            k=model.config.memory_k, stride_minutes=5,
            device=device, dtype=embeddings.dtype,
        )
        bank.build_from_windows(embeddings=embeddings, windows=dataset.windows)
    logits = {task: np.empty(len(dataset), dtype=np.float32) for task in TASKS}
    regression = {
        str(h): np.empty(len(dataset), dtype=np.float32) for h in REGRESSION_HORIZONS
    }
    loader = deep_training._loader(
        dataset, batch_size=2, num_workers=0, source_balanced=False
    )
    model.eval()
    with torch.no_grad():
        for batch in loader:
            indices = batch["dataset_index"].long()
            representation = embeddings[indices].to(device)
            metadata = [dataset.metadata(int(index)) for index in indices.tolist()]
            memory, padding = (None, None) if bank is None else bank.retrieve(
                query_embeddings=representation, query_metadata=metadata
            )
            outputs = model.forward_from_representation(
                representation, memory_embeddings=memory, memory_padding_mask=padding
            )
            for task in TASKS:
                logits[task][indices] = outputs["logits"][task].float().cpu().numpy()
            for horizon in REGRESSION_HORIZONS:
                regression[str(horizon)][indices] = (
                    outputs["regression"][str(horizon)].float().cpu().numpy()
                )
    predictions = dataset.windows.copy()
    for task in TASKS:
        predictions[f"{task}_logit"] = logits[task]
        predictions[f"{task}_probability"] = (
            1.0 / (1.0 + np.exp(-np.clip(logits[task], -50, 50)))
        )
    for horizon in REGRESSION_HORIZONS:
        predictions[f"cgm_{horizon}_prediction_scaled"] = regression[str(horizon)]
    return predictions


@pytest.mark.parametrize("model_class", [LSTMBaseline, TCNBaseline])
@pytest.mark.parametrize("with_memory", [False, True])
def test_prediction_metadata_branch_preserves_exact_output_and_torch_rng(
    model_class, with_memory
):
    torch.manual_seed(17)
    model = model_class(d_model=4, layers=1)
    if with_memory:
        model.attach_memory(memory_k=2, n_heads=1, dropout=0.1)
    dataset = _TinyPredictionDataset()
    state = torch.get_rng_state().clone()
    expected = _legacy_predict(model, dataset)
    expected_rng = torch.get_rng_state().clone()
    assert dataset.metadata_calls == list(range(len(dataset)))
    assert not torch.equal(state, expected_rng), "oracle must exercise DataLoader RNG"
    dataset.metadata_calls.clear()
    torch.set_rng_state(state)
    actual = deep_training._predict_dataset(
        model, dataset, device=torch.device("cpu"), batch_size=2,
        num_workers=0, memory_stride_minutes=5,
    )
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert dataset.metadata_calls == (list(range(len(dataset))) if with_memory else [])
