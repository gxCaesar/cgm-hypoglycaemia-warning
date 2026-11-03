"""Array-backed sample assembly for the window datasets.

`WindowSequenceDataset.__getitem__` reconstructs a sample from pandas on every
call: a 51-column row Series, a 72-row frame slice, a `pd.to_datetime` parse,
`to_numpy` per feature, and a Python loop for the time-since-observation
channel.  Measured on the frozen protocol that is about 390 microseconds per
sample, or roughly 41 Python-level calls, against a model step of about 10
milliseconds for 256 samples -- so about ninety percent of training wall clock
is sample assembly rather than compute.

This module precomputes the same values once into contiguous arrays and reduces
`__getitem__` to slicing.  It changes no number: `FastWindowSequenceDataset` is
required to return tensors bit-identical to the dataset it mirrors, and
`tests/test_fast_sequences.py` asserts that against the original implementation
rather than against a reimplementation of it.

The one channel that cannot be precomputed pointwise is time-since-observation.
`_time_since_observation` seeds its counter with the *window* length and
normalises by it, so the value at a position depends on the window it is being
read in, not only on the participant timeline.  It is reconstructed here from a
global last-observed index with an exact vectorised identity; see
`_time_since_from_last_observed`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from .sequences import (
    ACTION_FIELDS,
    ANCHOR_COLUMN,
    LOSS_WEIGHT_SUFFIX,
    REGRESSION_HORIZONS,
    SEQUENCE_FEATURES,
    TASKS,
    TRAINING_MASK_SUFFIX,
    ParticipantSequenceStore,
    SequenceScaler,
    WindowSequenceDataset,
    validate_optional_loss_weights,
    validate_optional_training_masks,
)

# Column offsets inside the assembled feature matrix, in SEQUENCE_FEATURES order:
# cgm_value, cgm_observed, then five columns per action field, then sin/cos.
_CGM_COLUMNS = 2
_COLUMNS_PER_ACTION = 5
_TIME_SINCE_OFFSET = 4  # value, available, observed, ambiguous, time_since


def _action_time_since_column(field_position: int) -> int:
    return _CGM_COLUMNS + field_position * _COLUMNS_PER_ACTION + _TIME_SINCE_OFFSET


def _last_observed_index(observed: np.ndarray) -> np.ndarray:
    """Index of the most recent True at or before each position, else -1."""

    positions = np.arange(len(observed), dtype=np.int64)
    marked = np.where(observed, positions, -1)
    return np.maximum.accumulate(marked)


def _time_since_from_last_observed(
    last_observed: np.ndarray,
    start: int,
    width: int,
) -> np.ndarray:
    """Reproduce `_time_since_observation` for one window, vectorised.

    The original seeds `since` with the window width and clips to it, so a
    position with no observation at or before it *inside the window* saturates
    at exactly 1.0 however far back the previous observation lies. Positions
    that do have one take `(j - last_observed[j]) / width`, which is always
    below 1.0 because the gap cannot exceed `width - 1`.
    """

    absolute = np.arange(start, start + width, dtype=np.int64)
    window_last = last_observed[start : start + width]
    gap = absolute - window_last
    gap = np.where(window_last >= start, gap, width)
    return (np.minimum(gap, width) / max(1, width)).astype(np.float32)


@dataclass(frozen=True)
class ParticipantFeatureBlock:
    """One participant's timeline, assembled once."""

    features: np.ndarray  # (readings, len(SEQUENCE_FEATURES)) float32
    action_last_observed: tuple[np.ndarray, ...]  # per action field, int64
    lower_censored: np.ndarray  # bool, for the per-window refusal


def build_participant_block(
    frame: pd.DataFrame,
    *,
    scaler: SequenceScaler,
) -> ParticipantFeatureBlock:
    """Assemble one participant's pointwise features exactly as the original does."""

    readings = len(frame)
    features = np.zeros((readings, len(SEQUENCE_FEATURES)), dtype=np.float64)

    cgm = frame["cgm_mg_dl"].to_numpy(dtype=float)
    cgm_observed = (
        frame["cgm_observed"].to_numpy(dtype=bool)
        if "cgm_observed" in frame
        else np.isfinite(cgm)
    )
    features[:, 0] = scaler.transform("cgm_mg_dl", cgm)
    features[:, 1] = cgm_observed.astype(float)

    action_last_observed: list[np.ndarray] = []
    for position, field in enumerate(ACTION_FIELDS):
        base = _CGM_COLUMNS + position * _COLUMNS_PER_ACTION
        observed = frame[f"{field}_observed"].to_numpy(dtype=bool)
        raw = frame[field].to_numpy(dtype=float)
        filled = np.where(observed, raw, scaler.values[field].mean)
        value = scaler.transform(field, filled)
        features[:, base + 0] = np.where(observed, value, 0.0)
        features[:, base + 1] = frame[f"{field}_available"].to_numpy(dtype=float)
        features[:, base + 2] = observed.astype(float)
        features[:, base + 3] = frame[f"{field}_zero_ambiguous"].to_numpy(dtype=float)
        # column base + 4 is time-since, filled per window
        action_last_observed.append(_last_observed_index(observed))

    local_column = (
        frame["timestamp_local"] if "timestamp_local" in frame else frame["timestamp"]
    )
    local = pd.to_datetime(local_column)
    minute = local.dt.hour.to_numpy() * 60 + local.dt.minute.to_numpy()
    phase = 2 * np.pi * minute / (24 * 60)
    features[:, -2] = np.sin(phase)
    features[:, -1] = np.cos(phase)

    lower_censored = (
        frame["cgm_lower_censored"].to_numpy(dtype=bool)
        if "cgm_lower_censored" in frame
        else np.zeros(readings, dtype=bool)
    )
    return ParticipantFeatureBlock(
        features=features.astype(np.float32),
        action_last_observed=tuple(action_last_observed),
        lower_censored=lower_censored,
    )


_BLOCK_CACHE_ATTRIBUTE = "_fast_feature_blocks"


def _scaler_identity(scaler: SequenceScaler) -> tuple:
    """Exact content identity, so a cache hit cannot cross two scalings."""

    return tuple(
        (field, values["mean"], values["std"])
        for field, values in sorted(scaler.as_dict().items())
    )


def _shared_participant_blocks(
    store: ParticipantSequenceStore,
    scaler: SequenceScaler,
    keys: list,
) -> list[ParticipantFeatureBlock]:
    """Build each participant's block once per (store, scaling), not per dataset.

    A training run constructs four datasets -- train, selection, calibration and
    the stream adapter -- over one participant store. Building blocks per
    instance would hold four copies of the same 46.6M-row feature matrix, about
    3.5 GB each, and the per-lane RAM ceiling that gates a launch is derived from
    exactly this peak. The cache lives on the store because that is what the
    blocks are a function of, and it is keyed by the scaler's contents rather
    than its identity so a differently fitted scaler can never hit it.
    """

    cache = getattr(store, _BLOCK_CACHE_ATTRIBUTE, None)
    if cache is None:
        cache = {}
        object.__setattr__(store, _BLOCK_CACHE_ATTRIBUTE, cache)
    signature = (_scaler_identity(scaler), tuple(keys))
    blocks = cache.get(signature)
    if blocks is None:
        blocks = [
            build_participant_block(store.participants[key], scaler=scaler)
            for key in keys
        ]
        cache[signature] = blocks
    return blocks


class FastWindowSequenceDataset(WindowSequenceDataset):
    """`WindowSequenceDataset` with sample assembly moved to arrays.

    Subclasses rather than replaces, so every construction-time validation --
    required columns, participant coverage, source indexing -- is inherited
    unchanged and only the hot path is overridden. The precompute mirrors the
    parent's own expressions line for line; where the parent computes in float64
    and narrows once at the end, so does this.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._prepare_arrays()

    # -- precompute ---------------------------------------------------------

    def _prepare_arrays(self) -> None:
        keys = sorted(self._participants)
        self._slot_of_key = {key: slot for slot, key in enumerate(keys)}
        self._blocks = _shared_participant_blocks(
            self.participant_store, self.scaler, keys
        )

        windows = self.windows
        count = len(windows)

        self._slot = np.array(
            [
                self._slot_of_key[
                    (str(source), str(participant))
                ]
                for source, participant in zip(  # noqa: B905
                    windows["source_id"].astype(str),
                    windows["participant_id"].astype(str),
                )
            ],
            dtype=np.int64,
        )
        self._end_index = windows["participant_end_index"].to_numpy(dtype=np.int64)

        self._labels = np.stack(
            [windows[f"{task}_label"].to_numpy(dtype=float) for task in TASKS],
            axis=1,
        ).astype(np.float32)

        training_columns = []
        for task in TASKS:
            target = f"{task}_training_target"
            column = target if target in windows.columns else f"{task}_label"
            training_columns.append(windows[column].to_numpy(dtype=float))
        self._training_labels = np.stack(training_columns, axis=1).astype(np.float32)
        if not np.isfinite(self._training_labels).all() or np.any(
            (self._training_labels < 0) | (self._training_labels > 1)
        ):
            raise ValueError("training labels must be finite probabilities")

        validate_optional_training_masks(windows)
        mask_columns = []
        for task in TASKS:
            override = f"{task}{TRAINING_MASK_SUFFIX}"
            if override in windows.columns:
                mask_columns.append(windows[override].to_numpy(dtype=bool))
                continue
            mask_columns.append(
                (~windows[f"{task.split('_')[0]}_currently_active"].to_numpy(bool))
                & windows[f"{task}_label"].notna().to_numpy()
            )
        self._label_mask = np.stack(mask_columns, axis=1)

        validate_optional_loss_weights(windows)
        if any(f"{task}{LOSS_WEIGHT_SUFFIX}" in windows.columns for task in TASKS):
            self._loss_weight: np.ndarray | None = np.stack(
                [
                    (
                        windows[f"{task}{LOSS_WEIGHT_SUFFIX}"].to_numpy(dtype=float)
                        if f"{task}{LOSS_WEIGHT_SUFFIX}" in windows.columns
                        else np.ones(len(windows), dtype=float)
                    )
                    for task in TASKS
                ],
                axis=1,
            ).astype(np.float32)
        else:
            self._loss_weight = None

        # Training frame only, same reason as the reference dataset.
        if ANCHOR_COLUMN in windows.columns:
            self._anchor: np.ndarray | None = windows[ANCHOR_COLUMN].to_numpy(
                dtype=np.float32
            )
        else:
            self._anchor = None

        available_columns = []
        raw_columns = []
        mean = self.scaler.values["cgm_mg_dl"].mean
        for horizon in REGRESSION_HORIZONS:
            target = f"cgm_{horizon}_target"
            present = windows[target].notna().to_numpy()
            flag = f"{target}_available"
            declared = (
                windows[flag].to_numpy(dtype=bool)
                if flag in windows.columns
                else present
            )
            available = declared & present
            available_columns.append(available)
            raw = windows[target].to_numpy(dtype=float)
            raw_columns.append(np.where(available, raw, mean))
        self._regression_available = np.stack(available_columns, axis=1)
        self._regression_targets = self.scaler.transform(
            "cgm_mg_dl", np.stack(raw_columns, axis=1)
        ).astype(np.float32)

        self._source_index = np.array(
            [
                self.source_to_index[str(source)]
                for source in windows["source_id"].astype(str)
            ],
            dtype=np.int64,
        )
        self._sampling_weight = windows["sampling_weight"].to_numpy(dtype=float).astype(
            np.float32
        )
        assert len(self._slot) == count

    # -- hot path -----------------------------------------------------------

    def _assemble_sequence(self, index: int) -> np.ndarray:
        block = self._blocks[self._slot[index]]
        end_index = int(self._end_index[index])
        width = self.context_samples
        start_index = end_index - width + 1
        if start_index < 0:
            raise IndexError("window context starts before participant data")
        if end_index + 1 > len(block.features):
            raise IndexError("window context has an unexpected length")
        if block.lower_censored[start_index : end_index + 1].any():
            raise ValueError("lower-censored CGM cannot enter model context features")

        sequence = block.features[start_index : end_index + 1].copy()
        for position in range(len(ACTION_FIELDS)):
            sequence[:, _action_time_since_column(position)] = (
                _time_since_from_last_observed(
                    block.action_last_observed[position], start_index, width
                )
            )
        if not np.isfinite(sequence).all():
            raise ValueError("sequence contains non-finite values after masking")
        return sequence

    def __getitem__(self, index: int) -> dict[str, object]:
        sequence = self._assemble_sequence(index)
        sample: dict[str, object] = {
            "sequence": torch.from_numpy(sequence),
            "labels": torch.from_numpy(self._labels[index].copy()),
            "training_labels": torch.from_numpy(self._training_labels[index].copy()),
            "label_mask": torch.from_numpy(self._label_mask[index].copy()),
            "regression_targets": torch.from_numpy(
                self._regression_targets[index].copy()
            ),
            "regression_mask": torch.from_numpy(
                self._regression_available[index].copy()
            ),
            "source_index": torch.tensor(
                int(self._source_index[index]), dtype=torch.long
            ),
            "sampling_weight": torch.tensor(
                float(self._sampling_weight[index]), dtype=torch.float32
            ),
            "dataset_index": torch.tensor(index, dtype=torch.long),
        }
        if self._loss_weight is not None:
            sample["loss_weight"] = torch.from_numpy(
                self._loss_weight[index].copy()
            )
        if self._anchor is not None:
            sample["anchor"] = torch.tensor(
                float(self._anchor[index]), dtype=torch.float32
            )
        return sample
