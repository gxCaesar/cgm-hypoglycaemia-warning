from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .schema import ACTION_FIELDS

TASKS = ("hypo_30", "hypo_60", "hyper_30", "hyper_60")
TRAINING_MASK_SUFFIX = "_training_mask"
LOSS_WEIGHT_SUFFIX = "_loss_weight"
# Kept here rather than imported from `anchor_composition` so this module has no
# new import edge; the two names are asserted equal by test.
ANCHOR_COLUMN = "anchor_value"
REVISION_CONTRACT = "2026-11-04-fixed-frame"
INPUT_FEATURE_MODES = ("multimodal", "cgm_mask_clock")


def validate_input_feature_mode(mode: str) -> str:
    if mode not in INPUT_FEATURE_MODES:
        raise ValueError(f"input_feature_mode must be one of {INPUT_FEATURE_MODES}")
    return mode


def training_target_arrays(windows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Resolve exactly the retained targets and masks for the revision loss.

    Explicit targets must carry explicit boolean masks. Missing outcomes may be
    present on a prediction timeline, but only a masked missing value is filled
    with zero for tensor assembly; the original outcome column is never changed.
    """

    validate_optional_training_masks(windows)
    targets, masks = [], []
    for task in TASKS:
        target_column = f"{task}_training_target"
        mask_column = f"{task}_training_mask"
        if target_column in windows and mask_column not in windows:
            raise ValueError(f"{target_column} requires {mask_column}")
        column = target_column if target_column in windows else f"{task}_label"
        values = windows[column].to_numpy(dtype=float)
        mask = (
            windows[mask_column].to_numpy(dtype=bool)
            if mask_column in windows
            else (
                ~windows[f"{task.split('_')[0]}_currently_active"].to_numpy(bool)
                & windows[column].notna().to_numpy()
            )
        )
        invalid = ~np.isfinite(values) | (values < 0) | (values > 1)
        if (
            (invalid & mask).any()
            or ((values < 0) | (values > 1)).any()
            or (invalid.any() and mask_column not in windows)
        ):
            raise ValueError(
                f"unmasked training targets for {task} must be probabilities"
            )
        targets.append(np.where(np.isfinite(values), values, 0.0))
        masks.append(mask)
    return np.stack(targets, axis=1).astype(np.float32), np.stack(masks, axis=1)


def validate_optional_training_masks(windows: pd.DataFrame) -> None:
    """Reject a per-task loss mask that is not a null-free boolean column.

    The mask is optional and is read per row with ``Series.get``, so a null would
    reach ``bool(nan)``, evaluate true, and silently train on a window the caller
    meant to drop. Both dataset classes check it once here instead of per row, so
    the eager and the fast path cannot disagree about it.
    """

    for task in TASKS:
        column = f"{task}{TRAINING_MASK_SUFFIX}"
        if column not in windows.columns:
            continue
        values = windows[column]
        if values.isna().any():
            raise ValueError(f"{column} contains nulls")
        if values.dtype != bool:
            raise ValueError(f"{column} must be boolean, got {values.dtype}")


def validate_optional_loss_weights(windows: pd.DataFrame) -> None:
    """Reject a per-task event-loss weight that is not finite and non-negative.

    A negative weight would reward the model for getting a window wrong, and a
    non-finite one would poison the loss without naming itself. Checked once per
    dataset so the eager and the fast path cannot disagree about it.
    """

    for task in TASKS:
        column = f"{task}{LOSS_WEIGHT_SUFFIX}"
        if column not in windows.columns:
            continue
        values = windows[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{column} contains non-finite weights")
        if np.any(values < 0):
            raise ValueError(f"{column} contains negative weights")
REGRESSION_HORIZONS = (30, 60)
SEQUENCE_FEATURES = (
    "cgm_value",
    "cgm_observed",
    *(
        feature
        for field in ACTION_FIELDS
        for feature in (
            f"{field}_value",
            f"{field}_available",
            f"{field}_observed",
            f"{field}_zero_ambiguous",
            f"{field}_time_since",
        )
    ),
    "time_of_day_sin",
    "time_of_day_cos",
)

_PARTICIPANT_SEQUENCE_BASE_COLUMNS = (
    "timestamp",
    "timestamp_local",
    "cgm_mg_dl",
    "cgm_observed",
    "cgm_interpolated",
    "cgm_lower_censored",
)


@dataclass(frozen=True)
class ParticipantSequenceStore:
    """Immutable participant lookup shared by sequence datasets in one lane."""

    participants: Mapping[tuple[str, str], pd.DataFrame]

    @classmethod
    def from_frame(
        cls, frame: pd.DataFrame, *, input_feature_mode: str = "multimodal"
    ) -> ParticipantSequenceStore:
        validate_input_feature_mode(input_feature_mode)
        required = {
            "source_id",
            "participant_id",
            "timestamp",
            "cgm_mg_dl",
        }
        if input_feature_mode == "multimodal":
            required.update(
                column
                for field in ACTION_FIELDS
                for column in (
                    field, f"{field}_available", f"{field}_observed",
                    f"{field}_zero_ambiguous",
                )
            )
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(
                f"participant sequence frame is missing columns: {missing}"
            )
        value_columns = [
            column
            for column in (
                *_PARTICIPANT_SEQUENCE_BASE_COLUMNS,
                *(
                    column
                    for field in ACTION_FIELDS
                    for column in (
                        field,
                        f"{field}_available",
                        f"{field}_observed",
                        f"{field}_zero_ambiguous",
                    )
                ),
            )
            if column in frame.columns
        ]
        participants = {
            (str(source), str(participant)): group[value_columns]
            .sort_values("timestamp")
            .reset_index(drop=True)
            for (source, participant), group in frame.groupby(
                ["source_id", "participant_id"],
                sort=False,
                observed=True,
            )
        }
        if not participants:
            raise ValueError("participant sequence frame is empty")
        return cls(participants=participants)


def revision_anchor_mask(
    windows: pd.DataFrame, store: ParticipantSequenceStore
) -> np.ndarray:
    """Bind window anchors by exact normalized time and validate their provenance.

    The end index is checked against the timestamp join, never used as a finite
    glucose fallback. Callers may use the returned mask before dataset creation;
    revision datasets require every retained anchor to pass it.
    """

    required = {"source_id", "participant_id", "context_end", "participant_end_index"}
    missing = sorted(required.difference(windows.columns))
    if missing:
        raise ValueError(f"revision windows are missing anchor columns: {missing}")
    eligible = np.zeros(len(windows), dtype=bool)
    provenance = ("cgm_observed", "cgm_interpolated", "cgm_lower_censored")
    for (source, participant), positions in windows.groupby(
        ["source_id", "participant_id"], sort=False, observed=True
    ).indices.items():
        key = (str(source), str(participant))
        if key not in store.participants:
            raise ValueError(f"revision anchor participant is missing: {key}")
        frame = store.participants[key]
        for column in provenance:
            if column not in frame:
                raise ValueError(
                    f"revision anchors require provenance column {column}"
                )
            if frame[column].isna().any() or frame[column].dtype != bool:
                raise ValueError(
                    f"revision anchor provenance {column} must be null-free boolean"
                )
        timestamps = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"], utc=True))
        if timestamps.hasnans or timestamps.has_duplicates:
            raise ValueError("revision anchor timestamps must be unique and non-null")
        rows = windows.iloc[positions]
        ends = pd.DatetimeIndex(pd.to_datetime(rows["context_end"], utc=True))
        indices = timestamps.get_indexer(ends)
        if np.any(indices < 0):
            raise ValueError("revision context_end has no exact participant timestamp")
        supplied = pd.to_numeric(
            rows["participant_end_index"], errors="coerce"
        ).to_numpy()
        if not np.array_equal(supplied, indices):
            raise ValueError(
                "revision participant_end_index conflicts with context_end"
            )
        anchors = frame.iloc[indices]
        eligible[positions] = (
            anchors["cgm_observed"].to_numpy(bool)
            & ~anchors["cgm_interpolated"].to_numpy(bool)
            & ~anchors["cgm_lower_censored"].to_numpy(bool)
            & np.isfinite(anchors["cgm_mg_dl"].to_numpy(float))
        )
    return eligible


@dataclass(frozen=True)
class ScaleValue:
    mean: float
    std: float


@dataclass(frozen=True)
class SequenceScaler:
    values: dict[str, ScaleValue]

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        participant_keys: set[tuple[str, str]],
        *,
        input_feature_mode: str = "multimodal",
    ) -> SequenceScaler:
        validate_input_feature_mode(input_feature_mode)
        keys = list(
            zip(  # noqa: B905
                frame["source_id"].astype(str),
                frame["participant_id"].astype(str),
            )
        )
        selected = frame.loc[[key in participant_keys for key in keys]]
        if selected.empty:
            raise ValueError("cannot fit sequence scaler on an empty training split")
        values: dict[str, ScaleValue] = {}
        cgm_selected = selected
        if "cgm_lower_censored" in selected:
            cgm_selected = selected.loc[
                ~selected["cgm_lower_censored"].astype(bool)
            ]
        cgm = pd.to_numeric(
            cgm_selected["cgm_mg_dl"],
            errors="coerce",
        ).dropna()
        values["cgm_mg_dl"] = _scale_value(cgm)
        for field in ACTION_FIELDS:
            if input_feature_mode == "cgm_mask_clock":
                values[field] = ScaleValue(mean=0.0, std=1.0)
                continue
            observed = selected[f"{field}_observed"].astype(bool)
            numeric = pd.to_numeric(
                selected.loc[observed, field],
                errors="coerce",
            ).dropna()
            values[field] = _scale_value(numeric)
        return cls(values=values)

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, dict[str, float]],
    ) -> SequenceScaler:
        return cls(
            values={
                field: ScaleValue(
                    mean=float(values["mean"]),
                    std=float(values["std"]),
                )
                for field, values in payload.items()
            }
        )

    def transform(self, field: str, values: np.ndarray) -> np.ndarray:
        scale = self.values[field]
        return (values - scale.mean) / scale.std

    def as_dict(self) -> dict[str, dict[str, float]]:
        return {
            field: {"mean": scale.mean, "std": scale.std}
            for field, scale in self.values.items()
        }


def _scale_value(values: pd.Series) -> ScaleValue:
    if values.empty:
        return ScaleValue(mean=0.0, std=1.0)
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    if not np.isfinite(std) or std < 1e-6:
        std = 1.0
    return ScaleValue(mean=mean, std=std)


def sample_training_windows(
    windows: pd.DataFrame,
    *,
    negative_ratio: float,
    seed: int,
    use_training_targets: bool = False,
) -> pd.DataFrame:
    if windows.empty:
        raise ValueError("cannot sample an empty window table")
    rng = np.random.default_rng(seed)
    output: list[pd.DataFrame] = []
    if use_training_targets:
        positive, usable = training_window_sampling_masks(windows)
        positive_mask = pd.Series(positive, index=windows.index)
        windows = windows.loc[usable]
        if windows.empty:
            raise ValueError(
                "no training windows have a usable classification or regression target"
            )
    else:
        positive_mask = (
            windows[[f"{task}_label" for task in TASKS]].fillna(0).any(axis=1)
        )
    for _, source_frame in windows.groupby(
        "source_id",
        sort=True,
        observed=False,
    ):
        source_positive_mask = positive_mask.loc[source_frame.index]
        positives = source_frame.loc[source_positive_mask].copy()
        negative_indices = source_frame.index[~source_positive_mask].to_numpy()
        negative_count = len(negative_indices)
        if positives.empty:
            keep_negative = min(negative_count, 1000)
        else:
            keep_negative = min(
                negative_count,
                int(np.ceil(len(positives) * negative_ratio)),
            )
        if keep_negative < negative_count:
            chosen = rng.choice(
                negative_indices,
                size=keep_negative,
                replace=False,
            )
            sampled_negatives = source_frame.loc[np.sort(chosen)].copy()
        else:
            sampled_negatives = source_frame.loc[~source_positive_mask].copy()
        negative_weight = (
            float(negative_count / len(sampled_negatives))
            if len(sampled_negatives)
            else 1.0
        )
        positives["sampling_weight"] = 1.0
        sampled_negatives["sampling_weight"] = negative_weight
        output.extend((positives, sampled_negatives))
    sampled = pd.concat(output, ignore_index=False)
    return sampled.sort_index().reset_index(drop=True)


def training_window_sampling_masks(
    windows: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Return revision positives and windows that can contribute any loss."""

    targets, masks = training_target_arrays(windows)
    positive = ((targets == 1.0) & masks).any(axis=1)
    usable = masks.any(axis=1)
    for horizon in REGRESSION_HORIZONS:
        target = f"cgm_{horizon}_target"
        available = f"{target}_available"
        values = windows[target].to_numpy(float)
        declared = (
            windows[available].to_numpy(bool)
            if available in windows else np.ones(len(windows), dtype=bool)
        )
        usable |= np.isfinite(values) & declared
    return positive, usable


class WindowSequenceDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        frame: pd.DataFrame,
        windows: pd.DataFrame,
        *,
        scaler: SequenceScaler,
        context_samples: int,
        source_to_index: dict[str, int] | None = None,
        participant_store: ParticipantSequenceStore | None = None,
        copy_windows: bool = True,
        input_feature_mode: str = "multimodal",
        revision_contract: str | None = None,
    ) -> None:
        self.input_feature_mode = validate_input_feature_mode(input_feature_mode)
        if revision_contract not in (None, REVISION_CONTRACT):
            raise ValueError(f"unsupported revision_contract: {revision_contract}")
        self.revision_contract = revision_contract
        if windows.empty:
            raise ValueError("sequence dataset requires at least one window")
        if copy_windows:
            self.windows = windows.reset_index(drop=True).copy()
        else:
            # Controlled training callers keep the source frame immutable; a
            # shallow frame isolates index/column additions while sharing data.
            self.windows = windows.copy(deep=False)
            self.windows.index = pd.RangeIndex(len(self.windows))
        if "sampling_weight" not in self.windows:
            self.windows["sampling_weight"] = 1.0
        validate_optional_training_masks(self.windows)
        validate_optional_loss_weights(self.windows)
        self._revision_training_arrays = (
            training_target_arrays(self.windows) if revision_contract else None
        )
        self._loss_weight_columns: tuple[str | None, ...] = tuple(
            f"{task}{LOSS_WEIGHT_SUFFIX}"
            if f"{task}{LOSS_WEIGHT_SUFFIX}" in self.windows.columns
            else None
            for task in TASKS
        )
        self._has_loss_weight = any(
            column is not None for column in self._loss_weight_columns
        )
        # Present on the training frame alone. `attach_anchor_values` is called
        # on `train_windows`, so the selection and calibration datasets never
        # carry the column and never emit the key -- which is what keeps the
        # anchor out of inference by construction rather than by a flag.
        self._has_anchor = ANCHOR_COLUMN in self.windows.columns
        self.scaler = scaler
        self.context_samples = int(context_samples)
        # Deduplicate before stringifying: `.astype(str)` first would build one
        # Python string per window row, and this runs once per dataset. The
        # sorted result is what indexes sources into the model, so the order
        # must not move -- `.unique()` on a categorical returns the observed
        # values only, which is what the previous form also produced.
        sources = sorted(map(str, self.windows["source_id"].unique()))
        self.source_to_index = source_to_index or {
            source: index for index, source in enumerate(sources)
        }
        resolved_store = participant_store or ParticipantSequenceStore.from_frame(
            frame, input_feature_mode=self.input_feature_mode
        )
        required_participants = {
            (str(row.source_id), str(row.participant_id))
            for row in self.windows[
                ["source_id", "participant_id"]
            ].drop_duplicates().itertuples(index=False)
        }
        missing_participants = sorted(
            required_participants.difference(resolved_store.participants)
        )
        if missing_participants:
            raise ValueError(
                "participant sequence store is missing window participants: "
                f"{missing_participants[:5]}"
            )
        self.participant_store = resolved_store
        self._participants = resolved_store.participants
        if revision_contract and not revision_anchor_mask(
            self.windows, resolved_store
        ).all():
            raise ValueError(
                "revision windows require observed noninterpolated uncensored anchors"
            )

    def __len__(self) -> int:
        return len(self.windows)

    def metadata(self, index: int) -> dict[str, Any]:
        row = self.windows.iloc[index]
        return {
            "source_id": str(row["source_id"]),
            "participant_id": str(row["participant_id"]),
            "context_start": pd.Timestamp(row["context_start"]),
            "context_end": pd.Timestamp(row["context_end"]),
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.windows.iloc[index]
        key = (str(row["source_id"]), str(row["participant_id"]))
        participant = self._participants[key]
        end_index = int(row["participant_end_index"])
        start_index = end_index - self.context_samples + 1
        if start_index < 0:
            raise IndexError("window context starts before participant data")
        context = participant.iloc[start_index : end_index + 1]
        if len(context) != self.context_samples:
            raise IndexError("window context has an unexpected length")

        sequence = self._tensorize_context(context)
        labels = np.array(
            [float(row[f"{task}_label"]) for task in TASKS],
            dtype=np.float32,
        )
        training_labels = np.array(
            [
                float(row.get(f"{task}_training_target", row[f"{task}_label"]))
                for task in TASKS
            ],
            dtype=np.float32,
        )
        if self._revision_training_arrays is not None:
            training_labels = self._revision_training_arrays[0][index].copy()
        if not np.isfinite(training_labels).all() or np.any(
            (training_labels < 0) | (training_labels > 1)
        ):
            raise ValueError("training labels must be finite probabilities")
        label_mask = np.array(
            [
                bool(
                    row.get(
                        f"{task}{TRAINING_MASK_SUFFIX}",
                        not bool(row[f"{task.split('_')[0]}_currently_active"])
                        and pd.notna(row[f"{task}_label"]),
                    )
                )
                for task in TASKS
            ],
            dtype=bool,
        )
        if self._revision_training_arrays is not None:
            label_mask = self._revision_training_arrays[1][index].copy()
        regression_available = np.array(
            [
                bool(
                    row.get(
                        f"cgm_{horizon}_target_available",
                        pd.notna(row[f"cgm_{horizon}_target"]),
                    )
                )
                and pd.notna(row[f"cgm_{horizon}_target"])
                for horizon in REGRESSION_HORIZONS
            ],
            dtype=bool,
        )
        regression_raw = np.array(
            [
                (
                    float(row[f"cgm_{horizon}_target"])
                    if available
                    else self.scaler.values["cgm_mg_dl"].mean
                )
                for horizon, available in zip(  # noqa: B905
                    REGRESSION_HORIZONS,
                    regression_available,
                )
            ],
            dtype=float,
        )
        regression_targets = self.scaler.transform(
            "cgm_mg_dl",
            regression_raw,
        ).astype(np.float32)
        sample = {
            "sequence": torch.from_numpy(sequence),
            "labels": torch.from_numpy(labels),
            "training_labels": torch.from_numpy(training_labels),
            "label_mask": torch.from_numpy(label_mask),
            "regression_targets": torch.from_numpy(regression_targets),
            "regression_mask": torch.from_numpy(regression_available),
            "source_index": torch.tensor(
                self.source_to_index[key[0]],
                dtype=torch.long,
            ),
            "sampling_weight": torch.tensor(
                float(row["sampling_weight"]),
                dtype=torch.float32,
            ),
            "dataset_index": torch.tensor(index, dtype=torch.long),
        }
        if self._has_loss_weight:
            sample["loss_weight"] = torch.from_numpy(
                np.array(
                    [
                        1.0 if column is None else float(row[column])
                        for column in self._loss_weight_columns
                    ],
                    dtype=np.float32,
                )
            )
        if self._has_anchor:
            sample["anchor"] = torch.tensor(
                float(row[ANCHOR_COLUMN]),
                dtype=torch.float32,
            )
        return sample

    def _tensorize_context(self, context: pd.DataFrame) -> np.ndarray:
        features: list[np.ndarray] = []
        if (
            "cgm_lower_censored" in context
            and context["cgm_lower_censored"].astype(bool).any()
        ):
            raise ValueError(
                "lower-censored CGM cannot enter model context features"
            )
        cgm = context["cgm_mg_dl"].to_numpy(dtype=float)
        cgm_observed = (
            context["cgm_observed"].to_numpy(dtype=bool)
            if "cgm_observed" in context
            else np.isfinite(cgm)
        )
        cgm_value = self.scaler.transform("cgm_mg_dl", cgm)
        features.extend((cgm_value, cgm_observed.astype(float)))

        for field in ACTION_FIELDS:
            if self.input_feature_mode == "cgm_mask_clock":
                features.extend(np.zeros(len(context)) for _ in range(5))
                continue
            observed = context[f"{field}_observed"].to_numpy(dtype=bool)
            raw = context[field].to_numpy(dtype=float)
            filled = np.where(observed, raw, self.scaler.values[field].mean)
            value = self.scaler.transform(field, filled)
            value = np.where(observed, value, 0.0)
            available = context[f"{field}_available"].to_numpy(dtype=float)
            ambiguous = context[f"{field}_zero_ambiguous"].to_numpy(dtype=float)
            time_since = _time_since_observation(observed)
            features.extend(
                (
                    value,
                    available,
                    observed.astype(float),
                    ambiguous,
                    time_since,
                )
            )

        local_column = (
            context["timestamp_local"]
            if "timestamp_local" in context
            else context["timestamp"]
        )
        local = pd.to_datetime(local_column)
        minute = local.dt.hour.to_numpy() * 60 + local.dt.minute.to_numpy()
        phase = 2 * np.pi * minute / (24 * 60)
        features.extend((np.sin(phase), np.cos(phase)))
        sequence = np.stack(features, axis=1).astype(np.float32)
        if sequence.shape[1] != len(SEQUENCE_FEATURES):
            raise AssertionError("sequence feature width does not match contract")
        if not np.isfinite(sequence).all():
            raise ValueError("sequence contains non-finite values after masking")
        return sequence


class ForecastWindowSequenceDataset(WindowSequenceDataset):
    """Window dataset with a contiguous future-CGM trajectory target.

    Prepared window shards remain the authority for eligibility, partition,
    and the decision-point index. Future targets are read only from the same
    source-scoped participant frame and never alter the sample set.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        windows: pd.DataFrame,
        *,
        scaler: SequenceScaler,
        context_samples: int,
        prediction_samples: int,
        sample_minutes: int,
        source_to_index: dict[str, int] | None = None,
        participant_store: ParticipantSequenceStore | None = None,
        copy_windows: bool = True,
    ) -> None:
        super().__init__(
            frame,
            windows,
            scaler=scaler,
            context_samples=context_samples,
            source_to_index=source_to_index,
            participant_store=participant_store,
            copy_windows=copy_windows,
        )
        if prediction_samples <= 0 or sample_minutes <= 0:
            raise ValueError("prediction_samples and sample_minutes must be positive")
        self.prediction_samples = int(prediction_samples)
        self.sample_minutes = int(sample_minutes)

    def __getitem__(self, index: int) -> dict[str, Any]:
        output = super().__getitem__(index)
        row = self.windows.iloc[index]
        key = (str(row["source_id"]), str(row["participant_id"]))
        participant = self._participants[key]
        end_index = int(row["participant_end_index"])
        future = participant.iloc[
            end_index + 1 : end_index + self.prediction_samples + 1
        ]
        if len(future) != self.prediction_samples:
            raise IndexError("forecast target extends beyond participant data")
        future_time = pd.to_datetime(future["timestamp"], utc=True)
        expected_start = pd.Timestamp(row["context_end"]) + pd.Timedelta(
            minutes=self.sample_minutes
        )
        if future_time.iloc[0] != expected_start or not future_time.diff().dropna().eq(
            pd.Timedelta(minutes=self.sample_minutes)
        ).all():
            raise ValueError("forecast target is not contiguous after context_end")

        raw = pd.to_numeric(future["cgm_mg_dl"], errors="coerce").to_numpy(
            dtype=float
        )
        available = np.isfinite(raw)
        if "cgm_lower_censored" in future:
            available &= ~future["cgm_lower_censored"].astype(bool).to_numpy()
        scale = self.scaler.values["cgm_mg_dl"]
        filled = np.where(available, raw, scale.mean)
        normalized = self.scaler.transform("cgm_mg_dl", filled).astype(np.float32)
        output.update(
            {
                "forecast_targets": torch.from_numpy(normalized),
                "forecast_targets_mg_dl": torch.from_numpy(
                    filled.astype(np.float32)
                ),
                "forecast_mask": torch.from_numpy(available.astype(bool)),
            }
        )
        return output


def _time_since_observation(observed: np.ndarray) -> np.ndarray:
    elapsed = np.empty(len(observed), dtype=np.float32)
    since = len(observed)
    for index, value in enumerate(observed):
        if value:
            since = 0
        else:
            since += 1
        elapsed[index] = min(since, len(observed)) / max(1, len(observed))
    return elapsed
