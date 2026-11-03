"""Train-only participant-CF5 source-reference fitting.

The module intentionally separates the artifact-free synthetic path from two
production artifact contracts.  ``fit_production_reference_artifact`` retains
the immutable v1.1 sampled-record estimator, while
``fit_production_weighted_reference_artifact`` emits the successor v1.2
sampling-weight-sum estimator.  Both production paths fix the eleven sources,
five folds, thirty epochs, optimizer, batching, and immutable artifact write.

Neither entry point reads project files, contacts a remote host, or selects a
shared-model checkpoint.  Callers supply already-loaded train-only frames.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset

from .baselines import build_temporal_baseline
from .checkpoint_state import checkpoint_state_sha256
from .deep_training import (
    _action_dropout_probabilities,
    _drop_action_groups,
    _positive_weights,
    _seed_everything,
    _unweighted_multitask_loss,
)
from .mero_reference import (
    ParticipantFoldAssignment,
    ReferenceFoldRecord,
    assert_reference_scope,
    assign_source_scoped_folds,
    build_reference_artifact,
    derive_reference_model_seed,
    select_reference_checkpoint,
    write_reference_artifact,
)
from .mero_weighted_reference import (
    WeightedReferenceFoldRecord,
    build_weighted_reference_artifact,
    write_weighted_reference_artifact,
)
from .models import count_parameters
from .schema import ACTION_FIELDS, REQUIRED_COLUMNS
from .sequences import (
    REGRESSION_HORIZONS,
    TASKS,
    ParticipantSequenceStore,
    SequenceScaler,
    WindowSequenceDataset,
    sample_training_windows,
)

PRODUCTION_SOURCE_ORDER = (
    "AZT1D",
    "BrisT1D",
    "CTR3",
    "DCLP3",
    "DCLP5",
    "Flair",
    "IOBP2",
    "PEDAP",
    "ReplaceBG",
    "Shanghai T1DM",
    "T1D-UOM",
)
"""The lexical source order frozen by the v1.1 reference specification."""

_ORIGINAL_ROW_INDEX = "__cf5_original_window_row_index"
_EVALUATION_CONTEXT = "__cf5_evaluation_context_end"
_EXPECTED_LSTM_PARAMETER_COUNT = 268_166

SAMPLED_RECORD_COUNT_V1_1 = "sampled_record_count_v1_1"
SAMPLING_WEIGHT_SUM_V1_2 = "sampling_weight_sum_v1_2"
ReferenceLossDenominator = Literal[
    "sampled_record_count_v1_1",
    "sampling_weight_sum_v1_2",
]


@dataclass(frozen=True)
class SyntheticReferenceFitConfig:
    """Explicitly lower-level settings for an artifact-free synthetic fit.

    The defaults are deliberately small.  Production settings are represented
    by :class:`ProductionReferenceFitConfig` and are checked independently of
    this class, so a reduced setting cannot be used to create a production
    reference artifact.
    """

    source_order: tuple[str, ...]
    fold_count: int = 2
    maximum_epochs: int = 1
    reference_seed: int = 20260805
    data_sampling_seed: int = 42
    negative_ratio: float = 20.0
    physical_batch_size: int = 4
    effective_batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    feature_group_dropout_max: float = 0.5
    auxiliary_regression_weight: float = 0.1
    context_samples: int = 72
    num_workers: int = 0
    device: str = "cpu"
    loss_denominator_contract: ReferenceLossDenominator = SAMPLING_WEIGHT_SUM_V1_2


@dataclass(frozen=True)
class ProductionReferenceFitConfig:
    """The non-negotiable legacy v1.1 production fitting contract."""

    source_order: tuple[str, ...] = PRODUCTION_SOURCE_ORDER
    fold_count: int = 5
    maximum_epochs: int = 30
    reference_seed: int = 20260805
    data_sampling_seed: int = 42
    negative_ratio: float = 20.0
    physical_batch_size: int = 256
    effective_batch_size: int = 1024
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    feature_group_dropout_max: float = 0.5
    auxiliary_regression_weight: float = 0.1
    context_samples: int = 72
    num_workers: int = 8
    device: str = "cpu"
    loss_denominator_contract: ReferenceLossDenominator = SAMPLED_RECORD_COUNT_V1_1


@dataclass(frozen=True)
class WeightedProductionReferenceFitConfig(ProductionReferenceFitConfig):
    """The non-negotiable successor sampling-weight-sum contract."""

    loss_denominator_contract: ReferenceLossDenominator = SAMPLING_WEIGHT_SUM_V1_2


@dataclass(frozen=True)
class Float64LossSummary:
    """A float64 CPU weighted-loss sum and explicit denominator contract."""

    total_loss: float
    mean_loss: float
    record_count: int
    sampling_weight_sum: float
    denominator: float
    denominator_contract: ReferenceLossDenominator


@dataclass(frozen=True)
class ReferenceEvaluation:
    """A deterministic full-fit or held-fold evaluation result."""

    loss: Float64LossSummary
    original_window_row_indices: tuple[int, ...]


@dataclass(frozen=True)
class SyntheticReferenceFoldResult:
    """One source/fold fit ledger.  It cannot serialize a reference artifact."""

    source_id: str
    fold_id: int
    fit_participant_ids: tuple[str, ...]
    held_participant_ids: tuple[str, ...]
    fit_raw_window_count: int
    fit_sampled_window_count: int
    held_raw_window_count: int
    held_sampled_window_count: int
    fit_evaluation_denominator: float
    held_evaluation_denominator: float
    loss_denominator_contract: ReferenceLossDenominator
    derived_model_seed: int
    checkpoint_epoch: int
    checkpoint_objective: float
    epoch_objectives: tuple[float, ...]
    fold_oof_loss: float
    fit_evaluation_row_indices: tuple[int, ...]
    held_evaluation_row_indices: tuple[int, ...]
    action_dropout_probabilities: tuple[tuple[str, float], ...]
    checkpoint_state: Mapping[str, torch.Tensor]
    scaler: SequenceScaler

    def as_reference_fold_record(self) -> ReferenceFoldRecord:
        """Project the completed synthetic ledger into the production schema.

        This conversion is intentionally not an artifact writer.  Only the
        production entry point supplies it to ``build_reference_artifact``.
        """

        if self.loss_denominator_contract != SAMPLED_RECORD_COUNT_V1_1:
            raise ValueError(
                "the immutable v1.1 artifact accepts only its sampled-record "
                "denominator contract"
            )
        return ReferenceFoldRecord(
            source_id=self.source_id,
            fold_id=self.fold_id,
            fit_participant_count=len(self.fit_participant_ids),
            held_participant_count=len(self.held_participant_ids),
            fit_raw_window_count=self.fit_raw_window_count,
            fit_sampled_window_count=self.fit_sampled_window_count,
            held_raw_window_count=self.held_raw_window_count,
            held_sampled_window_count=self.held_sampled_window_count,
            derived_model_seed=self.derived_model_seed,
            checkpoint_epoch=self.checkpoint_epoch,
            checkpoint_objective=self.checkpoint_objective,
            epoch_objectives=self.epoch_objectives,
            fold_oof_loss=self.fold_oof_loss,
        )

    def as_weighted_reference_fold_record(self) -> WeightedReferenceFoldRecord:
        """Project this fold into the successor weighted artifact schema."""

        if self.loss_denominator_contract != SAMPLING_WEIGHT_SUM_V1_2:
            raise ValueError(
                "the weighted successor artifact requires its "
                "sampling-weight-sum denominator contract"
            )
        return WeightedReferenceFoldRecord(
            source_id=self.source_id,
            fold_id=self.fold_id,
            fit_participant_count=len(self.fit_participant_ids),
            held_participant_count=len(self.held_participant_ids),
            fit_raw_window_count=self.fit_raw_window_count,
            fit_sampled_window_count=self.fit_sampled_window_count,
            held_raw_window_count=self.held_raw_window_count,
            held_sampled_window_count=self.held_sampled_window_count,
            fit_sampling_weight_denominator=self.fit_evaluation_denominator,
            held_sampling_weight_denominator=self.held_evaluation_denominator,
            loss_denominator_contract=self.loss_denominator_contract,
            derived_model_seed=self.derived_model_seed,
            checkpoint_epoch=self.checkpoint_epoch,
            checkpoint_state_sha256=checkpoint_state_sha256(
                self.checkpoint_state
            ),
            checkpoint_objective=self.checkpoint_objective,
            epoch_objectives=self.epoch_objectives,
            fold_oof_loss=self.fold_oof_loss,
        )


@dataclass(frozen=True)
class SyntheticReferenceFitResult:
    """Artifact-free result of all requested source-scoped synthetic folds."""

    assignments: tuple[ParticipantFoldAssignment, ...]
    global_class_positive_weights: tuple[float, ...]
    folds: tuple[SyntheticReferenceFoldResult, ...]
    per_source_reference_risks: dict[str, float]
    loss_denominator_contract: ReferenceLossDenominator


@dataclass(frozen=True)
class ProductionWeightedReferenceFitBundle:
    """Weighted parent artifact plus its exact selected fold states."""

    artifact: Mapping[str, Any]
    fit_result: SyntheticReferenceFitResult


def _require_integer(value: Any, *, field: str, minimum: int = 0) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError(f"{field} must be an integer")
    resolved = int(value)
    if resolved < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return resolved


def _require_finite_real(value: Any, *, field: str, minimum: float = 0.0) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{field} must be numeric")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < minimum:
        raise ValueError(f"{field} must be finite and at least {minimum}")
    return resolved


def _validate_fit_config(
    config: (
        SyntheticReferenceFitConfig
        | ProductionReferenceFitConfig
        | WeightedProductionReferenceFitConfig
    ),
    *,
    production: bool,
) -> tuple[str, ...]:
    source_order = tuple(config.source_order)
    # ``assert_reference_scope`` delegates identifier and lexical-order checks
    # to the canonical reference module without needing a synthetic row.
    assert_reference_scope((), source_order=source_order)
    _require_integer(config.fold_count, field="fold_count", minimum=2)
    _require_integer(config.maximum_epochs, field="maximum_epochs", minimum=1)
    _require_integer(config.reference_seed, field="reference_seed")
    _require_integer(config.data_sampling_seed, field="data_sampling_seed")
    _require_integer(config.physical_batch_size, field="physical_batch_size", minimum=1)
    _require_integer(
        config.effective_batch_size,
        field="effective_batch_size",
        minimum=1,
    )
    _require_integer(config.context_samples, field="context_samples", minimum=1)
    _require_integer(config.num_workers, field="num_workers")
    _require_finite_real(config.negative_ratio, field="negative_ratio", minimum=0.0)
    _require_finite_real(config.learning_rate, field="learning_rate", minimum=0.0)
    _require_finite_real(config.weight_decay, field="weight_decay", minimum=0.0)
    _require_finite_real(
        config.gradient_clip_norm,
        field="gradient_clip_norm",
        minimum=0.0,
    )
    _require_finite_real(
        config.feature_group_dropout_max,
        field="feature_group_dropout_max",
        minimum=0.0,
    )
    _require_finite_real(
        config.auxiliary_regression_weight,
        field="auxiliary_regression_weight",
        minimum=0.0,
    )
    if config.effective_batch_size < config.physical_batch_size:
        raise ValueError("effective_batch_size must be at least physical_batch_size")
    if config.effective_batch_size % config.physical_batch_size:
        raise ValueError("effective_batch_size must be a physical-batch multiple")
    if config.feature_group_dropout_max > 1:
        raise ValueError("feature_group_dropout_max must be at most one")
    if config.gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be positive")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if not isinstance(config.device, str) or not config.device:
        raise ValueError("device must be a non-empty torch device string")
    torch.device(config.device)
    if config.loss_denominator_contract not in {
        SAMPLED_RECORD_COUNT_V1_1,
        SAMPLING_WEIGHT_SUM_V1_2,
    }:
        raise ValueError("unsupported reference loss denominator contract")

    if production:
        expected = (
            ProductionReferenceFitConfig()
            if config.loss_denominator_contract == SAMPLED_RECORD_COUNT_V1_1
            else WeightedProductionReferenceFitConfig()
        )
        locked_fields = (
            "source_order",
            "fold_count",
            "maximum_epochs",
            "reference_seed",
            "data_sampling_seed",
            "negative_ratio",
            "physical_batch_size",
            "effective_batch_size",
            "learning_rate",
            "weight_decay",
            "gradient_clip_norm",
            "feature_group_dropout_max",
            "auxiliary_regression_weight",
            "context_samples",
            "num_workers",
            "loss_denominator_contract",
        )
        for field in locked_fields:
            if getattr(config, field) != getattr(expected, field):
                raise ValueError(
                    f"production reference {field} disagrees with contract"
                )
    return source_order


def _required_window_columns() -> tuple[str, ...]:
    return (
        "partition",
        "source_id",
        "participant_id",
        "context_end",
        "participant_end_index",
        *(f"{task}_label" for task in TASKS),
        "hypo_currently_active",
        "hyper_currently_active",
        *(f"cgm_{horizon}_target" for horizon in REGRESSION_HORIZONS),
    )


def _validate_train_only_inputs(
    frame: pd.DataFrame,
    windows: pd.DataFrame,
    *,
    source_order: tuple[str, ...],
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or not isinstance(windows, pd.DataFrame):
        raise TypeError("frame and windows must be pandas DataFrames")
    if frame.empty or windows.empty:
        raise ValueError("reference fitting requires non-empty frame and windows")
    missing_frame = sorted(set(REQUIRED_COLUMNS).difference(frame.columns))
    if missing_frame:
        raise ValueError(
            f"reference frame is missing required columns: {missing_frame}"
        )
    missing_windows = sorted(
        set(_required_window_columns()).difference(windows.columns)
    )
    if missing_windows:
        raise ValueError(
            "reference windows are missing required task or window columns: "
            f"{missing_windows}"
        )
    if _ORIGINAL_ROW_INDEX in windows:
        raise ValueError(f"{_ORIGINAL_ROW_INDEX} is reserved by the reference fitter")
    if "partition" in frame and not frame["partition"].eq("train").all():
        raise ValueError("reference frame contains a non-train partition row")

    assert_reference_scope(
        windows[["partition", "source_id", "participant_id"]].to_dict("records"),
        source_order=source_order,
        forbidden_sources=("Loop Observational Study", "HUPA-UCM", "OhioT1DM"),
    )
    frame_scope = frame[["source_id", "participant_id"]].copy()
    frame_scope["partition"] = "train"
    assert_reference_scope(
        frame_scope[["partition", "source_id", "participant_id"]].to_dict(
            "records"
        ),
        source_order=source_order,
        forbidden_sources=("Loop Observational Study", "HUPA-UCM", "OhioT1DM"),
    )
    observed_sources = tuple(sorted(windows["source_id"].unique()))
    if observed_sources != source_order:
        raise ValueError("reference windows are missing or add an ineligible source")
    parsed_context = pd.to_datetime(windows["context_end"], utc=True, errors="coerce")
    if parsed_context.isna().any():
        raise ValueError("reference windows contain an invalid context_end")
    end_indices = pd.to_numeric(windows["participant_end_index"], errors="coerce")
    if end_indices.isna().any() or not np.equal(
        end_indices,
        np.floor(end_indices),
    ).all():
        raise ValueError(
            "reference windows contain a non-integer participant_end_index"
        )

    window_keys = set(
        zip(  # noqa: B905
            windows["source_id"].astype(str),
            windows["participant_id"].astype(str),
        )
    )
    frame_keys = set(
        zip(  # noqa: B905
            frame["source_id"].astype(str),
            frame["participant_id"].astype(str),
        )
    )
    if not window_keys.issubset(frame_keys):
        raise ValueError("reference frame is missing a window participant")

    input_indices = pd.to_numeric(pd.Series(windows.index), errors="coerce")
    if input_indices.isna().any() or not np.equal(
        input_indices,
        np.floor(input_indices),
    ).all():
        raise ValueError("reference windows require integer original row indices")
    if input_indices.duplicated().any():
        raise ValueError("reference windows have duplicate original row indices")
    work = windows.copy()
    # This preserves the prepared shard's integer row index through the
    # existing ratio-20 sampler, whose own reset_index would otherwise erase
    # the prescribed evaluation tie-break.
    work[_ORIGINAL_ROW_INDEX] = input_indices.to_numpy(dtype=np.int64)
    return work


def deterministic_reference_evaluation_order(
    windows: pd.DataFrame,
    *,
    original_row_index_column: str = _ORIGINAL_ROW_INDEX,
) -> pd.DataFrame:
    """Return the frozen participant/time/original-row evaluation ordering."""

    if original_row_index_column not in windows:
        raise ValueError("reference evaluation requires a preserved row index")
    if windows.empty:
        raise ValueError("reference evaluation requires at least one window")
    output = windows.copy()
    parsed_context = pd.to_datetime(output["context_end"], utc=True, errors="coerce")
    if parsed_context.isna().any():
        raise ValueError("reference evaluation contains an invalid context_end")
    original_indices = pd.to_numeric(
        output[original_row_index_column],
        errors="coerce",
    )
    if original_indices.isna().any() or not np.equal(
        original_indices,
        np.floor(original_indices),
    ).all():
        raise ValueError("reference evaluation has a non-integer original row index")
    if original_indices.duplicated().any():
        raise ValueError("reference evaluation has duplicate original row indices")
    output[original_row_index_column] = original_indices.astype(np.int64)
    output[_EVALUATION_CONTEXT] = parsed_context
    return (
        output.sort_values(
            ["participant_id", _EVALUATION_CONTEXT, original_row_index_column],
            kind="mergesort",
        )
        .drop(columns=[_EVALUATION_CONTEXT])
        .reset_index(drop=True)
    )


def accumulate_float64_record_losses(
    loss_chunks: Iterable[torch.Tensor],
) -> Float64LossSummary:
    """Reproduce the frozen v1.1 sampled-record denominator exactly."""

    return accumulate_float64_weighted_losses(
        (
            (chunk, torch.ones_like(chunk, dtype=torch.float32))
            for chunk in loss_chunks
        ),
        denominator_contract=SAMPLED_RECORD_COUNT_V1_1,
    )


def accumulate_float64_weighted_losses(
    loss_and_weight_chunks: Iterable[tuple[torch.Tensor, torch.Tensor]],
    *,
    denominator_contract: ReferenceLossDenominator,
) -> Float64LossSummary:
    """Accumulate unweighted losses with an explicit population denominator."""

    if denominator_contract not in {
        SAMPLED_RECORD_COUNT_V1_1,
        SAMPLING_WEIGHT_SUM_V1_2,
    }:
        raise ValueError("unsupported reference loss denominator contract")

    total = np.float64(0.0)
    weight_total = np.float64(0.0)
    count = 0
    for loss_chunk, weight_chunk in loss_and_weight_chunks:
        if (
            not isinstance(loss_chunk, torch.Tensor)
            or not loss_chunk.is_floating_point()
        ):
            raise TypeError("per-record losses must be floating-point tensors")
        if (
            not isinstance(weight_chunk, torch.Tensor)
            or not weight_chunk.is_floating_point()
        ):
            raise TypeError("sampling weights must be floating-point tensors")
        values = (
            loss_chunk.detach().reshape(-1).to(device="cpu", dtype=torch.float32)
        )
        weights = (
            weight_chunk.detach().reshape(-1).to(device="cpu", dtype=torch.float32)
        )
        if values.shape != weights.shape:
            raise ValueError("per-record losses and sampling weights must align")
        if values.numel() == 0:
            raise ValueError("per-record loss chunks must be non-empty")
        if not bool(torch.isfinite(values).all()):
            raise FloatingPointError("non-finite per-record reference loss")
        if not bool(torch.isfinite(weights).all()) or bool((weights <= 0).any()):
            raise ValueError("sampling weights must be finite and positive")
        for value, weight in zip(values.numpy(), weights.numpy(), strict=True):
            total = np.float64(
                total + np.float64(value) * np.float64(weight)
            )
            weight_total = np.float64(weight_total + np.float64(weight))
        count += int(values.numel())
    if count == 0:
        raise ValueError("at least one per-record reference loss is required")
    denominator = (
        np.float64(count)
        if denominator_contract == SAMPLED_RECORD_COUNT_V1_1
        else weight_total
    )
    if not math.isfinite(float(denominator)) or denominator <= 0:
        raise FloatingPointError("reference loss denominator became non-finite")
    return Float64LossSummary(
        total_loss=float(total),
        mean_loss=float(total / denominator),
        record_count=count,
        sampling_weight_sum=float(weight_total),
        denominator=float(denominator),
        denominator_contract=denominator_contract,
    )


def evaluate_reference_dataset(
    model: nn.Module,
    frame: pd.DataFrame,
    windows: pd.DataFrame,
    *,
    scaler: SequenceScaler,
    context_samples: int,
    source_to_index: dict[str, int],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    positive_weights: torch.Tensor,
    auxiliary_regression_weight: float,
    loss_denominator_contract: ReferenceLossDenominator = SAMPLING_WEIGHT_SUM_V1_2,
) -> ReferenceEvaluation:
    """Evaluate every sampled window once in frozen deterministic order."""

    return _evaluate_windows(
        model,
        frame,
        windows,
        scaler=scaler,
        context_samples=context_samples,
        source_to_index=source_to_index,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        positive_weights=positive_weights,
        auxiliary_regression_weight=auxiliary_regression_weight,
        loss_denominator_contract=loss_denominator_contract,
    )


def _evaluate_windows(
    model: nn.Module,
    frame: pd.DataFrame,
    windows: pd.DataFrame,
    *,
    scaler: SequenceScaler,
    context_samples: int,
    source_to_index: dict[str, int],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    positive_weights: torch.Tensor,
    auxiliary_regression_weight: float,
    loss_denominator_contract: ReferenceLossDenominator,
    participant_store: ParticipantSequenceStore | None = None,
) -> ReferenceEvaluation:
    """Evaluate a frame once; action dropout is deliberately absent here.

    `participant_store` is optional so the public `evaluate_reference_dataset`
    wrapper keeps working without one. Passing `None` restores the original
    behaviour, where `WindowSequenceDataset` builds its own; the fold loop always
    passes the shared store, which is where the 1,815 rebuilds were.
    """

    ordered_windows = deterministic_reference_evaluation_order(windows)
    dataset = WindowSequenceDataset(
        frame,
        ordered_windows,
        scaler=scaler,
        context_samples=context_samples,
        source_to_index=source_to_index,
        participant_store=participant_store,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    model.eval()

    def loss_and_weight_chunks() -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
        # Yield one batch at a time so a multi-million-window reference
        # evaluation never retains the complete GPU loss vector in memory.
        with torch.no_grad():
            for batch in loader:
                batch = _move_batch(batch, device=device)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    outputs = model(batch["sequence"])
                    try:
                        loss = _unweighted_multitask_loss(
                            outputs,
                            batch,
                            positive_weights=positive_weights,
                            auxiliary_weight=auxiliary_regression_weight,
                        )
                    except FloatingPointError as error:
                        raise FloatingPointError(
                            "non-finite reference evaluation loss"
                        ) from error
                    yield loss, batch["sampling_weight"]

    summary = accumulate_float64_weighted_losses(
        loss_and_weight_chunks(),
        denominator_contract=loss_denominator_contract,
    )
    ordered_indices = tuple(
        int(value) for value in ordered_windows[_ORIGINAL_ROW_INDEX].tolist()
    )
    if summary.record_count != len(ordered_indices):
        raise AssertionError("reference evaluation did not visit every sampled record")
    return ReferenceEvaluation(
        loss=summary,
        original_window_row_indices=ordered_indices,
    )


def _move_batch(
    batch: dict[str, Any],
    *,
    device: torch.device,
) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _optimizer_and_scheduler(
    model: nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
    total_updates: int,
) -> tuple[AdamW, LambdaLR]:
    """Match the existing 5% linear-warmup plus cosine-decay schedule."""

    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    warmup_steps = max(1, int(round(total_updates * 0.05)))

    def lr_multiplier(step: int) -> float:
        if step < warmup_steps:
            return max(1e-8, step / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_updates - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return optimizer, LambdaLR(optimizer, lr_lambda=lr_multiplier)


def _assert_finite_gradients(model: nn.Module, *, source_id: str, fold_id: int) -> None:
    for parameter in model.parameters():
        if parameter.grad is not None and not bool(
            torch.isfinite(parameter.grad).all()
        ):
            raise FloatingPointError(
                "non-finite reference gradient "
                f"for source={source_id!r}, fold={fold_id}"
            )


def _fit_epoch(
    model: nn.Module,
    dataset: WindowSequenceDataset,
    *,
    epoch_index: int,
    derived_model_seed: int,
    device: torch.device,
    config: SyntheticReferenceFitConfig | ProductionReferenceFitConfig,
    optimizer: AdamW,
    scheduler: LambdaLR,
    positive_weights: torch.Tensor,
    action_dropout_probabilities: tuple[float, ...],
    source_id: str,
    fold_id: int,
) -> int:
    """Visit a fixed sampled population under the explicit loss denominator."""

    generator = torch.Generator(device="cpu").manual_seed(
        derived_model_seed + epoch_index
    )
    shuffled_indices = torch.randperm(len(dataset), generator=generator).tolist()
    shuffled_dataset = Subset(dataset, shuffled_indices)
    loader = DataLoader(
        shuffled_dataset,
        batch_size=config.physical_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    accumulation_steps = config.effective_batch_size // config.physical_batch_size
    update_count = 0
    physical_group: list[dict[str, Any]] = []
    model.train()
    optimizer.zero_grad(set_to_none=True)

    def complete_update() -> None:
        nonlocal update_count
        record_count = sum(int(batch["sequence"].shape[0]) for batch in physical_group)
        if record_count <= 0:
            raise AssertionError("reference effective update has no records")
        loss_chunks: list[torch.Tensor] = []
        weight_chunks: list[torch.Tensor] = []
        for stored_batch in physical_group:
            sequence = _drop_action_groups(
                stored_batch["sequence"],
                probabilities=action_dropout_probabilities,
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                outputs = model(sequence)
                try:
                    per_example = _unweighted_multitask_loss(
                        outputs,
                        stored_batch,
                        positive_weights=positive_weights,
                        auxiliary_weight=config.auxiliary_regression_weight,
                    )
                except FloatingPointError as error:
                    raise FloatingPointError(
                        "non-finite reference loss "
                        f"for source={source_id!r}, fold={fold_id}, "
                        f"epoch={epoch_index}"
                    ) from error
            if not bool(torch.isfinite(per_example).all()):
                raise FloatingPointError(
                    "non-finite reference loss "
                    f"for source={source_id!r}, fold={fold_id}, epoch={epoch_index}"
                )
            loss_chunks.append(per_example)
            weights = stored_batch["sampling_weight"]
            if (
                not weights.is_floating_point()
                or not bool(torch.isfinite(weights).all())
                or bool((weights <= 0).any())
            ):
                raise ValueError(
                    "reference sampling weights must be finite and positive"
                )
            weight_chunks.append(weights)
        losses = torch.cat(loss_chunks).to(dtype=torch.float32)
        sampling_weights = torch.cat(weight_chunks).to(dtype=torch.float32)
        denominator = (
            losses.new_tensor(float(record_count))
            if config.loss_denominator_contract == SAMPLED_RECORD_COUNT_V1_1
            else sampling_weights.sum()
        )
        if not bool(torch.isfinite(denominator)) or bool(denominator <= 0):
            raise FloatingPointError(
                "reference effective-update denominator became non-finite"
            )
        loss = torch.sum(losses * sampling_weights) / denominator
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(
                "non-finite reference loss "
                f"for source={source_id!r}, fold={fold_id}, epoch={epoch_index}"
            )
        loss.backward()
        _assert_finite_gradients(model, source_id=source_id, fold_id=fold_id)
        gradient_norm = nn.utils.clip_grad_norm_(
            model.parameters(),
            config.gradient_clip_norm,
        )
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError(
                "non-finite reference gradient norm "
                f"for source={source_id!r}, fold={fold_id}"
            )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        update_count += 1

    for batch_index, batch in enumerate(loader):
        physical_group.append(_move_batch(batch, device=device))
        if (
            (batch_index + 1) % accumulation_steps == 0
            or batch_index + 1 == len(loader)
        ):
            complete_update()
            physical_group.clear()
    if physical_group:
        raise AssertionError("reference effective-update group was not finalized")
    return update_count


def _model_state_copy(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _fit_one_source_fold(
    frame: pd.DataFrame,
    source_windows: pd.DataFrame,
    *,
    source_id: str,
    fold_id: int,
    fit_participant_ids: tuple[str, ...],
    held_participant_ids: tuple[str, ...],
    global_positive_weights: torch.Tensor,
    config: SyntheticReferenceFitConfig | ProductionReferenceFitConfig,
    model_factory: Callable[[], nn.Module],
    production: bool,
    participant_store: ParticipantSequenceStore,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> SyntheticReferenceFoldResult:
    fit_windows = source_windows.loc[
        source_windows["participant_id"].isin(fit_participant_ids)
    ].copy()
    held_windows = source_windows.loc[
        source_windows["participant_id"].isin(held_participant_ids)
    ].copy()
    if fit_windows.empty or held_windows.empty:
        raise ValueError(
            "reference source="
            f"{source_id!r}, fold={fold_id} has an empty fit or held scope"
        )
    fit_keys = {(source_id, participant) for participant in fit_participant_ids}
    held_keys = {(source_id, participant) for participant in held_participant_ids}
    if fit_keys & held_keys:
        raise ValueError("reference fit and held participants overlap")
    source_keys = {
        (source_id, participant)
        for participant in source_windows["participant_id"].astype(str).unique()
    }
    if fit_keys | held_keys != source_keys:
        raise ValueError("reference fold participants disagree with source ledger")

    sampled_fit = sample_training_windows(
        fit_windows,
        negative_ratio=config.negative_ratio,
        seed=config.data_sampling_seed,
    )
    if sampled_fit.empty:
        raise ValueError("reference fit sampler retained no windows")
    scaler = SequenceScaler.fit(frame, fit_keys)
    dropout_by_field = _action_dropout_probabilities(
        frame,
        fit_keys,
        maximum=config.feature_group_dropout_max,
    )
    action_dropout_probabilities = tuple(
        float(dropout_by_field[field]) for field in ACTION_FIELDS
    )
    source_to_index = {source_id: 0}
    fit_dataset = WindowSequenceDataset(
        frame,
        sampled_fit,
        scaler=scaler,
        context_samples=config.context_samples,
        source_to_index=source_to_index,
        participant_store=participant_store,
    )
    derived_model_seed = derive_reference_model_seed(
        reference_seed=config.reference_seed,
        source_id=source_id,
        fold_id=fold_id,
    )
    _seed_everything(derived_model_seed)
    device = torch.device(config.device)
    model = model_factory().to(device)
    if production and count_parameters(model) != _EXPECTED_LSTM_PARAMETER_COUNT:
        raise ValueError(
            "production reference model is not the frozen 268166-parameter LSTM"
        )
    positive_weights = global_positive_weights.to(device)
    updates_per_epoch = math.ceil(len(fit_dataset) / config.effective_batch_size)
    total_updates = updates_per_epoch * config.maximum_epochs
    optimizer, scheduler = _optimizer_and_scheduler(
        model,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        total_updates=total_updates,
    )

    epoch_objectives: list[float] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_objective = float("inf")
    observed_updates = 0
    for epoch_index in range(config.maximum_epochs):
        observed_updates += _fit_epoch(
            model,
            fit_dataset,
            epoch_index=epoch_index,
            derived_model_seed=derived_model_seed,
            device=device,
            config=config,
            optimizer=optimizer,
            scheduler=scheduler,
            positive_weights=positive_weights,
            action_dropout_probabilities=action_dropout_probabilities,
            source_id=source_id,
            fold_id=fold_id,
        )
        evaluation = _evaluate_windows(
            model,
            frame,
            sampled_fit,
            scaler=scaler,
            context_samples=config.context_samples,
            source_to_index=source_to_index,
            device=device,
            batch_size=config.physical_batch_size,
            num_workers=config.num_workers,
            positive_weights=positive_weights,
            auxiliary_regression_weight=config.auxiliary_regression_weight,
            loss_denominator_contract=config.loss_denominator_contract,
            participant_store=participant_store,
        )
        objective = evaluation.loss.mean_loss
        if not math.isfinite(objective):
            raise FloatingPointError("non-finite deterministic fit objective")
        epoch_objectives.append(objective)
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "reference_fit_epoch",
                    "source_id": source_id,
                    "fold_id": fold_id,
                    "epoch": epoch_index + 1,
                    "fit_objective": objective,
                    "optimizer_updates": observed_updates,
                }
            )
        # Strict comparison leaves an exactly tied earliest checkpoint intact.
        if objective < best_objective:
            best_objective = objective
            best_epoch = epoch_index + 1
            best_state = _model_state_copy(model)

    if observed_updates != total_updates:
        raise AssertionError(
            "reference optimizer-update ledger disagrees with schedule"
        )
    selected_epoch, selected_objective = select_reference_checkpoint(
        enumerate(epoch_objectives, start=1)
    )
    if selected_epoch != best_epoch or selected_objective != best_objective:
        raise AssertionError(
            "reference checkpoint ledger disagrees with earliest minimum"
        )
    if best_state is None:
        raise AssertionError("reference fit completed without a checkpoint state")
    model.load_state_dict(best_state, strict=True)

    # The held sampler is intentionally called only after checkpoint selection.
    sampled_held = sample_training_windows(
        held_windows,
        negative_ratio=config.negative_ratio,
        seed=config.data_sampling_seed,
    )
    if sampled_held.empty:
        raise ValueError("reference held sampler retained no windows")
    held_evaluation = _evaluate_windows(
        model,
        frame,
        sampled_held,
        scaler=scaler,
        context_samples=config.context_samples,
        source_to_index=source_to_index,
        device=device,
        batch_size=config.physical_batch_size,
        num_workers=config.num_workers,
        positive_weights=positive_weights,
        auxiliary_regression_weight=config.auxiliary_regression_weight,
        loss_denominator_contract=config.loss_denominator_contract,
        participant_store=participant_store,
    )
    fit_evaluation = _evaluate_windows(
        model,
        frame,
        sampled_fit,
        scaler=scaler,
        context_samples=config.context_samples,
        source_to_index=source_to_index,
        device=device,
        batch_size=config.physical_batch_size,
        num_workers=config.num_workers,
        positive_weights=positive_weights,
        auxiliary_regression_weight=config.auxiliary_regression_weight,
        loss_denominator_contract=config.loss_denominator_contract,
        participant_store=participant_store,
    )
    if not math.isclose(
        fit_evaluation.loss.mean_loss,
        selected_objective,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        # A checkpoint is selected from this exact, deterministic estimator.
        # Re-evaluating it must produce the same float64 value before OOF use.
        raise AssertionError("reloaded reference checkpoint changed fit objective")

    expected_fit_denominator = (
        len(sampled_fit)
        if config.loss_denominator_contract == SAMPLED_RECORD_COUNT_V1_1
        else len(fit_windows)
    )
    expected_held_denominator = (
        len(sampled_held)
        if config.loss_denominator_contract == SAMPLED_RECORD_COUNT_V1_1
        else len(held_windows)
    )
    for label, observed, expected in (
        (
            "fit",
            fit_evaluation.loss.denominator,
            expected_fit_denominator,
        ),
        (
            "held",
            held_evaluation.loss.denominator,
            expected_held_denominator,
        ),
    ):
        if not math.isclose(
            observed,
            float(expected),
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise AssertionError(
                f"reference {label} denominator disagrees with its population"
            )

    return SyntheticReferenceFoldResult(
        source_id=source_id,
        fold_id=fold_id,
        fit_participant_ids=fit_participant_ids,
        held_participant_ids=held_participant_ids,
        fit_raw_window_count=len(fit_windows),
        fit_sampled_window_count=len(sampled_fit),
        held_raw_window_count=len(held_windows),
        held_sampled_window_count=len(sampled_held),
        fit_evaluation_denominator=fit_evaluation.loss.denominator,
        held_evaluation_denominator=held_evaluation.loss.denominator,
        loss_denominator_contract=config.loss_denominator_contract,
        derived_model_seed=derived_model_seed,
        checkpoint_epoch=selected_epoch,
        checkpoint_objective=selected_objective,
        epoch_objectives=tuple(epoch_objectives),
        fold_oof_loss=held_evaluation.loss.mean_loss,
        fit_evaluation_row_indices=fit_evaluation.original_window_row_indices,
        held_evaluation_row_indices=held_evaluation.original_window_row_indices,
        action_dropout_probabilities=tuple(
            (field, dropout_by_field[field]) for field in ACTION_FIELDS
        ),
        checkpoint_state=_model_state_copy(model),
        scaler=scaler,
    )


def _fit_reference_folds(
    frame: pd.DataFrame,
    windows: pd.DataFrame,
    *,
    config: SyntheticReferenceFitConfig | ProductionReferenceFitConfig,
    model_factory: Callable[[], nn.Module],
    production: bool,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> SyntheticReferenceFitResult:
    source_order = _validate_fit_config(config, production=production)
    work = _validate_train_only_inputs(
        frame,
        windows,
        source_order=source_order,
    )
    participants_by_source = {
        source: tuple(
            sorted(
                work.loc[work["source_id"].eq(source), "participant_id"].unique()
            )
        )
        for source in source_order
    }
    # One whole-cohort lookup for the entire run. `WindowSequenceDataset` builds
    # its own when none is passed, and this fitter constructs a dataset per fit and
    # per evaluation, so leaving it unset costs `folds x (1 + epochs + 2)` eager
    # groupby-and-sort passes over the full frame: 1,815 of them under the frozen
    # production contract, against the one `deep_training` builds and shares. The
    # store is immutable and read-only, so this is a cost change and not a value
    # change; `test_reference_fit_is_bitwise_unchanged_by_store_sharing` pins that.
    participant_store = ParticipantSequenceStore.from_frame(frame)
    assignments = assign_source_scoped_folds(
        participants_by_source,
        source_order=source_order,
        fold_count=config.fold_count,
        reference_seed=config.reference_seed,
    )
    sampled_global = sample_training_windows(
        work,
        negative_ratio=config.negative_ratio,
        seed=config.data_sampling_seed,
    )
    positive_weight_population = (
        sampled_global
        if config.loss_denominator_contract == SAMPLED_RECORD_COUNT_V1_1
        else work
    )
    global_positive_weights_tensor = _positive_weights(positive_weight_population)
    if not bool(torch.isfinite(global_positive_weights_tensor).all()):
        raise FloatingPointError("non-finite global reference class-positive weight")
    global_positive_weights = tuple(
        float(value) for value in global_positive_weights_tensor.tolist()
    )
    if len(global_positive_weights) != len(TASKS) or min(global_positive_weights) <= 0:
        raise ValueError("reference requires one positive global weight per task")

    fold_results: list[SyntheticReferenceFoldResult] = []
    per_source_reference_risks: dict[str, float] = {}
    for source_id in source_order:
        source_assignments = tuple(
            assignment
            for assignment in assignments
            if assignment.source_id == source_id
        )
        source_windows = work.loc[work["source_id"].eq(source_id)].copy()
        source_results: list[SyntheticReferenceFoldResult] = []
        for fold_id in range(config.fold_count):
            held_participant_ids = tuple(
                sorted(
                    assignment.participant_id
                    for assignment in source_assignments
                    if assignment.fold_id == fold_id
                )
            )
            fit_participant_ids = tuple(
                sorted(
                    assignment.participant_id
                    for assignment in source_assignments
                    if assignment.fold_id != fold_id
                )
            )
            result = _fit_one_source_fold(
                frame,
                source_windows,
                source_id=source_id,
                fold_id=fold_id,
                fit_participant_ids=fit_participant_ids,
                held_participant_ids=held_participant_ids,
                global_positive_weights=global_positive_weights_tensor,
                config=config,
                model_factory=model_factory,
                production=production,
                participant_store=participant_store,
                progress_callback=progress_callback,
            )
            source_results.append(result)
            fold_results.append(result)
        weighted = np.float64(0.0)
        held_denominator = np.float64(0.0)
        for result in source_results:
            weighted = np.float64(
                weighted
                + np.float64(result.fold_oof_loss)
                * np.float64(result.held_evaluation_denominator)
            )
            held_denominator = np.float64(
                held_denominator
                + np.float64(result.held_evaluation_denominator)
            )
        if held_denominator <= 0:
            raise AssertionError("reference source OOF denominator is empty")
        risk = float(weighted / held_denominator)
        if not math.isfinite(risk) or risk < 0:
            raise FloatingPointError("non-finite source reference OOF risk")
        per_source_reference_risks[source_id] = risk
    return SyntheticReferenceFitResult(
        assignments=assignments,
        global_class_positive_weights=global_positive_weights,
        folds=tuple(fold_results),
        per_source_reference_risks=per_source_reference_risks,
        loss_denominator_contract=config.loss_denominator_contract,
    )


def fit_synthetic_reference(
    frame: pd.DataFrame,
    windows: pd.DataFrame,
    *,
    config: SyntheticReferenceFitConfig,
    model_factory: Callable[[], nn.Module] | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> SyntheticReferenceFitResult:
    """Fit reduced synthetic source/fold references without writing an artifact."""

    factory = model_factory or (lambda: build_temporal_baseline("lstm"))
    return _fit_reference_folds(
        frame,
        windows,
        config=config,
        model_factory=factory,
        production=False,
        progress_callback=progress_callback,
    )


def fit_production_reference_artifact(
    frame: pd.DataFrame,
    windows: pd.DataFrame,
    *,
    artifact_path: str | Path,
    input_manifest_sha256: str,
    window_manifest_sha256: str,
    reference_config_sha256: str,
    loss_contract_sha256: str,
    config: ProductionReferenceFitConfig | None = None,
) -> dict[str, Any]:
    """Fit and immutably serialize exactly the frozen 11-source CF5 reference."""

    resolved_config = ProductionReferenceFitConfig() if config is None else config
    if resolved_config.loss_denominator_contract != SAMPLED_RECORD_COUNT_V1_1:
        raise ValueError(
            "legacy production reference requires sampled_record_count_v1_1"
        )
    result = _fit_reference_folds(
        frame,
        windows,
        config=resolved_config,
        model_factory=lambda: build_temporal_baseline("lstm"),
        production=True,
    )
    if len(result.folds) != len(PRODUCTION_SOURCE_ORDER) * 5:
        raise AssertionError(
            "production reference did not complete all 55 source-fold fits"
    )
    artifact = build_reference_artifact(
        source_order=resolved_config.source_order,
        assignments=result.assignments,
        fold_records=tuple(
            fold.as_reference_fold_record() for fold in result.folds
        ),
        per_source_reference_risks=result.per_source_reference_risks,
        reference_seed=resolved_config.reference_seed,
        data_sampling_seed=resolved_config.data_sampling_seed,
        fold_count=resolved_config.fold_count,
        input_manifest_sha256=input_manifest_sha256,
        window_manifest_sha256=window_manifest_sha256,
        reference_config_sha256=reference_config_sha256,
        global_class_positive_weights=result.global_class_positive_weights,
        loss_contract_sha256=loss_contract_sha256,
    )
    write_reference_artifact(artifact, artifact_path)
    return artifact


def fit_production_weighted_reference_artifact(
    frame: pd.DataFrame,
    windows: pd.DataFrame,
    *,
    artifact_path: str | Path,
    input_manifest_sha256: str,
    window_manifest_sha256: str,
    reference_config_sha256: str,
    loss_contract_sha256: str,
    config: WeightedProductionReferenceFitConfig | None = None,
) -> dict[str, Any]:
    """Fit and serialize the exact 11-source weighted successor reference."""

    return dict(
        fit_production_weighted_reference_bundle(
            frame,
            windows,
            artifact_path=artifact_path,
            input_manifest_sha256=input_manifest_sha256,
            window_manifest_sha256=window_manifest_sha256,
            reference_config_sha256=reference_config_sha256,
            loss_contract_sha256=loss_contract_sha256,
            config=config,
        ).artifact
    )


def fit_production_weighted_reference_bundle(
    frame: pd.DataFrame,
    windows: pd.DataFrame,
    *,
    artifact_path: str | Path,
    input_manifest_sha256: str,
    window_manifest_sha256: str,
    reference_config_sha256: str,
    loss_contract_sha256: str,
    config: WeightedProductionReferenceFitConfig | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> ProductionWeightedReferenceFitBundle:
    """Fit once while retaining the selected states needed for event OOF."""

    resolved_config = (
        WeightedProductionReferenceFitConfig() if config is None else config
    )
    if not isinstance(resolved_config, WeightedProductionReferenceFitConfig):
        raise ValueError(
            "sampling_weight_sum_v1_2 production reference requires "
            "WeightedProductionReferenceFitConfig before fitting"
        )
    if resolved_config.loss_denominator_contract != SAMPLING_WEIGHT_SUM_V1_2:
        raise ValueError(
            "weighted production reference requires sampling_weight_sum_v1_2"
        )
    result = _fit_reference_folds(
        frame,
        windows,
        config=resolved_config,
        model_factory=lambda: build_temporal_baseline("lstm"),
        production=True,
        progress_callback=progress_callback,
    )
    if result.loss_denominator_contract != SAMPLING_WEIGHT_SUM_V1_2:
        raise AssertionError("weighted production fit used the wrong denominator")
    if len(result.folds) != len(PRODUCTION_SOURCE_ORDER) * 5:
        raise AssertionError(
            "weighted production reference did not complete all 55 source-fold fits"
        )
    artifact = build_weighted_reference_artifact(
        source_order=resolved_config.source_order,
        assignments=result.assignments,
        fold_records=tuple(
            fold.as_weighted_reference_fold_record() for fold in result.folds
        ),
        per_source_reference_risks=result.per_source_reference_risks,
        reference_seed=resolved_config.reference_seed,
        data_sampling_seed=resolved_config.data_sampling_seed,
        input_manifest_sha256=input_manifest_sha256,
        window_manifest_sha256=window_manifest_sha256,
        reference_config_sha256=reference_config_sha256,
        loss_contract_sha256=loss_contract_sha256,
        global_class_positive_weights=result.global_class_positive_weights,
    )
    write_weighted_reference_artifact(artifact, artifact_path)
    return ProductionWeightedReferenceFitBundle(
        artifact=artifact,
        fit_result=result,
    )
