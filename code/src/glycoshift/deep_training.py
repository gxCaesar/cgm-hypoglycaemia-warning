from __future__ import annotations

import json
import math
import random
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from sklearn.metrics import average_precision_score
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset

from .anchor_composition import (
    ANCHORED_TASKS,
    attach_anchor_values,
    fit_glucose_anchor,
    summarize_anchor,
)
from .baselines import build_temporal_baseline
from .calibration import (
    apply_temperature,
    fit_conformal_threshold,
    fit_temperature,
)
from .config import resolve_path, save_config
from .fast_sequences import FastWindowSequenceDataset
from .io import load_frame, write_json
from .lead_time_weight import (
    attach_lead_time_loss_weights,
    summarize_lead_time_weights,
)
from .memory import ContiguousStrictPastMemoryBank
from .metrics import select_alarm_threshold
from .models import GlycoShift, GlycoShiftModelConfig, count_parameters
from .pipeline import (
    evaluate_predictions,
    load_audited_development_windows,
    prepare_dataset,
    prepare_evaluation_dataset,
    validate_training_output_isolation,
)
from .sampling import SourceBalancedSampler
from .schema import ACTION_FIELDS
from .sequences import (
    REGRESSION_HORIZONS,
    REVISION_CONTRACT,
    SEQUENCE_FEATURES,
    TASKS,
    ParticipantSequenceStore,
    SequenceScaler,
    WindowSequenceDataset,
    sample_training_windows,
    training_target_arrays,
    training_window_sampling_masks,
    validate_input_feature_mode,
)
from .severe_target import (
    attach_severe_training_targets,
    summarize_severe_targets,
)
from .sota_objectives import attach_tls_exp_targets, summarize_tls_targets
from .target_alignment import severe_source_logit_column


@dataclass
class GroupDRO:
    """Group-distributionally-robust loss aggregation over source groups.

    Two rules are available, and which one is used is a scientific choice
    recorded in the config as ``training.group_dro_rule``.

    ``cumulative`` (default) is Sagawa et al. 2020 exactly: exponentiated
    gradient on the simplex, ``w <- w * exp(eta * L)`` renormalised. The
    multiplication accumulates across steps, so a group whose loss stays
    highest has its weight grow without bound. That is the intended no-regret
    behaviour when group difficulty is something the model can equalise.

    **It is not benign when group difficulty is intrinsic.** Measured on this
    project's eleven CGM sources, whose per-group losses span 0.28 to 1.19,
    the weight vector collapsed to 1.0000 on a single source (AZT1D) at every
    epoch of every seed, so an arm described as cross-source training was in
    fact single-source training, and it scored 0.062 BELOW plain ERM on a
    same-seed same-epoch paired comparison. A float64 log-space oracle ruled
    out float32 underflow as the cause: the collapse is what the unconstrained
    objective specifies.

    ``softmax`` is the fix: ``w = softmax(L / tau)``, the one-step closed form
    of KL-ball DRO, which depends only on the CURRENT losses and therefore
    cannot accumulate. It interpolates continuously between the two regimes
    this project has already measured as endpoints:

        tau -> infinity   uniform weights, i.e. ERM
        tau -> 0          all mass on the worst group

    ``temperature`` is tau. ``eta`` is retained for the cumulative rule so the
    original arm remains exactly reproducible; ``eta = 0`` under the cumulative
    rule is exact group-balanced ERM (bitwise uniform), which is how the ERM
    control arm is built.
    """

    weights: torch.Tensor
    eta: float
    rule: str = "cumulative"
    temperature: float = 1.0

    @classmethod
    def create(
        cls,
        *,
        group_count: int,
        eta: float,
        device: torch.device,
        rule: str = "cumulative",
        temperature: float = 1.0,
    ) -> GroupDRO:
        if rule not in {"cumulative", "softmax"}:
            raise ValueError(f"unsupported group_dro_rule: {rule}")
        if rule == "softmax" and not temperature > 0:
            raise ValueError("softmax rule requires a positive temperature")
        return cls(
            weights=torch.full(
                (group_count,),
                1.0 / group_count,
                device=device,
            ),
            eta=float(eta),
            rule=str(rule),
            temperature=float(temperature),
        )

    def aggregate(
        self,
        losses: torch.Tensor,
        groups: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[int, float]]:
        group_losses: list[torch.Tensor] = []
        present: list[int] = []
        observed: dict[int, float] = {}
        for group in range(len(self.weights)):
            mask = groups == group
            if mask.any():
                value = losses[mask].mean()
                group_losses.append(value)
                present.append(group)
                observed[group] = float(value.detach().cpu())
        if not group_losses:
            raise ValueError("GroupDRO batch has no source groups")
        indices = torch.tensor(present, device=losses.device, dtype=torch.long)
        stacked = torch.stack(group_losses)

        if self.rule == "cumulative":
            with torch.no_grad():
                self.weights[indices] *= torch.exp(self.eta * stacked.detach())
                self.weights /= self.weights.sum()
            active_weights = self.weights[indices]
            active_weights = active_weights / active_weights.sum()
        else:
            # Memoryless: current losses only. softmax is computed over the
            # groups PRESENT in this batch, then written back so the logged
            # weight vector stays interpretable; absent groups keep no mass
            # this step rather than inheriting stale mass.
            with torch.no_grad():
                active = torch.softmax(
                    stacked.detach() / self.temperature,
                    dim=0,
                )
                self.weights.zero_()
                self.weights[indices] = active
            active_weights = active
        return torch.sum(active_weights * stacked), observed


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(config: dict[str, Any]) -> torch.device:
    requested = str(config["training"].get("device", "auto"))
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


# The source-balanced sampler draws in 65,536-record chunks, so a chunk boundary
# is crossed every chunk_size/batch_size steps. Left on the global RNG the
# sampler and the feature-group dropout in the training loop interleave their
# draws, and DataLoader prefetch depth (2 * num_workers) shifts that interleaving
# -- measured 2026-08-13: index and dropout streams both diverge between
# num_workers 0, 2, 4 and 8. An owned generator decouples the two streams so the
# realized sample order depends on the seed alone.
_SAMPLER_SEED_OFFSET = 0x5EED


def _window_dataset_class(config: Mapping[str, Any]) -> type[WindowSequenceDataset]:
    """Pick the sample-assembly implementation named by the config.

    The array-backed path is now the default, on the evidence its adoption was
    made conditional on: `scripts/audit_fast_sequences_equivalence.py` compared
    20,000 real calibration windows across all nine returned tensors on cityu
    and reported `bit_identical: true` with every maximum absolute difference
    zero and no dtype or shape mismatch, at 111.4 microseconds per sample
    against 1,883.1. Synthetic fixtures had already shown the same on two hosts
    with different floating-point environments. The audit report is retained at
    `experiments/20260817-glycoshift-v1-5-baseline-closure-v1/fast-sequences-equivalence-audit.json`.

    This is the same discipline the participant memory bank followed: the
    reference implementation stays, the optimised one becomes production only
    after exact equivalence is proven on real shards, and either can be selected
    explicitly. `training.fast_sample_assembly: false` restores the original,
    which is what to set if this path is ever suspected.
    """

    training = config.get("training") or {}
    if bool(training.get("fast_sample_assembly", True)):
        return FastWindowSequenceDataset
    return WindowSequenceDataset


def _loader(
    dataset: WindowSequenceDataset,
    *,
    batch_size: int,
    num_workers: int,
    source_balanced: bool,
    sampler_seed: int | None = None,
) -> DataLoader:
    if source_balanced:
        source_indices = dataset.windows["source_id"].map(
            dataset.source_to_index
        )
        if source_indices.isna().any():
            raise ValueError("training windows contain an unknown source")
        if sampler_seed is None:
            raise ValueError("source-balanced sampling requires a sampler seed")
        sampler = SourceBalancedSampler(
            source_indices.to_numpy(dtype=np.int64, copy=False),
            num_sources=len(dataset.source_to_index),
            num_samples=len(dataset),
            generator=torch.Generator(device="cpu").manual_seed(int(sampler_seed)),
        )
    else:
        sampler = None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=sampler is None and source_balanced,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def _positive_weights(
    windows: pd.DataFrame,
    *,
    mode: str = "prevalence",
    use_training_targets: bool = False,
) -> torch.Tensor:
    """Per-task positive-class weight.

    `prevalence`, the default and the frozen behaviour, returns
    `min(100, negatives / positives)` measured on the frame it is given. That
    frame has already been negative-subsampled, so positives are up-weighted
    once by sampling and again here.

    `uniform` returns ones. It is C1's removal ablation and it removes the
    second up-weighting only, leaving the sampling ratio untouched. The
    null hypothesis of a control is *not doing* the thing, not doing it
    differently, so this goes all the way to one rather than to some
    intermediate value chosen after looking.

    The deficit is measured rather than supposed. True severe-event rates
    differ across glucose bands by about 290x, so a single scalar up-weight
    distorts the bands unequally, and the atlas records the resulting
    over-prediction rising strictly monotonically from 64x to 918x with
    glucose. M0's own record attributes 12.69 of a 55-fold over-prediction to
    this weighting against 4.34 to the target mismatch it was built to fix.

    ``use_training_targets`` is the dated revision opt-in. It counts only the
    actual retained training targets/masks and keeps the same ratio cap/rule.
    The default preserves historical M0's ordinary-label weighting defect so
    retained results can still be reconstructed without changing their recipe.
    """

    if mode not in {"prevalence", "uniform"}:
        raise ValueError(
            f"training.positive_weight_mode must be prevalence or uniform, got {mode!r}"
        )
    resolved = training_target_arrays(windows) if use_training_targets else None
    weights: list[float] = []
    for position, task in enumerate(TASKS):
        if mode == "uniform":
            weights.append(1.0)
            continue
        if resolved is None:
            kind = task.split("_")[0]
            eligible = windows.loc[~windows[f"{kind}_currently_active"]]
            labels = eligible[f"{task}_label"].to_numpy(dtype=int)
        else:
            targets, masks = resolved
            labels = targets[masks[:, position], position].astype(float)
        positives = max(1.0, float(labels.sum()))
        negatives = max(1.0, float(len(labels) - labels.sum()))
        weights.append(min(100.0, negatives / positives))
    return torch.tensor(weights, dtype=torch.float32)


def _training_target_audit(
    windows: pd.DataFrame, positive_weights: torch.Tensor
) -> dict[str, Any]:
    targets, masks = training_target_arrays(windows)
    tasks = {}
    for position, task in enumerate(TASKS):
        target = f"{task}_training_target"
        mask = f"{task}_training_mask"
        retained = targets[masks[:, position], position].astype(float)
        tasks[task] = {
            "target_column": target if target in windows else f"{task}_label",
            "mask_column": (
                mask if mask in windows else "legacy_currently_active_and_label_present"
            ),
            "retained_count": int(len(retained)),
            "positive_count": float(retained.sum()),
            "negative_count": float(len(retained) - retained.sum()),
            "positive_weight": float(positive_weights[position]),
        }
    return {"sampled_window_count": len(windows), "tasks": tasks}


def _drop_action_groups(
    sequence: torch.Tensor,
    *,
    probabilities: tuple[float, ...],
) -> torch.Tensor:
    if len(probabilities) != len(ACTION_FIELDS):
        raise ValueError("one feature-group dropout probability is required per action")
    if max(probabilities, default=0.0) <= 0:
        return sequence
    output = sequence.clone()
    batch_size = sequence.shape[0]
    for field_index, probability in enumerate(probabilities):
        start = 2 + field_index * 5
        drop = torch.rand(batch_size, device=sequence.device) < probability
        if not drop.any():
            continue
        output[drop, :, start : start + 4] = 0
        output[drop, :, start + 4] = 1
    return output


def _action_dropout_probabilities(
    frame: pd.DataFrame,
    participant_keys: set[tuple[str, str]],
    *,
    maximum: float,
) -> dict[str, float]:
    if not 0 <= maximum <= 1:
        raise ValueError("feature_group_dropout_max must be between zero and one")
    row_keys = list(
        zip(  # noqa: B905
            frame["source_id"].astype(str),
            frame["participant_id"].astype(str),
        )
    )
    selected = frame.loc[[key in participant_keys for key in row_keys]]
    probabilities: dict[str, float] = {}
    for field in ACTION_FIELDS:
        available = selected[f"{field}_available"].astype(bool)
        if not available.any():
            probabilities[field] = 0.0
            continue
        unreliable = (
            ~selected.loc[available, f"{field}_observed"].astype(bool)
            | selected.loc[
                available,
                f"{field}_zero_ambiguous",
            ].astype(bool)
        )
        probabilities[field] = float(min(maximum, unreliable.mean()))
    return probabilities


# The composed probability is clamped before its logarithm. `anchor` is strictly
# inside (0, 1) by construction and `sigmoid` is in [0, 1], so the convex
# combination can only reach a boundary when `sigmoid` underflows in float32.
_ANCHOR_EPSILON = 1e-7


def _anchored_classification(
    logits: torch.Tensor,
    labels: torch.Tensor,
    positive: torch.Tensor,
    anchor: torch.Tensor,
    anchor_weight: float,
) -> torch.Tensor:
    """Return the classification loss taken through the deployed composition.

    The deployed scorer is a composition of this member with a current-glucose
    threshold, and `RESULT-a-better-member-makes-the-combination-worse.md`
    measured that being good alone and being good in that composition are
    different objectives here by eleven sigma of swing. This puts the anchor
    inside the loss so the member is trained against the second one.

    The gradient is what carries the claim. With `p = w*sigma + (1-w)*A`,
    `dL/df = w * sigma * (1 - sigma) * (p - y) / (p * (1 - p))`, so a window
    where the anchor is already high and the label is positive contributes
    almost nothing, while the anchor's false alarms and the events it misses
    both contribute strongly. The member is not taught to relearn current
    glucose.

    Only `ANCHORED_TASKS` are composed. `hyper_*` keep the original
    `logsigmoid` form exactly, so their values and gradients are unchanged.
    """

    probability = anchor_weight * torch.sigmoid(logits) + (1.0 - anchor_weight) * anchor
    probability = probability.clamp(_ANCHOR_EPSILON, 1.0 - _ANCHOR_EPSILON)
    return -(
        positive * labels * torch.log(probability)
        + (1 - labels) * torch.log1p(-probability)
    )


def _unweighted_multitask_loss(
    outputs: dict[str, Any],
    batch: dict[str, torch.Tensor],
    *,
    positive_weights: torch.Tensor,
    auxiliary_weight: float,
    anchor_weight: float | None = None,
) -> torch.Tensor:
    logits = torch.stack([outputs["logits"][task] for task in TASKS], dim=1)
    labels = batch.get("training_labels", batch["labels"])
    mask = batch["label_mask"].float()
    positive = positive_weights.unsqueeze(0)
    classification = -(
        positive * labels * functional.logsigmoid(logits)
        + (1 - labels) * functional.logsigmoid(-logits)
    )
    anchor = batch.get("anchor")
    if anchor_weight is not None and anchor is not None:
        # Absent unless the training frame carries an anchor column, so the
        # default path never reaches this branch and is bitwise the code it
        # was. The task selection is a fixed boolean over TASKS rather than a
        # config field: which head carries the endpoint's severe score is a
        # property of the endpoint, not a tuning choice.
        anchored = torch.tensor(
            [task in ANCHORED_TASKS for task in TASKS],
            device=logits.device,
        ).unsqueeze(0)
        composed = _anchored_classification(
            logits,
            labels,
            positive,
            anchor.unsqueeze(1),
            float(anchor_weight),
        )
        classification = torch.where(anchored, composed, classification)
    event_weight = batch.get("loss_weight")
    if event_weight is not None:
        # Absent unless a caller attached per-task weights, so the default path
        # never executes this multiply and is bitwise the code it was. Present,
        # it scales each positive window's event loss by the warning time that
        # window delivers. The weights average one over positives, so this
        # redistributes emphasis among them without altering the aggregate weight
        # positives carry against negatives.
        classification = classification * event_weight
    classification = (classification * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    regression = torch.stack(
        [outputs["regression"][str(value)] for value in REGRESSION_HORIZONS],
        dim=1,
    )
    regression_elementwise = functional.smooth_l1_loss(
        regression,
        batch["regression_targets"],
        reduction="none",
    )
    regression_mask = batch.get(
        "regression_mask",
        torch.ones_like(regression_elementwise, dtype=torch.bool),
    ).float()
    regression_loss = (
        regression_elementwise * regression_mask
    ).sum(dim=1) / regression_mask.sum(dim=1).clamp_min(1)
    loss = classification + auxiliary_weight * regression_loss
    if not bool(torch.isfinite(loss).all()):
        raise FloatingPointError("multitask loss became non-finite")
    return loss


def _multitask_loss(
    outputs: dict[str, Any],
    batch: dict[str, torch.Tensor],
    *,
    positive_weights: torch.Tensor,
    auxiliary_weight: float,
    anchor_weight: float | None = None,
) -> torch.Tensor:
    """Return the legacy numerator-weighted per-record multitask loss."""

    return _unweighted_multitask_loss(
        outputs,
        batch,
        positive_weights=positive_weights,
        auxiliary_weight=auxiliary_weight,
        anchor_weight=anchor_weight,
    ) * batch["sampling_weight"]


def _encode_dataset(
    model: nn.Module,
    dataset: WindowSequenceDataset,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    dataset_indices: np.ndarray | None = None,
) -> torch.Tensor:
    selected_indices = (
        None
        if dataset_indices is None
        else np.asarray(dataset_indices, dtype=np.int64)
    )
    if selected_indices is not None:
        if (
            selected_indices.ndim != 1
            or len(selected_indices) == 0
            or selected_indices.min() < 0
            or selected_indices.max() >= len(dataset)
        ):
            raise ValueError("dataset indices are outside the sequence dataset")
        encoding_dataset = Subset(dataset, selected_indices)
    else:
        encoding_dataset = dataset
    loader = _loader(
        encoding_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        source_balanced=False,
    )
    output = torch.empty(
        len(encoding_dataset),
        model.config.d_model,
        dtype=torch.float32,
    )
    model.eval()
    offset = 0
    with torch.no_grad():
        for batch in loader:
            indices = batch["dataset_index"].long()
            sequence = batch["sequence"].to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                representation = model.encode_sequence(sequence)[:, -1]
            if selected_indices is None:
                output[indices] = representation.float().cpu()
            else:
                stop = offset + len(indices)
                expected = torch.from_numpy(
                    selected_indices[offset:stop]
                ).long()
                if not torch.equal(indices, expected):
                    raise AssertionError(
                        "anchor encoder changed the requested dataset order"
                    )
                output[offset:stop] = representation.float().cpu()
                offset = stop
    if selected_indices is not None and offset != len(selected_indices):
        raise AssertionError("anchor encoder did not visit every requested row")
    return output


def _memory_bank(
    model: nn.Module,
    dataset: WindowSequenceDataset,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    stride_minutes: int,
) -> ContiguousStrictPastMemoryBank:
    embeddings = _encode_dataset(
        model,
        dataset,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    bank = ContiguousStrictPastMemoryBank(
        k=model.config.memory_k,
        stride_minutes=stride_minutes,
        device=device,
        dtype=embeddings.dtype,
    )
    bank.build_from_windows(
        embeddings=embeddings,
        windows=dataset.windows,
    )
    return bank


def _predict_dataset(
    model: nn.Module,
    dataset: WindowSequenceDataset,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    memory_stride_minutes: int,
) -> pd.DataFrame:
    embeddings = _encode_dataset(
        model,
        dataset,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    if bool(getattr(model, "supports_memory", False)):
        bank = ContiguousStrictPastMemoryBank(
            k=model.config.memory_k,
            stride_minutes=memory_stride_minutes,
            device=device,
            dtype=embeddings.dtype,
        )
        bank.build_from_windows(
            embeddings=embeddings,
            windows=dataset.windows,
        )
    else:
        bank = None
    logits = {
        task: np.empty(len(dataset), dtype=np.float32)
        for task in TASKS
    }
    regression = {
        str(value): np.empty(len(dataset), dtype=np.float32)
        for value in REGRESSION_HORIZONS
    }
    loader = _loader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        source_balanced=False,
    )
    model.eval()
    with torch.no_grad():
        for batch in loader:
            indices = batch["dataset_index"].long()
            representation = embeddings[indices].to(device)
            if bank is None:
                memory = None
                padding = None
            else:
                query_metadata = [
                    dataset.metadata(int(index))
                    for index in indices.tolist()
                ]
                memory, padding = bank.retrieve(
                    query_embeddings=representation,
                    query_metadata=query_metadata,
                )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                outputs = model.forward_from_representation(
                    representation,
                    memory_embeddings=memory,
                    memory_padding_mask=padding,
                )
            for task in TASKS:
                logits[task][indices] = (
                    outputs["logits"][task].float().cpu().numpy()
                )
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


def _selection_score(
    predictions: pd.DataFrame, *, use_training_targets: bool = False
) -> float:
    scores: list[float] = []
    resolved = training_target_arrays(predictions) if use_training_targets else None
    for position, task in enumerate(TASKS):
        if resolved is None:
            kind = task.split("_")[0]
            mask = ~predictions[f"{kind}_currently_active"]
            labels = predictions[f"{task}_label"].loc[mask].to_numpy(dtype=int)
        else:
            targets, masks = resolved
            mask = masks[:, position]
            labels = targets[mask, position]
        if len(np.unique(labels)) < 2:
            continue
        scores.append(
            float(
                average_precision_score(
                    labels,
                    predictions[f"{task}_probability"].loc[mask],
                )
            )
        )
    if not scores:
        raise ValueError("selection split has no task with both classes")
    return float(np.mean(scores))


def _restore_regression_scale(
    predictions: pd.DataFrame,
    scaler: SequenceScaler,
) -> None:
    scale = scaler.values["cgm_mg_dl"]
    for horizon in REGRESSION_HORIZONS:
        predictions[f"cgm_{horizon}_prediction"] = (
            predictions[f"cgm_{horizon}_prediction_scaled"] * scale.std
            + scale.mean
        )


def _attach_severe_scores(predictions: pd.DataFrame) -> None:
    """Bind the frozen severe-event score to the matching hypo-head logit."""

    for horizon in REGRESSION_HORIZONS:
        # Read the binding from its single source of truth rather than spelling
        # it out again here; see `target_alignment` for why it is recorded once.
        logit_column = severe_source_logit_column(horizon)
        if logit_column not in predictions:
            raise ValueError(f"prediction frame is missing {logit_column}")
        logits = pd.to_numeric(
            predictions[logit_column], errors="coerce"
        ).to_numpy(dtype=float)
        if not np.isfinite(logits).all():
            raise ValueError("severe-hypoglycemia checkpoint scores are non-finite")
        predictions[f"severe_hypo_{horizon}_raw_logit"] = logits
        predictions[f"severe_hypo_{horizon}_raw_probability"] = 1.0 / (
            1.0 + np.exp(-np.clip(logits, -50.0, 50.0))
        )


def _selection_operational_candidate_count(
    config: dict[str, Any],
) -> int | None:
    value = config["evaluation"].get("selection_operational_candidate_count")
    if value is None:
        return None
    if not bool(config["project"].get("synthetic", False)):
        raise ValueError(
            "real-data checkpoint proxy requires all unique selection scores"
        )
    resolved = int(value)
    if resolved < 2:
        raise ValueError(
            "selection operational candidate count must be at least two"
        )
    return resolved


def _append_log(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _record_segment_epoch(
    *,
    segment_trainer: Any,
    model: nn.Module,
    epoch: int,
    scan_limit: int,
    records: list[dict[str, Any]],
    run_dir: Path,
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    """Run and persist the frozen per-epoch G4 gate before advancing the dual."""

    if scan_limit <= 0:
        raise ValueError("G4 scan_limit must be positive")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("G4 tolerance must be finite and positive")
    log_path = run_dir / "segment_surrogate_log.json"

    for update_index in range(scan_limit):
        try:
            comparison = segment_trainer.stream_agreement(
                model,
                epoch=epoch,
                update_index=update_index,
            )
        except Exception as exc:
            failure = {
                "event": "stream_agreement",
                "epoch": int(epoch),
                "scan_update_index": int(update_index),
                "segments": 0,
                "tolerance": float(tolerance),
                "status": "fail",
                "reason": "comparison_error",
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            }
            records.append(failure)
            write_json({"records": records}, log_path)
            raise

        segments = comparison.get("segments")
        if (
            not isinstance(segments, int)
            or isinstance(segments, bool)
            or segments < 0
        ):
            failure = {
                "event": "stream_agreement",
                "epoch": int(epoch),
                "scan_update_index": int(update_index),
                "segments": segments,
                "tolerance": float(tolerance),
                "status": "fail",
                "reason": "invalid_segment_count",
            }
            records.append(failure)
            write_json({"records": records}, log_path)
            raise RuntimeError(
                "G4 stream agreement returned an invalid segment count"
            )
        if segments == 0:
            continue

        try:
            mean_error = float(comparison["mean_relative_error"])
            max_error = float(comparison["max_relative_error"])
            warmup = int(comparison["warmup_decisions"])
        except (KeyError, TypeError, ValueError) as exc:
            failure = {
                "event": "stream_agreement",
                "epoch": int(epoch),
                "scan_update_index": int(update_index),
                "segments": int(segments),
                "tolerance": float(tolerance),
                "status": "fail",
                "reason": "incomplete_comparison",
            }
            records.append(failure)
            write_json({"records": records}, log_path)
            raise RuntimeError(
                "G4 stream agreement lacks required comparison fields"
            ) from exc

        agreement = {
            "event": "stream_agreement",
            "epoch": int(epoch),
            "scan_update_index": int(update_index),
            "segments": int(segments),
            "mean_relative_error": mean_error,
            "max_relative_error": max_error,
            "warmup_decisions": warmup,
            "tolerance": float(tolerance),
            "status": "pass",
        }
        if (
            not math.isfinite(mean_error)
            or not math.isfinite(max_error)
            or mean_error < 0
            or max_error < 0
            or mean_error > max_error
        ):
            agreement.update(status="fail", reason="nonfinite_or_invalid_error")
            records.append(agreement)
            write_json({"records": records}, log_path)
            raise FloatingPointError(
                f"non-finite or invalid G4 stream agreement at epoch={epoch}"
            )
        if max_error > tolerance:
            agreement.update(status="fail", reason="tolerance_exceeded")
            records.append(agreement)
            write_json({"records": records}, log_path)
            raise RuntimeError(
                f"G4 stream agreement failed at epoch={epoch}: "
                f"max_relative_error={max_error} > {tolerance}"
            )

        records.append(agreement)
        records.append(
            {"event": "dual_update", **segment_trainer.end_epoch(epoch)}
        )
        write_json({"records": records}, log_path)
        return agreement

    failure = {
        "event": "stream_agreement",
        "epoch": int(epoch),
        "scan_update_index": None,
        "scan_limit": int(scan_limit),
        "segments": 0,
        "tolerance": float(tolerance),
        "status": "fail",
        "reason": "no_eligible_burden_segment",
    }
    records.append(failure)
    write_json({"records": records}, log_path)
    raise RuntimeError(
        f"G4 stream agreement found no eligible burden segment at epoch={epoch}"
    )


def _load_development_training_data(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if bool(config["project"].get("synthetic", False)):
        return prepare_dataset(config, include_holdouts=False)
    development_sources = set(
        map(str, config["data"]["development_sources"])
    )
    frame = load_frame(config, sources=development_sources)
    windows, audit = load_audited_development_windows(config)
    return frame, windows, audit


def run_deep_training(
    config: dict[str, Any],
    *,
    architecture: str = "glycoshift",
    prepared_data: tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]] | None = None,
    revision_checkpoint_proxy: Callable | None = None,
    revision_calibrator: Callable | None = None,
    revision_reporter: Callable | None = None,
) -> Path:
    revision_contract = config["training"].get("revision_contract")
    if revision_contract not in (None, REVISION_CONTRACT):
        raise ValueError(f"unsupported training.revision_contract: {revision_contract}")
    revision = revision_contract is not None
    input_feature_mode = validate_input_feature_mode(
        str(config["model"].get("input_feature_mode", "multimodal"))
    )
    if revision and (
        prepared_data is None
        or revision_checkpoint_proxy is None
        or revision_calibrator is None
        or revision_reporter is None
    ):
        raise ValueError(
            "revision training requires prepared_data and fixed-frame callbacks"
        )
    if not revision and any(
        callback is not None
        for callback in (
            revision_checkpoint_proxy, revision_calibrator, revision_reporter
        )
    ):
        raise ValueError("revision callbacks require training.revision_contract")
    if prepared_data is not None and not revision:
        raise ValueError("prepared_data requires training.revision_contract")
    seed = int(
        config["training"].get("seed", config["split"]["seed"])
    )
    _seed_everything(seed)
    run_dir = resolve_path(config, config["output"]["run_dir"])
    if run_dir is None:
        raise ValueError("output.run_dir is required")
    protected_artifacts = (
        run_dir / "model_summary.json",
        run_dir / "predictions.parquet",
        run_dir / "ckpts" / "best.pt",
    )
    if (
        not bool(config["project"].get("synthetic", False))
        and any(path.exists() for path in protected_artifacts)
    ):
        raise FileExistsError(
            f"refusing to overwrite an existing real-data run: {run_dir}"
        )
    validate_training_output_isolation(config, run_dir)
    frame, windows, audit = (
        prepared_data
        if prepared_data is not None
        else _load_development_training_data(config)
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / "ckpts"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train.jsonl"
    save_config(config, run_dir / "config_resolved.yaml")
    write_json(audit, run_dir / "cohort_audit.json")

    train_windows = windows.loc[windows["partition"] == "train"].copy()
    selection_windows = windows.loc[windows["partition"] == "selection"].copy()
    calibration_windows = windows.loc[windows["partition"] == "calibration"].copy()
    if any(
        part.empty
        for part in (train_windows, selection_windows, calibration_windows)
    ):
        raise ValueError(
            "deep training requires train, selection, and calibration windows"
        )
    window_counts = {
        "train_windows_before_sampling": len(train_windows),
        "selection_windows": len(selection_windows),
        "calibration_windows": len(calibration_windows),
    }
    del windows
    source_order = tuple(
        sorted(map(str, config["data"]["development_sources"]))
    )
    observed_sources = tuple(
        sorted(train_windows["source_id"].astype(str).unique())
    )
    if observed_sources != source_order:
        raise ValueError("deep train split is missing a development source")
    selection_candidate_count = _selection_operational_candidate_count(config)
    # C5. Absent, this is the frozen `area` rule every completed arm was
    # selected under; the branch below is not entered and nothing changes.
    checkpoint_selection_rule = str(
        (config.get("training") or {}).get("checkpoint_selection_rule", "area")
    )
    checkpoint_budget = float(
        (config.get("training") or {}).get("checkpoint_budget", 0.5)
    )
    training = config["training"]
    event_objective = str(training.get("event_objective", "bce"))
    if revision and event_objective != "severe_retarget":
        raise ValueError(
            "fixed-frame revision requires event_objective severe_retarget"
        )
    if event_objective == "tls_exp":
        train_windows = attach_tls_exp_targets(
            train_windows,
            sample_minutes=int(config["protocol"]["sample_minutes"]),
            gamma_per_hour=float(training["tls_gamma_per_hour"]),
            copy=False,
        )
        tls_audit = summarize_tls_targets(train_windows)
        tls_audit.update(
            {
                "objective": "tls_exp",
                "gamma_per_hour": float(training["tls_gamma_per_hour"]),
                "support_horizon_multiple": 2,
            }
        )
        write_json(tls_audit, run_dir / "tls_target_audit.json")
    elif event_objective == "severe_retarget":
        # The scored head is `hypo_<h>`; `_attach_severe_scores` binds the severe
        # score to its logit. This points that head at the severe label and widens
        # its loss mask to the endpoint's own eligibility rule. Every `_label`
        # column, and so the negative-subsampling positive mask below, is left
        # alone, which is what keeps this arm a single-variable change from its
        # control.
        if revision:
            # The timeline builder owns horizon-specific base-evaluability masks.
            # Require its explicit targets/masks instead of overwriting them here.
            for current_windows in (
                train_windows, selection_windows, calibration_windows
            ):
                required = {
                    f"hypo_{horizon}_training_{suffix}"
                    for horizon in (30, 60)
                    for suffix in ("target", "mask")
                }
                missing = sorted(required.difference(current_windows.columns))
                if missing:
                    raise ValueError(
                        f"revision timeline is missing target/mask columns: {missing}"
                    )
                targets, masks = training_target_arrays(current_windows)
                for position, horizon in enumerate((30, 60)):
                    severe = current_windows[
                        f"severe_hypo_{horizon}_label"
                    ].to_numpy(float)
                    retained = masks[:, position]
                    if not np.array_equal(
                        targets[retained, position], severe[retained]
                    ):
                        raise ValueError(
                            f"revision hypo_{horizon} target differs from severe label"
                        )
        else:
            train_windows = attach_severe_training_targets(train_windows, copy=False)
        severe_audit = summarize_severe_targets(train_windows)
        severe_audit["objective"] = "severe_retarget"
        write_json(severe_audit, run_dir / "severe_target_audit.json")
    anchor_weight = training.get("anchor_weight")
    if anchor_weight is not None:
        # M6. The anchor is fitted on `train_windows`, which this function has
        # already restricted to the train partition, and `fit_glucose_anchor`
        # refuses any other partition rather than trusting that. The column is
        # attached here and nowhere else, so the selection and calibration
        # datasets built below cannot carry it.
        anchor_weight = float(anchor_weight)
        anchor = fit_glucose_anchor(train_windows)
        train_windows = attach_anchor_values(train_windows, anchor, copy=False)
        write_json(
            summarize_anchor(train_windows, anchor, anchor_weight=anchor_weight),
            run_dir / "anchor_audit.json",
        )
    elif event_objective == "lead_time_weighted":
        # A weight, not a target: the Bayes-optimal score stays the event
        # probability, so this can act only through finite capacity and finite
        # data. That bound is stated in the pre-registration and a null is a live
        # outcome. It inverts the published TLS direction, which values a
        # five-minute lead 1.913 times a thirty-minute one.
        train_windows = attach_lead_time_loss_weights(train_windows, copy=False)
        lead_audit = summarize_lead_time_weights(train_windows)
        lead_audit["objective"] = "lead_time_weighted"
        lead_audit["weight_form"] = "v(d) = d / mean(d over positives)"
        write_json(lead_audit, run_dir / "lead_time_weight_audit.json")
    train_keys = set(
        zip(  # noqa: B905
            train_windows["source_id"].astype(str),
            train_windows["participant_id"].astype(str),
        )
    )
    scaler = SequenceScaler.fit(
        frame, train_keys, input_feature_mode=input_feature_mode
    )
    write_json(scaler.as_dict(), run_dir / "scaler.json")
    dropout_probabilities = (
        {field: 0.0 for field in ACTION_FIELDS}
        if input_feature_mode == "cgm_mask_clock"
        else _action_dropout_probabilities(
            frame, train_keys,
            maximum=float(training.get("feature_group_dropout_max", 0.5)),
        )
    )
    write_json(
        dropout_probabilities,
        run_dir / "feature_group_dropout.json",
    )
    sampled_train = sample_training_windows(
        train_windows,
        negative_ratio=float(config["training"]["negative_ratio"]),
        seed=int(training.get("data_sampling_seed", seed)),
        use_training_targets=revision,
    )
    window_counts["train_windows_after_sampling"] = len(sampled_train)
    positive_weights = _positive_weights(
        sampled_train,
        mode=str(training.get("positive_weight_mode", "prevalence")),
        use_training_targets=revision,
    )
    target_audit = (
        _training_target_audit(sampled_train, positive_weights) if revision else None
    )
    if target_audit is not None:
        sampling_positive, sampling_usable = training_window_sampling_masks(
            train_windows
        )
        retained_positive, _ = training_window_sampling_masks(sampled_train)
        target_audit["sampling"] = {
            "positive_basis": "any_training_target_equal_one_and_task_mask",
            "input_window_count": len(train_windows),
            "usable_window_count": int(sampling_usable.sum()),
            "excluded_no_loss_count": int((~sampling_usable).sum()),
            "input_positive_count": int(sampling_positive.sum()),
            "sampled_positive_count": int(retained_positive.sum()),
            "sampled_negative_count": int((~retained_positive).sum()),
            "negative_ratio": float(training["negative_ratio"]),
            "data_sampling_seed": int(training.get("data_sampling_seed", seed)),
        }
        target_audit["positive_weight_mode"] = str(
            training.get("positive_weight_mode", "prevalence")
        )
        target_audit["positive_weight_basis"] = "retained_training_targets_and_masks"
        target_audit["positive_weight_cap"] = 100.0
    input_contract = {
        "input_feature_mode": input_feature_mode,
        "feature_width": len(SEQUENCE_FEATURES),
        "active_features": list(
            (SEQUENCE_FEATURES[:2] + SEQUENCE_FEATURES[-2:])
            if input_feature_mode == "cgm_mask_clock" else SEQUENCE_FEATURES
        ),
        "zeroed_features": (
            list(SEQUENCE_FEATURES[2:-2])
            if input_feature_mode == "cgm_mask_clock" else []
        ),
        "action_dropout_applied": any(
            value > 0 for value in dropout_probabilities.values()
        ),
    }
    if revision:
        write_json(target_audit, run_dir / "training_target_audit.json")
        write_json(
            {
                "revision_contract": revision_contract,
                **input_contract, "targets": target_audit,
            },
            run_dir / "effective_training_contract.json",
        )
    context_samples = int(
        float(config["protocol"]["context_hours"])
        * 60
        / int(config["protocol"]["sample_minutes"])
    )
    source_to_index = {
        source: index
        for index, source in enumerate(
            sorted(sampled_train["source_id"].astype(str).unique())
        )
    }
    participant_store = ParticipantSequenceStore.from_frame(
        frame, input_feature_mode=input_feature_mode
    )
    dataset_class = _window_dataset_class(config)
    train_dataset = dataset_class(
        frame,
        sampled_train,
        scaler=scaler,
        context_samples=context_samples,
        source_to_index=source_to_index,
        participant_store=participant_store,
        copy_windows=False,
        input_feature_mode=input_feature_mode,
        revision_contract=revision_contract,
    )
    selection_dataset = dataset_class(
        frame,
        selection_windows,
        scaler=scaler,
        context_samples=context_samples,
        participant_store=participant_store,
        copy_windows=False,
        input_feature_mode=input_feature_mode,
        revision_contract=revision_contract,
    )
    calibration_dataset = dataset_class(
        frame,
        calibration_windows,
        scaler=scaler,
        context_samples=context_samples,
        participant_store=participant_store,
        copy_windows=False,
        input_feature_mode=input_feature_mode,
        revision_contract=revision_contract,
    )
    del (
        frame,
        train_windows,
        sampled_train,
        selection_windows,
        calibration_windows,
    )

    device = _device(config)
    if architecture == "glycoshift":
        model_config = GlycoShiftModelConfig.variant(
            str(config["model"]["variant"]),
            context_samples=context_samples,
            memory_k=int(config["model"]["memory_k"]),
            dropout=float(config["model"]["dropout"]),
        )
        model = GlycoShift(model_config).to(device)
        # Participant-memory ablation. `_fuse_memory` already falls back to the
        # learned population_memory when no retrieval is supplied, so clearing
        # this flag disables retrieval while leaving architecture, parameter
        # count and attention untouched — the only thing that changes is
        # whether memory is retrieved per participant or globally shared.
        #
        # Both the training loop and `_predict_dataset` gate on
        # `model.supports_memory`, so setting it here disables retrieval
        # consistently on both paths. Defaults to True, so existing configs
        # are unaffected.
        if not bool(config["model"].get("use_participant_memory", True)):
            model.supports_memory = False
    else:
        # C3. `attach_memory` is applied after the encoder and heads already
        # hold their weights, so a baseline built with memory is bitwise the
        # same baseline without it plus new parameters. That makes the removal
        # ablation exact rather than approximate. Defaults to 0, so every
        # existing config builds the model it always built.
        model = build_temporal_baseline(
            architecture,
            memory_k=int(config["model"].get("temporal_memory_k", 0)),
        ).to(device)
        model_config = model.config
    uses_participant_memory = bool(getattr(model, "supports_memory", False))
    batch_size = int(training["batch_size"])
    num_workers = int(training.get("num_workers", 0))
    effective_batch_size = int(training.get("effective_batch_size", batch_size))
    accumulation_steps = max(1, math.ceil(effective_batch_size / batch_size))
    loader = _loader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        source_balanced=True,
        sampler_seed=seed + _SAMPLER_SEED_OFFSET,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    maximum_epochs = int(training["maximum_epochs"])
    total_steps = max(1, math.ceil(len(loader) / accumulation_steps) * maximum_epochs)
    warmup_steps = max(1, int(round(total_steps * 0.05)))

    def lr_multiplier(step: int) -> float:
        if step < warmup_steps:
            return max(1e-8, step / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = LambdaLR(optimizer, lr_lambda=lr_multiplier)
    group_dro = GroupDRO.create(
        group_count=len(source_to_index),
        eta=float(training["group_dro_eta"]),
        device=device,
        # Defaults reproduce the original arm exactly, so existing configs and
        # every completed run are unaffected.
        rule=str(training.get("group_dro_rule", "cumulative")),
        temperature=float(training.get("group_dro_temperature", 1.0)),
    )
    positive_weights = positive_weights.to(device)
    # Local import avoids a module cycle: forecast_training reuses the generic
    # deep data and prediction helpers, while all temporal baselines must share
    # the same frozen operational checkpoint proxy.
    from .forecast_training import (
        _calibrate_operational_endpoint,
        _checkpoint_proxy,
    )
    if revision:
        _checkpoint_proxy = revision_checkpoint_proxy

    # The segment-sampled operational surrogate. Absent from every existing config,
    # so every completed run and every other arm is unaffected. When present it adds
    # a term whose estimand IS the frozen primary endpoint, sampled on contiguous
    # warmed segments rather than on complete participant trajectories, which
    # `FINDING-the-complete-stream-adapter-is-not-affordable.md` prices out of reach.
    surrogate_spec = training.get("segment_surrogate")
    segment_trainer = None
    segment_weight = 0.0
    segment_period = 0
    if surrogate_spec is not None:
        from .segment_surrogate import (
            SegmentPackSampler,
            SegmentSurrogateTrainer,
            build_participant_segment_index,
        )

        if uses_participant_memory:
            raise ValueError(
                "the segment surrogate scores segments without a memory bank; "
                "a memory-enabled arm would train and score through different paths"
            )
        segment_weight = float(surrogate_spec["weight"])
        if segment_weight <= 0:
            raise ValueError("segment_surrogate.weight must be positive")
        # The dataset's own frame, not the pre-dataset one: its index is a
        # RangeIndex, so the positions this returns ARE dataset indices and
        # `dataset[position]` addresses the right window by construction.
        segment_index = build_participant_segment_index(
            train_dataset.windows,
            source_order=source_order,
            sample_minutes=int(config["protocol"]["sample_minutes"]),
        )
        segment_sampler = SegmentPackSampler(
            segment_index,
            source_order=source_order,
            segment_decisions=int(surrogate_spec["segment_decisions"]),
            seed=int(surrogate_spec["pack_seed"]),
        )
        segment_trainer = SegmentSurrogateTrainer(
            dataset=train_dataset,
            sampler=segment_sampler,
            device=device,
            target_fapd=float(surrogate_spec.get("target_fapd", 0.5)),
            forward_chunk=int(surrogate_spec.get("forward_chunk", 1024)),
            # The same workers the main loader gets. Sequence construction is the
            # CPU-bound half of this package; a single-threaded surrogate fetch would
            # sit an order of magnitude behind the loop it rides on.
            num_workers=int(surrogate_spec.get("num_workers", num_workers)),
        )
        segment_period = max(
            1, len(loader) // max(1, int(surrogate_spec["updates_per_epoch"]))
        )
        write_json(
            {
                "objective": "segment_operational_surrogate",
                "weight": segment_weight,
                "segment_decisions": int(surrogate_spec["segment_decisions"]),
                "warmup_decisions": int(segment_sampler.warmup),
                "updates_per_epoch": int(surrogate_spec["updates_per_epoch"]),
                "batches_per_surrogate_update": int(segment_period),
                "num_workers": int(surrogate_spec.get("num_workers", num_workers)),
                "forward_chunk": int(surrogate_spec.get("forward_chunk", 1024)),
                "pack_seed": int(surrogate_spec["pack_seed"]),
                "target_fapd": float(surrogate_spec.get("target_fapd", 0.5)),
                "participants_indexed": len(segment_index),
                "sources": list(source_order),
            },
            run_dir / "segment_surrogate_spec.json",
        )
    segment_log: list[dict[str, Any]] = []

    best_key = (-float("inf"), -float("inf"), -float("inf"), -float("inf"))
    best_epoch = -1
    patience = 0
    update_step = 0
    first_loss_logged = False
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(maximum_epochs):
        bank = (
            _memory_bank(
                model,
                train_dataset,
                device=device,
                batch_size=batch_size,
                num_workers=num_workers,
                stride_minutes=int(config["model"]["memory_stride_minutes"]),
            )
            if bool(getattr(model, "supports_memory", False))
            and epoch >= int(training["warmup_memory_epochs"])
            else None
        )
        model.train()
        epoch_losses: list[float] = []
        for batch_index, batch in enumerate(loader):
            batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            sequence = _drop_action_groups(
                batch["sequence"],
                probabilities=tuple(
                    dropout_probabilities[field]
                    for field in ACTION_FIELDS
                ),
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                representation = model.encode_sequence(sequence)[:, -1]
                if bank is None:
                    memory = None
                    padding = None
                else:
                    query_metadata = [
                        train_dataset.metadata(int(index))
                        for index in batch["dataset_index"].tolist()
                    ]
                    memory, padding = bank.retrieve(
                        query_embeddings=representation.detach(),
                        query_metadata=query_metadata,
                    )
                outputs = model.forward_from_representation(
                    representation,
                    memory_embeddings=memory,
                    memory_padding_mask=padding,
                )
                per_example = _multitask_loss(
                    outputs,
                    batch,
                    positive_weights=positive_weights,
                    auxiliary_weight=float(training["auxiliary_regression_weight"]),
                    anchor_weight=anchor_weight,
                )
                loss, source_losses = group_dro.aggregate(
                    per_example,
                    batch["source_index"],
                )
                scaled_loss = loss / accumulation_steps
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite loss at epoch={epoch} batch={batch_index}"
                )
            if not first_loss_logged:
                first_loss_payload = {
                    "event": "first_finite_loss",
                    "epoch": epoch,
                    "batch": batch_index,
                    "loss": float(loss.detach().cpu()),
                }
                _append_log(log_path, first_loss_payload)
                print(json.dumps(first_loss_payload, sort_keys=True), flush=True)
                first_loss_logged = True
            scaled_loss.backward()
            epoch_losses.append(float(loss.detach().cpu()))
            if segment_trainer is not None and batch_index % segment_period == 0:
                # Backward separately so gradients accumulate into the same buffers
                # before the step, leaving the multitask path untouched. The
                # surrogate runs outside autocast: its inner loops are thousands of
                # tiny reductions where bf16 would cost accuracy and buy nothing.
                surrogate_result = segment_trainer.step(
                    model,
                    epoch=epoch,
                    update_index=batch_index // segment_period,
                )
                surrogate_term = (
                    segment_weight * surrogate_result.loss / accumulation_steps
                )
                if not torch.isfinite(surrogate_term):
                    raise FloatingPointError(
                        f"non-finite segment surrogate at epoch={epoch} "
                        f"batch={batch_index}"
                    )
                surrogate_term.backward()
                if len(segment_log) < 4096:
                    segment_log.append(
                        {
                            "epoch": int(epoch),
                            "batch": int(batch_index),
                            "surrogate_loss": float(
                                surrogate_result.loss.detach().cpu()
                            ),
                            "event_miss": float(
                                surrogate_result.event_miss.detach().cpu()
                            ),
                            "fapd": float(
                                surrogate_result.false_alarms_per_patient_day
                                .detach()
                                .cpu()
                            ),
                            "dual_value": surrogate_result.dual_value,
                            "scored_events": surrogate_result.scored_event_count,
                        }
                    )
            should_step = (
                (batch_index + 1) % accumulation_steps == 0
                or batch_index + 1 == len(loader)
            )
            if should_step:
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(training["gradient_clip_norm"]),
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                update_step += 1

        if segment_trainer is not None:
            assert isinstance(surrogate_spec, Mapping)
            _record_segment_epoch(
                segment_trainer=segment_trainer,
                model=model,
                epoch=epoch,
                scan_limit=int(surrogate_spec["updates_per_epoch"]),
                records=segment_log,
                run_dir=run_dir,
            )
        del bank, memory, padding
        selection_predictions = _predict_dataset(
            model,
            selection_dataset,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            memory_stride_minutes=int(config["model"]["memory_stride_minutes"]),
        )
        _attach_severe_scores(selection_predictions)
        operational_score, tie_fapd, proxy = _checkpoint_proxy(
            selection_predictions,
            source_order=source_order,
            candidate_count=selection_candidate_count,
            selection_rule=checkpoint_selection_rule,
            budget=checkpoint_budget,
        )
        window_score = _selection_score(
            selection_predictions, use_training_targets=revision
        )
        key = (operational_score, window_score, -tie_fapd, -float(epoch))
        epoch_payload = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)),
            "selection_operational_auc": operational_score,
            "selection_macro_window_auprc": window_score,
            "selection_proxy_tie_fapd": tie_fapd,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "group_weights": group_dro.weights.detach().cpu().tolist(),
            "last_batch_source_losses": source_losses,
            "update_step": update_step,
        }
        _append_log(log_path, epoch_payload)
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "model_config": model_config.__dict__,
                "architecture": architecture,
                "uses_participant_memory": uses_participant_memory,
                "selection_operational_score": operational_score,
                "selection_macro_window_auprc": window_score,
                "selection_proxy_tie_fapd": tie_fapd,
                "selection_operational_proxy": proxy,
            },
            checkpoint_dir / "last.pt",
        )
        if key > best_key:
            best_key = key
            best_epoch = epoch
            patience = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "model_config": model_config.__dict__,
                    "architecture": architecture,
                    "uses_participant_memory": uses_participant_memory,
                    "selection_operational_score": operational_score,
                    "selection_macro_window_auprc": window_score,
                    "selection_proxy_tie_fapd": tie_fapd,
                    "selection_operational_proxy": proxy,
                },
                checkpoint_dir / "best.pt",
            )
        else:
            patience += 1
            if patience >= int(training["early_stopping_patience"]):
                break

    checkpoint = torch.load(checkpoint_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    selection_predictions = _predict_dataset(
        model,
        selection_dataset,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        memory_stride_minutes=int(config["model"]["memory_stride_minutes"]),
    )
    calibration_predictions = _predict_dataset(
        model,
        calibration_dataset,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        memory_stride_minutes=int(config["model"]["memory_stride_minutes"]),
    )
    for current_predictions in (
        selection_predictions,
        calibration_predictions,
    ):
        _attach_severe_scores(current_predictions)
        _restore_regression_scale(current_predictions, scaler)
    if revision:
        task_summary, operational = revision_calibrator(
            selection_predictions, calibration_predictions,
            config=config, source_order=source_order,
        )
    else:
        task_summary: dict[str, Any] = {}
        for task in TASKS:
            kind, horizon_text = task.split("_")
            horizon = int(horizon_text)
            eligible = calibration_predictions.loc[
                ~calibration_predictions[f"{kind}_currently_active"]
            ].copy()
            temperature = fit_temperature(
                eligible[f"{task}_label"].to_numpy(dtype=int),
                eligible[f"{task}_logit"].to_numpy(dtype=float),
            )
            for predictions in (selection_predictions, calibration_predictions):
                predictions[f"{task}_probability"] = apply_temperature(
                    predictions[f"{task}_logit"].to_numpy(dtype=float),
                    temperature,
                )
            conformal_threshold = fit_conformal_threshold(
                eligible[f"{task}_label"].to_numpy(dtype=int),
                apply_temperature(
                    eligible[f"{task}_logit"].to_numpy(dtype=float),
                    temperature,
                ),
                alpha=0.1,
            )
            thresholds: dict[str, float] = {}
            for target in config["evaluation"]["false_alarms_per_patient_day"]:
                thresholds[str(target)] = select_alarm_threshold(
                    calibration_predictions.loc[
                        ~calibration_predictions[f"{kind}_currently_active"]
                    ],
                    probability_column=f"{task}_probability",
                    label_column=f"{task}_label",
                    event_id_column=f"{task}_event_id",
                    event_onset_column=f"{task}_event_onset",
                    horizon_minutes=horizon,
                    target_false_alarms_per_day=float(target),
                    candidate_count=int(
                        config["evaluation"].get("threshold_candidate_count", 2001)
                    ),
                    workers=int(
                        config["evaluation"].get("alarm_sweep_workers", 1)
                    ),
                )
            task_summary[task] = {
                "temperature": temperature,
                "conformal_nonconformity_threshold": conformal_threshold,
                "threshold_partition": "calibration",
                "thresholds_from_calibration": thresholds,
                "thresholds_from_validation": thresholds,
            }

        operational = _calibrate_operational_endpoint(
            selection_predictions,
            calibration_predictions,
            config=config,
            source_order=source_order,
        )

    predictions = pd.concat(
        (
            selection_predictions.assign(partition="selection"),
            calibration_predictions.assign(partition="calibration"),
        ),
        ignore_index=True,
    )
    predictions.to_parquet(run_dir / "predictions.parquet", index=False)
    model_summary = {
        "status": (
            "synthetic_architecture_smoke"
            if bool(config["project"].get("synthetic", False))
            else "development_revision" if revision
            else "development_only_external_untouched"
        ),
        "variant": config["model"]["variant"],
        "architecture": architecture,
        "parameters": count_parameters(model),
        "best_epoch": best_epoch,
        "best_selection_operational_auc": float(best_key[0]),
        "best_selection_macro_window_auprc": float(best_key[1]),
        "training_seed": seed,
        "split_seed": int(config["split"]["seed"]),
        "tasks": task_summary,
        "revision_contract": revision_contract,
        "training_targets": target_audit,
        **input_contract,
    }
    write_json(model_summary, run_dir / "model_summary.json")
    write_json(operational, run_dir / "operational_metrics.json")
    write_json(
        checkpoint["selection_operational_proxy"],
        run_dir / "selection_operational_proxy.json",
    )
    reporter = revision_reporter if revision else evaluate_predictions
    metrics = reporter(
        predictions,
        config=config,
        model_summary=model_summary,
    )
    metrics["operational_fixed_frame" if revision else "operational_v1_3"] = operational
    metrics["legacy_pooled_evaluator_reportable"] = False
    write_json(metrics, run_dir / "metrics.json")
    write_json(
        {
            "device": str(device),
            "training_seed": seed,
            "split_seed": int(config["split"]["seed"]),
            "parameters": count_parameters(model),
            "best_epoch": best_epoch,
            "best_operational_score": float(best_key[0]),
            "best_selection_macro_window_auprc": float(best_key[1]),
            **window_counts,
            "feature_group_dropout_probabilities": dropout_probabilities,
            "external_evaluated": False,
            "revision_contract": revision_contract,
            "training_targets": target_audit,
            **input_contract,
        },
        run_dir / "training_summary.json",
    )
    return run_dir


def _load_frozen_model(
    run_dir: Path,
    *,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint_path = run_dir / "ckpts" / "best.pt"
    summary_path = run_dir / "model_summary.json"
    if not checkpoint_path.exists() or not summary_path.exists():
        raise FileNotFoundError("run is missing a frozen model or model summary")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    architecture = str(checkpoint.get("architecture", summary.get("architecture", "")))
    if architecture == "glycoshift":
        model = GlycoShift(
            GlycoShiftModelConfig(**checkpoint["model_config"])
        )
        # Restore the ablation setting from the checkpoint rather than the
        # config. A memory-ablated run must not silently regain retrieval at
        # evaluation time, and binding it to the weights makes that impossible
        # even if evaluated with a different config. Absent key means an older
        # checkpoint, which always used retrieval.
        model.supports_memory = bool(
            checkpoint.get("uses_participant_memory", True)
        )
    elif architecture in {"lstm", "tcn", "patchtst"}:
        model = build_temporal_baseline(architecture)
    else:
        raise ValueError(f"unsupported frozen architecture: {architecture}")
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval(), summary


def _external_gate(
    windows: pd.DataFrame,
    *,
    minimum_participants: int,
    minimum_events_per_class: int,
) -> dict[str, int]:
    participants = int(
        windows[["source_id", "participant_id"]].drop_duplicates().shape[0]
    )
    event_counts = {
        kind: int(windows[f"{kind}_60_event_id"].dropna().nunique())
        for kind in ("hypo", "hyper")
    }
    if participants < minimum_participants:
        raise ValueError(
            f"external gate failed: {participants} participants < "
            f"{minimum_participants}"
        )
    for kind, count in event_counts.items():
        if count < minimum_events_per_class:
            raise ValueError(
                f"external gate failed: {kind} events {count} < "
                f"{minimum_events_per_class}"
            )
    return {
        "participants": participants,
        "hypo_events": event_counts["hypo"],
        "hyper_events": event_counts["hyper"],
    }


def run_frozen_evaluation(
    config: dict[str, Any],
    *,
    run_dir: str | Path,
    partition: str,
) -> Path:
    if config["training"].get("revision_contract") is not None:
        raise ValueError("revision evaluation requires the fixed-frame runner")
    if partition not in {"external_loop", "stress_hupa_ucm"}:
        raise ValueError(
            "frozen evaluation is limited to external_loop or "
            "stress_hupa_ucm"
        )
    output_dir = Path(run_dir).expanduser().resolve()
    predictions_path = output_dir / f"predictions_{partition}.parquet"
    metrics_path = output_dir / f"metrics_{partition}.json"
    if predictions_path.exists() or metrics_path.exists():
        raise FileExistsError(
            f"refusing to repeat or overwrite frozen evaluation: {partition}"
        )
    frame, partition_windows, audit = prepare_evaluation_dataset(
        config,
        partition=partition,
    )
    if partition_windows.empty:
        raise ValueError(f"no eligible windows for {partition}")
    if partition == "external_loop":
        gate = _external_gate(
            partition_windows,
            minimum_participants=int(
                config["evaluation"]["external_minimum_participants"]
            ),
            minimum_events_per_class=int(
                config["evaluation"]["external_minimum_events_per_class"]
            ),
        )
    else:
        gate = {
            "participants": int(
                partition_windows[
                    ["source_id", "participant_id"]
                ].drop_duplicates().shape[0]
            ),
            "hypo_events": int(
                partition_windows["hypo_60_event_id"].dropna().nunique()
            ),
            "hyper_events": int(
                partition_windows["hyper_60_event_id"].dropna().nunique()
            ),
        }
    with (output_dir / "scaler.json").open("r", encoding="utf-8") as handle:
        scaler = SequenceScaler.from_dict(json.load(handle))
    context_samples = int(
        float(config["protocol"]["context_hours"])
        * 60
        / int(config["protocol"]["sample_minutes"])
    )
    device = _device(config)
    model, summary = _load_frozen_model(output_dir, device=device)
    trained_mode = str(summary.get("input_feature_mode", "multimodal"))
    requested_mode = str(config["model"].get("input_feature_mode", trained_mode))
    if requested_mode != trained_mode:
        raise ValueError("frozen evaluation input_feature_mode differs from training")
    if summary.get("revision_contract") is not None:
        raise ValueError("revision evaluation requires the fixed-frame runner")
    dataset = WindowSequenceDataset(
        frame,
        partition_windows,
        scaler=scaler,
        context_samples=context_samples,
        input_feature_mode=trained_mode,
    )
    predictions = _predict_dataset(
        model,
        dataset,
        device=device,
        batch_size=int(config["training"]["batch_size"]),
        num_workers=int(config["training"].get("num_workers", 0)),
        memory_stride_minutes=int(config["model"]["memory_stride_minutes"]),
    )
    _restore_regression_scale(predictions, scaler)
    for task in TASKS:
        temperature = float(summary["tasks"][task]["temperature"])
        predictions[f"{task}_probability"] = apply_temperature(
            predictions[f"{task}_logit"].to_numpy(dtype=float),
            temperature,
        )
    predictions["partition"] = partition
    predictions.to_parquet(predictions_path, index=False)
    metrics = evaluate_predictions(
        predictions,
        config=config,
        model_summary=summary,
    )
    metrics["external_gate"] = gate
    metrics["cohort_audit"] = audit
    write_json(metrics, metrics_path)
    return metrics_path
