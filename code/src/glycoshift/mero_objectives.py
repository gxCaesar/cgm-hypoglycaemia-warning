from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Literal

import torch

ObjectiveMode = Literal["all_source_erm", "hardmax_zero", "mero"]


@dataclass(frozen=True)
class EffectiveObjectiveResult:
    objective: torch.Tensor
    raw_source_risks: torch.Tensor
    reference_risks: torch.Tensor
    excess_source_risks: torch.Tensor
    active_source_weights: torch.Tensor
    source_counts: torch.Tensor
    source_sampling_weight_sums: torch.Tensor
    record_count: int
    sampling_weight_sum: float


class AllSourceLossAccumulator:
    """Aggregate one effective update before applying an all-source objective.

    The accumulator retains the differentiable per-record loss graph across
    physical batches. It deliberately forms the source means and objective
    exactly once; averaging independent physical-batch maxima is not exposed.
    """

    MODES = frozenset({"all_source_erm", "hardmax_zero", "mero"})

    def __init__(
        self,
        *,
        source_order: tuple[str, ...],
        mode: ObjectiveMode,
        reference_risks: torch.Tensor | None,
        expected_records: int | None = None,
    ) -> None:
        if any(not isinstance(source, str) for source in source_order):
            raise TypeError("source_order entries must be strings")
        if not source_order or len(set(source_order)) != len(source_order):
            raise ValueError("source_order must be non-empty and unique")
        if source_order != tuple(sorted(source_order)):
            raise ValueError("source_order must use lexical order")
        if mode not in self.MODES:
            raise ValueError(f"unsupported MERO objective mode: {mode}")
        if expected_records is not None:
            if not isinstance(expected_records, Integral) or isinstance(
                expected_records,
                bool,
            ):
                raise TypeError("expected_records must be an integer")
            if expected_records <= 0:
                raise ValueError("expected_records must be positive when supplied")

        source_count = len(source_order)
        if mode == "all_source_erm":
            if reference_risks is not None:
                raise ValueError("all_source_erm must not receive a reference")
            references = torch.zeros(source_count, dtype=torch.float32)
        else:
            if reference_risks is None:
                raise ValueError(f"{mode} requires an explicit reference vector")
            references = reference_risks.detach().clone().to(dtype=torch.float32)
            if references.ndim != 1 or len(references) != source_count:
                raise ValueError("reference vector must match source_order")
            if not bool(torch.isfinite(references).all()) or bool(
                (references < 0).any()
            ):
                raise ValueError("reference risks must be finite and non-negative")
            if mode == "hardmax_zero" and bool((references != 0).any()):
                raise ValueError("hardmax_zero requires an explicit zero vector")

        self.source_order = source_order
        self.mode: ObjectiveMode = mode
        self.reference_risks = references
        self.expected_records = (
            None if expected_records is None else int(expected_records)
        )
        self._loss_chunks: list[torch.Tensor] = []
        self._group_chunks: list[torch.Tensor] = []
        self._sampling_weight_chunks: list[torch.Tensor] = []
        self._device: torch.device | None = None
        self._finalized = False

    def add(
        self,
        losses: torch.Tensor,
        groups: torch.Tensor,
        *,
        sampling_weights: torch.Tensor,
    ) -> None:
        if self._finalized:
            raise RuntimeError("cannot add losses after finalizing the update")
        if (
            losses.ndim != 1
            or groups.ndim != 1
            or sampling_weights.ndim != 1
            or losses.shape != groups.shape
            or losses.shape != sampling_weights.shape
        ):
            raise ValueError(
                "losses, groups, and sampling_weights must be equal-length vectors"
            )
        if len(losses) == 0:
            raise ValueError("physical-batch losses must be non-empty")
        if not losses.is_floating_point() or not bool(torch.isfinite(losses).all()):
            raise ValueError("losses must be finite floating-point values")
        if groups.dtype != torch.long:
            raise ValueError("groups must use torch.long indices")
        if (
            groups.device != losses.device
            or sampling_weights.device != losses.device
        ):
            raise ValueError(
                "losses, groups, and sampling_weights must share a device"
            )
        if (
            not sampling_weights.is_floating_point()
            or not bool(torch.isfinite(sampling_weights).all())
            or bool((sampling_weights <= 0).any())
        ):
            raise ValueError(
                "sampling_weights must be finite positive floating-point values"
            )
        source_count = len(self.source_order)
        if bool((groups < 0).any()) or bool((groups >= source_count).any()):
            raise ValueError("group index is outside the frozen source order")
        if self._device is None:
            self._device = losses.device
        elif losses.device != self._device:
            raise ValueError("all physical batches must use the same device")
        self._loss_chunks.append(losses)
        self._group_chunks.append(groups)
        self._sampling_weight_chunks.append(sampling_weights)

    def finalize(self) -> EffectiveObjectiveResult:
        if self._finalized:
            raise RuntimeError("effective update has already been finalized")
        if not self._loss_chunks:
            raise ValueError("effective update contains no physical batches")
        self._finalized = True

        losses = torch.cat(self._loss_chunks).to(dtype=torch.float32)
        groups = torch.cat(self._group_chunks)
        sampling_weights = torch.cat(self._sampling_weight_chunks).to(
            dtype=torch.float32
        )
        record_count = int(losses.numel())
        if self.expected_records is not None and record_count != self.expected_records:
            raise ValueError(
                "effective update record count mismatch: "
                f"expected={self.expected_records}, observed={record_count}"
            )

        source_count = len(self.source_order)
        source_sums = torch.zeros(
            source_count,
            device=losses.device,
            dtype=torch.float32,
        ).scatter_add(0, groups, losses * sampling_weights)
        source_sampling_weight_sums = torch.zeros(
            source_count,
            device=losses.device,
            dtype=torch.float32,
        ).scatter_add(0, groups, sampling_weights)
        counts = torch.bincount(groups, minlength=source_count)
        if bool((counts == 0).any()):
            missing = torch.nonzero(counts == 0, as_tuple=False).flatten().tolist()
            raise ValueError(
                f"effective update requires every source group; missing={missing}"
            )
        if not bool(torch.isfinite(source_sampling_weight_sums).all()) or bool(
            (source_sampling_weight_sums <= 0).any()
        ):
            raise ValueError(
                "every source requires a finite positive sampling-weight denominator"
            )
        raw_risks = source_sums / source_sampling_weight_sums
        references = self.reference_risks.to(device=losses.device).detach()
        excess = raw_risks - references

        if self.mode == "all_source_erm":
            objective = raw_risks.mean()
            weights = torch.full_like(raw_risks, 1.0 / source_count)
        else:
            objective, active_index = torch.max(excess, dim=0)
            weights = torch.zeros_like(raw_risks)
            weights[active_index] = 1.0

        return EffectiveObjectiveResult(
            objective=objective,
            raw_source_risks=raw_risks,
            reference_risks=references,
            excess_source_risks=excess,
            active_source_weights=weights,
            source_counts=counts,
            source_sampling_weight_sums=source_sampling_weight_sums,
            record_count=record_count,
            sampling_weight_sum=float(sampling_weights.sum().detach().cpu()),
        )
