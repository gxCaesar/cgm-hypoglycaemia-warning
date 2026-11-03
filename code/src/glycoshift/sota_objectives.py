from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import torch

from .sequences import TASKS


def tls_exp_target(
    time_to_event_minutes: np.ndarray | list[float] | float,
    *,
    horizon_minutes: int,
    gamma_per_hour: float,
) -> np.ndarray:
    """Return exponential temporal-label-smoothing targets.

    The ICML 2023 TLS paper defines support ``(h_min, h_max) = (0, 2h)``.
    Its official implementation divides step distances by steps-per-hour
    before applying ``gamma``, so the rate here is explicitly per hour.
    """

    horizon = int(horizon_minutes)
    gamma = float(gamma_per_hour)
    if horizon <= 0:
        raise ValueError("TLS horizon_minutes must be positive")
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("TLS gamma_per_hour must be finite and positive")

    distance = np.asarray(time_to_event_minutes, dtype=float)
    distance_hours = distance / 60
    horizon_hours = horizon / 60
    target = np.zeros_like(distance_hours, dtype=float)
    target[distance_hours <= 0] = 1.0
    interior = (
        np.isfinite(distance_hours)
        & (distance_hours > 0)
        & (distance_hours < 2 * horizon_hours)
    )
    denominator = -np.expm1(-2 * gamma * horizon_hours)
    target[interior] = (
        np.exp(-gamma * distance_hours[interior])
        - np.exp(-2 * gamma * horizon_hours)
    ) / denominator
    return np.clip(target, 0.0, 1.0)


def attach_tls_exp_targets(
    windows: pd.DataFrame,
    *,
    sample_minutes: int,
    gamma_per_hour: float,
    copy: bool = True,
) -> pd.DataFrame:
    """Attach training-only TLS targets without changing hard event labels.

    Event onsets are reconstructed from the complete, unsampled training
    window table. This must run before negative-window subsampling because TLS
    has support out to twice the task horizon.
    """

    if windows.empty:
        raise ValueError("TLS requires a non-empty window table")
    if sample_minutes <= 0:
        raise ValueError("sample_minutes must be positive")
    required = {"source_id", "participant_id", "context_end"}
    missing = required.difference(windows.columns)
    if missing:
        raise ValueError(f"TLS windows are missing columns: {sorted(missing)}")

    output = windows.copy() if copy else windows
    query_times = pd.to_datetime(
        output["context_end"],
        utc=True,
        errors="coerce",
    )
    if query_times.isna().any():
        raise ValueError("TLS context_end contains unparseable timestamps")

    groups = output.groupby(
        ["source_id", "participant_id"],
        sort=False,
    ).indices
    for task in TASKS:
        kind, horizon_text = task.rsplit("_", 1)
        horizon = int(horizon_text)
        onset_columns = [
            column
            for column in output.columns
            if column.startswith(f"{kind}_")
            and column.endswith("_event_onset")
            and column[len(kind) + 1 : -len("_event_onset")].isdigit()
        ]
        if not onset_columns:
            raise ValueError(f"TLS cannot reconstruct {kind} event onsets")

        distances = np.full(len(output), np.inf, dtype=float)
        for positions in groups.values():
            position_array = np.asarray(positions, dtype=np.int64)
            participant = output.iloc[position_array]
            onsets = pd.concat(
                [
                    pd.to_datetime(
                        participant[column],
                        utc=True,
                        errors="coerce",
                    )
                    for column in onset_columns
                ],
                ignore_index=True,
            ).dropna()
            if onsets.empty:
                continue
            event_ns = np.sort(onsets.astype("int64").unique())
            query_ns = query_times.iloc[position_array].astype("int64").to_numpy()
            next_indices = np.searchsorted(event_ns, query_ns, side="right")
            has_future = next_indices < len(event_ns)
            if not has_future.any():
                continue
            selected_positions = position_array[has_future]
            delta_ns = (
                event_ns[next_indices[has_future]] - query_ns[has_future]
            )
            delta_minutes = delta_ns.astype(float) / (60 * 1_000_000_000)
            nearest_grid = np.rint(delta_minutes / sample_minutes)
            if not np.allclose(
                delta_minutes,
                nearest_grid * sample_minutes,
                atol=1e-8,
                rtol=0,
            ):
                raise ValueError("TLS event/query times do not share the sample grid")
            distances[selected_positions] = delta_minutes

        output[f"{task}_training_target"] = tls_exp_target(
            distances,
            horizon_minutes=horizon,
            gamma_per_hour=gamma_per_hour,
        ).astype(np.float32)
    return output


def summarize_tls_targets(windows: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {"training_window_count": int(len(windows)), "tasks": {}}
    for task in TASKS:
        target_column = f"{task}_training_target"
        if target_column not in windows:
            raise ValueError(f"missing TLS target column: {target_column}")
        targets = windows[target_column].to_numpy(dtype=float)
        if not np.isfinite(targets).all() or np.any((targets < 0) | (targets > 1)):
            raise ValueError(f"invalid TLS targets for {task}")
        summary["tasks"][task] = {
            "hard_positive_count": int(windows[f"{task}_label"].sum()),
            "soft_positive_count": int(np.count_nonzero(targets > 0)),
            "target_mean": float(targets.mean()),
            "target_min": float(targets.min()),
            "target_max": float(targets.max()),
        }
    return summary


def _validate_sha256(value: str, *, field: str) -> str:
    normalized = str(value).lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return normalized


@dataclass(frozen=True)
class FrozenSourceRiskReference:
    """Provenance-complete training-only reference for a MERO-style arm."""

    METHOD: ClassVar[str] = "participant-cross-fitted-mero-style-v1"
    SPLIT: ClassVar[str] = "train"

    source_order: tuple[str, ...]
    reference_risks: tuple[float, ...]
    fold_count: int
    reference_seed: int
    data_sampling_seed: int
    input_manifest_sha256: str
    window_manifest_sha256: str
    training_config_sha256: str

    def __post_init__(self) -> None:
        if not self.source_order or len(set(self.source_order)) != len(
            self.source_order
        ):
            raise ValueError("source_order must be non-empty and unique")
        if len(self.reference_risks) != len(self.source_order):
            raise ValueError("one reference risk is required per source")
        risks = np.asarray(self.reference_risks, dtype=float)
        if not np.isfinite(risks).all() or np.any(risks < 0):
            raise ValueError("reference risks must be finite and non-negative")
        if self.fold_count < 2:
            raise ValueError("cross-fitted references require at least two folds")
        if self.reference_seed < 0 or self.data_sampling_seed < 0:
            raise ValueError("reference and data-sampling seeds must be non-negative")
        for field in (
            "input_manifest_sha256",
            "window_manifest_sha256",
            "training_config_sha256",
        ):
            object.__setattr__(
                self,
                field,
                _validate_sha256(getattr(self, field), field=field),
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.METHOD,
            "split": self.SPLIT,
            "source_order": list(self.source_order),
            "reference_risks": list(self.reference_risks),
            "fold_count": self.fold_count,
            "reference_seed": self.reference_seed,
            "data_sampling_seed": self.data_sampling_seed,
            "input_manifest_sha256": self.input_manifest_sha256,
            "window_manifest_sha256": self.window_manifest_sha256,
            "training_config_sha256": self.training_config_sha256,
        }

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.as_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FrozenSourceRiskReference:
        if payload.get("method") != cls.METHOD or payload.get("split") != cls.SPLIT:
            raise ValueError("reference artifact method or split is incompatible")
        return cls(
            source_order=tuple(map(str, payload["source_order"])),
            reference_risks=tuple(map(float, payload["reference_risks"])),
            fold_count=int(payload["fold_count"]),
            reference_seed=int(payload["reference_seed"]),
            data_sampling_seed=int(payload["data_sampling_seed"]),
            input_manifest_sha256=str(payload["input_manifest_sha256"]),
            window_manifest_sha256=str(payload["window_manifest_sha256"]),
            training_config_sha256=str(payload["training_config_sha256"]),
        )

    @classmethod
    def read(cls, path: str | Path) -> FrozenSourceRiskReference:
        with Path(path).open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("reference artifact root must be a mapping")
        return cls.from_dict(payload)

    def write(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


class MinimaxExcessRisk:
    """Exact full-group empirical/cross-fitted MERO-style aggregation.

    This is deliberately not named MMR: it implements the empirical excess
    risk objective, not the convex per-iteration QP algorithm in the MMR paper.
    """

    def __init__(
        self,
        reference_risks: torch.Tensor,
        *,
        source_order: tuple[str, ...],
        require_all_groups: bool = True,
    ) -> None:
        references = reference_risks.detach().clone().to(dtype=torch.float32)
        if references.ndim != 1 or len(references) == 0:
            raise ValueError("reference_risks must be a non-empty vector")
        if not torch.isfinite(references).all() or (references < 0).any():
            raise ValueError("reference_risks must be finite and non-negative")
        if len(source_order) != len(references) or len(set(source_order)) != len(
            source_order
        ):
            raise ValueError("source_order must uniquely match reference_risks")
        self.reference_risks = references
        self.source_order = tuple(source_order)
        self.require_all_groups = bool(require_all_groups)

    @classmethod
    def from_reference(
        cls,
        reference: FrozenSourceRiskReference,
        *,
        require_all_groups: bool = True,
    ) -> MinimaxExcessRisk:
        return cls(
            torch.tensor(reference.reference_risks, dtype=torch.float32),
            source_order=reference.source_order,
            require_all_groups=require_all_groups,
        )

    def aggregate(
        self,
        losses: torch.Tensor,
        groups: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        dict[int, float],
        dict[int, float],
        torch.Tensor,
    ]:
        if losses.ndim != 1 or groups.ndim != 1 or losses.shape != groups.shape:
            raise ValueError("losses and groups must be equal-length vectors")
        if groups.dtype != torch.long:
            raise ValueError("groups must use torch.long indices")
        if len(losses) == 0 or not torch.isfinite(losses).all():
            raise ValueError("losses must be non-empty and finite")
        group_count = len(self.reference_risks)
        if (groups < 0).any() or (groups >= group_count).any():
            raise ValueError("group index is outside the frozen source order")

        present = [
            group for group in range(group_count) if bool((groups == group).any())
        ]
        if self.require_all_groups and len(present) != group_count:
            missing = sorted(set(range(group_count)).difference(present))
            raise ValueError(
                "exact MERO aggregation requires every source group; "
                f"missing={missing}"
            )
        if not present:
            raise ValueError("MERO batch has no source groups")

        raw = torch.stack([losses[groups == group].mean() for group in present])
        reference = self.reference_risks.to(device=losses.device, dtype=losses.dtype)
        indices = torch.tensor(present, device=losses.device, dtype=torch.long)
        excess = raw - reference[indices]
        objective = torch.amax(excess)

        with torch.no_grad():
            active = torch.isclose(excess, objective.detach()).to(losses.dtype)
            active /= active.sum()
            weights = torch.zeros(
                group_count,
                device=losses.device,
                dtype=losses.dtype,
            )
            weights[indices] = active
        raw_observed = {
            group: float(value.detach().cpu())
            for group, value in zip(present, raw, strict=True)
        }
        excess_observed = {
            group: float(value.detach().cpu())
            for group, value in zip(present, excess, strict=True)
        }
        return objective, raw_observed, excess_observed, weights
