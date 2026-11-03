"""Train a member through the composition it is deployed in.

Every arm this project has trained optimised the member's own score, while the
deployed scorer is a composition of that member with a current-glucose
threshold. Those are measurably different objectives here: swapping a member
that gains ``+4.92 sigma`` alone into the combination costs ``-6.50 sigma``
(``RESULT-a-better-member-makes-the-combination-worse.md``), because the better
member learned to rank severe events directly, which is what the threshold
already does.

This module supplies the fixed anchor that lets the training loss see the
composition. It is deliberately small and does two things: fit a monotone
current-glucose anchor on the training partition, and attach its value to the
training windows. Nothing here runs at inference -- the anchor column is
attached to the training frame only, exactly as ``attach_lead_time_loss_weights``
is, so the selection and calibration datasets never carry it.

The pre-registration is
``experiments/20260817-glycoshift-v1-5-baseline-closure-v1/PREREG-anchor-aware-member-training.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "ANCHOR_COLUMN",
    "GlucoseAnchor",
    "attach_anchor_values",
    "fit_glucose_anchor",
    "summarize_anchor",
]

ANCHOR_COLUMN = "anchor_value"
ANCHOR_SOURCE_COLUMN = "cgm_last"

# The tasks whose logits carry the endpoint's severe score. `hyper_*` are left
# alone: the anchor is a statement about hypoglycaemia risk and composing it
# into a hyperglycaemia head would be a different claim with no evidence behind
# it.
ANCHORED_TASKS = ("hypo_30", "hypo_60")


@dataclass(frozen=True)
class GlucoseAnchor:
    """A monotone non-increasing map from current glucose into ``(0, 1)``.

    The deployed combination ranks each member through its calibration ECDF and
    averages. This is the same construction fitted on the partition a training
    loss is allowed to see, so the training-time anchor and the deployed
    threshold member are the same function of the same column, differing only in
    which partition supplied the ECDF.

    ``grid`` is ascending glucose and ``values`` is the matching anchor, so
    ``np.interp`` reads it directly. Values are ``(rank + 0.5) / n`` rather than
    ``rank / n`` so the anchor is strictly inside ``(0, 1)`` and the composed
    probability cannot be pinned at a boundary before clamping.
    """

    grid: np.ndarray
    values: np.ndarray
    fitted_rows: int
    partition: str

    def __post_init__(self) -> None:
        if self.grid.ndim != 1 or self.values.ndim != 1:
            raise ValueError("anchor grid and values must be one-dimensional")
        if self.grid.size != self.values.size:
            raise ValueError("anchor grid and values must have equal length")
        if self.grid.size < 2:
            raise ValueError("anchor needs at least two distinct glucose values")
        if not np.all(np.diff(self.grid) > 0):
            raise ValueError("anchor grid must be strictly ascending in glucose")
        if not np.all(np.diff(self.values) <= 0):
            raise ValueError("anchor must be non-increasing in glucose")
        if not (np.all(self.values > 0.0) and np.all(self.values < 1.0)):
            raise ValueError("anchor values must lie strictly inside (0, 1)")

    def evaluate(self, glucose: np.ndarray) -> np.ndarray:
        """Return the anchor at ``glucose``, clamped to the fitted range.

        ``np.interp`` holds the end values outside the grid, which is the
        intended behaviour: a glucose below anything seen in training is at
        least as dangerous as the lowest fitted point.
        """

        values = np.asarray(glucose, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("anchor cannot be evaluated at non-finite glucose")
        return np.interp(values, self.grid, self.values)

    def as_dict(self) -> dict[str, object]:
        return {
            "grid": [float(value) for value in self.grid],
            "values": [float(value) for value in self.values],
            "fitted_rows": int(self.fitted_rows),
            "partition": str(self.partition),
        }


def fit_glucose_anchor(
    windows: pd.DataFrame,
    *,
    maximum_grid_points: int = 512,
) -> GlucoseAnchor:
    """Fit the anchor on ``windows``, which must be a single partition.

    The partition check is the gate: a training-time anchor fitted on
    calibration or selection rows would put held-out data inside the loss, and
    that is exactly the failure this refuses rather than documents.
    """

    if ANCHOR_SOURCE_COLUMN not in windows:
        raise KeyError(
            f"anchor needs the {ANCHOR_SOURCE_COLUMN!r} column; the arm that "
            "lacks it cannot be anchored"
        )
    if "partition" in windows:
        partitions = set(windows["partition"].astype(str).unique())
        if partitions != {"train"}:
            raise ValueError(
                "the training anchor must be fitted on train rows alone, got "
                f"{sorted(partitions)}"
            )
    glucose = pd.to_numeric(windows[ANCHOR_SOURCE_COLUMN], errors="coerce")
    observed = np.asarray(glucose.dropna(), dtype=np.float64)
    if observed.size < 2:
        raise ValueError("anchor needs at least two observed current-glucose values")

    ordered = np.sort(observed)
    total = float(ordered.size)
    # Probability that a training window sits at or above this glucose, i.e. the
    # ECDF of -cgm_last, shifted by half a rank so the range is open.
    quantiles = np.linspace(0.0, 1.0, min(maximum_grid_points, ordered.size))
    grid = np.quantile(ordered, quantiles)
    grid = np.unique(grid)
    if grid.size < 2:
        raise ValueError("anchor grid collapsed to a single glucose value")
    below = np.searchsorted(ordered, grid, side="left").astype(np.float64)
    values = (total - below + 0.5) / (total + 1.0)
    # `searchsorted` is non-decreasing in glucose, so `values` is non-increasing
    # by construction; assert it rather than assume it.
    values = np.minimum.accumulate(values)
    return GlucoseAnchor(
        grid=grid,
        values=values,
        fitted_rows=int(observed.size),
        partition="train",
    )


def attach_anchor_values(
    windows: pd.DataFrame,
    anchor: GlucoseAnchor,
    *,
    copy: bool = True,
) -> pd.DataFrame:
    """Attach the anchor column, mirroring ``attach_lead_time_loss_weights``."""

    frame = windows.copy() if copy else windows
    glucose = pd.to_numeric(frame[ANCHOR_SOURCE_COLUMN], errors="coerce")
    if not np.isfinite(np.asarray(glucose, dtype=np.float64)).all():
        raise ValueError(
            "every training window needs an observed current glucose to be anchored"
        )
    frame[ANCHOR_COLUMN] = anchor.evaluate(
        np.asarray(glucose, dtype=np.float64)
    ).astype(np.float64)
    return frame


def summarize_anchor(
    windows: pd.DataFrame,
    anchor: GlucoseAnchor,
    *,
    anchor_weight: float,
) -> dict[str, object]:
    """Return the audit record written beside the run."""

    values = np.asarray(windows[ANCHOR_COLUMN], dtype=np.float64)
    glucose = np.asarray(
        pd.to_numeric(windows[ANCHOR_SOURCE_COLUMN], errors="coerce"),
        dtype=np.float64,
    )
    return {
        "objective": "anchor_aware",
        "anchor_weight": float(anchor_weight),
        "anchored_tasks": list(ANCHORED_TASKS),
        "anchor_source_column": ANCHOR_SOURCE_COLUMN,
        "fitted_partition": anchor.partition,
        "fitted_rows": int(anchor.fitted_rows),
        "grid_points": int(anchor.grid.size),
        "anchored_windows": int(values.size),
        "anchor_minimum": float(values.min()),
        "anchor_maximum": float(values.max()),
        "anchor_mean": float(values.mean()),
        "glucose_minimum": float(glucose.min()),
        "glucose_maximum": float(glucose.max()),
        "anchor_at_54_mg_dl": float(anchor.evaluate(np.array([54.0]))[0]),
        "anchor_at_61_mg_dl": float(anchor.evaluate(np.array([61.0]))[0]),
        "anchor_at_70_mg_dl": float(anchor.evaluate(np.array([70.0]))[0]),
    }
