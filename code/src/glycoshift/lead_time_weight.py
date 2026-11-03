"""Weight each positive window's event loss by the warning time it delivers.

Training treats every pre-event window as equally valuable, and the measurements
in the v1.5 bundle say the resulting models warn late: the arm that wins the
frozen endpoint gives a median warning of five minutes, the leading deep arm
fifteen, and the endpoint scores whether an event was matched and not how early,
so it cannot see the difference.

**The published scheme that shapes by lead time points the other way.**
``sota_objectives.attach_tls_exp_targets`` sets a target that is monotone
*decreasing* in lead time; at the ``gamma_per_hour = 0.2`` frozen in the TLS arm
config a five-minute lead carries target ``0.908818`` against ``0.475021`` at
thirty minutes, a ratio of ``1.913``. That is not a criticism of the scheme,
whose decay follows from a premise about label certainty and whose purpose is to
soften a hard horizon boundary. It is the reason the opposite direction has never
been tried here, and that arm missed its predeclared gate by ``0.000331342``.

The weight is deliberately *not* a target. Multiplying a positive window's loss
by ``v(d)`` leaves the Bayes-optimal score at the event probability and can act
only through finite capacity and finite data, so a null result is a live and
expected outcome rather than a sign of a bad run. A scheme that moved the target
would change the optimal ranking, and that is a different, stronger and untested
claim. See ``PREREG-lead-time-weighted-training.md``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .sequences import LOSS_WEIGHT_SUFFIX, TASKS

MEAN_WEIGHT_TOLERANCE = 1e-9


def _lead_minutes(windows: pd.DataFrame, task: str) -> np.ndarray:
    """Return lead time in minutes, NaN where the window has no future event."""

    onset = pd.to_datetime(
        windows[f"{task}_event_onset"], utc=True, errors="coerce"
    )
    query = pd.to_datetime(windows["context_end"], utc=True, errors="coerce")
    if query.isna().any():
        raise ValueError("context_end contains unparseable timestamps")
    return (onset - query).dt.total_seconds().to_numpy(dtype=float) / 60.0


def attach_lead_time_loss_weights(
    windows: pd.DataFrame,
    *,
    copy: bool = True,
) -> pd.DataFrame:
    """Attach ``v(d) = d / dbar`` on positives and ``1`` on negatives.

    ``dbar`` is the mean lead over that task's positive windows, so the weights
    sum to the positive count exactly. **That is the property the comparison
    rests on**: the weighting redistributes emphasis *among* positives without
    altering the aggregate weight positives carry against negatives, so the arm
    differs from its control in the distribution of the event loss and in nothing
    else. A task with no positive window is left at a uniform weight of one.
    """

    if windows.empty:
        raise ValueError("lead-time weighting requires a non-empty window table")
    required = {"context_end"}
    for task in TASKS:
        required.add(f"{task}_label")
        required.add(f"{task}_event_onset")
    missing = sorted(required.difference(windows.columns))
    if missing:
        raise ValueError(f"lead-time weighting is missing columns: {missing}")

    output = windows.copy() if copy else windows
    for task in TASKS:
        horizon = int(task.rsplit("_", 1)[1])
        positive = output[f"{task}_label"].fillna(0).to_numpy(dtype=float) > 0.5
        lead = _lead_minutes(output, task)

        # `windows.py` writes the label and the onset together, so a positive
        # window without an onset is a corrupt frame rather than a missing value.
        if positive.any() and not np.isfinite(lead[positive]).all():
            raise ValueError(f"{task} has a positive window without an onset")
        # `next_event_within` accepts an onset in `(query, query + horizon]`, so
        # this interval is what the label means. Checking it here is what would
        # catch a sign error or a timezone slip before it looked like a null.
        if positive.any() and (
            np.any(lead[positive] <= 0) or np.any(lead[positive] > horizon)
        ):
            raise ValueError(
                f"{task} lead times fall outside (0, {horizon}] minutes"
            )

        weight = np.ones(len(output), dtype=float)
        if positive.any():
            mean_lead = float(lead[positive].mean())
            if not np.isfinite(mean_lead) or mean_lead <= 0:
                raise ValueError(f"{task} mean lead time is not positive")
            weight[positive] = lead[positive] / mean_lead
            realized = float(weight[positive].mean())
            if abs(realized - 1.0) > MEAN_WEIGHT_TOLERANCE:
                raise ValueError(
                    f"{task} positive weights average {realized!r}, not one"
                )
        output[f"{task}{LOSS_WEIGHT_SUFFIX}"] = weight.astype(np.float32)

    return output


def summarize_lead_time_weights(windows: pd.DataFrame) -> dict[str, Any]:
    """Report the realised weighting, including the direction it points.

    ``weight_ratio_far_over_near`` is the assertion the whole arm turns on: above
    one, a late warning is worth less than an early one, which is the inversion
    of the published temporal-label-smoothing direction.
    """

    summary: dict[str, Any] = {"window_count": int(len(windows)), "tasks": {}}
    for task in TASKS:
        column = f"{task}{LOSS_WEIGHT_SUFFIX}"
        if column not in windows:
            raise ValueError(f"missing lead-time weight column: {column}")
        weight = windows[column].to_numpy(dtype=float)
        if not np.isfinite(weight).all() or np.any(weight < 0):
            raise ValueError(f"invalid lead-time weights for {task}")

        positive = windows[f"{task}_label"].fillna(0).to_numpy(dtype=float) > 0.5
        entry: dict[str, Any] = {
            "positive_window_count": int(positive.sum()),
            "weight_min": float(weight.min()),
            "weight_max": float(weight.max()),
        }
        if positive.any():
            lead = _lead_minutes(windows, task)[positive]
            positive_weight = weight[positive]
            entry.update(
                {
                    "mean_lead_minutes": float(lead.mean()),
                    "median_lead_minutes": float(np.median(lead)),
                    "positive_weight_mean": float(positive_weight.mean()),
                    "positive_weight_sum": float(positive_weight.sum()),
                    "weight_ratio_far_over_near": (
                        float(positive_weight.max() / positive_weight.min())
                        if positive_weight.min() > 0
                        else None
                    ),
                }
            )
        summary["tasks"][task] = entry
    return summary
