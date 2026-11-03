"""Retarget the scored hypoglycaemia heads onto the severe-event label.

The primary endpoint scores severe hypoglycaemia, glucose below 54 mg/dL, but no
model in this package has a severe head: ``TASKS`` holds four heads and
``_attach_severe_scores`` (``deep_training.py:585``) binds the severe score to the
70 mg/dL ``hypo_<h>`` logit verbatim. So every arm has been optimised for one
event and scored on another. ``FINDING-no-arm-was-ever-trained-on-the-endpoint.md``
dates that as drift rather than design and measures its size: a mean predicted
probability of 0.19880329 against an observed severe rate of 0.00361157.

This module changes exactly one thing -- what the scored head is asked to predict --
by writing the two optional per-window columns the sequence datasets already read:

``<task>_training_target``  replaces the loss target, leaving ``<task>_label`` alone
``<task>_training_mask``    replaces that task's loss mask

**Extending ``TASKS`` instead would change three other things at once**, which is
why this route was taken. The negative-subsampling mask at ``sequences.py:189``
is built from ``<task>_label`` over ``TASKS``, so a fifth and sixth task would
alter the sampled training set and confound the target change with a data change.
The per-task mask resolves its activity column as ``task.split("_")[0]``, which
yields ``severe`` for ``severe_hypo_30`` while the column is
``severe_hypo_currently_active``. And ``TASKS`` is read at thirty-four sites across
ten modules, several of them on frozen paths.

Both columns are optional everywhere they are read, so a frame this module has not
touched trains exactly as before.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

SCORED_HORIZONS: tuple[int, ...] = (30, 60)
ACTIVE_COLUMN = "severe_hypo_currently_active"


def _scored_task(horizon: int) -> str:
    """Return the head whose logit becomes the severe score at this horizon."""

    return f"hypo_{horizon}"


def attach_severe_training_targets(
    windows: pd.DataFrame,
    *,
    copy: bool = True,
) -> pd.DataFrame:
    """Point each scored hypo head at its severe label, without moving anything else.

    The mask is taken from ``severe_hypo_currently_active`` rather than from
    ``hypo_currently_active``. That is deliberate and it is the substantive half of
    this function. The default mask drops every window in which a hypo event is
    already running, which is exactly the 54-to-70 mg/dL band where the endpoint is
    decided -- the leading trivial arm alarms at 61.0 mg/dL and its whole advantage
    lies between 54 and 60. Retargeting the label while keeping the default mask
    would train the head on the severe event and hide from it the region that
    settles the benchmark. The replacement matches the endpoint's own eligibility
    rule, which excludes only windows whose severe event is already active.
    """

    if windows.empty:
        raise ValueError("severe retargeting requires a non-empty window table")

    required = {ACTIVE_COLUMN}
    for horizon in SCORED_HORIZONS:
        required.add(f"severe_hypo_{horizon}_label")
        required.add(f"{_scored_task(horizon)}_label")
    missing = sorted(required.difference(windows.columns))
    if missing:
        raise ValueError(f"severe retargeting is missing columns: {missing}")

    output = windows.copy() if copy else windows
    active = output[ACTIVE_COLUMN]
    if active.isna().any():
        raise ValueError(f"{ACTIVE_COLUMN} contains nulls")
    active_values = active.to_numpy(dtype=bool)

    for horizon in SCORED_HORIZONS:
        task = _scored_task(horizon)
        severe = output[f"severe_hypo_{horizon}_label"]
        present = severe.notna().to_numpy()
        target = severe.fillna(0.0).to_numpy(dtype=float)
        if not np.isfinite(target).all() or np.any((target < 0) | (target > 1)):
            raise ValueError(f"severe_hypo_{horizon}_label is not a 0/1 label")
        # A masked window's target is never read by the loss, but the sequence
        # datasets validate every target as a finite probability before the mask
        # is applied, so absent labels are written as 0 rather than left NaN.
        output[f"{task}_training_target"] = target.astype(np.float32)
        output[f"{task}_training_mask"] = (~active_values) & present

    return output


def summarize_severe_targets(windows: pd.DataFrame) -> dict[str, Any]:
    """Report what the retargeting changed, including the mask it widened.

    ``newly_unmasked`` is the count this whole change exists for: windows the
    default mask discards because a hypo event is running, and which the severe
    eligibility rule keeps.
    """

    summary: dict[str, Any] = {"window_count": int(len(windows)), "tasks": {}}
    for horizon in SCORED_HORIZONS:
        task = _scored_task(horizon)
        target_column = f"{task}_training_target"
        mask_column = f"{task}_training_mask"
        for column in (target_column, mask_column):
            if column not in windows:
                raise ValueError(f"missing severe retargeting column: {column}")

        target = windows[target_column].to_numpy(dtype=float)
        if not np.isfinite(target).all() or np.any((target < 0) | (target > 1)):
            raise ValueError(f"invalid severe targets for {task}")
        mask = windows[mask_column].to_numpy(dtype=bool)

        default_mask = (
            ~windows[f"{task.split('_')[0]}_currently_active"].to_numpy(dtype=bool)
        ) & windows[f"{task}_label"].notna().to_numpy()

        summary["tasks"][task] = {
            "severe_positive_count": int(target.sum()),
            "default_positive_count": int(windows[f"{task}_label"].fillna(0).sum()),
            "trained_window_count": int(mask.sum()),
            "default_trained_window_count": int(default_mask.sum()),
            "newly_unmasked": int((mask & ~default_mask).sum()),
            "newly_masked": int((~mask & default_mask).sum()),
        }
    return summary
