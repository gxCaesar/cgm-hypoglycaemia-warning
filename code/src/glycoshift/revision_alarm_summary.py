"""Anonymous CGM-at-alarm summaries from the unchanged emitted alarm ledger."""

from __future__ import annotations

import numpy as np
import pandas as pd


def summarize_alarm_glucose(
    alarms: pd.DataFrame, *, source_order: tuple[str, ...]
) -> list[dict]:
    """Summarize emissions without fitting, rematching or dropping unevaluable ones.

    Each source and the pooled population have separate emitted, evaluable,
    matched and unmatched summaries. Pooling here weights alarms, not sources
    or participants. No participant identifier or timestamp is exported.
    """
    required = {
        "source_id",
        "participant_id",
        "context_end",
        "cgm_last",
        "base_evaluable",
        "matched",
        "unmatched",
    }
    if not required.issubset(alarms):
        raise ValueError("alarm ledger is missing required fields")
    if not source_order or len(source_order) != len(set(source_order)):
        raise ValueError("source order must be nonempty and unique")
    if set(alarms["source_id"]).difference(source_order):
        raise ValueError("alarm ledger contains an undeclared source")
    keys = ["source_id", "participant_id", "context_end"]
    if alarms[keys].isna().any().any() or alarms.duplicated(keys).any():
        raise ValueError("alarm identity is missing or duplicated")
    for field in ("base_evaluable", "matched", "unmatched"):
        values = alarms[field]
        if len(values) and (
            not pd.api.types.is_bool_dtype(values.dtype) or values.isna().any()
        ):
            raise ValueError("alarm status must be complete boolean values")
    glucose = pd.to_numeric(alarms["cgm_last"], errors="coerce").to_numpy(float)
    if not np.isfinite(glucose).all():
        raise ValueError("CGM at emitted alarms must be finite")
    base = alarms["base_evaluable"].to_numpy(bool)
    matched = alarms["matched"].to_numpy(bool)
    unmatched = alarms["unmatched"].to_numpy(bool)
    if np.any(matched & ~base) or not np.array_equal(unmatched, base & ~matched):
        raise ValueError("alarm matching status contradicts evaluability")
    rows = []
    masks = {
        "emitted": np.ones(len(alarms), dtype=bool),
        "evaluable": base,
        "matched": matched,
        "unmatched": unmatched,
    }
    for source in (*source_order, None):
        membership = (
            np.ones(len(alarms), dtype=bool)
            if source is None
            else alarms["source_id"].eq(source).to_numpy()
        )
        for category, status in masks.items():
            values = glucose[membership & status]
            row = {
                "scope": "pooled_alarms" if source is None else "source",
                "source_id": source,
                "alarm_category": category,
                "alarm_count": int(values.size),
                "unit": "mg/dL",
                "mean": float(values.mean()) if values.size else None,
            }
            for name, q in (
                ("minimum", 0),
                ("q25", 0.25),
                ("median", 0.5),
                ("q75", 0.75),
                ("maximum", 1),
            ):
                row[name] = float(np.quantile(values, q)) if values.size else None
            rows.append(row)
    return rows
