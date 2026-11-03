from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _participant_frame(
    frame: pd.DataFrame,
    source_id: str,
    participant_id: str,
) -> pd.DataFrame:
    selected = frame.loc[
        (frame["source_id"].astype(str) == source_id)
        & (frame["participant_id"].astype(str) == participant_id)
    ]
    return selected.sort_values("timestamp").reset_index(drop=True)


def extract_nested_cgm_context(
    frame: pd.DataFrame,
    *,
    source_id: str,
    participant_id: str,
    participant_end_index: int,
    context_hours: float = 24,
    sample_minutes: int = 5,
) -> np.ndarray:
    """Extract one exact, finite CGM-only context for a frozen encoder."""

    if context_hours <= 0 or sample_minutes <= 0:
        raise ValueError("context_hours and sample_minutes must be positive")
    context_samples = int(context_hours * 60 / sample_minutes)
    if context_samples * sample_minutes != context_hours * 60:
        raise ValueError("context_hours must be divisible by sample_minutes")
    participant = _participant_frame(frame, str(source_id), str(participant_id))
    if participant.empty:
        raise ValueError("participant is absent from the frame")
    end_index = int(participant_end_index)
    start_index = end_index - context_samples + 1
    if start_index < 0 or end_index >= len(participant):
        raise ValueError("nested context has insufficient participant history")
    context = participant.iloc[start_index : end_index + 1]
    timestamps = pd.to_datetime(context["timestamp"], utc=True, errors="coerce")
    expected = pd.Timedelta(minutes=sample_minutes)
    if timestamps.isna().any() or not timestamps.diff().dropna().eq(expected).all():
        raise ValueError("nested context is not an exact sample-time grid")
    glucose = pd.to_numeric(context["cgm_mg_dl"], errors="coerce").to_numpy(
        dtype=float
    )
    if not np.isfinite(glucose).all():
        raise ValueError("nested context contains non-finite CGM")
    if "cgm_lower_censored" in context and context[
        "cgm_lower_censored"
    ].astype(bool).any():
        raise ValueError("nested context contains lower-censored CGM")
    return glucose.astype(np.float32)


def select_nested_context_windows(
    frame: pd.DataFrame,
    windows: pd.DataFrame,
    *,
    context_hours: float = 24,
    sample_minutes: int = 5,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select a strict 24-hour subset without changing split or labels."""

    required_frame = {
        "source_id",
        "participant_id",
        "timestamp",
        "cgm_mg_dl",
    }
    required_windows = {
        "source_id",
        "participant_id",
        "context_end",
        "participant_end_index",
        "partition",
    }
    missing_frame = required_frame.difference(frame.columns)
    missing_windows = required_windows.difference(windows.columns)
    if missing_frame or missing_windows:
        raise ValueError(
            "nested-context inputs are missing columns: "
            f"frame={sorted(missing_frame)}, windows={sorted(missing_windows)}"
        )
    if context_hours <= 0 or sample_minutes <= 0:
        raise ValueError("context_hours and sample_minutes must be positive")
    context_samples = int(context_hours * 60 / sample_minutes)
    if context_samples * sample_minutes != context_hours * 60:
        raise ValueError("context_hours must be divisible by sample_minutes")

    participants = {
        (str(source), str(participant)): group.sort_values(
            "timestamp"
        ).reset_index(drop=True)
        for (source, participant), group in frame.groupby(
            ["source_id", "participant_id"],
            sort=False,
        )
    }
    reasons = np.full(len(windows), "", dtype=object)
    expected_ns = int(pd.Timedelta(minutes=sample_minutes).value)
    grouped = windows.groupby(
        ["source_id", "participant_id"],
        sort=False,
    ).indices

    def assign(local: np.ndarray, mask: np.ndarray, reason: str) -> None:
        local[(local == "") & mask] = reason

    for key, positions in grouped.items():
        position_array = np.asarray(positions, dtype=np.int64)
        local = np.full(len(position_array), "", dtype=object)
        participant = participants.get((str(key[0]), str(key[1])))
        if participant is None:
            local[:] = "participant_missing"
            reasons[position_array] = local
            continue

        raw_end = pd.to_numeric(
            windows.iloc[position_array]["participant_end_index"],
            errors="coerce",
        ).to_numpy(dtype=float)
        integral = np.isfinite(raw_end) & np.equal(raw_end, np.floor(raw_end))
        assign(local, ~integral, "invalid_end_index")
        end = np.where(integral, raw_end, 0).astype(np.int64)
        in_range = integral & (end >= 0) & (end < len(participant))
        assign(local, integral & ~in_range, "end_index_out_of_range")
        has_history = in_range & (end >= context_samples - 1)
        assign(local, in_range & ~has_history, "insufficient_history")

        participant_times = pd.to_datetime(
            participant["timestamp"],
            utc=True,
            errors="coerce",
        )
        if participant_times.isna().any():
            assign(local, has_history, "participant_timestamp_invalid")
            reasons[position_array] = local
            continue
        participant_ns = participant_times.astype("int64").to_numpy()
        query_times = pd.to_datetime(
            windows.iloc[position_array]["context_end"],
            utc=True,
            errors="coerce",
        )
        query_valid = ~query_times.isna().to_numpy()
        assign(local, has_history & ~query_valid, "context_end_invalid")
        comparable = has_history & query_valid
        end_match = np.zeros(len(local), dtype=bool)
        end_match[comparable] = (
            participant_ns[end[comparable]]
            == query_times.iloc[np.flatnonzero(comparable)].astype("int64").to_numpy()
        )
        assign(local, comparable & ~end_match, "context_end_mismatch")

        glucose = pd.to_numeric(
            participant["cgm_mg_dl"],
            errors="coerce",
        ).to_numpy(dtype=float)
        invalid_cgm = ~np.isfinite(glucose)
        lower_censored = (
            participant["cgm_lower_censored"].astype(bool).to_numpy()
            if "cgm_lower_censored" in participant
            else np.zeros(len(participant), dtype=bool)
        )
        bad_edge = np.zeros(len(participant), dtype=np.int64)
        if len(participant) > 1:
            bad_edge[1:] = np.diff(participant_ns) != expected_ns
        invalid_prefix = np.concatenate([[0], np.cumsum(invalid_cgm)])
        censored_prefix = np.concatenate([[0], np.cumsum(lower_censored)])
        edge_prefix = np.concatenate([[0], np.cumsum(bad_edge)])

        candidates = local == ""
        candidate_indices = np.flatnonzero(candidates)
        if len(candidate_indices):
            starts = end[candidate_indices] - context_samples + 1
            stops = end[candidate_indices] + 1
            bad_cgm_count = invalid_prefix[stops] - invalid_prefix[starts]
            censored_count = censored_prefix[stops] - censored_prefix[starts]
            bad_edge_count = edge_prefix[stops] - edge_prefix[starts + 1]
            candidate_mask = np.zeros(len(local), dtype=bool)
            candidate_mask[candidate_indices[bad_cgm_count > 0]] = True
            assign(local, candidate_mask, "nonfinite_cgm")
            candidate_mask[:] = False
            candidate_mask[candidate_indices[censored_count > 0]] = True
            assign(local, candidate_mask, "lower_censored_cgm")
            candidate_mask[:] = False
            candidate_mask[candidate_indices[bad_edge_count > 0]] = True
            assign(local, candidate_mask, "time_grid_gap")
        reasons[position_array] = local

    eligible_mask = reasons == ""
    selected = windows.iloc[np.flatnonzero(eligible_mask)].copy()
    excluded, counts = np.unique(reasons[~eligible_mask], return_counts=True)
    audit = {
        "status": "passed" if len(selected) else "empty",
        "context_hours": float(context_hours),
        "sample_minutes": int(sample_minutes),
        "context_samples": int(context_samples),
        "input_window_count": int(len(windows)),
        "eligible_window_count": int(len(selected)),
        "excluded_window_count": int((~eligible_mask).sum()),
        "exclusion_reasons": {
            str(reason): int(count)
            for reason, count in zip(excluded, counts, strict=True)
        },
        "input_partition_counts": {
            str(key): int(value)
            for key, value in windows["partition"].value_counts().sort_index().items()
        },
        "eligible_partition_counts": {
            str(key): int(value)
            for key, value in selected["partition"].value_counts().sort_index().items()
        },
    }
    return selected, audit
