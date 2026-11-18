"""Observed-anchor timelines and fixed evaluation populations for the revision.

Future observations define retrospective outcomes and evaluable exposure only.
They never determine whether a context-valid, observed-anchor decision exists.
Legacy window construction is intentionally unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .events import detect_events
from .schema import ACTION_FIELDS


@dataclass
class ReplayInputs:
    timeline: pd.DataFrame
    events: dict[int, pd.DataFrame]
    exposure: dict[int, pd.DataFrame]
    audit: dict


CONFIRMATION_CONTRACT = "2026-11-09-complete-confirmation"


def validate_confirmation_masks(timeline: pd.DataFrame, contract: str | None) -> None:
    """Reject stale task masks before fitting a corrected replay.

    The continuity flags are produced from the normalized time axis by the
    builder. This gate checks their downstream use without consulting outcomes
    to decide completeness or replacing established positive labels.
    """
    if contract is None:
        return
    if contract != CONFIRMATION_CONTRACT:
        raise ValueError("unsupported ascertainment contract")
    historical = _strict_bool(timeline, "base_evaluable")
    glucose = pd.to_numeric(timeline["cgm_last"], errors="coerce").to_numpy(float)
    if not np.isfinite(glucose).all():
        raise ValueError("confirmation contract requires finite anchor glucose")
    for horizon in (30, 60):
        complete = _strict_bool(timeline, f"confirmation_complete_{horizon}")
        base = _strict_bool(timeline, f"base_evaluable_{horizon}")
        if not np.array_equal(base, historical & complete):
            raise ValueError(f"{horizon} confirmation/base mask mismatch")
        if horizon == 30 and not np.array_equal(base, historical):
            raise ValueError("30-minute frame must retain historical completeness")
        for kind, label_kind in (("hypo", "severe_hypo"), ("hyper", "hyper")):
            allowed = (
                glucose >= 54 if kind == "hypo"
                else ~_strict_bool(timeline, "hyper_currently_active")
            )
            task = f"{kind}_{horizon}"
            mask = _strict_bool(timeline, f"{task}_training_mask")
            if not np.array_equal(mask, base & allowed):
                raise ValueError(f"{task} confirmation/training mask mismatch")
            target = timeline[f"{task}_training_target"].to_numpy(float)
            labels = timeline[f"{label_kind}_{horizon}_label"].to_numpy(float)
            if not np.isin(labels, (0, 1)).all() or not np.array_equal(target, labels):
                raise ValueError(f"{task} confirmed labels must remain unchanged")


def _strict_bool(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame:
        raise ValueError(f"missing observed-anchor provenance: {column}")
    values = frame[column]
    if values.isna().any() or not pd.api.types.is_bool_dtype(values.dtype):
        raise ValueError(f"{column} must contain complete boolean values")
    return values.to_numpy(dtype=bool)


def _slope(values: np.ndarray, samples: int) -> np.ndarray:
    if samples < 2:
        raise ValueError("a slope context requires at least two samples")
    weights = np.arange(samples, dtype=float) - (samples - 1) / 2
    result = np.full(len(values), np.nan)
    if len(values) >= samples:
        result[samples - 1 :] = np.convolve(
            values, weights[::-1], mode="valid"
        ) / np.dot(weights, weights)
    return result


def build_participant_replay(
    participant: pd.DataFrame,
    config: dict,
    *,
    partition: str,
    lower_censoring_absent_means_false: bool = False,
) -> ReplayInputs:
    """Build one participant without reading another source/partition.

    ``lower_censoring_absent_means_false`` is an explicit normalization-contract
    assertion, not inference from a finite glucose value. The caller must audit it.
    Original maximum-horizon completeness is retained for the base evaluation
    frame at both horizons, while the full online timeline includes its tails.
    """
    required = {"source_id", "participant_id", "timestamp", "cgm_mg_dl"}
    if not required.issubset(participant):
        raise ValueError(
            f"missing normalized columns: {sorted(required - set(participant))}"
        )
    if participant.empty:
        raise ValueError("participant frame is empty")
    identities = (
        participant[["source_id", "participant_id"]].astype(str).drop_duplicates()
    )
    if len(identities) != 1:
        raise ValueError(
            "replay builder requires exactly one source-scoped participant"
        )
    source_id, participant_id = identities.iloc[0]
    ordered = participant.sort_values("timestamp").reset_index(drop=True)
    times = pd.to_datetime(ordered["timestamp"], utc=True, errors="coerce")
    if times.isna().any() or times.duplicated().any():
        raise ValueError("normalized timestamps must be valid and unique")
    observed = _strict_bool(ordered, "cgm_observed")
    interpolated = _strict_bool(ordered, "cgm_interpolated")
    if np.any(observed & interpolated):
        raise ValueError("a CGM observation cannot also be interpolated")
    if "cgm_lower_censored" in ordered:
        censored = _strict_bool(ordered, "cgm_lower_censored")
    elif lower_censoring_absent_means_false:
        censored = np.zeros(len(ordered), dtype=bool)
    else:
        raise ValueError("lower-censor provenance is missing without a source contract")
    glucose = pd.to_numeric(ordered["cgm_mg_dl"], errors="coerce").to_numpy(float)
    if not np.isfinite(glucose).all():
        raise ValueError(
            "normalized CGM must be finite; unresolved gaps must be absent rows"
        )
    protocol = config["protocol"]
    ascertainment = protocol.get("ascertainment_contract")
    if ascertainment not in (None, "2026-11-09-complete-confirmation"):
        raise ValueError("unsupported ascertainment contract")
    sample = int(protocol["sample_minutes"])
    horizons = tuple(int(h) for h in protocol["horizons_minutes"])
    if ascertainment and (
        sample != 5 or horizons != (30, 60)
        or int(protocol["minimum_consecutive_samples"]) != 3
    ):
        raise ValueError(
            "complete-confirmation contract requires 5min/30,60/three readings"
        )
    ncontext = int(float(protocol["context_hours"]) * 60 / sample)
    if (
        sample <= 0
        or ncontext < 2
        or not horizons
        or any(h <= 0 or h % sample for h in horizons)
    ):
        raise ValueError("invalid sample/context/horizon contract")
    if len(set(horizons)) != len(horizons):
        raise ValueError("duplicate horizons")
    ns = times.astype("int64").to_numpy()
    step = sample * 60 * 1_000_000_000
    discontinuities = np.r_[0, np.diff(ns) != step].cumsum()
    end = np.arange(ncontext - 1, len(ordered), dtype=int)
    start = end - ncontext + 1
    context_ok = discontinuities[end] == discontinuities[start]
    censor_sum = np.r_[0, censored.astype(int).cumsum()]
    context_ok &= censor_sum[end + 1] == censor_sum[start]
    context_candidates = end[context_ok]
    anchor_ok = observed[end] & ~interpolated[end] & ~censored[end]
    end = end[context_ok & anchor_ok]
    start = end - ncontext + 1
    if "timestamp_local" in ordered:
        local = pd.to_datetime(ordered["timestamp_local"], errors="coerce")
        if local.isna().any():
            raise ValueError("timestamp_local must be complete and parseable")
        days = local.dt.floor("D")
    else:
        days = times.dt.tz_localize(None).dt.floor("D")
    counts = days.value_counts()
    eligible_days = counts.index[
        counts / (24 * 60 / sample) >= float(config["data"]["minimum_day_coverage"])
    ]
    participant_qc = len(eligible_days) >= int(config["data"]["minimum_days"])
    future_end = end + max(horizons) // sample
    future_ok = future_end < len(ordered)
    safe_future = np.minimum(future_end, len(ordered) - 1)
    future_ok &= discontinuities[safe_future] == discontinuities[end]
    base = future_ok & days.iloc[end].isin(eligible_days).to_numpy() & participant_qc
    rolling = pd.Series(glucose).rolling(ncontext)
    timeline = pd.DataFrame(
        {
            "source_id": source_id,
            "participant_id": participant_id,
            "partition": partition,
            "context_start": times.iloc[start].to_numpy(),
            "context_end": times.iloc[end].to_numpy(),
            "patient_day": days.iloc[end].to_numpy(),
            "participant_end_index": end,
            "cgm_last": glucose[end],
            "cgm_mean": rolling.mean().to_numpy()[end],
            "cgm_std": rolling.std(ddof=0).to_numpy()[end],
            "cgm_min": rolling.min().to_numpy()[end],
            "cgm_max": rolling.max().to_numpy()[end],
            "cgm_slope": _slope(glucose, ncontext)[end],
            "cgm_slope_30": _slope(glucose, min(6, ncontext))[end],
            "anchor_cgm_observed": observed[end],
            "anchor_cgm_interpolated": interpolated[end],
            "anchor_cgm_censored": censored[end],
            "base_evaluable": base,
        }
    )
    # Preserve the classical feature schema. Their admitted action features are
    # selected by the runner; the main CGM-only model does not use these values.
    for field in ACTION_FIELDS:
        for suffix, target in (
            ("observed", "recording_rate"),
            ("available", "available_rate"),
            ("zero_ambiguous", "ambiguous_zero_rate"),
        ):
            column = f"{field}_{suffix}"
            if column not in ordered:
                raise ValueError(f"normalized action metadata missing: {column}")
            timeline[f"{field}_{target}"] = (
                pd.Series(_strict_bool(ordered, column))
                .rolling(ncontext)
                .mean()
                .to_numpy()[end]
            )
    catalogues = {}
    for kind, threshold_key, direction in (
        ("hypo", "hypoglycemia", "below"),
        ("severe_hypo", "severe_hypoglycemia", "below"),
        ("hyper", "hyperglycemia", "above"),
    ):
        events = detect_events(
            ordered,
            threshold=float(protocol["thresholds_mg_dl"][threshold_key]),
            direction=direction,
            minimum_length=int(protocol["minimum_consecutive_samples"]),
            merge_gap_minutes=int(protocol["merge_event_gap_minutes"]),
            kind=kind,
            sample_minutes=sample,
        )
        catalogues[kind] = events
        onsets = np.array([e.onset.value for e in events], dtype=np.int64)
        endings = np.array([e.end.value for e in events], dtype=np.int64)
        query = ns[end]
        next_index = np.searchsorted(onsets, query, side="right")
        previous = next_index - 1
        active = np.zeros(len(end), dtype=bool)
        has_previous = previous >= 0
        active[has_previous] = query[has_previous] <= endings[previous[has_previous]]
        timeline[f"{kind}_currently_active"] = active
        for horizon in horizons:
            hit = np.zeros(len(end), dtype=bool)
            valid = next_index < len(events)
            hit[valid] = (
                onsets[next_index[valid]] <= query[valid] + horizon * 60 * 1_000_000_000
            )
            event_ids = np.full(len(end), None, dtype=object)
            event_onsets = np.full(
                len(end), np.datetime64("NaT"), dtype="datetime64[ns]"
            )
            hit_positions = np.flatnonzero(hit)
            event_ids[hit_positions] = [
                events[next_index[i]].event_id for i in hit_positions
            ]
            event_onsets[hit] = onsets[next_index[hit]].astype("datetime64[ns]")
            timeline[f"{kind}_{horizon}_label"] = hit.astype(int)
            timeline[f"{kind}_{horizon}_event_id"] = event_ids
            timeline[f"{kind}_{horizon}_event_onset"] = pd.to_datetime(
                event_onsets, utc=True
            )
    base_events, exposures = {}, {}
    for horizon in horizons:
        horizon_base = base.copy()
        if ascertainment:
            confirmation_end = end + max(60, horizon + 10) // sample
            confirmation_safe = np.minimum(confirmation_end, len(ordered) - 1)
            complete = (confirmation_end < len(ordered)) & (
                discontinuities[confirmation_safe] == discontinuities[end]
            )
            timeline[f"confirmation_complete_{horizon}"] = complete
            horizon_base &= complete
        timeline[f"base_evaluable_{horizon}"] = horizon_base
        target_index = end + horizon // sample
        target_exists = target_index < len(ordered)
        target_safe = np.minimum(target_index, len(ordered) - 1)
        target_exists &= discontinuities[target_safe] == discontinuities[end]
        target_exists &= ~censored[target_safe]
        target = np.full(len(end), np.nan)
        target[target_exists] = glucose[target_index[target_exists]]
        timeline[f"cgm_{horizon}_target"] = target
        timeline[f"cgm_{horizon}_target_available"] = target_exists & base
        for task_kind, target_kind in (("hypo", "severe_hypo"), ("hyper", "hyper")):
            timeline[f"{task_kind}_{horizon}_training_target"] = timeline[
                f"{target_kind}_{horizon}_label"
            ]
            allowed = (
                glucose[end] >= 54
                if task_kind == "hypo"
                else ~timeline["hyper_currently_active"].to_numpy(bool)
            )
            timeline[f"{task_kind}_{horizon}_training_mask"] = horizon_base & allowed
        lead_times = ns[end[horizon_base]]
        event_rows = []
        for event in catalogues["severe_hypo"]:
            first = np.searchsorted(
                lead_times, event.onset.value - horizon * 60 * 1_000_000_000
            )
            last = np.searchsorted(lead_times, event.onset.value)
            if first < last:
                event_rows.append(
                    {
                        "source_id": source_id,
                        "participant_id": participant_id,
                        "event_id": event.event_id,
                        "event_onset": event.onset,
                    }
                )
        base_events[horizon] = pd.DataFrame(
            event_rows,
            columns=["source_id", "participant_id", "event_id", "event_onset"],
        )
        exposure_rows = []
        if horizon_base.any():
            exposure_rows.append(
                {
                    "source_id": source_id,
                    "participant_id": participant_id,
                    "patient_days": int(
                        timeline.loc[horizon_base, "patient_day"].nunique()
                    ),
                    "observation_minutes": int(horizon_base.sum()) * sample,
                }
            )
        exposures[horizon] = pd.DataFrame(
            exposure_rows,
            columns=[
                "source_id",
                "participant_id",
                "patient_days",
                "observation_minutes",
            ],
        )
    validate_confirmation_masks(timeline, ascertainment)
    return ReplayInputs(
        timeline,
        base_events,
        exposures,
        {
            "source_id": source_id,
            "partition": partition,
            "normalized_records": len(ordered),
            "context_valid_anchors": len(context_candidates),
            "context_valid_interpolated_anchors": int(
                interpolated[context_candidates].sum()
            ),
            "observed_anchor_decisions": len(timeline),
            "base_evaluable_decisions": int(base.sum()),
            "eligible_calendar_days": len(eligible_days),
            "participant_passes_day_qc": participant_qc,
            "base_event_counts": {str(h): len(base_events[h]) for h in horizons},
            "base_uses_maximum_horizon_completeness": max(horizons),
            "ascertainment_contract": ascertainment,
            "lower_censor_column_absence_authorized": (
                lower_censoring_absent_means_false
            ),
        },
    )
