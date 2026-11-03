from __future__ import annotations

import numpy as np
import pandas as pd

from .events import detect_events, inside_event, next_event_within
from .schema import ACTION_FIELDS

FEATURE_COLUMNS = (
    "cgm_last",
    "cgm_mean",
    "cgm_std",
    "cgm_min",
    "cgm_max",
    "cgm_slope",
    "cgm_slope_30",
    *(f"{field}_recording_rate" for field in ACTION_FIELDS),
    *(f"{field}_available_rate" for field in ACTION_FIELDS),
    *(f"{field}_ambiguous_zero_rate" for field in ACTION_FIELDS),
)


def build_windows(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    protocol = config["protocol"]
    sample_minutes = int(protocol["sample_minutes"])
    context_samples = int(
        float(protocol["context_hours"]) * 60 / sample_minutes
    )
    horizons = [int(value) for value in protocol["horizons_minutes"]]
    minimum_length = int(protocol["minimum_consecutive_samples"])
    merge_gap = int(protocol["merge_event_gap_minutes"])
    thresholds = protocol["thresholds_mg_dl"]

    output: list[dict[str, object]] = []
    grouping = frame.groupby(["source_id", "participant_id"], sort=False)
    for (source_id, participant_id), participant in grouping:
        ordered = participant.sort_values("timestamp").reset_index(drop=True)
        ordered["timestamp"] = pd.to_datetime(ordered["timestamp"], utc=True)
        lower_censored = (
            ordered["cgm_lower_censored"].astype(bool)
            if "cgm_lower_censored" in ordered
            else pd.Series(False, index=ordered.index, dtype=bool)
        )
        if "timestamp_local" in ordered:
            local_time = pd.to_datetime(
                ordered["timestamp_local"],
                errors="coerce",
            )
            if local_time.isna().any():
                raise ValueError("timestamp_local contains unparseable values")
        else:
            local_time = ordered["timestamp"].dt.tz_localize(None)
        ordered["_local_day"] = local_time.dt.floor("D")
        expected_per_day = 24 * 60 // sample_minutes
        day_counts = ordered.groupby("_local_day", observed=True).size()
        eligible_days = set(
            day_counts.loc[
                day_counts / expected_per_day
                >= float(config["data"]["minimum_day_coverage"])
            ].index
        )
        if len(eligible_days) < int(config["data"]["minimum_days"]):
            continue
        hypo_events = detect_events(
            ordered,
            threshold=float(thresholds["hypoglycemia"]),
            direction="below",
            minimum_length=minimum_length,
            merge_gap_minutes=merge_gap,
            kind="hypo",
            sample_minutes=sample_minutes,
        )
        severe_hypo_events = detect_events(
            ordered,
            threshold=float(thresholds["severe_hypoglycemia"]),
            direction="below",
            minimum_length=minimum_length,
            merge_gap_minutes=merge_gap,
            kind="severe_hypo",
            sample_minutes=sample_minutes,
        )
        hyper_events = detect_events(
            ordered,
            threshold=float(thresholds["hyperglycemia"]),
            direction="above",
            minimum_length=minimum_length,
            merge_gap_minutes=merge_gap,
            kind="hyper",
            sample_minutes=sample_minutes,
        )

        final_end_index = len(ordered) - max(horizons) // sample_minutes
        for end_index in range(context_samples - 1, final_end_index):
            context = ordered.iloc[end_index - context_samples + 1 : end_index + 1]
            if lower_censored.iloc[
                end_index - context_samples + 1 : end_index + 1
            ].any():
                continue
            query_time = ordered["timestamp"].iloc[end_index]
            query_day = ordered["_local_day"].iloc[end_index]
            if query_day not in eligible_days:
                continue
            expected_span = pd.Timedelta(
                minutes=sample_minutes * (context_samples - 1)
            )
            observed_span = (
                context["timestamp"].iloc[-1] - context["timestamp"].iloc[0]
            )
            if observed_span != expected_span:
                continue
            future_end_index = end_index + max(horizons) // sample_minutes
            future = ordered.iloc[end_index : future_end_index + 1]
            if (
                future["timestamp"].iloc[-1] - query_time
                != pd.Timedelta(minutes=max(horizons))
            ):
                continue
            if not future["timestamp"].diff().dropna().eq(
                pd.Timedelta(minutes=sample_minutes)
            ).all():
                continue

            glucose = context["cgm_mg_dl"].to_numpy(dtype=float)
            short_glucose = glucose[-min(6, len(glucose)) :]
            row: dict[str, object] = {
                "source_id": str(source_id),
                "participant_id": str(participant_id),
                "context_start": context["timestamp"].iloc[0],
                "context_end": query_time,
                "patient_day": query_day,
                "cgm_last": float(glucose[-1]),
                "cgm_mean": float(np.mean(glucose)),
                "cgm_std": float(np.std(glucose)),
                "cgm_min": float(np.min(glucose)),
                "cgm_max": float(np.max(glucose)),
                "cgm_slope": float(np.polyfit(np.arange(len(glucose)), glucose, 1)[0]),
                "cgm_slope_30": float(
                    np.polyfit(
                        np.arange(len(short_glucose)),
                        short_glucose,
                        1,
                    )[0]
                ),
                "participant_end_index": int(end_index),
            }
            for field in ACTION_FIELDS:
                row[f"{field}_recording_rate"] = float(
                    context[f"{field}_observed"].mean()
                )
                row[f"{field}_available_rate"] = float(
                    context[f"{field}_available"].mean()
                )
                row[f"{field}_ambiguous_zero_rate"] = float(
                    context[f"{field}_zero_ambiguous"].mean()
                )

            for kind, events in (
                ("hypo", hypo_events),
                ("severe_hypo", severe_hypo_events),
                ("hyper", hyper_events),
            ):
                row[f"{kind}_currently_active"] = inside_event(events, query_time)
                for horizon in horizons:
                    event = next_event_within(events, query_time, horizon)
                    row[f"{kind}_{horizon}_label"] = int(event is not None)
                    row[f"{kind}_{horizon}_event_id"] = (
                        event.event_id if event is not None else None
                    )
                    row[f"{kind}_{horizon}_event_onset"] = (
                        event.onset if event is not None else pd.NaT
                    )
            for horizon in horizons:
                target_index = end_index + horizon // sample_minutes
                target_available = not bool(
                    lower_censored.iloc[target_index]
                )
                row[f"cgm_{horizon}_target_available"] = target_available
                row[f"cgm_{horizon}_target"] = (
                    float(ordered["cgm_mg_dl"].iloc[target_index])
                    if target_available
                    else np.nan
                )
            output.append(row)

    windows = pd.DataFrame(output)
    if windows.empty:
        raise ValueError("no eligible windows were generated")
    return windows
