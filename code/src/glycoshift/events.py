from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Event:
    event_id: str
    kind: str
    onset: pd.Timestamp
    end: pd.Timestamp


@dataclass(frozen=True)
class EventIndex:
    """Binary-search event queries with the same boundaries as the list oracle."""

    events: tuple[Event, ...]
    onset_nanoseconds: tuple[int, ...]

    @classmethod
    def build(cls, events: list[Event]) -> EventIndex:
        frozen = tuple(events)
        for index, event in enumerate(frozen):
            if event.end < event.onset:
                raise ValueError("event end precedes onset")
            if index and frozen[index - 1].onset > event.onset:
                raise ValueError("events must be sorted by onset")
            if index and frozen[index - 1].end >= event.onset:
                raise ValueError("events must not overlap")
        return cls(
            events=frozen,
            onset_nanoseconds=tuple(
                int(pd.Timestamp(event.onset).value)
                for event in frozen
            ),
        )

    def next_event_within(
        self,
        query_time: pd.Timestamp,
        horizon_minutes: int,
    ) -> Event | None:
        query = pd.Timestamp(query_time)
        index = bisect_right(self.onset_nanoseconds, int(query.value))
        if index == len(self.events):
            return None
        event = self.events[index]
        horizon_end = query + pd.Timedelta(minutes=horizon_minutes)
        return event if event.onset <= horizon_end else None

    def inside_event(self, query_time: pd.Timestamp) -> bool:
        query = pd.Timestamp(query_time)
        index = bisect_right(
            self.onset_nanoseconds,
            int(query.value),
        ) - 1
        return index >= 0 and query <= self.events[index].end


def _true_runs(mask: np.ndarray, minimum_length: int) -> list[tuple[int, int]]:
    padded = np.concatenate(([False], mask.astype(bool), [False]))
    changes = np.diff(padded.astype(int))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return [
        (int(start), int(end))
        for start, end in zip(starts, ends)  # noqa: B905
        if end - start + 1 >= minimum_length
    ]


def detect_events(
    participant: pd.DataFrame,
    *,
    threshold: float,
    direction: str,
    minimum_length: int,
    merge_gap_minutes: int,
    kind: str,
    sample_minutes: int = 5,
) -> list[Event]:
    ordered = participant.sort_values("timestamp").reset_index(drop=True)
    glucose = ordered["cgm_mg_dl"].to_numpy(dtype=float)
    if direction == "below":
        mask = glucose < threshold
    elif direction == "above":
        mask = glucose > threshold
    else:
        raise ValueError(f"unsupported direction: {direction}")

    timestamps = pd.to_datetime(ordered["timestamp"], utc=True)
    segment = (
        timestamps.diff().gt(pd.Timedelta(minutes=sample_minutes)).cumsum()
    )
    runs: list[tuple[int, int]] = []
    for _, indices in segment.groupby(segment).groups.items():
        positions = np.asarray(list(indices), dtype=int)
        for start, end in _true_runs(mask[positions], minimum_length):
            runs.append((int(positions[start]), int(positions[end])))
    merged: list[tuple[int, int]] = []
    for start, end in runs:
        if not merged:
            merged.append((start, end))
            continue
        previous_start, previous_end = merged[-1]
        gap = timestamps.iloc[start] - timestamps.iloc[previous_end]
        continuous_gap = timestamps.iloc[start] - timestamps.iloc[previous_end]
        expected_samples = int(
            continuous_gap / pd.Timedelta(minutes=sample_minutes)
        )
        observed_samples = start - previous_end
        has_unresolved_gap = expected_samples != observed_samples
        if (
            gap <= pd.Timedelta(minutes=merge_gap_minutes)
            and not has_unresolved_gap
        ):
            merged[-1] = (previous_start, end)
        else:
            merged.append((start, end))

    prefix = (
        f"{ordered['source_id'].iloc[0]}:"
        f"{ordered['participant_id'].iloc[0]}:{kind}"
    )
    return [
        Event(
            event_id=f"{prefix}:{index:04d}",
            kind=kind,
            onset=timestamps.iloc[start],
            end=timestamps.iloc[end],
        )
        for index, (start, end) in enumerate(merged)
    ]


def next_event_within(
    events: list[Event],
    query_time: pd.Timestamp,
    horizon_minutes: int,
) -> Event | None:
    horizon_end = query_time + pd.Timedelta(minutes=horizon_minutes)
    for event in events:
        if query_time < event.onset <= horizon_end:
            return event
    return None


def inside_event(events: list[Event], query_time: pd.Timestamp) -> bool:
    return any(event.onset <= query_time <= event.end for event in events)
