import pandas as pd
import pytest

from glycoshift.events import (
    Event,
    EventIndex,
    detect_events,
    inside_event,
    next_event_within,
)


def test_three_sample_event_and_future_onset() -> None:
    frame = pd.DataFrame(
        {
            "source_id": ["s"] * 8,
            "participant_id": ["p"] * 8,
            "timestamp": pd.date_range(
                "2026-01-01", periods=8, freq="5min", tz="UTC"
            ),
            "cgm_mg_dl": [100, 90, 69, 68, 67, 80, 60, 90],
        }
    )
    events = detect_events(
        frame,
        threshold=70,
        direction="below",
        minimum_length=3,
        merge_gap_minutes=5,
        kind="hypo",
    )
    assert len(events) == 1
    assert events[0].onset == frame["timestamp"].iloc[2]
    assert next_event_within(events, frame["timestamp"].iloc[0], 15) == events[0]
    assert next_event_within(events, frame["timestamp"].iloc[0], 5) is None


def test_event_index_matches_list_oracle_at_every_boundary() -> None:
    origin = pd.Timestamp("2026-01-01", tz="UTC")
    events = [
        Event(
            event_id="hypo:0",
            kind="hypo",
            onset=origin + pd.Timedelta(minutes=10),
            end=origin + pd.Timedelta(minutes=20),
        ),
        Event(
            event_id="hypo:1",
            kind="hypo",
            onset=origin + pd.Timedelta(minutes=30),
            end=origin + pd.Timedelta(minutes=35),
        ),
        Event(
            event_id="hypo:2",
            kind="hypo",
            onset=origin + pd.Timedelta(minutes=50),
            end=origin + pd.Timedelta(minutes=60),
        ),
    ]
    index = EventIndex.build(events)

    for minute in range(-5, 66):
        query = origin + pd.Timedelta(minutes=minute)
        assert index.inside_event(query) == inside_event(events, query)
        for horizon in (0, 5, 15, 30, 60):
            assert index.next_event_within(
                query,
                horizon,
            ) == next_event_within(events, query, horizon)


def test_event_index_rejects_unsorted_or_overlapping_events() -> None:
    origin = pd.Timestamp("2026-01-01", tz="UTC")
    first = Event(
        event_id="hypo:0",
        kind="hypo",
        onset=origin + pd.Timedelta(minutes=10),
        end=origin + pd.Timedelta(minutes=20),
    )
    overlapping = Event(
        event_id="hypo:1",
        kind="hypo",
        onset=origin + pd.Timedelta(minutes=15),
        end=origin + pd.Timedelta(minutes=25),
    )
    earlier = Event(
        event_id="hypo:2",
        kind="hypo",
        onset=origin,
        end=origin + pd.Timedelta(minutes=5),
    )
    with pytest.raises(ValueError, match="must not overlap"):
        EventIndex.build([first, overlapping])
    with pytest.raises(ValueError, match="sorted by onset"):
        EventIndex.build([first, earlier])
