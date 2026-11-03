"""Categorical grouping keys must not fabricate events or reject clean data.

Window shards arrive dictionary-encoded: `_compact_window_table` makes
`source_id`, `participant_id` and every `*_event_id` column categorical. A
pandas grouped *aggregation* over categorical keys reindexes its result to the
full cartesian product of categories, so absent combinations appear as rows with
zero-length aggregates. Iterating a groupby does not, which is why only the
aggregating call sites are covered here.

This defect stopped the 2026-08-17 baseline campaign at its first trial:
`_eligible_operational_frame` read 545 real events on one calibration slice as
95,375 groups, 94,830 of them empty, and raised "one event ID maps to multiple
event onsets" against data that was in fact clean. The same shape in
`_event_table` is worse because it does not raise -- it silently inflates the
denominator of every event sensitivity.

Each test below fails if `observed=True` is removed from the call site it
covers.
"""

from __future__ import annotations

import ast
import pathlib

import numpy as np
import pandas as pd
import pytest

import glycoshift
from glycoshift.metrics import _event_table
from glycoshift.operational_metrics import (
    OperationalEndpointSpec,
    _eligible_operational_frame,
)


def _categorical_positive_frame(
    *, sources: int = 4, participants: int = 12, events_each: int = 3
) -> pd.DataFrame:
    """Positive rows with the dictionary encoding real shards carry."""

    rows = []
    base = pd.Timestamp("2026-01-01T00:00:00Z")
    for source in range(sources):
        for participant in range(participants):
            for event in range(events_each):
                onset = base + pd.Timedelta(hours=event + 24 * participant)
                for window in range(2):
                    rows.append(
                        {
                            "source_id": f"src_{source}",
                            "participant_id": f"src_{source}-p{participant:03d}",
                            "event_id": f"src_{source}-p{participant:03d}-e{event}",
                            "event_onset": onset,
                            "context_end": onset - pd.Timedelta(minutes=10 * window),
                        }
                    )
    frame = pd.DataFrame(rows)
    for column in ("source_id", "participant_id", "event_id"):
        frame[column] = frame[column].astype("category")
    return frame


def test_event_table_does_not_fabricate_rows_for_categorical_keys() -> None:
    frame = _categorical_positive_frame()
    real_events = frame.groupby(
        ["source_id", "participant_id", "event_id"], observed=True
    ).ngroups

    events = _event_table(
        frame, event_id_column="event_id", event_onset_column="event_onset"
    )

    assert len(events) == real_events
    assert not events["event_onset"].isna().any(), (
        "a NaT onset means the cartesian product of categories leaked in"
    )
    # The categorical product is what the assertion above is guarding against;
    # show it is genuinely larger, so the test cannot pass vacuously.
    product = (
        frame["source_id"].cat.categories.size
        * frame["participant_id"].cat.categories.size
        * frame["event_id"].cat.categories.size
    )
    assert product > real_events * 10


def test_event_table_matches_the_stringified_frame() -> None:
    """Encoding must not change the answer, only its representation."""

    frame = _categorical_positive_frame()
    stringified = frame.copy()
    for column in ("source_id", "participant_id", "event_id"):
        stringified[column] = stringified[column].astype(str)

    from_categorical = _event_table(
        frame, event_id_column="event_id", event_onset_column="event_onset"
    ).reset_index(drop=True)
    from_strings = _event_table(
        stringified, event_id_column="event_id", event_onset_column="event_onset"
    ).reset_index(drop=True)

    assert len(from_categorical) == len(from_strings)
    pd.testing.assert_series_equal(
        from_categorical["event_onset"],
        from_strings["event_onset"],
        check_names=False,
    )
    assert (
        from_categorical["event_id"].astype(str).tolist()
        == from_strings["event_id"].astype(str).tolist()
    )


def _operational_frame() -> tuple[pd.DataFrame, OperationalEndpointSpec]:
    positives = _categorical_positive_frame(
        sources=3, participants=8, events_each=2
    )
    frame = positives.rename(
        columns={
            "event_id": "severe_hypo_30_event_id",
            "event_onset": "severe_hypo_30_event_onset",
        }
    ).copy()
    frame["severe_hypo_30_label"] = 1
    frame["severe_hypo_currently_active"] = False
    frame["severe_hypo_30_probability"] = np.linspace(0.1, 0.9, len(frame))
    frame["patient_day"] = frame["context_end"].dt.tz_convert(None).dt.normalize()
    # decision keys must be unique
    frame["context_end"] = frame["context_end"] + pd.to_timedelta(
        np.arange(len(frame)), unit="s"
    )
    spec = OperationalEndpointSpec(
        score_column="severe_hypo_30_probability",
        label_column="severe_hypo_30_label",
        event_id_column="severe_hypo_30_event_id",
        event_onset_column="severe_hypo_30_event_onset",
        active_column="severe_hypo_currently_active",
        horizon_minutes=30,
        source_order=tuple(sorted(frame["source_id"].astype(str).unique())),
    )
    return frame, spec


def test_eligible_frame_accepts_clean_categorical_data() -> None:
    """The campaign-stopping refusal must not fire on data that is correct."""

    frame, spec = _operational_frame()
    eligible = _eligible_operational_frame(frame, spec=spec)
    assert len(eligible) == len(frame)


def test_eligible_frame_still_rejects_a_genuine_onset_conflict() -> None:
    """Fixing the false positive must not disarm the check."""

    frame, spec = _operational_frame()
    conflicting = frame.copy()
    target = conflicting.index[0]
    conflicting.loc[target, spec.event_onset_column] = conflicting.loc[
        target, spec.event_onset_column
    ] + pd.Timedelta(hours=5)
    with pytest.raises(ValueError, match="one event ID maps to multiple event onsets"):
        _eligible_operational_frame(conflicting, spec=spec)


# ---------------------------------------------------------------------------
# Package-wide inventory gate
# ---------------------------------------------------------------------------
#
# The three tests above pin the two call sites that were actually wrong. This
# one stops the fourth occurrence from being written at all.
#
# The distinguishing property, and the whole reason an inventory is enough: a
# grouped *aggregation* fabricates a row for every absent category combination,
# and that row then enters a count, a denominator or a uniqueness check. An
# iteration, a `.transform`, and a `.groups` / `.indices` walk cannot -- an
# empty group yields an empty frame whose loop body appends nothing, and
# `transform` broadcasts per row so unobserved categories contribute none.
#
# Every entry below was read and classified on 2026-08-17. Each is a `for ...
# in`, a `.transform(`, a `.groups.items()`, or a name consumed by one of those.
# None aggregates.

_GROUPBY_WITHOUT_OBSERVED = {
    "adapters.py": [
        'grouping = normalized.groupby(["source_id", "participant_id"], sort=False)',
        'for source, source_frame in frame.groupby("source_id", sort=True):',
        'by_source = values.groupby(normalized["source_id"]).transform(',
    ],
    "c2_reference_pipeline.py": [
        "for (source, participant), positions in dataset.windows.groupby(",
    ],
    "c2_runtime.py": [
        "for (source, participant), positions in dataset.windows.groupby(",
    ],
    "events.py": [
        "for _, indices in segment.groupby(segment).groups.items():",
    ],
    "foundation_contract.py": [
        "for (source, participant), group in frame.groupby(",
        "grouped = windows.groupby(",
    ],
    "metrics.py": [
        "for key, group in keyed.groupby(",
        "for key, group in keyed.groupby(",
    ],
    "operational_metrics.py": [
        "for (source, _participant), participant in frame.groupby(",
        "for (source_id, participant_id), participant in frame.groupby(",
    ],
    "role_ingest.py": [
        "for raw_source, group in frame.groupby(",
        'for key, participant in ready.groupby("_participant_key", sort=False):',
    ],
    "sota_objectives.py": [
        "groups = output.groupby(",
    ],
    "streaming.py": [
        'for key, participant in ready.groupby("_participant_key", sort=False):',
    ],
    "training_workload.py": [
        'for source, source_frame in work.groupby("source_id", sort=False):',
    ],
    "windows.py": [
        'grouping = frame.groupby(["source_id", "participant_id"], sort=False)',
    ],
}


def _groupby_sites_without_observed() -> dict[str, list[str]]:
    package = pathlib.Path(glycoshift.__file__).parent
    found: dict[str, list[str]] = {}
    for path in sorted(package.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        lines = source.splitlines()
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "groupby":
                continue
            if any(keyword.arg == "observed" for keyword in node.keywords):
                continue
            found.setdefault(path.name, []).append(lines[node.lineno - 1].strip())
    return {name: sorted(texts) for name, texts in sorted(found.items())}


def test_no_new_groupby_omits_observed() -> None:
    """A new `groupby` without `observed=` must be classified by a human.

    If this fails, resolve it one of two ways and never a third:

    * the call **aggregates** (`.agg`, `.size`, `.nunique`, `.ngroups`, or a
      column selection followed by any of those) -- pass `observed=True`, and
      add a test that fails when it is removed;
    * the call **iterates** or `.transform`s -- append its line to
      ``_GROUPBY_WITHOUT_OBSERVED`` in the same sorted position, having read it.

    Do not silence this by widening the inventory without reading the site. The
    v1.4 campaign died on exactly this defect, at a call site that had passed
    401 tests because every fixture reached it with string keys.
    """

    expected = {
        name: sorted(texts)
        for name, texts in sorted(_GROUPBY_WITHOUT_OBSERVED.items())
    }
    assert _groupby_sites_without_observed() == expected
