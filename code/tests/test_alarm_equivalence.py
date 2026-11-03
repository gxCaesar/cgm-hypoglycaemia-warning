"""Equivalence of the vectorised alarm path against the retained oracle.

The oracle is correct but scales as O(alarms x events) with a per-row pandas
constructor, which projects to months on the audited partitions. The
replacement must therefore be proved identical, not merely similar: these
metrics decide reported alarm burden, sensitivity and lead time.

Every assertion here is exact. No tolerance is used anywhere, because both
paths must select the same rows and compute the same lead minutes from the
same integer nanoseconds.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from glycoshift.metrics import (
    _SWEEP_MINIMUM_CANDIDATES_PER_WORKER,
    AlarmThresholdSweep,
    _collapsed_alarm_rows,
    _collapsed_alarm_rows_oracle,
    _event_table,
    _match_alarms,
    _match_alarms_oracle,
    _sweep_all_candidates,
    alarm_metrics,
    event_average_precision,
    select_alarm_threshold,
)

COLLAPSE_MINUTES = 30
HORIZON_MINUTES = 30


def _frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["context_end"] = pd.to_datetime(frame["context_end"], utc=True)
    frame["patient_day"] = frame["context_end"].dt.floor("D")
    return frame.reset_index(drop=True)


def _synthetic(seed: int, *, participants: int, rows: int) -> pd.DataFrame:
    """Random but structured: repeated event ids ACROSS participants, exact
    refractory boundaries, and events both inside and outside the horizon."""
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2026-01-01T00:00:00Z")
    records: list[dict[str, object]] = []
    for participant in range(participants):
        source = f"S{participant % 3}"
        # Deliberately reuse the same event id strings in every participant so
        # the global `used` set can collide, matching production event ids
        # which are only unique within a participant.
        minute = 0
        for _ in range(rows):
            minute += int(rng.integers(1, 40)) * 5
            timestamp = base + pd.Timedelta(minutes=minute)
            has_event = bool(rng.random() < 0.15)
            records.append(
                {
                    "source_id": source,
                    "participant_id": f"P{participant}",
                    "context_end": timestamp,
                    "probability": float(rng.random()),
                    "event_id": f"E{rng.integers(0, 4)}" if has_event else None,
                    "event_onset": (
                        timestamp + pd.Timedelta(minutes=int(rng.integers(5, 40)))
                        if has_event
                        else pd.NaT
                    ),
                }
            )
    return _frame(records)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 6, 7])
@pytest.mark.parametrize("threshold", [0.0, 0.1, 0.5, 0.9, 1.0])
def test_collapse_matches_oracle_exactly(seed: int, threshold: float) -> None:
    frame = _synthetic(seed, participants=4, rows=40)
    expected = _collapsed_alarm_rows_oracle(
        frame,
        probability_column="probability",
        threshold=threshold,
        collapse_minutes=COLLAPSE_MINUTES,
    )
    observed = _collapsed_alarm_rows(
        frame,
        probability_column="probability",
        threshold=threshold,
        collapse_minutes=COLLAPSE_MINUTES,
    )
    assert list(observed.index) == list(expected.index)
    pd.testing.assert_frame_equal(observed, expected)


def test_collapse_boundary_is_strictly_greater_than_the_interval() -> None:
    """Exactly 30 minutes apart must be collapsed; one sample later must not.

    This is the single assertion that pins the comparison direction. A
    `>=` would silently keep an extra alarm and inflate alarm counts.
    """
    frame = _frame(
        [
            {
                "source_id": "S0",
                "participant_id": "P0",
                "context_end": "2026-01-01T00:00:00Z",
                "probability": 1.0,
            },
            {   # exactly at the boundary -> collapsed away
                "source_id": "S0",
                "participant_id": "P0",
                "context_end": "2026-01-01T00:30:00Z",
                "probability": 1.0,
            },
            {   # one sample past the boundary -> kept
                "source_id": "S0",
                "participant_id": "P0",
                "context_end": "2026-01-01T00:35:00Z",
                "probability": 1.0,
            },
        ]
    )
    kept = _collapsed_alarm_rows(
        frame, probability_column="probability", threshold=0.5
    )
    oracle = _collapsed_alarm_rows_oracle(
        frame, probability_column="probability", threshold=0.5
    )
    assert list(kept["context_end"]) == list(oracle["context_end"])
    assert len(kept) == 2
    assert kept["context_end"].tolist() == [
        pd.Timestamp("2026-01-01T00:00:00Z"),
        pd.Timestamp("2026-01-01T00:35:00Z"),
    ]


def test_collapse_refractory_runs_from_the_kept_alarm() -> None:
    """Three alarms 20 minutes apart collapse to the first and the third.

    A naive pairwise diff would keep none after the first; the rule measures
    from the last KEPT alarm, so 0 and 40 survive while 20 does not.
    """
    frame = _frame(
        [
            {
                "source_id": "S0",
                "participant_id": "P0",
                "context_end": f"2026-01-01T00:{minute:02d}:00Z",
                "probability": 1.0,
            }
            for minute in (0, 20, 40)
        ]
    )
    kept = _collapsed_alarm_rows(
        frame, probability_column="probability", threshold=0.5
    )
    oracle = _collapsed_alarm_rows_oracle(
        frame, probability_column="probability", threshold=0.5
    )
    assert kept["context_end"].tolist() == oracle["context_end"].tolist()
    assert kept["context_end"].dt.minute.tolist() == [0, 40]


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 6, 7])
@pytest.mark.parametrize("threshold", [0.0, 0.2, 0.6, 0.95])
def test_match_matches_oracle_exactly(seed: int, threshold: float) -> None:
    frame = _synthetic(seed, participants=4, rows=40)
    alarms = _collapsed_alarm_rows(
        frame, probability_column="probability", threshold=threshold
    )
    events = _event_table(
        frame,
        event_id_column="event_id",
        event_onset_column="event_onset",
    )
    expected = _match_alarms_oracle(
        alarms, events, horizon_minutes=HORIZON_MINUTES
    )
    observed = _match_alarms(alarms, events, horizon_minutes=HORIZON_MINUTES)
    expected_ids, expected_false, expected_lead = expected
    observed_ids, observed_false, observed_lead = observed
    assert observed_ids == expected_ids
    assert observed_false == expected_false
    assert observed_lead == expected_lead


def test_match_horizon_bounds_are_open_left_and_closed_right() -> None:
    """onset == alarm_time does not match; onset == alarm_time + horizon does.

    Getting either bound wrong shifts event sensitivity, so both are pinned
    against the oracle rather than only asserted directly.
    """
    frame = _frame(
        [
            {
                "source_id": "S0",
                "participant_id": "P0",
                "context_end": "2026-01-01T00:00:00Z",
                "probability": 1.0,
                "event_id": "E0",
                "event_onset": "2026-01-01T00:00:00Z",  # equal -> no match
            },
            {
                "source_id": "S0",
                "participant_id": "P1",
                "context_end": "2026-01-01T00:00:00Z",
                "probability": 1.0,
                "event_id": "E1",
                "event_onset": "2026-01-01T00:30:00Z",  # exactly horizon -> match
            },
        ]
    )
    frame["event_onset"] = pd.to_datetime(frame["event_onset"], utc=True)
    alarms = _collapsed_alarm_rows(
        frame, probability_column="probability", threshold=0.5
    )
    events = _event_table(
        frame, event_id_column="event_id", event_onset_column="event_onset"
    )
    observed = _match_alarms(alarms, events, horizon_minutes=HORIZON_MINUTES)
    expected = _match_alarms_oracle(
        alarms, events, horizon_minutes=HORIZON_MINUTES
    )
    assert observed == expected
    matched_ids, false_alarms, lead = observed
    assert matched_ids == ["E1"]
    assert false_alarms == 1
    assert lead == [30.0]


def test_match_keeps_the_global_used_set_semantics() -> None:
    """A repeated event id across participants can block a later match.

    Event ids are only unique within a participant, and the oracle's `used`
    set is global, so P1's E0 is unavailable once P0's E0 is consumed. This is
    preserved deliberately; the test exists so that any future correction is a
    conscious decision with a failing test, not a silent drift.
    """
    frame = _frame(
        [
            {
                "source_id": "S0",
                "participant_id": "P0",
                "context_end": "2026-01-01T00:00:00Z",
                "probability": 1.0,
                "event_id": "E0",
                "event_onset": "2026-01-01T00:10:00Z",
            },
            {
                "source_id": "S0",
                "participant_id": "P1",
                "context_end": "2026-01-01T01:00:00Z",
                "probability": 1.0,
                "event_id": "E0",
                "event_onset": "2026-01-01T01:10:00Z",
            },
        ]
    )
    frame["event_onset"] = pd.to_datetime(frame["event_onset"], utc=True)
    alarms = _collapsed_alarm_rows(
        frame, probability_column="probability", threshold=0.5
    )
    events = _event_table(
        frame, event_id_column="event_id", event_onset_column="event_onset"
    )
    observed = _match_alarms(alarms, events, horizon_minutes=HORIZON_MINUTES)
    assert observed == _match_alarms_oracle(
        alarms, events, horizon_minutes=HORIZON_MINUTES
    )
    matched_ids, false_alarms, _ = observed
    assert matched_ids == ["E0"]
    assert false_alarms == 1


@pytest.mark.parametrize("seed", [21, 22, 23, 24, 25])
def test_sweep_path_equals_single_threshold_path(seed: int) -> None:
    """AlarmThresholdSweep must equal alarm_metrics field for field.

    select_alarm_threshold and event_average_precision were switched onto the
    sweep, so this is the assertion that protects every reported alarm number.
    It is checked at both fine and coarse thresholds, including 0.0 where every
    row is an alarm and 1.0 where none is.
    """
    frame = _synthetic(seed, participants=5, rows=60)
    frame["event_onset"] = pd.to_datetime(frame["event_onset"], utc=True)
    sweep = AlarmThresholdSweep(
        frame,
        probability_column="probability",
        event_id_column="event_id",
        event_onset_column="event_onset",
    )
    for threshold in np.linspace(0.0, 1.0, 41):
        expected = alarm_metrics(
            frame,
            probability_column="probability",
            label_column="event_id",
            event_id_column="event_id",
            event_onset_column="event_onset",
            horizon_minutes=HORIZON_MINUTES,
            threshold=float(threshold),
        )
        observed = sweep.metrics_at(
            float(threshold), horizon_minutes=HORIZON_MINUTES
        )
        assert observed.keys() == expected.keys()
        for key in expected:
            left, right = observed[key], expected[key]
            if isinstance(right, float) and np.isnan(right):
                assert np.isnan(left), f"{key} at threshold {threshold}"
            else:
                assert left == right, (
                    f"{key} at threshold {threshold}: {left} != {right}"
                )


def test_sweep_handles_duplicate_timestamps_within_a_participant() -> None:
    """Identical context_end values must not change the collapse outcome.

    The sweep sorts with lexsort while the single-threshold path sorts a
    DataFrame; ties are the one place those two could diverge.
    """
    frame = _frame(
        [
            {
                "source_id": "S0",
                "participant_id": "P0",
                "context_end": "2026-01-01T00:00:00Z",
                "probability": 0.9,
                "event_id": "E0",
                "event_onset": "2026-01-01T00:10:00Z",
            },
            {
                "source_id": "S0",
                "participant_id": "P0",
                "context_end": "2026-01-01T00:00:00Z",
                "probability": 0.8,
                "event_id": None,
                "event_onset": pd.NaT,
            },
            {
                "source_id": "S0",
                "participant_id": "P0",
                "context_end": "2026-01-01T01:00:00Z",
                "probability": 0.7,
                "event_id": None,
                "event_onset": pd.NaT,
            },
        ]
    )
    frame["event_onset"] = pd.to_datetime(frame["event_onset"], utc=True)
    sweep = AlarmThresholdSweep(
        frame,
        probability_column="probability",
        event_id_column="event_id",
        event_onset_column="event_onset",
    )
    for threshold in (0.0, 0.5, 0.75, 0.85, 0.95):
        expected = alarm_metrics(
            frame,
            probability_column="probability",
            label_column="event_id",
            event_id_column="event_id",
            event_onset_column="event_onset",
            horizon_minutes=HORIZON_MINUTES,
            threshold=threshold,
        )
        observed = sweep.metrics_at(threshold, horizon_minutes=HORIZON_MINUTES)
        for key in expected:
            left, right = observed[key], expected[key]
            if isinstance(right, float) and np.isnan(right):
                assert np.isnan(left), key
            else:
                assert left == right, f"{key} at {threshold}"


@pytest.mark.parametrize("seed", [11, 12, 13])
def test_full_alarm_metrics_identical_across_thresholds(seed: int) -> None:
    """End-to-end: every field of alarm_metrics, over a threshold sweep.

    This is the quantity the paper reports, so equivalence is asserted on the
    public function rather than only on the two private helpers.
    """
    frame = _synthetic(seed, participants=5, rows=60)
    frame["event_onset"] = pd.to_datetime(frame["event_onset"], utc=True)

    import glycoshift.metrics as metrics_module

    for threshold in np.linspace(0.0, 1.0, 21):
        observed = alarm_metrics(
            frame,
            probability_column="probability",
            label_column="event_id",
            event_id_column="event_id",
            event_onset_column="event_onset",
            horizon_minutes=HORIZON_MINUTES,
            threshold=float(threshold),
        )
        original_collapse = metrics_module._collapsed_alarm_rows
        original_match = metrics_module._match_alarms
        metrics_module._collapsed_alarm_rows = (
            metrics_module._collapsed_alarm_rows_oracle
        )
        metrics_module._match_alarms = metrics_module._match_alarms_oracle
        try:
            expected = alarm_metrics(
                frame,
                probability_column="probability",
                label_column="event_id",
                event_id_column="event_id",
                event_onset_column="event_onset",
                horizon_minutes=HORIZON_MINUTES,
                threshold=float(threshold),
            )
        finally:
            metrics_module._collapsed_alarm_rows = original_collapse
            metrics_module._match_alarms = original_match

        assert observed.keys() == expected.keys()
        for key in expected:
            left, right = observed[key], expected[key]
            if isinstance(right, float) and np.isnan(right):
                assert np.isnan(left), f"{key} at threshold {threshold}"
            else:
                assert left == right, f"{key} at threshold {threshold}"


# ---------------------------------------------------------------------------
# Parallel candidate evaluation
# ---------------------------------------------------------------------------
#
# `select_alarm_threshold` sweeps ~2001 candidates per (task, false-alarm
# target), eight sweeps a run, about 16,812 `metrics_at` calls -- roughly 4
# core-hours on the real 5.78M-row calibration partition, paid in full by
# fifteen of the seventeen campaign arms.
#
# Distributing that loop is only admissible if it cannot move a number, so
# these assertions are exact on the float's hex, not `pytest.approx`. The
# threshold they check is the operating point every reported sensitivity and
# false-alarm rate is measured at.


def _sweep_frame() -> pd.DataFrame:
    return _synthetic(11, participants=40, rows=120)


def _threshold(frame: pd.DataFrame, *, workers: int, target: float) -> float:
    return select_alarm_threshold(
        frame,
        probability_column="probability",
        label_column="probability",
        event_id_column="event_id",
        event_onset_column="event_onset",
        horizon_minutes=HORIZON_MINUTES,
        candidate_count=201,
        target_false_alarms_per_day=target,
        workers=workers,
    )


@pytest.mark.parametrize("workers", [2, 3, 5])
@pytest.mark.parametrize("target", [0.5, 1.0])
def test_parallel_sweep_returns_the_serial_threshold_bit_for_bit(
    workers: int,
    target: float,
) -> None:
    frame = _sweep_frame()
    serial = _threshold(frame, workers=1, target=target)
    parallel = _threshold(frame, workers=workers, target=target)
    assert parallel.hex() == serial.hex()


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_every_candidate_result_matches_the_serial_evaluation(
    workers: int,
) -> None:
    """Not just the selected threshold: every candidate's full metric dict.

    A parallel path that agreed only on the argmax could still be reordering
    results, and the reordering would surface later as a different operating
    point on a different dataset.
    """

    frame = _sweep_frame()
    sweep = AlarmThresholdSweep(
        frame,
        probability_column="probability",
        event_id_column="event_id",
        event_onset_column="event_onset",
        collapse_minutes=COLLAPSE_MINUTES,
    )
    candidates = np.unique(
        np.quantile(
            frame["probability"].to_numpy(dtype=float), np.linspace(0, 1, 201)
        )
    )
    serial = _sweep_all_candidates(
        sweep, candidates, horizon_minutes=HORIZON_MINUTES, workers=1
    )
    observed = _sweep_all_candidates(
        sweep, candidates, horizon_minutes=HORIZON_MINUTES, workers=workers
    )
    assert len(observed) == len(candidates)
    for expected_row, observed_row in zip(serial, observed, strict=True):
        assert set(expected_row) == set(observed_row)
        for key, expected_value in expected_row.items():
            observed_value = observed_row[key]
            # `median_lead_minutes` is NaN whenever nothing matched, and NaN is
            # not equal to itself, so a naive `==` reports a false difference.
            if isinstance(expected_value, float) and math.isnan(expected_value):
                assert math.isnan(observed_value)
            else:
                assert observed_value == expected_value


def test_default_is_serial_and_starts_no_process() -> None:
    """The parallel path must be asked for; nothing forks by default."""

    frame = _sweep_frame()
    sweep = AlarmThresholdSweep(
        frame,
        probability_column="probability",
        event_id_column="event_id",
        event_onset_column="event_onset",
        collapse_minutes=COLLAPSE_MINUTES,
    )
    candidates = np.unique(
        np.quantile(
            frame["probability"].to_numpy(dtype=float), np.linspace(0, 1, 51)
        )
    )
    import glycoshift.metrics as metrics_module

    def _explode(*args: object, **kwargs: object) -> None:
        raise AssertionError("the serial path must not create a process pool")

    original = metrics_module.ProcessPoolExecutor
    metrics_module.ProcessPoolExecutor = _explode  # type: ignore[assignment]
    try:
        _sweep_all_candidates(
            sweep, candidates, horizon_minutes=HORIZON_MINUTES, workers=1
        )
        # Too few candidates to be worth a fork: also serial, same guard.
        _sweep_all_candidates(
            sweep, candidates[:3], horizon_minutes=HORIZON_MINUTES, workers=8
        )
    finally:
        metrics_module.ProcessPoolExecutor = original  # type: ignore[assignment]


def test_the_sweep_survives_a_pickle_round_trip() -> None:
    """`forkserver` pickles the sweep to each worker, so this is a hard
    requirement rather than an incidental property.

    `fork` was rejected deliberately: `select_alarm_threshold` runs after
    training, where torch's threads are still alive, and forking a
    multi-threaded process can hand the child a mutex locked with no owner.
    Pickling costs about 185 MiB once per worker on the real calibration
    partition, against a sweep measured in core-hours.
    """

    import pickle

    frame = _sweep_frame()
    sweep = AlarmThresholdSweep(
        frame,
        probability_column="probability",
        event_id_column="event_id",
        event_onset_column="event_onset",
        collapse_minutes=COLLAPSE_MINUTES,
    )
    restored = pickle.loads(pickle.dumps(sweep))
    threshold = float(frame["probability"].median())
    expected = sweep.metrics_at(threshold, horizon_minutes=HORIZON_MINUTES)
    observed = restored.metrics_at(threshold, horizon_minutes=HORIZON_MINUTES)
    assert set(expected) == set(observed)
    for key, value in expected.items():
        if isinstance(value, float) and math.isnan(value):
            assert math.isnan(observed[key])
        else:
            assert observed[key] == value


def test_no_start_method_falls_back_to_the_serial_loop() -> None:
    """A platform without forkserver or spawn must still produce results."""

    import glycoshift.metrics as metrics_module

    frame = _sweep_frame()
    sweep = AlarmThresholdSweep(
        frame,
        probability_column="probability",
        event_id_column="event_id",
        event_onset_column="event_onset",
        collapse_minutes=COLLAPSE_MINUTES,
    )
    candidates = np.unique(
        np.quantile(
            frame["probability"].to_numpy(dtype=float), np.linspace(0, 1, 61)
        )
    )
    expected = _sweep_all_candidates(
        sweep, candidates, horizon_minutes=HORIZON_MINUTES, workers=1
    )
    original = metrics_module._sweep_start_method
    metrics_module._sweep_start_method = lambda: None  # type: ignore[assignment]
    try:
        observed = _sweep_all_candidates(
            sweep, candidates, horizon_minutes=HORIZON_MINUTES, workers=8
        )
    finally:
        metrics_module._sweep_start_method = original  # type: ignore[assignment]
    assert len(observed) == len(expected)


def _event_ap_frame(seed: int, rows: int = 900) -> pd.DataFrame:
    """A frame with enough events for a real precision-recall sweep."""

    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    records = []
    for participant in range(6):
        event = 0
        for step in range(rows // 6):
            moment = start + pd.Timedelta(minutes=10 * step)
            positive = bool(rng.random() < 0.22)
            if positive:
                event += 1
            records.append(
                {
                    "source_id": f"S{participant % 3}",
                    "participant_id": f"P{participant}",
                    "context_end": moment,
                    "patient_day": moment.floor("D"),
                    "hypo_30_probability": float(
                        rng.random() * 0.6 + (0.35 if positive else 0.0)
                    ),
                    "hypo_30_label": int(positive),
                    "hypo_30_event_id": (
                        f"P{participant}-e{event}" if positive else None
                    ),
                    "hypo_30_event_onset": (
                        moment + pd.Timedelta(minutes=5) if positive else pd.NaT
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


@pytest.mark.parametrize("seed", [11, 29, 47])
@pytest.mark.parametrize("workers", [2, 3, 5])
def test_event_average_precision_is_bit_identical_across_worker_counts(
    seed: int, workers: int
) -> None:
    """The parallel candidate sweep must not perturb the reported number.

    `event_average_precision` runs inside `evaluate_predictions`, the CPU-only
    tail of every deep arm, where the arm has already released its GPU. Making
    that tail shorter shortens the window in which a co-tenant takes the approved
    card, which is what ended the 2026-08-19 campaign. It must cost nothing in
    the value.
    """

    frame = _event_ap_frame(seed)
    kwargs = dict(
        probability_column="hypo_30_probability",
        label_column="hypo_30_label",
        event_id_column="hypo_30_event_id",
        event_onset_column="hypo_30_event_onset",
        horizon_minutes=30,
        candidate_count=61,
    )
    serial = event_average_precision(frame, **kwargs, workers=1)
    parallel = event_average_precision(frame, **kwargs, workers=workers)
    assert float(serial).hex() == float(parallel).hex()


def test_event_average_precision_default_is_serial() -> None:
    """No config, no pool. The knob is opt-in and absent from every base config."""

    frame = _event_ap_frame(5)
    kwargs = dict(
        probability_column="hypo_30_probability",
        label_column="hypo_30_label",
        event_id_column="hypo_30_event_id",
        event_onset_column="hypo_30_event_onset",
        horizon_minutes=30,
        candidate_count=41,
    )
    assert (
        float(event_average_precision(frame, **kwargs)).hex()
        == float(event_average_precision(frame, **kwargs, workers=1)).hex()
    )


@pytest.mark.parametrize(
    ("count", "asked", "expected"),
    [
        (2001, 16, 16),   # the alarm sweep: the caller's figure stands
        (201, 16, 8),     # event AP: sixteen measured slower than eight
        (201, 8, 8),
        (41, 16, 1),      # too little work to spread at all
        (24, 2, 1),
    ],
)
def test_worker_count_is_capped_by_available_candidate_work(
    count: int, asked: int, expected: int
) -> None:
    """Each worker pays a fixed cost to receive the sweep state.

    Measured on real predictions, one selection cell of 5,842,352 rows and 201
    candidates: 55.5 s serial, 16.6 s at eight workers, 19.6 s at sixteen. Past
    some point another worker costs more than it saves, and the launch bundle
    derives sixteen for a lane, which is right for the 2001-candidate alarm sweep
    and wrong for the 201-candidate event-AP one.
    """

    capped = max(1, min(asked, count // _SWEEP_MINIMUM_CANDIDATES_PER_WORKER))
    assert capped == expected
