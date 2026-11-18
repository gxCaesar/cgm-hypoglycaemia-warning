"""Independent common-frame/source oracles; no real participant data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from test_revision_inputs import fixture

from glycoshift.events import Event, detect_events
from glycoshift.revision_inputs import build_participant_replay
from glycoshift.revision_round2_sensitivity import (
    build_elapsed_span_catalogue,
    common_completeness_frame,
    rematch_catalogues_on_common_frame,
    summarize_common_source_support,
)


def _elapsed_input(values, *, gap_after=None):
    frame, _ = fixture()
    frame = frame.iloc[:len(values)].copy()
    frame["cgm_mg_dl"] = values
    if gap_after is not None:
        frame.loc[gap_after + 1:, "timestamp"] += pd.Timedelta(minutes=5)
    return frame


@pytest.mark.parametrize("count", [1, 2, 3])
def test_elapsed_unconfirmed_low_runs_do_not_become_events(count):
    assert build_elapsed_span_catalogue(_elapsed_input([50] * count + [70] * 4)) == []


def test_elapsed_four_low_readings_confirm_first_reading_as_onset():
    frame = _elapsed_input([50] * 4)
    event, = build_elapsed_span_catalogue(frame)
    assert event.onset == frame.timestamp.iloc[0]
    assert event.onset_confirmed_at == frame.timestamp.iloc[3]
    assert (event.onset_confirmed_at - event.onset).total_seconds() == 15 * 60
    assert event.end == frame.timestamp.iloc[3]
    assert event.right_censored and event.censor_reason == "record_end"


def test_elapsed_end_is_fourth_recovery_reading_and_70_is_inclusive():
    frame = _elapsed_input([50] * 4 + [70] * 4)
    event, = build_elapsed_span_catalogue(frame)
    assert event.end == frame.timestamp.iloc[7]
    assert not event.right_censored and event.censor_reason is None


def test_elapsed_54_does_not_confirm_low_and_sub70_interrupts_recovery():
    frame = _elapsed_input([50, 50, 54] + [50] * 4 + [70] * 3 + [69] + [70] * 4)
    event, = build_elapsed_span_catalogue(frame)
    assert event.onset == frame.timestamp.iloc[3]
    assert event.onset_confirmed_at == frame.timestamp.iloc[6]
    assert event.end == frame.timestamp.iloc[14]


def test_elapsed_gap_censors_confirmed_event_and_resets_recovery_and_onset():
    frame = _elapsed_input([50] * 4 + [70] * 3 + [70] * 4 + [50] * 4 + [70] * 4,
                           gap_after=6)
    first, second = build_elapsed_span_catalogue(frame)
    assert first.onset == frame.timestamp.iloc[0]
    assert first.end == frame.timestamp.iloc[6]
    assert first.right_censored and first.censor_reason == "time_gap"
    assert second.onset == frame.timestamp.iloc[11]
    assert second.end == frame.timestamp.iloc[18]
    assert not second.right_censored


def test_elapsed_gap_discards_unconfirmed_low_run_and_restarts_at_next_reading():
    frame = _elapsed_input([50] * 3 + [50] * 4 + [70] * 4, gap_after=2)
    event, = build_elapsed_span_catalogue(frame)
    assert event.onset == frame.timestamp.iloc[3]
    assert event.onset_confirmed_at == frame.timestamp.iloc[6]
    assert event.end == frame.timestamp.iloc[10]


def test_elapsed_does_not_apply_primary_30_minute_merge():
    frame = _elapsed_input([50] * 4 + [70] * 4 + [50] * 4 + [70] * 4)
    first, second = build_elapsed_span_catalogue(frame)
    assert second.onset - first.end == pd.Timedelta(minutes=5)
    assert not first.right_censored and not second.right_censored


def test_elapsed_terminal_incomplete_recovery_remains_right_censored():
    frame = _elapsed_input([50] * 4 + [70] * 3)
    event, = build_elapsed_span_catalogue(frame)
    assert event.end == frame.timestamp.iloc[-1]
    assert event.right_censored and event.censor_reason == "record_end"


def _window_oracle(frame):
    """Enumerate confirming windows, then pair first onset/recovery per segment."""
    segments = []
    start = 0
    for index in range(1, len(frame)):
        gap = frame.timestamp.iloc[index] - frame.timestamp.iloc[index - 1]
        if gap != pd.Timedelta(minutes=5):
            segments.append((start, index))
            start = index
    segments.append((start, len(frame)))
    expected = []
    for start, stop in segments:
        low = [index for index in range(start, stop - 3)
               if all(value < 54 for value in frame.cgm_mg_dl.iloc[index:index + 4])]
        recovery = [
            index + 3 for index in range(start, stop - 3)
            if all(value >= 70 for value in frame.cgm_mg_dl.iloc[index:index + 4])
        ]
        cursor = start
        while any(index >= cursor for index in low):
            onset = min(index for index in low if index >= cursor)
            recovery_ends = [end for end in recovery if end > onset + 3]
            if recovery_ends:
                end, reason = min(recovery_ends), None
            else:
                end = stop - 1
                reason = "time_gap" if stop < len(frame) else "record_end"
            expected.append((
                frame.timestamp.iloc[onset], frame.timestamp.iloc[onset + 3],
                frame.timestamp.iloc[end], reason,
            ))
            cursor = end + 1
    return expected


def test_elapsed_catalogue_matches_independent_confirming_window_oracle():
    rng = np.random.default_rng(7)
    for trial in range(12):
        values = rng.choice([50, 54, 69, 70, 90], size=60)
        values[5:9] = 50
        values[14:18] = 70
        values[30:34] = 50
        values[44:48] = 70
        frame = _elapsed_input(values, gap_after=25 if trial % 2 else None)
        expected = _window_oracle(frame)
        actual = [
            (event.onset, event.onset_confirmed_at, event.end, event.censor_reason)
            for event in build_elapsed_span_catalogue(frame)
        ]
        assert actual == expected


def test_elapsed_catalogue_rematches_on_common_frame_with_censor_metadata():
    normalized, config = fixture()
    normalized.loc[30:33, "cgm_mg_dl"] = 50
    replay = build_participant_replay(normalized, config, partition="selection")
    primary = detect_events(
        normalized, threshold=54, direction="below", minimum_length=3,
        merge_gap_minutes=30, kind="severe_hypo", sample_minutes=5,
    )
    elapsed = build_elapsed_span_catalogue(normalized)
    assert primary[0].end == normalized.timestamp.iloc[33]
    assert elapsed[0].end == normalized.timestamp.iloc[37]
    alarms = _alarms(replay.timeline, normalized.timestamp.iloc[[25, 59]])
    _, results = rematch_catalogues_on_common_frame(
        normalized, replay.timeline, {"primary": primary, "elapsed_span": elapsed},
        alarms, horizon=60, glucose_cut=54,
    )
    for result in results.values():
        assert result.alarms.context_end.tolist() == alarms.context_end.tolist()
        assert result.alarms.matched.tolist() == [True, False]
        assert result.participants.unevaluable_alarm_count.iloc[0] == 1
    ledger = results["elapsed_span"].events
    assert ledger.onset_confirmed_at.tolist() == [normalized.timestamp.iloc[33]]
    assert ledger.right_censored.tolist() == [False]
    assert ledger.censor_reason.isna().all()


def _alarms(timeline, timestamps):
    columns = ["source_id", "participant_id", "context_end", "patient_day",
               "cgm_last", "base_evaluable"]
    table = timeline.loc[timeline.context_end.isin(timestamps), columns].copy()
    table = table.reset_index(drop=True)
    table["frozen_score"] = 0.9
    table["matched"] = False
    table["unmatched"] = table.base_evaluable
    table["matched_event_id"] = None
    table["matched_event_onset"] = pd.NaT
    table["lead_minutes"] = np.nan
    return table


@pytest.mark.parametrize("horizon,width", [(30, 60), (60, 75)])
def test_common_frame_matches_independent_continuity_oracle(horizon, width):
    normalized, config = fixture()
    normalized = normalized.drop(index=40).reset_index(drop=True)
    replay = build_participant_replay(normalized, config, partition="selection")
    common = common_completeness_frame(normalized, replay.timeline, horizon=horizon)
    timestamps = set(normalized.timestamp)
    expected = []
    for row in replay.timeline.itertuples():
        required = pd.date_range(row.context_end, periods=width // 5 + 1, freq="5min")
        expected.append(
            bool(row.base_evaluable) and all(t in timestamps for t in required)
        )
    assert common.timeline.base_evaluable.tolist() == expected
    assert len(common.timeline) == len(replay.timeline)
    assert common.exposure.observation_minutes.iloc[0] == sum(expected) * 5
    assert common.audit["required_future_minutes"] == width


def test_common_frame_never_reads_normalized_glucose_or_outcomes():
    normalized, config = fixture()
    timeline = build_participant_replay(
        normalized, config, partition="selection"
    ).timeline
    first = common_completeness_frame(normalized, timeline, horizon=60)
    metadata_only = normalized[["source_id", "participant_id", "timestamp"]]
    second = common_completeness_frame(metadata_only, timeline, horizon=60)
    changed = normalized.copy()
    changed["cgm_mg_dl"] = np.nan
    changed["any_future_outcome"] = True
    third = common_completeness_frame(changed, timeline, horizon=60)
    for actual in (second, third):
        pd.testing.assert_frame_equal(first.timeline, actual.timeline)
        pd.testing.assert_frame_equal(first.exposure, actual.exposure)


def test_t_plus_75_boundary_is_strict_and_common_for_both_catalogues():
    normalized, config = fixture()
    replay = build_participant_replay(normalized, config, partition="selection")
    common = common_completeness_frame(normalized, replay.timeline, horizon=60)
    actual = common.timeline.set_index("context_end").base_evaluable
    assert actual.loc[normalized.timestamp.iloc[56]]
    assert not actual.loc[normalized.timestamp.iloc[57]]
    # The historical t+60 frame includes both anchors.
    assert replay.timeline.set_index("context_end").base_evaluable.loc[
        normalized.timestamp.iloc[57]
    ]


def test_common_frame_rematch_keeps_fixed_online_emissions_and_unevaluable_alarms():
    normalized, config = fixture()
    replay = build_participant_replay(normalized, config, partition="selection")
    alarms = _alarms(replay.timeline, normalized.timestamp.iloc[[50, 59]])
    catalogue = [Event(
        "primary-e", "primary", normalized.timestamp.iloc[60],
        normalized.timestamp.iloc[64],
    )]
    alternative = [Event(
        "other-e", "other", normalized.timestamp.iloc[60],
        normalized.timestamp.iloc[66],
    )]
    common, results = rematch_catalogues_on_common_frame(
        normalized, replay.timeline, {"primary": catalogue, "alternative": alternative},
        alarms, horizon=60, glucose_cut=54,
    )
    assert len(common.timeline) == len(replay.timeline)
    for result in results.values():
        assert len(result.alarms) == len(alarms) == 2
        pd.testing.assert_series_equal(result.alarms.context_end, alarms.context_end)
        pd.testing.assert_series_equal(result.alarms.frozen_score, alarms.frozen_score)
        assert result.alarms.base_evaluable.tolist() == [True, False]
        assert result.alarms.matched.tolist() == [True, False]
        assert result.alarms.unmatched.tolist() == [False, False]
        assert result.alarms.accepted_primary_base_evaluable.tolist() == [True, True]
        assert result.participants.unevaluable_alarm_count.iloc[0] == 1
        assert result.participants.base_event_count.iloc[0] == 1


def test_zero_common_exposure_retains_participant_and_all_emissions():
    normalized, config = fixture()
    config["data"]["minimum_days"] = 7
    replay = build_participant_replay(normalized, config, partition="selection")
    alarms = _alarms(replay.timeline, normalized.timestamp.iloc[[50, 59]])
    _, results = rematch_catalogues_on_common_frame(
        normalized, replay.timeline, {"primary": [], "alternative": []},
        alarms, horizon=60, glucose_cut=54,
    )
    for result in results.values():
        assert result.participants.source_id.tolist() == ["s"]
        assert result.participants.patient_days.iloc[0] == 0
        assert result.participants.unevaluable_alarm_count.iloc[0] == 2
        assert not result.alarms.unmatched.any()
        assert np.isnan(result.participants.fixed_sensitivity.iloc[0])


def _metrics(*, observed=False, sources=("A", "B"), different_population=False):
    rows = []
    for horizon in (30, 60):
        for cut in (54, 70):
            for arm, seeds in (("lstm", (17, 23)), ("persistence", (None,))):
                for seed in seeds:
                    for source in sources:
                        # B has no observed-only events at h60, while its h30
                        # support remains positive. The overall common set is A.
                        base = (
                            0 if observed and source == sources[-1] and horizon == 60
                            else 2
                        )
                        if arm == "persistence":
                            detected = 1
                        elif source == sources[0]:
                            detected = (1 if seed == 17 else 0) if cut == 54 else 2
                        else:
                            detected = 2 if cut == 54 else 0
                        participant = (
                            "observed-person" if observed and different_population
                            else "p"
                        )
                        rows.append({
                            "source_id": source, "participant_id": participant,
                            "arm": arm, "model_seed": seed, "horizon_minutes": horizon,
                            "glucose_cut": cut, "base_event_count": base,
                            "detected_event_count": detected if base else 0,
                        })
    return pd.DataFrame(rows)


def _overall(summary, scope, catalogue):
    rows = summary.contrasts
    return rows.loc[
        rows.scope.eq(scope) & rows.catalogue.eq(catalogue)
        & rows.aggregation.eq("equal_horizon_equal_model_seed"),
        "interaction_70_minus_54",
    ].iloc[0]


def test_common_support_is_fixed_across_horizons_cuts_seeds_and_catalogues():
    summary = summarize_common_source_support(
        _metrics(), _metrics(observed=True), source_order=("A", "B")
    )
    assert summary.audit["common_source_order"] == ("A",)
    diagnostic = summary.source_support.set_index(["horizon_minutes", "source_id"])
    assert diagnostic.loc[(30, "B"), "common_positive_support"]
    assert not diagnostic.loc[(60, "B"), "common_positive_support"]
    assert _overall(summary, "common_support", "primary") == pytest.approx(0.75)
    assert _overall(summary, "common_support", "observed_only") == pytest.approx(0.75)
    # Hand oracle: mean seed change in A is .75 and B is -1, then equal sources.
    assert _overall(summary, "full_sources", "primary") == pytest.approx(-0.125)
    assert np.isnan(_overall(summary, "full_sources", "observed_only"))
    common_points = summary.points.loc[summary.points.scope.eq("common_support")]
    assert common_points.source_count.eq(1).all()
    assert set(common_points.glucose_cut) == {54, 70}


def test_full_eleven_source_observed_only_undefined_is_preserved():
    sources = tuple(f"source-{index}" for index in range(11))
    summary = summarize_common_source_support(
        _metrics(sources=sources), _metrics(observed=True, sources=sources),
        source_order=sources,
    )
    assert len(summary.audit["common_source_order"]) == 10
    full = summary.points.loc[
        summary.points.scope.eq("full_sources")
        & summary.points.catalogue.eq("observed_only")
        & summary.points.aggregation.eq("equal_horizon_equal_model_seed")
    ]
    assert full.source_count.eq(11).all()
    assert full.fixed_sensitivity.isna().all()
    assert all(row == (sources[-1],) for row in full.undefined_sources)


def test_common_source_summary_reports_different_populations_without_intersection():
    summary = summarize_common_source_support(
        _metrics(), _metrics(observed=True, different_population=True),
        source_order=("A", "B"),
    )
    population = summary.population_audit
    assert not population.same_participant_population.any()
    assert population.shared_participants.eq(0).all()
    assert not summary.audit["participant_intersection_applied"]
    assert not summary.audit["onset_intersection_applied"]
    assert _overall(summary, "common_support", "observed_only") == pytest.approx(0.75)


@pytest.mark.parametrize(
    "defect", ["missing_source", "truth_changes", "duplicate", "missing_cut"]
)
def test_common_source_rejects_incomplete_or_condition_dependent_support(defect):
    primary, observed = _metrics(), _metrics(observed=True)
    if defect == "missing_source":
        observed = observed.drop(index=1)
    elif defect == "truth_changes":
        observed.loc[0, "base_event_count"] = 3
    elif defect == "duplicate":
        observed = pd.concat([observed, observed.iloc[[0]]], ignore_index=True)
    else:
        observed = observed.loc[~observed.glucose_cut.eq(70)]
    with pytest.raises(
        ValueError, match="source|support/population|duplicate|both horizons"
    ):
        summarize_common_source_support(primary, observed, source_order=("A", "B"))
