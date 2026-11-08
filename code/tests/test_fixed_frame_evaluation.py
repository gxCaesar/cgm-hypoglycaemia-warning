from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import glycoshift.fixed_frame_evaluation as fixed_frame_module
from glycoshift.fixed_frame_evaluation import (
    FixedFrameEvaluation,
    FixedFrameKey,
    FixedFrameSpec,
    evaluate_fixed_frame,
    fit_fixed_frame_alarm_policy,
    joint_fixed_frame_bootstrap,
    selection_fixed_frame_curve,
)

_ORIGIN = pd.Timestamp("2026-01-01T00:00:00Z")


def _timeline(rows: list[tuple]) -> pd.DataFrame:
    output = []
    for source, participant, minute, glucose, score, evaluable in rows:
        timestamp = _ORIGIN + pd.Timedelta(minutes=minute)
        output.append({
            "source_id": source, "participant_id": participant,
            "context_end": timestamp, "patient_day": timestamp.normalize(),
            "cgm_last": glucose, "score": score, "base_evaluable": evaluable,
            "anchor_cgm_observed": True, "anchor_cgm_interpolated": False,
            "anchor_cgm_censored": False,
        })
    return pd.DataFrame(output)


def _events(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "source_id": source, "participant_id": participant,
            "event_id": event, "event_onset": _ORIGIN + pd.Timedelta(minutes=minute),
        }
        for source, participant, event, minute in rows
    ], columns=["source_id", "participant_id", "event_id", "event_onset"])


def _exposure(timeline: pd.DataFrame) -> pd.DataFrame:
    return timeline.loc[timeline["base_evaluable"]].groupby(
        ["source_id", "participant_id"], as_index=False
    ).agg(patient_days=("patient_day", "nunique"), observation_minutes=(
        "context_end", lambda values: len(values) * 5.0
    ))


def _evaluate(
    timeline: pd.DataFrame,
    events: pd.DataFrame,
    *,
    cut: float = 54,
    horizon: int = 30,
    threshold: float = 0.5,
    exposure: pd.DataFrame | None = None,
) -> FixedFrameEvaluation:
    return evaluate_fixed_frame(
        timeline, events, _exposure(timeline) if exposure is None else exposure,
        spec=FixedFrameSpec(
            "score", horizon, tuple(sorted(timeline["source_id"].unique()))
        ),
        threshold=threshold, glucose_cut=cut,
    )


def _normal_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    return _timeline([
        ("A", "p1", 0, 65, 0.9, True), ("A", "p1", 60, 90, 0.1, True),
        ("A", "p2", 0, 65, 0.1, True), ("A", "p3", 0, 90, 0.9, True),
        ("B", "p1", 0, 90, 0.9, True), ("B", "p1", 60, 90, 0.9, True),
    ]), _events([
        ("A", "p1", "same", 10), ("A", "p1", "second", 70),
        ("A", "p2", "same", 10), ("B", "p1", "same", 10),
        ("B", "p1", "second", 70),
    ])


def test_fixed_event_and_exposure_denominators_and_aggregations() -> None:
    timeline, events = _normal_inputs()
    base = _evaluate(timeline, events)
    restricted = _evaluate(timeline, events, cut=70)
    assert base.summary["equal_source_participant_macro_fixed_sensitivity"] == 0.625
    assert base.summary["participant_pooled_sensitivity"] == 0.5
    assert base.summary["event_pooled_sensitivity"] == 3 / 5
    assert base.summary[
        "equal_source_participant_macro_unmatched_alarms_per_patient_day"
    ] == pytest.approx(1 / 6)
    assert base.summary["ppv"] == 3 / 4
    assert base.summary["mean_participant_ppv"] == pytest.approx(2 / 3)
    assert base.summary["pooled_unmatched_alarms_per_patient_day"] == 1 / 4
    assert base.summary["effective_24h_days"] == 30 / 1440
    assert restricted.summary["equal_source_participant_macro_fixed_sensitivity"] == 0.5
    for result in (base, restricted):
        assert result.summary["participant_count"] == 4
        assert result.summary["base_event_count"] == 5
        assert result.summary["patient_days"] == 4
        assert result.summary["observation_minutes"] == 30
    assert restricted.participants.loc[
        restricted.participants["source_id"].eq("A")
        & restricted.participants["participant_id"].eq("p2"), "fixed_sensitivity"
    ].item() == 0
    assert events["event_id"].nunique() < len(events)
    assert base.events["detected"].sum() == 3  # IDs scoped to source and participant.


def test_zero_eligible_source_and_all_zero_opportunities_are_retained() -> None:
    timeline, events = _normal_inputs()
    timeline.loc[timeline["source_id"].eq("A"), "cgm_last"] = 65
    result = _evaluate(timeline, events, cut=70)
    assert result.sources.set_index("source_id").loc["A", "fixed_sensitivity"] == 0
    assert result.summary["equal_source_participant_macro_fixed_sensitivity"] == 0.5
    none = _evaluate(timeline, events, cut=100)
    assert none.summary["base_event_count"] == 5
    assert none.summary["reachable_event_count"] == 0
    assert none.summary["detected_event_count"] == 0
    assert none.summary["equal_source_participant_macro_fixed_sensitivity"] == 0
    assert none.participants["conditional_sensitivity"].isna().all()
    assert none.alarms.empty
    assert none.participants["ppv"].isna().all()
    assert none.summary["pooled_unmatched_alarms_per_patient_day"] == 0


def test_posthoc_censoring_preserves_alarm_history_and_exact_30_boundary() -> None:
    timeline = _timeline([
        ("A", "p", minute, 90, 0.9, minute != 0)
        for minute in (0, 20, 30, 31, 60, 61, 62)
    ])
    events = _events([("A", "p", "e", 45)])
    result = _evaluate(timeline, events)
    minutes = (result.alarms["context_end"] - _ORIGIN).dt.total_seconds() / 60
    assert minutes.tolist() == [0, 31, 62]
    assert result.summary["emitted_alarm_count"] == 3
    assert result.summary["delivered_alarm_count"] == 2
    assert result.summary["unevaluable_alarm_count"] == 1
    assert result.summary["unmatched_alarm_count"] == 1
    assert result.alarms["unmatched"].tolist() == [False, False, True]
    uncensored = timeline.assign(base_evaluable=True)
    changed = _evaluate(uncensored, events)
    pd.testing.assert_series_equal(
        result.alarms["context_end"], changed.alarms["context_end"]
    )
    event = result.events.iloc[0]
    assert event["lead_minutes"] == 14
    assert event["coverage_5"] and event["coverage_10"]
    assert not event["coverage_15"] and not event["coverage_20"]


def test_future_flags_and_future_extension_do_not_change_past_emissions() -> None:
    timeline = _timeline([
        ("A", "p", minute, 90, 0.9, True) for minute in (0, 20, 40)
    ])
    events = _events([("A", "p", "e", 50)])
    timeline["currently_active"] = [False, True, True]
    original = _evaluate(timeline, events)
    changed = _evaluate(timeline.assign(currently_active=False), events)
    pd.testing.assert_frame_equal(original.alarms, changed.alarms)
    extension = pd.concat([
        timeline, _timeline([("A", "p", 100, 90, 0.9, True)])
    ], ignore_index=True)
    future = _evaluate(extension, events)
    pd.testing.assert_series_equal(
        original.alarms["context_end"],
        future.alarms.loc[future.alarms["context_end"] <= _ORIGIN + pd.Timedelta(
            minutes=40
        ), "context_end"],
    )


def test_greedy_one_to_one_matching_and_horizon_boundary() -> None:
    timeline = _timeline([
        ("A", "p", -10, 90, 0.0, True),
        ("A", "p", 0, 90, 0.9, True),
        ("A", "p", 31, 90, 0.9, True),
    ])
    event = _events([("A", "p", "e", 50)])
    result = _evaluate(timeline, event, horizon=60)
    assert result.alarms["matched"].tolist() == [True, False]
    assert result.summary["detected_event_count"] == 1
    two = _events([("A", "p", "e", 50), ("A", "p", "f", 55)])
    result = _evaluate(timeline, two, horizon=60)
    assert result.alarms["matched_event_id"].tolist() == ["e", "f"]
    boundary = _events([("A", "p", "e", 0), ("A", "p", "f", 30)])
    result = _evaluate(timeline.iloc[:2], boundary)
    assert result.alarms["matched_event_id"].tolist() == ["f"]
    assert result.events["detected"].tolist() == [False, True]
    assert result.alarms["lead_minutes"].item() == 30


def test_late_matches_remain_matches_and_coverage_uses_unchanged_matches() -> None:
    timeline = _timeline([
        ("A", "p", 0, 90, 0.9, True), ("A", "p", 60, 90, 0.1, True)
    ])
    result = _evaluate(timeline, _events([
        ("A", "p", "late", 3), ("A", "p", "miss", 70)
    ]))
    assert result.summary["late_detection_count"] == 1
    assert result.summary["missed_event_count"] == 1
    assert result.summary["unmatched_alarm_count"] == 0
    assert result.summary["coverage_5_event_count"] == 0
    assert result.summary["ppv"] == 1
    assert result.alarms["cgm_last"].tolist() == [90]


@pytest.mark.parametrize("flag", [
    "anchor_cgm_observed", "anchor_cgm_interpolated", "anchor_cgm_censored",
    "base_evaluable",
])
def test_anchor_provenance_and_evaluability_are_strict_booleans(flag: str) -> None:
    timeline, events = _normal_inputs()
    exposure = _exposure(timeline)
    with pytest.raises(ValueError, match="missing columns"):
        _evaluate(timeline.drop(columns=flag), events, exposure=exposure)
    timeline[flag] = timeline[flag].astype(object)
    timeline.loc[0, flag] = "False"
    with pytest.raises(ValueError, match="complete boolean"):
        _evaluate(timeline, events, exposure=exposure)


def test_online_anchor_provenance_and_minimum_54_gate() -> None:
    timeline = _timeline([
        ("A", "p", minute, glucose, 0.9, True)
        for minute, glucose in ((0, 53), (40, 54), (80, 90), (120, 90), (160, 90))
    ])
    timeline.loc[2, "anchor_cgm_observed"] = False
    timeline.loc[3, "anchor_cgm_observed"] = False
    timeline.loc[3, "anchor_cgm_interpolated"] = True
    timeline.loc[4, "anchor_cgm_censored"] = True
    result = _evaluate(timeline, _events([]), cut=0)
    assert result.summary["eligible_decision_count"] == 1
    assert result.alarms["cgm_last"].tolist() == [54]
    assert not result.summary["sensitivity_defined"]
    assert result.summary["undefined_sensitivity_sources"] == ["A"]


def test_calendar_day_identifier_preserves_adapter_local_date() -> None:
    timeline, events = _normal_inputs()
    timeline["patient_day"] = pd.Timestamp("2026-01-01", tz="Asia/Shanghai")
    result = _evaluate(timeline, events)
    assert result.participants["base_patient_days"].iloc[0] == (
        pd.Timestamp("2026-01-01", tz="UTC"),
    )


@pytest.mark.parametrize("kind", [
    "duplicate_decision", "duplicate_event", "conflicting_onset", "invalid_onset",
    "foreign_event", "missing_participant", "missing_source", "bad_days",
    "bad_minutes", "no_lead_in", "nonfinite_score", "nonfinite_cgm",
    "contradictory_anchor", "horizon",
])
def test_known_bad_truth_membership_and_timestamps_fail_closed(kind: str) -> None:
    timeline, events = _normal_inputs()
    exposure = _exposure(timeline)
    if kind == "duplicate_decision":
        timeline = pd.concat([timeline, timeline.iloc[:1]])
    elif kind in ("duplicate_event", "conflicting_onset"):
        row = events.iloc[:1].copy()
        if kind == "conflicting_onset":
            row["event_onset"] += pd.Timedelta(minutes=1)
        events = pd.concat([events, row])
    elif kind == "invalid_onset":
        events.loc[0, "event_onset"] = pd.NaT
    elif kind == "foreign_event":
        events.loc[0, "participant_id"] = "not-a-base-participant"
    elif kind == "missing_participant":
        exposure = exposure.iloc[1:]
    elif kind == "missing_source":
        timeline = timeline.loc[timeline["source_id"].eq("A")]
        # Explicit two-source spec below must still reject the absent source.
    elif kind == "bad_days":
        exposure.loc[0, "patient_days"] = 2
    elif kind == "bad_minutes":
        exposure.loc[0, "observation_minutes"] = 1441
    elif kind == "no_lead_in":
        events.loc[0, "event_onset"] = _ORIGIN + pd.Timedelta(minutes=300)
    elif kind == "nonfinite_score":
        timeline.loc[0, "score"] = np.nan
    elif kind == "nonfinite_cgm":
        timeline["cgm_last"] = timeline["cgm_last"].astype(float)
        timeline.loc[0, "cgm_last"] = np.inf
    elif kind == "contradictory_anchor":
        timeline.loc[0, "anchor_cgm_interpolated"] = True
    elif kind == "horizon":
        events["horizon_minutes"] = 60
    with pytest.raises(ValueError):
        evaluate_fixed_frame(
            timeline, events, exposure, spec=FixedFrameSpec("score", 30, ("A", "B")),
            threshold=0.5,
        )


def test_calibration_fit_preserves_unique_candidates_and_high_threshold_tie() -> None:
    timeline = _timeline([
        (source, "p", minute, 90, score, True)
        for source in ("A", "B") for minute, score in ((0, 0.9), (60, 0.6))
    ])
    events = _events([(source, "p", "e", 10) for source in ("A", "B")])
    policy = fit_fixed_frame_alarm_policy(
        timeline, events, _exposure(timeline),
        spec=FixedFrameSpec("score", 30, ("A", "B")), partition="calibration",
        nominal_fapd=1,
    )
    assert policy.threshold == 0.9
    assert policy.threshold_count == 3
    assert policy.achieved_equal_source_participant_macro_fapd == 0
    assert policy.achieved_equal_source_participant_macro_fixed_sensitivity == 1
    assert policy.tie_rule == "maximum_fixed_sensitivity_then_highest_threshold"
    assert "sha256" not in str(policy.as_dict())
    for partition in ("selection", "external"):
        with pytest.raises(ValueError, match="calibration only"):
            fit_fixed_frame_alarm_policy(
                timeline, events, _exposure(timeline),
                spec=FixedFrameSpec("score", 30, ("A", "B")), partition=partition,
            )
    timeline["partition"] = ["calibration", "selection", "calibration", "calibration"]
    with pytest.raises(ValueError, match="mixed partitions"):
        fit_fixed_frame_alarm_policy(
            timeline, events, _exposure(timeline),
            spec=FixedFrameSpec("score", 30, ("A", "B")), partition="calibration",
        )


def test_calibration_zero_opportunity_source_and_no_alarm_sentinel() -> None:
    timeline, events = _normal_inputs()
    timeline.loc[timeline["source_id"].eq("A"), "cgm_last"] = 65
    kwargs = {
        "spec": FixedFrameSpec("score", 30, ("A", "B")), "partition": "calibration"
    }
    policy = fit_fixed_frame_alarm_policy(
        timeline, events, _exposure(timeline), glucose_cut=70, **kwargs
    )
    assert policy.achieved_equal_source_participant_macro_fixed_sensitivity == 0.5
    none = fit_fixed_frame_alarm_policy(
        timeline, events, _exposure(timeline), glucose_cut=100, **kwargs
    )
    assert none.threshold > timeline["score"].max()
    assert none.threshold_count == 1
    assert none.achieved_equal_source_participant_macro_fixed_sensitivity == 0
    assert none.participant_count == 4


@pytest.mark.parametrize("horizon", [30, 60])
@pytest.mark.parametrize("cut", [54, 70, 100])
@pytest.mark.parametrize("candidate_count", [None, 4])
def test_incremental_sweep_matches_full_ledger_at_every_candidate(
    horizon: int, cut: float, candidate_count: int | None,
) -> None:
    rng = np.random.default_rng(71)
    timeline = _timeline([
        (source, participant, minute,
         65 if source == "A" or rng.random() < 0.25 else 90,
         rng.choice([0.1, 0.3, 0.5, 0.7, 0.9]), minute % 40 != 0)
        for source in ("A", "B") for participant in ("p0", "p1", "zero")
        for minute in range(0, 181, 5)
    ])
    events = _events([
        (source, participant, str(index), onset)
        for source in ("A", "B") for participant in ("p0", "p1")
        for index, onset in enumerate((35, 95, 155))
    ])
    exposure = _exposure(timeline)
    spec = FixedFrameSpec("score", horizon, ("A", "B"))
    curve = selection_fixed_frame_curve(
        timeline, events, exposure, spec=spec, glucose_cut=cut,
        candidate_count=candidate_count, include_raw_points=True,
    )
    assert len(curve["raw_points"]) == curve["threshold_count"]
    assert curve["raw_points"][-1]["threshold"] > timeline["score"].max()
    for point in curve["raw_points"]:
        expected = _evaluate(
            timeline, events, horizon=horizon, cut=cut, threshold=point["threshold"]
        ).summary
        assert point["fixed_sensitivity"] == pytest.approx(
            expected["equal_source_participant_macro_fixed_sensitivity"], abs=1e-12
        )
        assert point["achieved_fapd"] == pytest.approx(
            expected["equal_source_participant_macro_unmatched_alarms_per_patient_day"],
            abs=1e-12,
        )
    policy = fit_fixed_frame_alarm_policy(
        timeline, events, exposure, spec=spec, glucose_cut=cut,
        candidate_count=candidate_count, partition="calibration", nominal_fapd=0.5,
    )
    feasible = [point for point in curve["raw_points"] if point["achieved_fapd"] <= 0.5]
    best = max(feasible, key=lambda point: (
        point["fixed_sensitivity"], point["threshold"]
    ))
    assert policy.threshold == best["threshold"]


def test_right_continuous_fixed_frame_frontier_has_hand_calculated_area() -> None:
    timeline, events = _normal_inputs()
    curve = selection_fixed_frame_curve(
        timeline, events, _exposure(timeline),
        spec=FixedFrameSpec("score", 30, ("A", "B")),
    )
    assert curve["raw_auc"] == pytest.approx(11 / 6)
    assert curve["normalized_auc"] == pytest.approx(11 / 12)
    assert curve["best_sensitivity_within_range"] == 1
    assert curve["fapd_at_best_sensitivity"] == pytest.approx(1 / 6)


def test_fit_builds_full_ledger_only_once_and_curve_uses_no_ledger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timeline, events = _normal_inputs()
    original = fixed_frame_module._evaluate_validated
    calls = []

    def wrapped(*args: object, **kwargs: object) -> FixedFrameEvaluation:
        calls.append(kwargs["threshold"])
        return original(*args, **kwargs)

    monkeypatch.setattr(fixed_frame_module, "_evaluate_validated", wrapped)
    spec = FixedFrameSpec("score", 30, ("A", "B"))
    selection_fixed_frame_curve(timeline, events, _exposure(timeline), spec=spec)
    assert not calls
    fit_fixed_frame_alarm_policy(
        timeline, events, _exposure(timeline), spec=spec, partition="calibration"
    )
    assert len(calls) == 1


def _joint_fixture(*, different_positivity: bool = False) -> dict:
    evaluations = {}
    for horizon in (30, 60):
        for seed in (17, 23):
            for arm in ("lstm", "control"):
                for cut in (54, 70):
                    rows = []
                    event_rows = []
                    for source in ("A", "B"):
                        for participant in ("p0", "p1", "zero"):
                            positive = participant != "zero"
                            if different_positivity:
                                positive = participant in (
                                    ("p0", "zero") if horizon == 30 else ("p1", "zero")
                                )
                            target = "p0" if (horizon == 30) == (seed == 17) else "p1"
                            detected = arm == "lstm" and participant == target
                            rows.append((
                                source, participant, 0, 90, float(detected), True
                            ))
                            if positive:
                                event_rows.append((source, participant, "e", 10))
                    timeline, events = _timeline(rows), _events(event_rows)
                    evaluations[FixedFrameKey(arm, cut, horizon, seed)] = _evaluate(
                        timeline, events, cut=cut, horizon=horizon
                    )
    return evaluations


def _bootstrap(evaluations: dict, **kwargs: object) -> dict:
    contrasts = {
        "base": {("lstm", 54): 1, ("control", 54): -1},
        "restricted": {("lstm", 70): 1, ("control", 70): -1},
        "interaction": {
            ("lstm", 70): 1, ("control", 70): -1,
            ("lstm", 54): -1, ("control", 54): 1,
        },
    }
    return joint_fixed_frame_bootstrap(
        evaluations, contrasts=contrasts, source_order=("A", "B"),
        model_seed_order=(17, 23), replicates=40, bootstrap_seed=20260810,
        include_replicates=True, **kwargs,
    )


def test_joint_two_horizon_two_seed_draw_is_shared_and_not_concatenated() -> None:
    result = _bootstrap(_joint_fixture())
    for name in ("base", "restricted"):
        contrast = result["contrasts"][name]
        assert contrast["estimate"] == 0.5
        assert contrast["ci_low"] == 0.5
        assert contrast["ci_high"] == 0.5
        assert set(contrast["replicate_estimates"]) == {0.5}
        cells = np.asarray(contrast["replicate_horizon_seed_estimates"])
        assert np.all(cells[:, 0, 0] + cells[:, 0, 1] == 1)
        assert np.all(cells[:, 0, 0] == cells[:, 1, 1])
    assert result["contrasts"]["interaction"]["ci_low"] == 0
    assert result["contrasts"]["interaction"]["ci_high"] == 0
    assert result["population"] == "event_positive_in_at_least_one_horizon"
    assert [entry["participant_count"] for entry in result["stratum_counts"]] == [2, 2]


def test_horizon_positivity_patterns_preserve_union_and_stratum_counts() -> None:
    result = _bootstrap(_joint_fixture(different_positivity=True))
    assert result["stratification"] == "source_by_horizon_positivity_pattern"
    assert len(result["stratum_counts"]) == 6
    assert all(entry["participant_count"] == 1 for entry in result["stratum_counts"])
    assert result["contrasts"]["base"]["estimate"] == 0.25
    assert result["contrasts"]["base"]["ci_low"] == 0.25
    assert result["contrasts"]["base"]["ci_high"] == 0.25


def test_burden_draw_separately_includes_zero_event_participants() -> None:
    result = _bootstrap(_joint_fixture(), metric="delivered_alarms_per_patient_day")
    assert result["population"] == "all_base_participants"
    assert result["stratification"] == "source"
    assert [entry["participant_count"] for entry in result["stratum_counts"]] == [3, 3]
    assert result["contrasts"]["base"]["estimate"] == pytest.approx(1 / 3)
    assert len(set(result["contrasts"]["base"]["replicate_estimates"])) > 1


@pytest.mark.parametrize("kind", [
    "missing_seed", "missing_source", "missing_participant", "event_count",
    "event_id", "event_onset", "patient_days", "observation_minutes",
])
def test_joint_bootstrap_known_bad_comparison_truth_fails(kind: str) -> None:
    evaluations = _joint_fixture()
    key = FixedFrameKey("lstm", 70, 30, 23)
    original = evaluations[key]
    if kind == "missing_seed":
        del evaluations[key]
    elif kind in ("missing_source", "missing_participant"):
        table = original.participants.copy()
        table = table.loc[table["source_id"].eq("A")] if kind == "missing_source" else (
            table.iloc[1:]
        )
        evaluations[key] = replace(original, participants=table)
    elif kind in ("event_id", "event_onset"):
        table = original.events.copy()
        if kind == "event_id":
            table.loc[0, "event_id"] = "wrong-event"
        else:
            table.loc[0, "event_onset"] += pd.Timedelta(minutes=1)
        evaluations[key] = replace(original, events=table)
    else:
        table = original.participants.copy()
        column = "base_event_count" if kind == "event_count" else kind
        table.loc[0, column] += 1
        evaluations[key] = replace(original, participants=table)
    with pytest.raises(ValueError):
        _bootstrap(evaluations)


def test_bootstrap_rejects_undefined_source_base_sensitivity() -> None:
    evaluations = _joint_fixture()
    for key, original in list(evaluations.items()):
        table = original.participants.copy()
        table.loc[table["source_id"].eq("B"), [
            "base_event_count", "detected_event_count"
        ]] = 0
        table.loc[table["source_id"].eq("B"), "fixed_sensitivity"] = np.nan
        events = original.events.loc[original.events["source_id"].eq("A")]
        evaluations[key] = replace(original, participants=table, events=events)
    with pytest.raises(ValueError, match="base sensitivity undefined: source=B"):
        _bootstrap(evaluations)


def test_default_bootstrap_requires_all_five_model_seeds() -> None:
    evaluations = {
        key: value for key, value in _joint_fixture().items() if key.glucose_cut == 54
    }
    with pytest.raises(ValueError, match="model seeds"):
        joint_fixed_frame_bootstrap(
            evaluations, source_order=("A", "B"),
            contrasts={"base": {("lstm", 54): 1, ("control", 54): -1}},
        )
