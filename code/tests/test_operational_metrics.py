from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from glycoshift.operational_metrics import (
    OperationalEndpointSpec,
    fit_operational_alarm_policy,
    operational_endpoint_metrics,
    participant_alarm_table,
    selection_operational_curve,
    source_stratified_paired_multiseed_operational_bootstrap_difference,
    source_stratified_paired_operational_bootstrap_difference,
)


def _spec(*, sources: tuple[str, ...] = ("A", "B")) -> OperationalEndpointSpec:
    return OperationalEndpointSpec(
        score_column="score",
        label_column="severe_hypo_30_label",
        event_id_column="severe_hypo_30_event_id",
        event_onset_column="severe_hypo_30_event_onset",
        active_column="severe_hypo_currently_active",
        horizon_minutes=30,
        source_order=sources,
    )


def _event_row(
    *,
    source: str,
    participant: str,
    day: int,
    event: int | None,
    score: float,
) -> dict[str, object]:
    context_end = pd.Timestamp("2026-01-01T00:00:00Z") + pd.Timedelta(days=day)
    event_id = None if event is None else f"event-{event}"
    return {
        "source_id": source,
        "participant_id": participant,
        "context_end": context_end,
        "patient_day": context_end.floor("D"),
        "score": score,
        "severe_hypo_30_label": int(event is not None),
        "severe_hypo_30_event_id": event_id,
        "severe_hypo_30_event_onset": (
            pd.NaT
            if event is None
            else context_end + pd.Timedelta(minutes=10)
        ),
        "severe_hypo_currently_active": False,
    }


def test_operational_endpoint_is_participant_macro_then_equal_source() -> None:
    rows = [
        _event_row(source="A", participant="p1", day=0, event=0, score=0.9),
        _event_row(source="A", participant="p1", day=1, event=1, score=0.1),
        _event_row(source="A", participant="p2", day=0, event=0, score=0.1),
        _event_row(source="A", participant="p3", day=0, event=None, score=0.9),
    ]
    rows.extend(
        _event_row(
            source="B",
            participant="p1",
            day=day,
            event=day,
            score=0.9,
        )
        for day in range(10)
    )
    metrics = operational_endpoint_metrics(
        pd.DataFrame(rows),
        spec=_spec(),
        threshold=0.5,
    )

    assert metrics[
        "equal_source_participant_macro_event_sensitivity"
    ] == pytest.approx(0.625)
    assert metrics[
        "equal_source_participant_macro_false_alarms_per_patient_day"
    ] == pytest.approx(1 / 6)
    assert metrics["event_positive_participant_count"] == 3
    assert metrics["participant_count"] == 4
    assert metrics["event_count"] == 13
    assert metrics["detected_event_count"] == 11
    assert 11 / 13 != pytest.approx(0.625)


def test_operational_matching_scopes_raw_event_ids_to_participant() -> None:
    rows = [
        _event_row(source=source, participant="same-id", day=0, event=0, score=1.0)
        for source in ("A", "B")
    ]
    metrics = operational_endpoint_metrics(
        pd.DataFrame(rows),
        spec=_spec(),
        threshold=0.5,
    )
    assert metrics["event_count"] == 2
    assert metrics["detected_event_count"] == 2
    assert metrics[
        "equal_source_participant_macro_event_sensitivity"
    ] == 1.0


def test_operational_contract_fails_on_inconsistent_truth_or_missing_source() -> None:
    frame = pd.DataFrame(
        [
            _event_row(source="A", participant="p", day=0, event=0, score=0.9),
            _event_row(source="B", participant="p", day=0, event=0, score=0.9),
        ]
    )
    inconsistent = frame.copy()
    inconsistent.loc[0, "severe_hypo_30_label"] = 0
    with pytest.raises(ValueError, match="label and event ID"):
        operational_endpoint_metrics(inconsistent, spec=_spec(), threshold=0.5)
    with pytest.raises(ValueError, match="source set mismatch"):
        operational_endpoint_metrics(
            frame.loc[frame["source_id"] == "A"],
            spec=_spec(),
            threshold=0.5,
        )
    with pytest.raises(ValueError, match="no event-positive participant"):
        operational_endpoint_metrics(
            pd.DataFrame(
                [
                    _event_row(
                        source="A",
                        participant="p",
                        day=0,
                        event=0,
                        score=0.9,
                    ),
                    _event_row(
                        source="B",
                        participant="p",
                        day=0,
                        event=None,
                        score=0.9,
                    ),
                ]
            ),
            spec=_spec(),
            threshold=0.5,
        )


def test_alarm_policy_uses_strict_no_alarm_sentinel() -> None:
    rows: list[dict[str, object]] = []
    for source in ("A", "B"):
        rows.extend(
            (
                _event_row(
                    source=source,
                    participant="p",
                    day=0,
                    event=0,
                    score=1.0,
                ),
                _event_row(
                    source=source,
                    participant="p",
                    day=1,
                    event=None,
                    score=1.0,
                ),
            )
        )
    frame = pd.DataFrame(rows)
    policy = fit_operational_alarm_policy(
        frame,
        spec=_spec(),
        nominal_fapd=0.0,
        candidate_count=None,
    )

    assert policy.threshold > 1.0
    assert policy.achieved_equal_source_participant_macro_fapd == 0.0
    assert policy.achieved_equal_source_participant_macro_event_sensitivity == 0.0
    assert len(policy.policy_sha256) == 64


def test_selection_curve_uses_unique_scores_and_feasibility_envelope() -> None:
    rows: list[dict[str, object]] = []
    for source in ("A", "B"):
        rows.extend(
            (
                _event_row(
                    source=source,
                    participant="p",
                    day=0,
                    event=0,
                    score=1.0,
                ),
                _event_row(
                    source=source,
                    participant="p",
                    day=1,
                    event=None,
                    score=0.0,
                ),
            )
        )
    curve = selection_operational_curve(
        pd.DataFrame(rows),
        spec=_spec(),
    )
    assert curve["threshold_count"] == 3
    assert curve["raw_auc"] == pytest.approx(2.0)
    assert curve["normalized_auc"] == pytest.approx(1.0)
    assert curve["candidate_rule"] == "all_unique_scores_plus_no_alarm_sentinel"


def test_source_stratified_paired_bootstrap_equal_weights_sources() -> None:
    rows: list[dict[str, object]] = []
    for index in range(10):
        rows.append(
            {
                "source_id": "A",
                "participant_id": f"p{index}",
                "event_count": 1,
                "patient_days": 1,
                "event_sensitivity": 1.0,
            }
        )
    rows.append(
        {
            "source_id": "B",
            "participant_id": "p0",
            "event_count": 1,
            "patient_days": 1,
            "event_sensitivity": 0.0,
        }
    )
    candidate = pd.DataFrame(rows)
    reference = candidate.copy()
    reference["event_sensitivity"] = 0.0
    result = source_stratified_paired_operational_bootstrap_difference(
        candidate,
        reference,
        source_order=("A", "B"),
        replicates=100,
        seed=17,
    )
    assert result["estimate"] == pytest.approx(0.5)
    assert result["ci_low"] == pytest.approx(0.5)
    assert result["ci_high"] == pytest.approx(0.5)
    assert not math.isclose(result["estimate"], 10 / 11)


def test_paired_bootstrap_samples_event_positive_population() -> None:
    rows: list[dict[str, object]] = []
    for source in ("A", "B"):
        rows.append(
            {
                "source_id": source,
                "participant_id": "event-positive",
                "event_count": 1,
                "patient_days": 1,
                "event_sensitivity": 1.0,
            }
        )
        rows.extend(
            {
                "source_id": source,
                "participant_id": f"zero-{index}",
                "event_count": 0,
                "patient_days": 1,
                "event_sensitivity": np.nan,
            }
            for index in range(9)
        )
    candidate = pd.DataFrame(rows)
    reference = candidate.copy()
    reference.loc[reference["event_count"] > 0, "event_sensitivity"] = 0.0
    result = source_stratified_paired_operational_bootstrap_difference(
        candidate,
        reference,
        source_order=("A", "B"),
        replicates=100,
        seed=17,
    )
    assert result["estimate"] == 1.0
    assert result["ci_low"] == 1.0
    assert result["ci_high"] == 1.0


def _multiseed_participant_table(
    differences: dict[tuple[str, str], float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    for (source, participant), difference in differences.items():
        reference_rows.append(
            {
                "source_id": source,
                "participant_id": participant,
                "event_count": 1,
                "patient_days": 2,
                "event_sensitivity": 0.0,
            }
        )
        candidate_rows.append(
            {
                "source_id": source,
                "participant_id": participant,
                "event_count": 1,
                "patient_days": 2,
                "event_sensitivity": difference,
            }
        )
    return pd.DataFrame(candidate_rows), pd.DataFrame(reference_rows)


def test_multiseed_bootstrap_reuses_one_participant_draw_across_seeds() -> None:
    seed_17 = _multiseed_participant_table(
        {
            (source, participant): float(participant == "p0")
            for source in ("A", "B")
            for participant in ("p0", "p1")
        }
    )
    seed_23 = _multiseed_participant_table(
        {
            (source, participant): float(participant == "p1")
            for source in ("A", "B")
            for participant in ("p0", "p1")
        }
    )
    result = source_stratified_paired_multiseed_operational_bootstrap_difference(
        {17: seed_17[0], 23: seed_23[0]},
        {17: seed_17[1], 23: seed_23[1]},
        source_order=("A", "B"),
        model_seed_order=(17, 23),
        replicates=200,
        bootstrap_seed=101,
    )
    assert result["estimate"] == pytest.approx(0.5)
    assert result["ci_low"] == pytest.approx(0.5)
    assert result["ci_high"] == pytest.approx(0.5)
    assert result["model_seeds"] == [17, 23]
    assert result["resampling_unit"] == (
        "event_positive_source_scoped_participant"
    )


def test_multiseed_bootstrap_rejects_seed_or_truth_drift() -> None:
    seed_17 = _multiseed_participant_table(
        {
            (source, participant): 0.2
            for source in ("A", "B")
            for participant in ("p0", "p1")
        }
    )
    seed_23 = _multiseed_participant_table(
        {
            (source, participant): 0.4
            for source in ("A", "B")
            for participant in ("p0", "p1")
        }
    )
    with pytest.raises(ValueError, match="exactly cover model_seed_order"):
        source_stratified_paired_multiseed_operational_bootstrap_difference(
            {17: seed_17[0]},
            {17: seed_17[1], 23: seed_23[1]},
            source_order=("A", "B"),
            model_seed_order=(17, 23),
            replicates=10,
            bootstrap_seed=101,
        )

    changed_candidate = seed_23[0].copy()
    changed_reference = seed_23[1].copy()
    changed_candidate.loc[0, "event_count"] = 2
    changed_reference.loc[0, "event_count"] = 2
    with pytest.raises(ValueError, match="truth or participant order differs"):
        source_stratified_paired_multiseed_operational_bootstrap_difference(
            {17: seed_17[0], 23: changed_candidate},
            {17: seed_17[1], 23: changed_reference},
            source_order=("A", "B"),
            model_seed_order=(17, 23),
            replicates=10,
            bootstrap_seed=101,
        )


def test_participant_table_keeps_zero_event_participant_for_burden() -> None:
    frame = pd.DataFrame(
        [
            _event_row(source="A", participant="p1", day=0, event=0, score=0.9),
            _event_row(source="A", participant="p2", day=0, event=None, score=0.9),
            _event_row(source="B", participant="p1", day=0, event=0, score=0.9),
        ]
    )
    table = participant_alarm_table(frame, spec=_spec(), threshold=0.5)
    zero_event = table.loc[
        (table["source_id"] == "A") & (table["participant_id"] == "p2")
    ].iloc[0]
    assert np.isnan(zero_event["event_sensitivity"])
    assert zero_event["false_alarms_per_patient_day"] == 1.0


def _random_sequence_frame(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2026-02-01T00:00:00Z")
    rows: list[dict[str, object]] = []
    for source in ("A", "B"):
        for participant_index in range(3):
            participant = f"p{participant_index}"
            event_minutes = (
                [75 + participant_index * 5, 205 + participant_index * 5]
                if participant_index < 2
                else []
            )
            for minute in range(0, 300, 5):
                timestamp = base + pd.Timedelta(minutes=minute)
                active = any(
                    onset <= minute <= onset + 10 for onset in event_minutes
                )
                future = next(
                    (
                        (event_index, onset)
                        for event_index, onset in enumerate(event_minutes)
                        if minute < onset <= minute + 30
                    ),
                    None,
                )
                rows.append(
                    {
                        "source_id": source,
                        "participant_id": participant,
                        "context_end": timestamp,
                        "patient_day": timestamp.floor("D"),
                        "score": float(rng.integers(0, 9) / 8),
                        "severe_hypo_30_label": int(future is not None),
                        "severe_hypo_30_event_id": (
                            None
                            if future is None
                            else f"{source}:{participant}:e{future[0]}"
                        ),
                        "severe_hypo_30_event_onset": (
                            pd.NaT
                            if future is None
                            else base + pd.Timedelta(minutes=future[1])
                        ),
                        "severe_hypo_currently_active": active,
                    }
                )
    return pd.DataFrame(rows)


@pytest.mark.parametrize("seed", range(10))
def test_incremental_unique_score_curve_matches_fixed_threshold_oracle(
    seed: int,
) -> None:
    frame = _random_sequence_frame(seed)
    curve = selection_operational_curve(
        frame,
        spec=_spec(),
        include_raw_points=True,
    )
    assert curve["threshold_count"] == frame["score"].nunique() + 1
    for point in curve["raw_points"]:
        expected = operational_endpoint_metrics(
            frame,
            spec=_spec(),
            threshold=point["threshold"],
        )
        assert point["event_sensitivity"] == pytest.approx(
            expected["equal_source_participant_macro_event_sensitivity"],
            abs=1e-12,
        )
        assert point["achieved_fapd"] == pytest.approx(
            expected[
                "equal_source_participant_macro_false_alarms_per_patient_day"
            ],
            abs=1e-12,
        )


def test_incremental_quantile_curve_matches_fixed_threshold_oracle() -> None:
    frame = _random_sequence_frame(101)
    curve = selection_operational_curve(
        frame,
        spec=_spec(),
        candidate_count=5,
        include_raw_points=True,
    )
    for point in curve["raw_points"]:
        expected = operational_endpoint_metrics(
            frame,
            spec=_spec(),
            threshold=point["threshold"],
        )
        assert point["event_sensitivity"] == pytest.approx(
            expected["equal_source_participant_macro_event_sensitivity"],
            abs=1e-12,
        )
        assert point["achieved_fapd"] == pytest.approx(
            expected[
                "equal_source_participant_macro_false_alarms_per_patient_day"
            ],
            abs=1e-12,
        )
