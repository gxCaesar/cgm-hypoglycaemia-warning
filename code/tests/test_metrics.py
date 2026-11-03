import math

import pandas as pd
import pytest

from glycoshift.metrics import (
    alarm_metrics,
    conformal_classification_metrics,
    holm_adjust,
    paired_participant_bootstrap_difference,
    participant_bootstrap_metrics,
    regression_metrics,
    select_alarm_threshold,
    selective_risk_curve,
)


def test_alarm_collapse_never_replaces_early_alarm_using_future_label() -> None:
    predictions = pd.DataFrame(
        {
            "source_id": ["s", "s"],
            "participant_id": ["p", "p"],
            "context_end": pd.to_datetime(
                ["2026-01-01T00:00:00Z", "2026-01-01T00:20:00Z"]
            ),
            "patient_day": pd.to_datetime(["2026-01-01", "2026-01-01"]),
            "hypo_10_probability": [0.9, 0.9],
            "hypo_10_label": [0, 1],
            "hypo_10_event_id": [None, "s:p:hypo:0000"],
            "hypo_10_event_onset": [pd.NaT, pd.Timestamp("2026-01-01T00:25:00Z")],
        }
    )
    metrics = alarm_metrics(
        predictions,
        probability_column="hypo_10_probability",
        label_column="hypo_10_label",
        event_id_column="hypo_10_event_id",
        event_onset_column="hypo_10_event_onset",
        horizon_minutes=10,
        threshold=0.5,
        collapse_minutes=30,
    )
    assert metrics["alarm_count"] == 1
    assert metrics["detected_event_count"] == 0
    assert metrics["false_alarms_per_patient_day"] == 1


def test_one_alarm_matches_at_most_one_event() -> None:
    onset = pd.Timestamp("2026-01-01T00:20:00Z")
    predictions = pd.DataFrame(
        {
            "source_id": ["s", "s"],
            "participant_id": ["p", "p"],
            "context_end": pd.to_datetime(
                ["2026-01-01T00:00:00Z", "2026-01-01T00:05:00Z"]
            ),
            "patient_day": pd.to_datetime(["2026-01-01", "2026-01-01"]),
            "hypo_30_probability": [0.9, 0.1],
            "hypo_30_label": [1, 1],
            "hypo_30_event_id": ["e1", "e2"],
            "hypo_30_event_onset": [onset, onset + pd.Timedelta(minutes=5)],
        }
    )
    metrics = alarm_metrics(
        predictions,
        probability_column="hypo_30_probability",
        label_column="hypo_30_label",
        event_id_column="hypo_30_event_id",
        event_onset_column="hypo_30_event_onset",
        horizon_minutes=30,
        threshold=0.5,
    )
    assert metrics["detected_event_count"] == 1
    assert metrics["event_count"] == 2


def test_alarm_threshold_has_a_strict_no_alarm_sentinel() -> None:
    predictions = pd.DataFrame(
        {
            "source_id": ["s"],
            "participant_id": ["p"],
            "context_end": pd.to_datetime(["2026-01-01T00:00:00Z"]),
            "patient_day": pd.to_datetime(["2026-01-01"]),
            "hypo_30_probability": [1.0],
            "hypo_30_label": [0],
            "hypo_30_event_id": [None],
            "hypo_30_event_onset": [pd.NaT],
        }
    )
    threshold = select_alarm_threshold(
        predictions,
        probability_column="hypo_30_probability",
        label_column="hypo_30_label",
        event_id_column="hypo_30_event_id",
        event_onset_column="hypo_30_event_onset",
        horizon_minutes=30,
        target_false_alarms_per_day=0.0,
    )
    metrics = alarm_metrics(
        predictions,
        probability_column="hypo_30_probability",
        label_column="hypo_30_label",
        event_id_column="hypo_30_event_id",
        event_onset_column="hypo_30_event_onset",
        horizon_minutes=30,
        threshold=threshold,
    )
    assert threshold > 1.0
    assert metrics["false_alarms_per_patient_day"] == 0.0


def test_holm_adjustment_is_monotone_and_preserves_missing_values() -> None:
    adjusted = holm_adjust(
        {
            "hypo_30": 0.01,
            "hypo_60": 0.04,
            "hyper_30": 0.03,
            "hyper_60": float("nan"),
        }
    )
    assert adjusted["hypo_30"] == 0.03
    assert adjusted["hyper_30"] == 0.06
    assert adjusted["hypo_60"] == 0.06
    assert math.isnan(adjusted["hyper_60"])


def test_conformal_and_selective_metrics_have_explicit_coverage() -> None:
    labels = [0, 0, 1, 1]
    probabilities = [0.05, 0.40, 0.60, 0.95]
    conformal = conformal_classification_metrics(
        labels,
        probabilities,
        threshold=0.45,
    )
    assert conformal["empirical_coverage"] == 1.0
    assert conformal["singleton_fraction"] == 1.0
    curve = selective_risk_curve(labels, probabilities)
    assert curve[-1]["coverage"] == 1.0
    assert curve[0]["classification_risk"] <= curve[-1]["classification_risk"]


def test_regression_metrics_use_only_finite_pairs() -> None:
    metrics = regression_metrics(
        [100.0, 120.0, float("nan")],
        [110.0, 120.0, 999.0],
    )
    assert metrics["n"] == 2
    assert metrics["mae"] == 5.0
    assert math.isclose(metrics["rmse"], math.sqrt(50))


def test_participant_bootstrap_keeps_repeated_draws_distinct() -> None:
    predictions = pd.DataFrame(
        {
            "source_id": ["s", "s"],
            "participant_id": ["p1", "p2"],
            "hypo_30_event_id": ["e1", "e2"],
        }
    )
    result = participant_bootstrap_metrics(
        predictions,
        metric=lambda frame: {
            "participants": float(frame["participant_id"].nunique()),
            "events": float(frame["hypo_30_event_id"].nunique()),
        },
        replicates=10,
        seed=3,
    )
    assert result["participants"]["ci_low"] == 2
    assert result["events"]["ci_low"] == 2


def test_paired_bootstrap_uses_the_same_participant_draws() -> None:
    reference = pd.DataFrame(
        {
            "source_id": ["s", "s"],
            "participant_id": ["p1", "p2"],
            "context_end": pd.to_datetime(
                ["2026-01-01", "2026-01-02"],
                utc=True,
            ),
            "hypo_30_label": [0, 1],
            "score": [0.0, 0.0],
        }
    )
    candidate = reference.assign(score=[1.0, 2.0])
    result = paired_participant_bootstrap_difference(
        candidate,
        reference,
        metric=lambda frame: float(frame["score"].mean()),
        replicates=20,
        seed=8,
    )
    assert result["estimate"] == 1.5
    assert result["ci_low"] >= 1.0
    with pytest.raises(ValueError, match="outcome mismatch"):
        paired_participant_bootstrap_difference(
            candidate.assign(hypo_30_label=[1, 1]),
            reference,
            metric=lambda frame: float(frame["score"].mean()),
            replicates=2,
            seed=8,
        )
