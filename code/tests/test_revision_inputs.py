import numpy as np
import pandas as pd
import pytest

from glycoshift.revision_inputs import build_participant_replay
from glycoshift.schema import ACTION_FIELDS
from glycoshift.windows import build_windows


def fixture():
    config = {
        "protocol": {
            "sample_minutes": 5,
            "context_hours": 0.5,
            "horizons_minutes": [30, 60],
            "minimum_consecutive_samples": 3,
            "merge_event_gap_minutes": 30,
            "thresholds_mg_dl": {
                "hypoglycemia": 70,
                "severe_hypoglycemia": 54,
                "hyperglycemia": 180,
            },
        },
        "data": {"minimum_day_coverage": 0.1, "minimum_days": 1},
    }
    frame = pd.DataFrame(
        {
            "source_id": "s",
            "participant_id": "p",
            "timestamp": pd.date_range("2026-01-01", periods=72, freq="5min", tz="UTC"),
            "cgm_mg_dl": 100.0,
            "cgm_observed": True,
            "cgm_interpolated": False,
            "cgm_lower_censored": False,
        }
    )
    for field in ACTION_FIELDS:
        frame[field] = np.nan
        for suffix in ("observed", "available", "zero_ambiguous"):
            frame[f"{field}_{suffix}"] = False
    return frame, config


def test_replay_preserves_legacy_base_keys_but_retains_censored_tail():
    frame, config = fixture()
    frame.loc[35:38, "cgm_mg_dl"] = 50.0
    old = build_windows(frame, config)
    replay = build_participant_replay(frame, config, partition="selection")
    base = replay.timeline.loc[replay.timeline.base_evaluable]
    pd.testing.assert_series_equal(
        base.context_end.reset_index(drop=True), old.context_end, check_names=False
    )
    assert len(replay.timeline) == len(frame) - 5
    assert (~replay.timeline.base_evaluable).sum() == 12
    assert len(replay.events[30]) == len(replay.events[60]) == 1
    for name in ("cgm_last", "cgm_mean", "cgm_std", "cgm_slope", "cgm_slope_30"):
        np.testing.assert_allclose(base[name], old[name], atol=1e-10)
    assert replay.exposure[30].observation_minutes.iloc[0] == len(base) * 5


def test_future_changes_may_change_truth_not_past_timeline_features():
    frame, config = fixture()
    one = build_participant_replay(frame, config, partition="selection")
    changed = frame.copy()
    changed.loc[40:, "cgm_mg_dl"] = 50.0
    changed.loc[40:, "cgm_observed"] = False
    changed.loc[40:, "cgm_interpolated"] = True
    two = build_participant_replay(changed, config, partition="selection")
    cutoff = frame.timestamp.iloc[39]
    columns = [
        "context_end",
        "cgm_last",
        "cgm_mean",
        "cgm_slope",
        "anchor_cgm_observed",
    ]
    pd.testing.assert_frame_equal(
        one.timeline.loc[one.timeline.context_end <= cutoff, columns],
        two.timeline.loc[two.timeline.context_end <= cutoff, columns],
    )


def test_observed_anchor_and_uncensored_context_are_required():
    frame, config = fixture()
    frame.loc[20, ["cgm_observed", "cgm_interpolated"]] = [False, True]
    frame.loc[30, "cgm_lower_censored"] = True
    replay = build_participant_replay(frame, config, partition="selection")
    ends = set(replay.timeline.participant_end_index)
    assert 20 not in ends
    assert 21 in ends  # a closed historical gap is available at this anchor
    assert not ends.intersection(range(30, 36))
    assert 36 in ends


def test_day_qc_does_not_remove_online_timeline():
    frame, config = fixture()
    one = build_participant_replay(frame, config, partition="selection")
    config["data"]["minimum_days"] = 7
    two = build_participant_replay(frame, config, partition="selection")
    pd.testing.assert_series_equal(one.timeline.context_end, two.timeline.context_end)
    assert not two.timeline.base_evaluable.any()
    assert two.events[30].empty and two.exposure[30].empty


def test_severe_training_mask_uses_observed_glucose_not_merged_event_state():
    frame, config = fixture()
    frame.loc[20:22, "cgm_mg_dl"] = 50.0
    frame.loc[26:28, "cgm_mg_dl"] = 50.0
    replay = build_participant_replay(frame, config, partition="train")
    row = replay.timeline.set_index("participant_end_index").loc[24]
    assert row.severe_hypo_currently_active  # retrospective merge bridges recovery
    assert row.cgm_last == 100
    assert row.hypo_30_training_mask and row.hypo_60_training_mask
    low = replay.timeline.set_index("participant_end_index").loc[21]
    assert not low.hypo_30_training_mask


def test_missing_or_conflicting_provenance_fails_closed():
    frame, config = fixture()
    with pytest.raises(ValueError, match="provenance"):
        build_participant_replay(
            frame.drop(columns="cgm_observed"), config, partition="selection"
        )
    frame.loc[20, "cgm_interpolated"] = True
    with pytest.raises(ValueError, match="also be interpolated"):
        build_participant_replay(frame, config, partition="selection")


def test_gap_restarts_context_and_prevents_unknown_future_from_becoming_exposure():
    frame, config = fixture()
    frame = frame.drop(index=30).reset_index(drop=True)
    replay = build_participant_replay(frame, config, partition="selection")
    gap = pd.Timestamp("2026-01-01T02:30:00Z")
    assert not replay.timeline.context_end.between(
        gap, gap + pd.Timedelta(minutes=25)
    ).any()
    preceding = replay.timeline.loc[
        replay.timeline.context_end.between(
            gap - pd.Timedelta(minutes=60), gap - pd.Timedelta(minutes=5)
        )
    ]
    assert not preceding.base_evaluable.any()
