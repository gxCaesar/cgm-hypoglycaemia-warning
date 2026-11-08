from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from glycoshift.calibration import apply_temperature
from glycoshift.revision_composition import (
    FrozenRevisionComposition,
    FrozenStepECDF,
    audit_development_mapping,
    build_fit_alias_ledger,
    compose_frozen_scores,
    fit_revision_composition,
    join_member_scores,
    load_frozen_composition,
    save_frozen_composition,
)
from glycoshift.revision_inputs import ReplayInputs
from glycoshift.revision_training import RevisionTrainingContext
from glycoshift.sequences import REVISION_CONTRACT


def _members(tmp_path):
    from test_revision_training_callbacks import _fixture

    replays, base, config = _fixture()
    members, metadata = {}, {}
    for member in ("lstm", "persistence"):
        predictions = deepcopy(base)
        if member == "persistence":
            for frame in predictions.values():
                logits = -frame.cgm_last.to_numpy() / 30
                for horizon in (30, 60):
                    frame[f"hypo_{horizon}_logit"] = logits
                    frame[f"severe_hypo_{horizon}_raw_logit"] = logits
        context = RevisionTrainingContext(replays, source_order=("A", "B"))
        tasks, _ = context.calibrate(
            predictions["selection"],
            predictions["calibration"],
            config=config,
            source_order=("A", "B"),
        )
        members[member] = pd.concat(predictions.values(), ignore_index=True)
        metadata[member] = {
            "run_dir": str(tmp_path / member),
            "summary": {
                "revision_contract": REVISION_CONTRACT,
                "training_seed": 17,
                "input_feature_mode": "cgm_mask_clock",
                "tasks": tasks,
                "training_targets": {
                    "tasks": {
                        f"hypo_{horizon}": {
                            "target_column": f"hypo_{horizon}_training_target",
                            "mask_column": f"hypo_{horizon}_training_mask",
                        }
                        for horizon in (30, 60)
                    }
                },
                **(
                    {"architecture": "lstm"}
                    if member == "lstm"
                    else {"baseline": "persistence"}
                ),
            },
        }
    return members, metadata, replays


def _fit(tmp_path, **kwargs):
    members, metadata, replays = _members(tmp_path)
    result = fit_revision_composition(
        members,
        member_metadata=metadata,
        replays=replays,
        source_order=("A", "B"),
        candidate_count=5,
        **kwargs,
    )
    return result, members, metadata, replays


def test_ecdf_ties_endpoints():
    mapping = FrozenStepECDF.fit(
        np.array([0.0, 0.0, 1.0, 2.0]), partition="calibration"
    )
    scores = np.array([-1, 0, 0.5, 1, 2, 3], dtype=float)
    np.testing.assert_array_equal(mapping.transform(scores), [0, 0.5, 0.5, 0.75, 1, 1])
    assert mapping.knots == (0, 1, 2)
    assert mapping.cumulative_counts == (2, 3, 4)
    with pytest.raises(ValueError, match="calibration only"):
        FrozenStepECDF.fit(scores, partition="selection")


























def test_ecdf_preserves_float64_adjacent_values_and_singleton_behavior():
    adjacent = np.nextafter(0.5, 1)
    mapping = FrozenStepECDF.fit(np.array([0.5, adjacent]), partition="calibration")
    assert mapping.knots[0] != mapping.knots[1]
    assert mapping.inline()["knots"] == [0.5, adjacent]
    np.testing.assert_array_equal(mapping.transform([0.5, adjacent]), [0.5, 1])
    singleton = FrozenStepECDF.fit(np.array([0.5]), partition="calibration")
    np.testing.assert_array_equal(singleton.transform([0, 0.5, 1]), [0, 1, 1])
    np.testing.assert_array_equal(
        singleton.legacy_interpolated(np.array([0, 0.5, 1])), [0, 0, 0]
    )
    with pytest.raises(ValueError, match="remote sidecars"):
        mapping.inline(max_knots=1)


@pytest.mark.parametrize(
    "knots,counts,n",
    [
        ((0.5, 0.1), (1, 2), 2),
        ((0.1, 0.5), (2, 1), 1),
        ((0.1, 0.5), (0, 2), 2),
        ((0.1, 0.5), (1, 2), 3),
        ((0.1, np.nan), (1, 2), 2),
        ((0.1, 0.5), (True, True), 1),
    ],
)
def test_invalid_frozen_ecdf_contract_fails_closed(knots, counts, n):
    with pytest.raises(ValueError, match="invalid frozen ECDF"):
        FrozenStepECDF(knots, counts, n)


def test_corrected_fixed_pair_uses_all_calibration_rows_and_fixed_universe(tmp_path):
    result, members, _, replays = _fit(tmp_path)
    assert "severe_hypo_30_probability" not in result.predictions
    assert "cgm_30_prediction" not in result.predictions
    for member in ("lstm", "persistence"):
        for horizon in (30, 60):
            assert result.frozen.mappings[member, horizon].sample_count == 12
    for (partition, horizon), evaluated in result.evaluations.items():
        assert evaluated.summary["base_event_count"] == len(
            replays[partition].events[horizon]
        )
        assert evaluated.summary["participant_count"] == 2
        assert evaluated.summary["patient_days"] == 2
        assert result.frozen.policies[horizon]["partition"] == "calibration"
        assert result.frozen.policies[horizon]["nominal_fapd"] == 0.5
        assert (
            result.frozen.policies[horizon]["endpoint"]["score_column"]
            == f"composition_{horizon}_score"
        )
    scored = compose_frozen_scores(
        result.frozen,
        {"lstm": members["lstm"].iloc[::-1], "persistence": members["persistence"]},
    )
    for horizon in (30, 60):
        np.testing.assert_array_equal(
            scored[f"composition_{horizon}_score"],
            result.predictions[f"composition_{horizon}_score"],
        )
    assert (
        result.frozen.inline()["score_interpretation"]
        == "rank_mean_not_calibrated_event_probability"
    )


def test_selection_changes_do_not_refit_member_maps_or_composition_policy(tmp_path):
    result, members, metadata, replays = _fit(tmp_path)
    changed = deepcopy(members)
    selection = changed["lstm"].partition.eq("selection")
    for horizon in (30, 60):
        column = f"hypo_{horizon}_logit"
        changed["lstm"].loc[selection, column] *= -4
        changed["lstm"].loc[selection, f"severe_hypo_{horizon}_raw_logit"] = changed[
            "lstm"
        ].loc[selection, column]
        temperature = metadata["lstm"]["summary"]["tasks"][f"hypo_{horizon}"][
            "temperature"
        ]
        values = apply_temperature(changed["lstm"].loc[selection, column], temperature)
        changed["lstm"].loc[selection, f"hypo_{horizon}_probability"] = values
        changed["lstm"].loc[selection, f"severe_hypo_{horizon}_probability"] = values
    other = fit_revision_composition(
        changed,
        member_metadata=metadata,
        replays=replays,
        source_order=("A", "B"),
        candidate_count=5,
    )
    assert other.frozen.mappings == result.frozen.mappings
    assert other.frozen.policies == result.frozen.policies
    assert not np.array_equal(
        other.predictions.composition_30_score, result.predictions.composition_30_score
    )


def test_prespecified_lstm_seed_can_share_control_without_relabelling_control_seed(
    tmp_path,
):
    members, metadata, replays = _members(tmp_path)
    metadata["lstm"]["summary"]["training_seed"] = 23
    result = fit_revision_composition(
        members,
        member_metadata=metadata,
        replays=replays,
        source_order=("A", "B"),
        model_seed=23,
        candidate_count=5,
    )
    assert result.frozen.lstm_seed == 23
    assert result.frozen.member_metadata["lstm"]["training_seed"] == 23
    assert result.frozen.member_metadata["persistence"]["training_seed"] == 17
    assert result.frozen.member_metadata["persistence"]["shared_control_fit"]


@pytest.mark.parametrize(
    "defect",
    [
        "missing",
        "duplicate",
        "timestamp",
        "partition",
        "truth",
        "probability",
        "target",
        "training_target",
        "seed",
        "external",
    ],
)
def test_exact_join_and_member_target_binding_fail_closed(tmp_path, defect):
    members, metadata, replays = _members(tmp_path)
    frame = members["persistence"]
    if defect == "missing":
        members["persistence"] = frame.iloc[1:]
    elif defect == "duplicate":
        members["persistence"] = pd.concat([frame, frame.iloc[:1]])
    elif defect == "timestamp":
        frame.loc[0, "context_end"] += pd.Timedelta(seconds=1)
    elif defect == "partition":
        frame.loc[0, "partition"] = "calibration"
    elif defect == "truth":
        frame.loc[0, "cgm_last"] += 1
    elif defect == "probability":
        frame.loc[0, "severe_hypo_30_probability"] += 0.01
    elif defect == "target":
        metadata["persistence"]["summary"]["tasks"]["hypo_30"]["target_column"] = (
            "hypo_30_label"
        )
    elif defect == "training_target":
        metadata["persistence"]["summary"]["training_targets"]["tasks"]["hypo_30"][
            "target_column"
        ] = "hypo_30_label"
    elif defect == "seed":
        metadata["lstm"]["summary"]["training_seed"] = 23
    else:
        frame["partition"] = "external_loop"
    with pytest.raises(ValueError):
        fit_revision_composition(
            members,
            member_metadata=metadata,
            replays=replays,
            source_order=("A", "B"),
            candidate_count=5,
        )


def test_join_compares_values_across_categorical_storage_metadata(tmp_path):
    members, _, _ = _members(tmp_path)
    members["lstm"]["partition"] = pd.Categorical(
        members["lstm"].partition, categories=["train", "selection", "calibration"]
    )
    members["persistence"]["partition"] = pd.Categorical(
        members["persistence"].partition, categories=["calibration", "selection"]
    )
    assert len(join_member_scores(members)) == len(members["lstm"])


def test_frozen_sidecars_roundtrip_precision_and_reject_changed_bindings(tmp_path):
    result, members, _, _ = _fit(tmp_path)
    destination = save_frozen_composition(result.frozen, tmp_path / "frozen")
    payload = json.loads(destination.read_text())
    for horizon in ("30", "60"):
        for member in ("lstm", "persistence"):
            reference = payload["references"][horizon][member]
            assert "knots" not in reference
            with np.load(reference["path"], allow_pickle=False) as arrays:
                assert arrays["knots"].dtype == np.float64
                assert arrays["cumulative_counts"].dtype == np.int64
    loaded = load_frozen_composition(destination)
    assert loaded.mappings == result.frozen.mappings
    inline = tmp_path / "inline-fixture.json"
    inline.write_text(json.dumps(result.frozen.inline(), allow_nan=False))
    assert load_frozen_composition(inline).mappings == loaded.mappings
    np.testing.assert_array_equal(
        compose_frozen_scores(loaded, members).composition_30_score,
        result.predictions.composition_30_score,
    )
    with pytest.raises(FileExistsError):
        save_frozen_composition(result.frozen, destination.parent)
    payload["references"]["30"]["lstm"]["size_bytes"] += 1
    destination.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="size/mtime"):
        load_frozen_composition(destination)


def test_hand_mapping_difference_changes_emissions_and_event_matches():
    origin = pd.Timestamp("2026-01-01", tz="UTC")
    members, replays, frames = {}, {}, []
    for partition, scores in (
        ("calibration", [0.0, 0.5, 1.0]),
        ("selection", [0.25, 0.5, 1.0]),
    ):
        timeline = pd.DataFrame(
            {
                "source_id": "A",
                "participant_id": partition,
                "partition": partition,
                "context_end": [origin + pd.Timedelta(minutes=m) for m in (0, 40, 80)],
                "patient_day": origin,
                "cgm_last": 90.0,
                "anchor_cgm_observed": True,
                "anchor_cgm_interpolated": False,
                "anchor_cgm_censored": False,
                "base_evaluable": True,
                "base_evaluable_30": True,
                "base_evaluable_60": True,
                "severe_hypo_30_probability": scores,
                "severe_hypo_60_probability": scores,
            }
        )
        events = pd.DataFrame(
            {
                "source_id": ["A"],
                "participant_id": [partition],
                "event_id": ["e"],
                "event_onset": [origin + pd.Timedelta(minutes=10)],
            }
        )
        exposure = pd.DataFrame(
            {
                "source_id": ["A"],
                "participant_id": [partition],
                "patient_days": [1],
                "observation_minutes": [15.0],
            }
        )
        replays[partition] = ReplayInputs(
            timeline, {30: events, 60: events}, {30: exposure, 60: exposure}, {}
        )
        frames.append(timeline)
    for member in ("lstm", "persistence"):
        members[member] = pd.concat(frames, ignore_index=True)
    mapping = FrozenStepECDF.fit(np.array([0.0, 0.5, 1.0]), partition="calibration")
    frozen = FrozenRevisionComposition(
        17,
        ("A",),
        {(m, h): mapping for m in members for h in (30, 60)},
        {30: {"threshold": 0.5}, 60: {"threshold": 0.5}},
        {},
    )
    result = audit_development_mapping(
        frozen,
        members,
        replays=replays,
        thresholds={h: {"step": 0.5, "linear": 0.25} for h in (30, 60)},
    )
    for horizon in ("30", "60"):
        calibration = result["horizons"][horizon]["partitions"]["calibration"]
        selection = result["horizons"][horizon]["partitions"]["selection"]
        assert calibration["threshold_crossing_disagreements"] == 0
        assert selection["threshold_crossing_disagreements"] == 1
        assert selection["linear_only_emitted_alarms"] == 1
        assert selection["base_events_with_detection_change"] == 1
        assert selection["base_events_with_matched_alarm_change"] == 1


def test_fit_alias_ledger_counts_paths_and_declared_pairings_not_numeric_identity(
    tmp_path,
):
    first, second = tmp_path / "fit-a", tmp_path / "fit-b"
    first.mkdir()
    second.mkdir()
    checkpoint = first / "best.pt"
    checkpoint.write_bytes(b"synthetic checkpoint")
    alias = tmp_path / "alias.pt"
    alias.symlink_to(checkpoint)
    entries = [
        {
            "entry_id": "a",
            "run_dir": str(first),
            "checkpoint_path": str(checkpoint),
            "numerical_equivalence_group": "same",
        },
        {
            "entry_id": "a_alias",
            "run_dir": str(tmp_path / "alias-run"),
            "producer_fit_path": None,
            "checkpoint_path": str(alias),
            "alias_of": "a",
        },
        {
            "entry_id": "b",
            "run_dir": str(second),
            "numerical_equivalence_group": "same",
        },
    ]
    ledger = build_fit_alias_ledger(
        entries, pairings=[("a", "b"), ("b", "a"), ("a_alias", "b")]
    )
    assert ledger["entry_count"] == 3 and ledger["unique_fit_count"] == 2
    assert ledger["pairing_entry_count"] == 3 and ledger["unique_fit_pair_count"] == 1
    assert ledger["confirmed_alias_groups"] == [["a", "a_alias"]]
    assert ledger["unverified_numerical_coincidences"] == [
        {"entries": ["a", "b"], "status": "not_alias_evidence"}
    ]
    assert ledger["alias_claims"][0]["confirmed_by_producer_path"]
    absent = build_fit_alias_ledger(entries)
    assert (
        absent["pairing_entry_count"] is None
        and absent["unique_fit_pair_count"] is None
    )
    assert absent["pairing_counts_status"] == "not_provided"


def test_fit_alias_ledger_resolves_shared_run_metadata_and_unverified_alias_claim(
    tmp_path,
):
    run = tmp_path / "accepted-run"
    entries = [
        {
            "entry_id": "with_artifact",
            "run_dir": str(run),
            "checkpoint_path": str(run / "best.pt"),
        },
        {"entry_id": "same_run", "run_dir": str(run)},
        {
            "entry_id": "unverified",
            "run_dir": str(tmp_path / "different"),
            "alias_of": "same_run",
        },
    ]
    result = build_fit_alias_ledger(entries, pairings=[])
    assert result["unique_fit_count"] == 2
    assert result["confirmed_alias_groups"] == [["with_artifact", "same_run"]]
    assert result["alias_claims"] == [
        {
            "entry_id": "unverified",
            "alias_of": "same_run",
            "confirmed_by_producer_path": False,
        }
    ]
    assert result["pairing_entry_count"] == 0 and result["unique_fit_pair_count"] == 0
