"""Known-answer G05 catalogues and frozen emissions; synthetic records only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import yaml
from test_revision_inputs import fixture

from glycoshift import fixed_frame_evaluation as evaluator
from glycoshift import revision_event_sensitivity as sensitivity
from glycoshift.events import detect_events
from glycoshift.revision_inputs import build_participant_replay
from glycoshift.sequences import REVISION_CONTRACT


def _primary(frame):
    return detect_events(
        frame,
        threshold=54,
        direction="below",
        minimum_length=3,
        merge_gap_minutes=30,
        kind="severe_hypo",
    )


def _alarms(timeline, positions):
    columns = [
        "source_id",
        "participant_id",
        "context_end",
        "patient_day",
        "cgm_last",
        "base_evaluable",
    ]
    table = timeline.iloc[positions][columns].copy().reset_index(drop=True)
    table["frozen_score"] = 0.8
    table["matched"] = False
    table["unmatched"] = table.base_evaluable
    table["matched_event_id"] = None
    table["matched_event_onset"] = pd.NaT
    table["lead_minutes"] = np.nan
    return table


def _truth(events, timeline, horizon):
    key = tuple(timeline[["source_id", "participant_id"]].iloc[0])
    return sensitivity._event_frame(
        sensitivity.base_catalogue(events, timeline, horizon), key
    )


def test_catalogue_summary_uses_only_observed_categorical_groups():
    differences = pd.DataFrame(
        {
            "source_id": pd.Categorical(["s1", "s1", "s2"], ["s1", "s2", "unused"]),
            "participant_id": ["p1", "p2", "p3"],
            "variant": pd.Categorical(["observed_only"] * 3, sensitivity.VARIANTS),
            "horizon_minutes": pd.Categorical([30, 30, 60], [30, 60]),
            "participant_count": [1, 1, 1],
            "base_event_count": [2, 3, 7],
        }
    )
    actual = sensitivity._catalogue_summary(differences)
    rows = {
        (row.source_id, row.variant, row.horizon_minutes): (
            row.participant_count,
            row.base_event_count,
        )
        for row in actual.itertuples()
    }
    assert len(actual) == 4
    assert rows == {
        ("s1", "observed_only", 30): (2, 5),
        ("s2", "observed_only", 60): (1, 7),
        ("pooled", "observed_only", 30): (2, 5),
        ("pooled", "observed_only", 60): (1, 7),
    }


def test_interpolated_confirmation_removed_and_observed_onset_shifts():
    frame, config = fixture()
    frame.loc[20:22, "cgm_mg_dl"] = 50.0
    frame.loc[20:22, "cgm_observed"] = False
    frame.loc[20:22, "cgm_interpolated"] = True
    assert len(_primary(frame)) == 1
    assert sensitivity.build_variant_catalogue(frame, "observed_only") == []
    frame.loc[26:28, "cgm_mg_dl"] = 50.0
    primary = _primary(frame)
    observed = sensitivity.build_variant_catalogue(frame, "observed_only")
    assert len(primary) == len(observed) == 1
    assert primary[0].onset == frame.timestamp.iloc[20]
    assert observed[0].onset == frame.timestamp.iloc[26]
    replay = build_participant_replay(frame, config, partition="selection")
    differences = sensitivity._catalogue_difference(
        primary, observed, frame, replay.timeline, 30
    )
    assert differences["primary_onset_confirmations_with_interpolated_reading"] == 1
    assert differences["shared_base_onset_count"] == 0
    assert differences["variant_base_events_overlapping_primary"] == 1
    # The original alarm is not moved to the later observed onset.
    position = (
        replay.timeline.context_end.eq(frame.timestamp.iloc[16]).to_numpy().nonzero()[0]
    )
    alarms = _alarms(replay.timeline, position)
    out30, _, metrics30 = sensitivity.rematch_emitted_alarms(
        replay.timeline,
        _truth(observed, replay.timeline, 30),
        replay.exposure[30],
        alarms,
        horizon=30,
        glucose_cut=54,
    )
    out60, _, metrics60 = sensitivity.rematch_emitted_alarms(
        replay.timeline,
        _truth(observed, replay.timeline, 60),
        replay.exposure[60],
        alarms,
        horizon=60,
        glucose_cut=54,
    )
    assert metrics30.detected_event_count.iloc[0] == 0
    assert metrics60.detected_event_count.iloc[0] == 1
    assert out60.lead_minutes.iloc[0] == 50
    pd.testing.assert_frame_equal(
        out30[alarms.columns.difference(sensitivity._MATCH)],
        alarms[alarms.columns.difference(sensitivity._MATCH)],
    )


def test_exact_thirty_merge_boundary_and_unchanged_emission_sequence():
    frame, config = fixture()
    frame.loc[20:22, "cgm_mg_dl"] = 50.0
    frame.loc[28:30, "cgm_mg_dl"] = 50.0
    primary = _primary(frame)
    strict = sensitivity.build_variant_catalogue(frame, "strict_merge_boundary")
    assert len(primary) == 1 and len(strict) == 2
    replay = build_participant_replay(frame, config, partition="selection")
    times = frame.timestamp.iloc[[16, 25, 70]]
    positions = np.flatnonzero(replay.timeline.context_end.isin(times))
    emitted = _alarms(replay.timeline, positions)
    original, _, _ = sensitivity.rematch_emitted_alarms(
        replay.timeline,
        _truth(primary, replay.timeline, 30),
        replay.exposure[30],
        emitted,
        horizon=30,
        glucose_cut=54,
    )
    original = original.drop(columns=[c for c in original if c.startswith("primary_")])
    changed, events, participant = sensitivity.rematch_emitted_alarms(
        replay.timeline,
        _truth(strict, replay.timeline, 30),
        replay.exposure[30],
        original,
        horizon=30,
        glucose_cut=54,
    )
    assert original.matched.sum() == 1 and changed.matched.sum() == 2
    assert len(changed) == 3 and changed.base_evaluable.tolist() == [True, True, False]
    assert not changed.iloc[-1].matched and not changed.iloc[-1].unmatched
    columns = [c for c in original if c not in sensitivity._MATCH]
    pd.testing.assert_frame_equal(changed[columns], original[columns])
    assert events.lead_minutes.tolist() == [20, 15]
    assert participant.coverage_20_event_count.iloc[0] == 1
    assert participant.coverage_15_event_count.iloc[0] == 2
    assert participant.unevaluable_alarm_count.iloc[0] == 1


def test_observed_merge_requires_every_grid_row_and_unaffected_case_agrees():
    frame, _ = fixture()
    frame.loc[20:22, "cgm_mg_dl"] = 50.0
    frame.loc[26:28, "cgm_mg_dl"] = 50.0
    primary = _primary(frame)
    for variant in sensitivity.VARIANTS:
        events = sensitivity.build_variant_catalogue(frame, variant)
        assert [(e.onset, e.end) for e in events] == [(e.onset, e.end) for e in primary]
    # One missing observed high reading prevents merging across that gap.
    frame.loc[24, "cgm_observed"] = False
    assert len(sensitivity.build_variant_catalogue(frame, "observed_only")) == 2
    # A censored reading also cannot support the observed-only confirmation.
    frame.loc[20, "cgm_lower_censored"] = True
    assert len(sensitivity.build_variant_catalogue(frame, "observed_only")) == 1


def test_boundary_only_variant_preserves_other_primary_gap_checks():
    frame, _ = fixture()
    frame.loc[20:22, "cgm_mg_dl"] = 50.0
    frame.loc[21, "timestamp"] -= pd.Timedelta(minutes=1)
    frame.loc[22, "timestamp"] -= pd.Timedelta(minutes=1)
    primary = _primary(frame)
    strict = sensitivity.build_variant_catalogue(frame, "strict_merge_boundary")
    assert [(e.onset, e.end) for e in strict] == [(e.onset, e.end) for e in primary]
    assert sensitivity.build_variant_catalogue(frame, "observed_only") == []


def test_zero_reachable_and_zero_event_participants_keep_exposure_and_burden():
    frame, config = fixture()
    frame.loc[:, "cgm_mg_dl"] = 60.0
    frame.loc[20:22, "cgm_mg_dl"] = 50.0
    replay = build_participant_replay(frame, config, partition="selection")
    empty = _alarms(replay.timeline, [])
    _, events, metrics = sensitivity.rematch_emitted_alarms(
        replay.timeline,
        replay.events[30],
        replay.exposure[30],
        empty,
        horizon=30,
        glucose_cut=70,
    )
    assert len(events) == 1 and not events.reachable.any()
    assert metrics.fixed_sensitivity.iloc[0] == 0
    assert np.isnan(metrics.conditional_sensitivity.iloc[0])
    assert metrics.patient_days.iloc[0] == replay.exposure[30].patient_days.iloc[0]
    frame.loc[:, "cgm_mg_dl"] = 100.0
    replay = build_participant_replay(frame, config, partition="selection")
    alarms = _alarms(replay.timeline, [0, len(replay.timeline) - 1])
    _, _, metrics = sensitivity.rematch_emitted_alarms(
        replay.timeline,
        replay.events[30],
        replay.exposure[30],
        alarms,
        horizon=30,
        glucose_cut=54,
    )
    assert metrics.base_event_count.iloc[0] == 0
    assert np.isnan(metrics.fixed_sensitivity.iloc[0])
    assert (
        metrics.delivered_alarm_count.iloc[0]
        == metrics.unmatched_alarm_count.iloc[0]
        == 1
    )
    _, summary = sensitivity._aggregate_participants(metrics, ("s",))
    assert summary["unmatched_alarm_count"] == 1
    assert summary["undefined_sensitivity_sources"] == ["s"]


@pytest.mark.parametrize(
    "defect", ["missing_mask", "duplicate_time", "duplicate_alarm", "cgm"]
)
def test_missing_provenance_and_changed_or_duplicate_keys_fail_closed(defect):
    frame, config = fixture()
    if defect == "missing_mask":
        with pytest.raises(ValueError, match="boolean provenance"):
            sensitivity.build_variant_catalogue(
                frame.drop(columns="cgm_observed"), "observed_only"
            )
        return
    if defect == "duplicate_time":
        with pytest.raises(ValueError, match="unique"):
            sensitivity.build_variant_catalogue(
                pd.concat([frame, frame.iloc[:1]]), "observed_only"
            )
        return
    replay = build_participant_replay(frame, config, partition="selection")
    alarms = _alarms(replay.timeline, [0])
    if defect == "duplicate_alarm":
        alarms = pd.concat([alarms, alarms], ignore_index=True)
    else:
        alarms.loc[0, "cgm_last"] += 1
    with pytest.raises(ValueError, match="duplicate|differs"):
        sensitivity.rematch_emitted_alarms(
            replay.timeline,
            replay.events[30],
            replay.exposure[30],
            alarms,
            horizon=30,
            glucose_cut=54,
        )


def _binding(path):
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _concat(tables):
    normalized = []
    for table in tables:
        table = table.copy()
        for column in (
            "context_end",
            "event_onset",
            "matched_event_onset",
            "alarm_context_end",
        ):
            if column in table:
                table[column] = pd.to_datetime(table[column], utc=True)
        normalized.append(pa.Table.from_pandas(table, preserve_index=False))
    return pa.concat_tables(
        normalized,
        promote_options="permissive",
    ).to_pandas()


def _accepted_fixture(tmp_path):
    normalized_root, prepared, run = (
        tmp_path / name for name in ("normalized", "prepared", "run")
    )
    for folder in (normalized_root, prepared, run):
        folder.mkdir()
    frame, replay_config = fixture()
    replay_config.update(
        project={"synthetic": True},
        split={"validation_fraction": 0.3},
        model={},
        evaluation={},
        output={},
    )
    replay_config["data"].update(
        development_sources=["s"], forbidden_development_sources=["OhioT1DM"]
    )
    config_path = tmp_path / "replay.yaml"
    config_path.write_text(yaml.safe_dump(replay_config))
    manifest, normalized_files, prepared_paths, event_tables, exposure_tables = (
        [],
        [],
        [],
        {},
        {},
    )
    all_alarms, all_events, all_metrics = {}, {}, {}
    for horizon in sensitivity.HORIZONS:
        event_tables[horizon], exposure_tables[horizon] = [], []
        all_alarms[horizon], all_events[horizon], all_metrics[horizon] = [], [], []
    for index, participant in enumerate(("synthetic-zero", "synthetic-event")):
        part = frame.copy()
        part["participant_id"] = participant
        if index:
            part.loc[20:22, "cgm_mg_dl"] = 50.0
            part.loc[20:22, "cgm_observed"] = False
            part.loc[20:22, "cgm_interpolated"] = True
            part.loc[26:28, "cgm_mg_dl"] = 50.0
        normalized_path = normalized_root / f"part-{index}.parquet"
        part.to_parquet(normalized_path, index=False)
        normalized_files.append(_binding(normalized_path))
        replay = build_participant_replay(part, replay_config, partition="selection")
        timeline_path = prepared / f"timeline-{index}.parquet"
        replay.timeline.to_parquet(timeline_path, index=False)
        prepared_paths.append(timeline_path)
        manifest.append(
            {
                "source_id": "s",
                "participant_id": participant,
                "partition": "selection",
                "timeline_path": str(timeline_path),
                "stored_rows": len(replay.timeline),
                "normalized_paths": [str(normalized_path)],
            }
        )
        scored = replay.timeline.copy()
        scored["frozen_score"] = scored.context_end.isin(
            part.timestamp.iloc[[16, 70]]
        ).astype(float)
        for horizon in sensitivity.HORIZONS:
            event_tables[horizon].append(replay.events[horizon])
            exposure_tables[horizon].append(replay.exposure[horizon])
            # The independent primary evaluator supplies the accepted fixture;
            # it is forbidden only during the sensitivity consumer itself.
            evaluated = evaluator.evaluate_fixed_frame(
                scored,
                replay.events[horizon],
                replay.exposure[horizon],
                spec=evaluator.FixedFrameSpec("frozen_score", horizon, ("s",)),
                threshold=0.5,
                glucose_cut=54,
            )
            all_alarms[horizon].append(evaluated.alarms)
            all_events[horizon].append(evaluated.events)
            all_metrics[horizon].append(evaluated.participants)
    started = max(item["mtime_ns"] for item in normalized_files) + 1
    paths, conditions = {}, []
    for horizon in sensitivity.HORIZONS:
        paths[f"selection/{horizon}"] = {}
        for category, tables in (
            ("events", event_tables),
            ("exposure", exposure_tables),
        ):
            path = prepared / f"{category}-{horizon}.parquet"
            _concat(tables[horizon]).to_parquet(path, index=False)
            paths[f"selection/{horizon}"][category] = str(path)
            prepared_paths.append(path)
        condition = {"horizon_minutes": horizon, "glucose_cut": 54, "nominal_fapd": 0.5}
        for category, tables in (
            ("alarms", all_alarms),
            ("events", all_events),
            ("participants", all_metrics),
        ):
            path = run / f"{category}-{horizon}.parquet"
            _concat(tables[horizon]).to_parquet(path, index=False)
            condition[category] = _binding(path)
        conditions.append(condition)
    # A nonexistent train path proves no original train shard is opened.
    manifest.append(
        {
            "source_id": "s",
            "participant_id": "never-read-train",
            "partition": "train",
            "timeline_path": str(prepared / "absent-train.parquet"),
        }
    )
    manifest_path = prepared / "participant-manifest.private.json"
    manifest_path.write_text(json.dumps({"participants": manifest}))
    audit_path = prepared / "input-audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "status": "passed",
                "revision_contract": REVISION_CONTRACT,
                "source_order": ["s"],
                "normalized_path": str(normalized_root),
                "endpoint_paths": paths,
            }
        )
    )
    prepared_paths.extend([audit_path, manifest_path])
    summary = run / "model_summary.json"
    summary.write_text(
        json.dumps(
            {
                "revision_contract": REVISION_CONTRACT,
                "baseline": "persistence",
                "training_seed": 42,
            }
        )
    )
    evidence = tmp_path / "synthetic-acceptance-only.json"
    evidence.write_text(json.dumps({"synthetic_fixture_only": True}))
    return {
        "revision_contract": REVISION_CONTRACT,
        "partition": "selection",
        "prepared_dir": str(prepared),
        "replay_config": _binding(config_path),
        "preparation_started_at": pd.Timestamp(started, tz="UTC").isoformat(),
        "output_dir": str(tmp_path / "output"),
        "input_acceptance": {
            "status": "accepted",
            "binding_time": "current_sensitivity_acceptance",
            "evidence": [_binding(evidence)],
            "normalized_files": normalized_files,
            "prepared_files": [_binding(p) for p in prepared_paths],
        },
        "runs": [
            {
                "arm": "persistence",
                "model_seed": None,
                "run_dir": str(run),
                "model_summary": _binding(summary),
                "accepted_evidence": [_binding(evidence)],
                "ledger_root": str(run),
                "conditions": conditions,
            }
        ],
    }


def test_selection_cli_consumer_keeps_alarm_sequence_exposure_and_private_records(
    tmp_path, monkeypatch
):
    config = _accepted_fixture(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "event sensitivity must not fit, score or rerun refractory"
        )

    monkeypatch.setattr(evaluator, "evaluate_fixed_frame", forbidden)
    monkeypatch.setattr(evaluator, "fit_fixed_frame_alarm_policy", forbidden)
    monkeypatch.setattr(evaluator, "_collapse_positions", forbidden)
    output = sensitivity.run_event_sensitivity(config)
    assert json.loads((output / "status.json").read_text())["status"] == "complete"
    payload = json.loads(
        (output / "exportable/event-sensitivity-summary.json").read_text()
    )
    assert payload["alarms_scores_policies_refractory_changed"] is False
    assert payload["base_exposure_changed"] is False
    assert payload["raw_acquisition_provenance_established"] is False
    assert "participant_id" not in json.dumps(payload)
    rows = [row for row in payload["rows"] if row["scope"] == "equal_source_and_pooled"]
    assert len(rows) == 4
    assert {row["participant_count"] for row in rows} == {2}
    assert {row["patient_days"] for row in rows} == {2}
    root = output / "private/evaluations/run-000/observed_only/54/30/0.5"
    actual = pd.read_parquet(root / "alarms")
    original = pd.read_parquet(config["runs"][0]["conditions"][0]["alarms"]["path"])
    columns = [c for c in original if c not in sensitivity._MATCH]
    sensitivity._equal(actual, original, columns, "exact test alarm sequence")
    assert len(pd.read_parquet(root / "events")) == 1  # First shard has no event.
    catalogue = pd.read_csv(output / "exportable/catalogue-differences.csv")
    observed = catalogue.loc[
        (catalogue.source_id.eq("pooled")) & catalogue.variant.eq("observed_only")
    ]
    assert observed.primary_only_base_onset_count.tolist() == [1, 1]
    with pytest.raises(FileExistsError):
        sensitivity.run_event_sensitivity(config)


@pytest.mark.parametrize(
    "defect",
    ["no_binding", "late_mtime", "parity", "external_role", "duplicate_ledger"],
)
def test_cli_rejects_missing_or_changed_primary_input_contract(tmp_path, defect):
    config = _accepted_fixture(tmp_path)
    if defect == "no_binding":
        config["input_acceptance"]["normalized_files"].pop()
        message = "bindings differ"
    elif defect == "late_mtime":
        config["preparation_started_at"] = "2000-01-01T00:00:00Z"
        message = "mtime must precede"
    elif defect == "parity":
        binding = config["input_acceptance"]["prepared_files"][0]
        path = Path(binding["path"])
        table = pd.read_parquet(path)
        table.loc[0, "cgm_mean"] += 1
        table.to_parquet(path, index=False)
        binding.update(_binding(path))
        message = "complete primary timeline.*differs"
    elif defect == "external_role":
        config["partition"] = "external_loop"
        message = "development selection"
    else:
        condition = config["runs"][0]["conditions"][0]
        path = Path(condition["alarms"]["path"])
        table = pd.read_parquet(path)
        pd.concat([table, table.iloc[:1]], ignore_index=True).to_parquet(
            path, index=False
        )
        condition["alarms"] = _binding(path)
        message = "duplicate"
    with pytest.raises(ValueError, match=message):
        sensitivity.run_event_sensitivity(config)


@pytest.mark.parametrize(
    "defect",
    [
        "missing_timeline",
        "missing_event",
        "duplicate_exposure",
        "ledger_role",
        "missing_horizon",
    ],
)
def test_cli_rejects_incomplete_primary_population_or_role(tmp_path, defect):
    config = _accepted_fixture(tmp_path)
    if defect == "missing_horizon":
        config["runs"][0]["conditions"].pop()
        message = "both original horizons"
    elif defect == "missing_timeline":
        binding = config["input_acceptance"]["prepared_files"][0]
        path = Path(binding["path"])
        pd.read_parquet(path).iloc[1:].to_parquet(path, index=False)
        binding.update(_binding(path))
        message = "row count/role"
    elif defect == "duplicate_exposure":
        binding = next(
            b
            for b in config["input_acceptance"]["prepared_files"]
            if Path(b["path"]).name == "exposure-30.parquet"
        )
        path = Path(binding["path"])
        table = pd.read_parquet(path)
        pd.concat([table, table.iloc[:1]], ignore_index=True).to_parquet(
            path, index=False
        )
        binding.update(_binding(path))
        message = "identities are duplicated"
    else:
        condition = config["runs"][0]["conditions"][0]
        category = "events" if defect == "missing_event" else "alarms"
        path = Path(condition[category]["path"])
        table = pd.read_parquet(path)
        if defect == "missing_event":
            table = table.iloc[:0]
            message = "event catalogue.*differs"
        else:
            table["partition"] = "calibration"
            message = "selection identities or roles"
        table.to_parquet(path, index=False)
        condition[category] = _binding(path)
    with pytest.raises(ValueError, match=message):
        sensitivity.run_event_sensitivity(config)


def test_composition_recipe_identity_needs_no_score_mapping_or_fit(tmp_path):
    config = _accepted_fixture(tmp_path)
    run = config["runs"][0]
    recipe = Path(run["run_dir"]) / "synthetic-recipe-identity.json"
    recipe.write_text(
        json.dumps(
            {
                "revision_contract": REVISION_CONTRACT,
                "algebra": "arithmetic_mean_of_two_right_step_member_ECDF_ranks",
                "comparison_lstm_seed": 17,
                "source_order": ["s"],
                "reference_partition": "calibration",
            }
        )
    )
    # These synthetic accepted emissions are inputs; no ECDF array is opened.
    run.pop("model_summary")
    run.update(arm="composition", model_seed=17, composition_recipe=_binding(recipe))
    output = sensitivity.run_event_sensitivity(config)
    report = json.loads(
        (output / "exportable/event-sensitivity-summary.json").read_text()
    )
    assert {row["arm"] for row in report["rows"]} == {"composition"}


def test_model_identity_cannot_be_relabelled_in_the_accepted_listing(tmp_path):
    config = _accepted_fixture(tmp_path)
    config["runs"][0]["arm"] = "lstm"
    with pytest.raises(ValueError, match="identity differs"):
        sensitivity.run_event_sensitivity(config)
