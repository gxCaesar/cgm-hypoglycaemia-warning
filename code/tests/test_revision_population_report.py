import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "summarise_revision_population.py"
SPEC = importlib.util.spec_from_file_location("revision_population_report", SCRIPT)
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def _write_json(path, value):
    path.write_text(json.dumps(value))


def prepared_bundle(tmp_path):
    """Two sources, all frozen roles, horizon-specific positivity and zero events."""
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    counts, manifest = [], []
    for source in ("source-a", "source-b"):
        for role in report.PARTITIONS:
            counts.append(
                {
                    "source_id": source,
                    "partition": role,
                    "participants": 2,
                    "base_evaluable_decisions": 8,
                    "observed_anchor_decisions": 12,
                    "legacy_interpolated_anchor_decisions": 3,
                    "base_uses_maximum_horizon_completeness": 120,
                    "private_input_path": "/private/raw-inputs",
                }
            )
            directory = prepared / "endpoints" / role
            directory.mkdir(parents=True, exist_ok=True)
            exposure = [
                {
                    "source_id": source,
                    "participant_id": f"{role}-{person}",
                    "patient_days": person + 1,
                    "observation_minutes": 20,
                }
                for person in (0, 1)
            ]
            for horizon in report.HORIZONS:
                # Participant 1 in source A is positive only at 60 min. The
                # corresponding source-B participant has no event at either H.
                positive_people = (
                    [0, 1] if source == "source-a" and horizon == 60 else [0]
                )
                events = pd.DataFrame(
                    [
                        {
                            "source_id": source,
                            "participant_id": f"{role}-{person}",
                            "event_id": f"{source}-event-{person}",
                            "event_onset": pd.Timestamp("2026-01-01T00:30Z")
                            + pd.Timedelta(minutes=person * 10),
                        }
                        for person in positive_people
                    ]
                )
                event_path = directory / f"events-{horizon}.parquet"
                exposure_path = directory / f"exposure-{horizon}.parquet"
                if event_path.exists():
                    events = pd.concat(
                        [pd.read_parquet(event_path), events], ignore_index=True
                    )
                    exposed = pd.concat(
                        [pd.read_parquet(exposure_path), pd.DataFrame(exposure)],
                        ignore_index=True,
                    )
                else:
                    exposed = pd.DataFrame(exposure)
                events.to_parquet(event_path, index=False)
                exposed.to_parquet(exposure_path, index=False)
            for person in (0, 1):
                frame = pd.DataFrame(
                    {
                        "source_id": source,
                        "participant_id": f"{role}-{person}",
                        "partition": role,
                        "context_end": pd.date_range(
                            "2026-01-01", periods=6, freq="5min", tz="UTC"
                        ),
                        "cgm_last": [50.0, 80, 80, 80, 80, 80],
                        "base_evaluable": [True, True, True, True, False, False],
                        "raw_private_value": np.arange(6),
                    }
                )
                for horizon in report.HORIZONS:
                    frame[f"base_evaluable_{horizon}"] = frame.base_evaluable
                    frame[f"hypo_{horizon}_training_mask"] = [
                        False,
                        True,
                        True,
                        True,
                        False,
                        False,
                    ]
                    retained = [0, 0, 0]
                    if person == 0:
                        retained = [1, 1, 0] if horizon == 30 else [1, 1, 1]
                    elif source == "source-a" and horizon == 60:
                        retained = [0, 1, 0]
                    frame[f"hypo_{horizon}_training_target"] = [
                        np.nan,
                        *retained,
                        np.nan,
                        np.nan,
                    ]
                if role == "train":
                    frame = frame.loc[frame.base_evaluable]
                path = prepared / "timelines" / f"{source}-{role}-{person}.parquet"
                path.parent.mkdir(exist_ok=True)
                frame.to_parquet(path, index=False)
                manifest.append(
                    {
                        "source_id": source,
                        "participant_id": f"{role}-{person}",
                        "partition": role,
                        "timeline_path": str(path.relative_to(prepared)),
                        "stored_rows": len(frame),
                        "private_path": "/private/person-data",
                    }
                )
    _write_json(
        prepared / "input-audit.json",
        {
            "revision_contract": "2026-11-04-fixed-frame",
            "status": "passed",
            "source_order": ["source-a", "source-b"],
            "original_participants": 12,
            "source_partition_counts": counts,
            "normalized_path": "/private/normalized-inputs",
        },
    )
    _write_json(
        prepared / "participant-manifest.private.json", {"participants": manifest}
    )
    return prepared


def test_population_oracle_retains_full_exposure_and_distinct_denominators(
    tmp_path, monkeypatch
):
    prepared = prepared_bundle(tmp_path)
    # Exercise ordering/identity checks across stream-batch boundaries.
    monkeypatch.setattr(report, "BATCH_ROWS", 2)
    output = tmp_path / "aggregate"
    result = report.summarise_population(prepared, output)
    assert result["participants_by_partition"] == dict.fromkeys(report.PARTITIONS, 4)
    assert result["rows"] == {
        "input_counts": 6,
        "endpoint_counts": 18,
        "training_window_counts": 18,
    }
    inputs = pd.read_csv(output / "input-counts.csv")
    assert inputs.base_future_completeness_minutes.eq(60).all()
    assert inputs.legacy_interpolated_anchor_decisions.eq(3).all()
    endpoint = pd.read_csv(output / "endpoint-counts.csv")
    totals = endpoint.loc[endpoint.row_type == "role_total"]
    assert totals.base_participants.eq(4).all()
    assert totals.loc[totals.horizon == 30, "event_positive_participants"].eq(2).all()
    assert totals.loc[totals.horizon == 60, "event_positive_participants"].eq(3).all()
    assert totals.loc[totals.horizon == 30, "base_events"].eq(2).all()
    assert totals.loc[totals.horizon == 60, "base_events"].eq(3).all()
    assert totals.calendar_days.eq(6).all()
    assert totals.observation_minutes.eq(80).all()
    assert totals.effective_24h_days.to_numpy() == pytest.approx(80 / 1440)
    windows = pd.read_csv(output / "training-window-counts.csv")
    source_windows = windows.loc[windows.row_type == "source"]
    assert source_windows.retained_training_windows.eq(6).all()
    assert (
        source_windows.loc[source_windows.horizon == 30, "positive_training_windows"]
        .eq(2)
        .all()
    )
    assert (
        source_windows.loc[
            (source_windows.horizon == 60) & (source_windows.source_id == "source-a"),
            "positive_training_windows",
        ]
        .eq(4)
        .all()
    )
    assert (
        source_windows.loc[
            (source_windows.horizon == 60) & (source_windows.source_id == "source-b"),
            "positive_training_windows",
        ]
        .eq(3)
        .all()
    )
    assert source_windows.base_evaluable_decisions.eq(8).all()
    assert (
        source_windows.loc[source_windows.partition == "train", "stored_rows"]
        .eq(8)
        .all()
    )
    assert (
        source_windows.loc[source_windows.partition != "train", "stored_rows"]
        .eq(12)
        .all()
    )
    # Aggregate exports contain no personal identity, individual times or paths.
    exported = "\n".join(path.read_text() for path in output.iterdir())
    for forbidden in (
        "participant_id",
        "context_end",
        "/private/",
        "train-0",
        "raw_private_value",
    ):
        assert forbidden not in exported
    assert "not source-macro" in result["definitions"]["role_totals"]


def _modify_table(prepared, relative, change):
    path = prepared / relative
    frame = change(pd.read_parquet(path))
    frame.to_parquet(path, index=False)


@pytest.mark.parametrize(
    "defect, message",
    [
        ("missing_exposure", "exposure participant count differs"),
        ("event_without_exposure", "event participant is missing"),
        ("duplicate_exposure", "duplicate exposure"),
        ("duplicate_event", "duplicate event identity"),
        ("nonpositive_days", "patient_days must be finite and positive"),
        ("changed_horizon_exposure", "base exposure differs between horizons"),
        (
            "changed_horizon_onset",
            "shared event identity has conflicting horizon onsets",
        ),
        ("partition_overlap", "overlaps partitions"),
    ],
)
def test_endpoint_known_defects_fail_closed(tmp_path, defect, message):
    prepared = prepared_bundle(tmp_path)
    exposure = "endpoints/train/exposure-30.parquet"
    events = "endpoints/train/events-30.parquet"
    if defect == "missing_exposure":
        _modify_table(prepared, exposure, lambda f: f.drop(index=3))
    elif defect == "event_without_exposure":
        _modify_table(prepared, exposure, lambda f: f.drop(index=0))
    elif defect == "duplicate_exposure":
        _modify_table(prepared, exposure, lambda f: pd.concat([f, f.iloc[[0]]]))
    elif defect == "duplicate_event":
        _modify_table(prepared, events, lambda f: pd.concat([f, f.iloc[[0]]]))
    elif defect == "nonpositive_days":
        _modify_table(prepared, exposure, lambda f: f.assign(patient_days=0))
    elif defect == "changed_horizon_exposure":
        _modify_table(
            prepared,
            "endpoints/train/exposure-60.parquet",
            lambda f: f.assign(observation_minutes=21),
        )
    elif defect == "changed_horizon_onset":
        _modify_table(
            prepared,
            "endpoints/train/events-60.parquet",
            lambda f: f.assign(event_onset=f.event_onset + pd.Timedelta(minutes=5)),
        )
    else:
        for horizon in report.HORIZONS:
            for table in ("events", "exposure"):
                _modify_table(
                    prepared,
                    f"endpoints/selection/{table}-{horizon}.parquet",
                    lambda f: f.assign(
                        participant_id=f.participant_id.str.replace(
                            "selection", "train"
                        )
                    ),
                )
    with pytest.raises(ValueError, match=message):
        report.summarise_population(prepared, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "defect, message",
    [
        ("missing_timeline", "manifest timeline is missing"),
        ("duplicate_manifest", "duplicate manifest participant"),
        (
            "missing_manifest_participant",
            "manifest is missing base exposure participants",
        ),
        ("wrong_identity", "timeline identity differs"),
        ("wrong_stored_rows", "stored row count differs"),
        ("duplicate_decision", "decision identity must be unique"),
        ("retained_nan", "retained level-2 training targets must be binary"),
        ("active_flag_mask", "training mask differs from base online eligibility"),
    ],
)
def test_timeline_known_defects_fail_closed(tmp_path, defect, message):
    prepared = prepared_bundle(tmp_path)
    manifest_path = prepared / "participant-manifest.private.json"
    manifest = json.loads(manifest_path.read_text())
    record = manifest["participants"][0]
    relative = record["timeline_path"]
    if defect == "missing_timeline":
        record["timeline_path"] = "missing.parquet"
    elif defect == "duplicate_manifest":
        manifest["participants"].append(record.copy())
    elif defect == "missing_manifest_participant":
        manifest["participants"].pop()
    elif defect == "wrong_stored_rows":
        record["stored_rows"] += 1
    elif defect == "wrong_identity":
        _modify_table(
            prepared, relative, lambda f: f.assign(participant_id="another-person")
        )
    elif defect == "duplicate_decision":

        def duplicate(frame):
            frame.loc[2, "context_end"] = frame.loc[1, "context_end"]
            return frame

        _modify_table(prepared, relative, duplicate)
    elif defect == "retained_nan":

        def nan_target(frame):
            frame.loc[1, "hypo_30_training_target"] = np.nan
            return frame

        _modify_table(prepared, relative, nan_target)
    else:
        _modify_table(
            prepared, relative, lambda f: f.assign(hypo_30_training_mask=False)
        )
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match=message):
        report.summarise_population(prepared, tmp_path / "out")


@pytest.mark.parametrize(
    "defect, message",
    [
        ("wrong_contract", "wrong revision contract"),
        ("missing_cell", "exactly one row"),
        ("duplicate_cell", "exactly one row"),
        ("wrong_total", "participant total differs"),
        ("completeness", "completeness metadata is inconsistent"),
    ],
)
def test_audit_grid_and_metadata_known_defects(tmp_path, defect, message):
    prepared = prepared_bundle(tmp_path)
    audit = json.loads((prepared / "input-audit.json").read_text())
    if defect == "wrong_contract":
        audit["revision_contract"] = "legacy"
    elif defect == "missing_cell":
        audit["source_partition_counts"].pop()
    elif defect == "duplicate_cell":
        audit["source_partition_counts"].append(audit["source_partition_counts"][0])
    elif defect == "wrong_total":
        audit["original_participants"] += 1
    else:
        audit["source_partition_counts"][0][
            "base_uses_maximum_horizon_completeness"
        ] = 60
    with pytest.raises(ValueError, match=message):
        report.input_counts(audit)


def test_missing_manifest_explicitly_omits_optional_window_report(tmp_path):
    prepared = prepared_bundle(tmp_path)
    # Preserve the fixture, using a second bundle view containing only endpoints.
    manifest = prepared / "participant-manifest.private.json"
    manifest.rename(prepared / "manifest-not-selected.json")
    result = report.summarise_population(prepared, tmp_path / "out")
    assert result["timeline_counts"] == {
        "status": "not_run",
        "reason": "participant manifest absent",
    }
    assert not (tmp_path / "out" / "training-window-counts.csv").exists()


def test_absolute_in_bundle_paths_and_empty_output_are_valid(tmp_path):
    prepared = prepared_bundle(tmp_path)
    path = prepared / "participant-manifest.private.json"
    manifest = json.loads(path.read_text())
    for record in manifest["participants"]:
        record["timeline_path"] = str(prepared / record["timeline_path"])
    _write_json(path, manifest)
    output = tmp_path / "out"
    output.mkdir()
    assert report.summarise_population(prepared, output)["status"] == "passed"


@pytest.mark.parametrize("absolute", [False, True])
def test_timeline_path_cannot_escape_selected_bundle(tmp_path, absolute):
    prepared = prepared_bundle(tmp_path)
    path = prepared / "participant-manifest.private.json"
    manifest = json.loads(path.read_text())
    outside = tmp_path / "outside.parquet"
    original = prepared / manifest["participants"][0]["timeline_path"]
    outside.write_bytes(original.read_bytes())
    manifest["participants"][0]["timeline_path"] = (
        str(outside) if absolute else "../outside.parquet"
    )
    _write_json(path, manifest)
    with pytest.raises(ValueError, match="path escapes bundle"):
        report.summarise_population(prepared, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "relative",
    [
        "input-audit.json",
        "participant-manifest.private.json",
        "endpoints/train/events-30.parquet",
        "endpoints/train/exposure-30.parquet",
        "timelines/source-a-train-0.parquet",
        "timelines",
    ],
)
def test_symlink_inputs_are_rejected_even_when_target_is_inside_bundle(
    tmp_path, relative
):
    prepared = prepared_bundle(tmp_path)
    path = prepared / relative
    target = path.with_name(path.name + ".original")
    path.rename(target)
    path.symlink_to(target, target_is_directory=target.is_dir())
    with pytest.raises(ValueError, match="must not use symlinks"):
        report.summarise_population(prepared, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_nonempty_output_with_unrecognised_filename_is_rejected(tmp_path):
    prepared = prepared_bundle(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    existing = output / "keep-this.txt"
    existing.write_text("retained")
    with pytest.raises(FileExistsError, match="must be absent or empty"):
        report.summarise_population(prepared, output)
    assert list(output.iterdir()) == [existing]
    assert existing.read_text() == "retained"
