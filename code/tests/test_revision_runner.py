import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest
from test_revision_prepare import crossfit_inputs, inputs

from glycoshift.config import load_config
from glycoshift.revision_crossfit_prepare import prepare_crossfit_replays
from glycoshift.revision_prepare import prepare_replays
from glycoshift.revision_runner import (
    load_crossfit_inner_bundle,
    load_crossfit_outer_bundle,
    load_replay_bundle,
    revision_config,
    run_revision_deep,
)


def test_bundle_preserves_partitions_and_rejects_changed_binding(tmp_path):
    normalized, legacy, config = inputs(tmp_path)
    config["project"] = {"synthetic": True}
    prepared = tmp_path / "prepared"
    prepare_replays(normalized, legacy, prepared, config)
    (_, windows, _), replays = load_replay_bundle(prepared, config, sequences=False)
    assert len(windows) == sum(len(r.timeline) for r in replays.values())
    assert set(windows.participant_id) == {"0", "1", "2"}
    assert len(replays["selection"].timeline) > len(replays["train"].timeline)
    manifest_path = prepared / "participant-manifest.private.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["participants"][0]["stored_rows"] += 1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="row count differs"):
        load_replay_bundle(prepared, config, sequences=False)


def test_revision_config_preserves_optimization_and_original(tmp_path):
    original = load_config(Path(__file__).parents[1] / "configs/smoke.yaml")
    config = revision_config(original, seed=17, output=tmp_path)
    for field in ("maximum_epochs", "batch_size", "learning_rate", "negative_ratio"):
        assert config["training"][field] == original["training"][field]
    assert "revision_contract" not in original["training"]
    with pytest.raises(ValueError, match="seed"):
        revision_config(original, seed=18, output=tmp_path)


def test_complete_revision_training_path_on_synthetic_replays(tmp_path, monkeypatch):
    normalized, legacy, fixture_config = inputs(tmp_path)
    original = load_config(Path(__file__).parents[1] / "configs/smoke.yaml")
    original["protocol"].update(fixture_config["protocol"])
    original["data"].update(fixture_config["data"])
    original["training"].update(
        maximum_epochs=1,
        batch_size=64,
        effective_batch_size=64,
        num_workers=0,
        data_sampling_seed=42,
        device="cpu",
    )
    original["evaluation"].update(
        threshold_candidate_count=5,
        event_ap_candidate_count=5,
        selection_operational_candidate_count=5,
    )
    config = revision_config(original, seed=17, output=tmp_path / "run")
    prepared = tmp_path / "prepared"
    prepare_replays(normalized, legacy, prepared, config)
    frame = pd.concat(
        [pd.read_parquet(normalized / f"part-{i}.parquet") for i in range(3)],
        ignore_index=True,
    )

    def synthetic_frame(_config, *, sources, participant_keys):
        assert sources == {"s"}
        assert participant_keys == {("s", str(i)) for i in range(3)}
        return frame

    monkeypatch.setattr("glycoshift.revision_runner.load_frame", synthetic_frame)
    run = run_revision_deep(config, prepared, architecture="lstm")
    log = [
        json.loads(line) for line in (run / "logs/train.jsonl").read_text().splitlines()
    ]
    first = next(row for row in log if row["event"] == "first_finite_loss")
    assert np.isfinite(first["loss"])
    summary = json.loads((run / "training_summary.json").read_text())
    assert summary["best_epoch"] == 0
    assert summary["revision_contract"] == "2026-11-04-fixed-frame"
    predictions = pd.read_parquet(run / "predictions.parquet")
    assert (~predictions.base_evaluable).sum() == 24
    for horizon in (30, 60):
        np.testing.assert_array_equal(
            predictions[f"hypo_{horizon}_probability"],
            predictions[f"severe_hypo_{horizon}_probability"],
        )
    assert (run / "metrics.json").is_file()


def _crossfit(tmp_path):
    normalized, legacy, config = crossfit_inputs(tmp_path)
    prepared = tmp_path / "crossfit"
    prepare_crossfit_replays(normalized, legacy, prepared, config)
    return prepared, config


def test_crossfit_inner_and_outer_roles_keep_complete_separate_populations(tmp_path):
    prepared, config = _crossfit(tmp_path)
    outer_all = []
    for fold in range(5):
        (_, windows, audit), inner = load_crossfit_inner_bundle(
            prepared, config, outer_fold=fold, sequences=False
        )
        frame, outer, outer_audit = load_crossfit_outer_bundle(
            prepared, config, outer_fold=fold, sequences=False
        )
        assert frame is None
        assert set(inner) == {"train", "selection", "calibration"}
        assert set(windows.partition) == set(inner)
        assert inner["train"].timeline.base_evaluable.all()
        assert (~inner["selection"].timeline.base_evaluable).any()
        assert (~inner["calibration"].timeline.base_evaluable).any()
        assert (~outer.timeline.base_evaluable).any()
        assert outer.timeline.partition.eq("outer_evaluation").all()
        assert audit["outer_fold"] == fold
        assert audit["crossfit_prepared_dir"] == str(prepared)
        assert outer_audit["role"] == "outer_evaluation"
        inner_ids = set(windows.participant_id)
        outer_ids = set(outer.timeline.participant_id)
        assert not inner_ids.intersection(outer_ids)
        assert inner_ids.union(outer_ids) == {str(i) for i in range(9)}
        outer_all.extend(outer_ids)
        for replay in [*inner.values(), outer]:
            assert set(replay.timeline.original_partition).issubset(set(inner))
            ids = set(replay.timeline.participant_id)
            for horizon in (30, 60):
                assert set(replay.events[horizon].participant_id) == ids
                assert set(replay.exposure[horizon].participant_id) == ids
                # One retained event and one exposure row per synthetic person;
                # endpoints are regrouped by the new role, not original split.
                assert len(replay.events[horizon]) == len(ids)
    assert sorted(outer_all) == [str(i) for i in range(9)]
    with pytest.raises(ValueError, match="cross-fitting loader"):
        load_replay_bundle(prepared / "shared", config, sequences=False)


def test_crossfit_inner_role_filter_precedes_timeline_and_normalized_value_reads(
    tmp_path, monkeypatch
):
    prepared, config = _crossfit(tmp_path)
    role_path = prepared / "folds/fold-0/participant-roles.private.json"
    roles = json.loads(role_path.read_text())["participants"]
    outer_ids = {
        entry["participant_id"] for entry in roles
        if entry["role"] == "outer_evaluation"
    }
    manifest_path = prepared / "shared/participant-manifest.private.json"
    manifest = json.loads(manifest_path.read_text())["participants"]
    forbidden_paths = {
        entry["timeline_path"] for entry in manifest
        if entry["participant_id"] in outer_ids
    }
    original_read = pq.ParquetFile
    observed_paths = []

    def guarded_parquet(path, *args, **kwargs):
        assert str(path) not in forbidden_paths
        observed_paths.append(str(path))
        return original_read(path, *args, **kwargs)

    def normalized(_config, *, sources, participant_keys):
        assert sources == {"s"}
        assert {participant for _, participant in participant_keys} == (
            {str(i) for i in range(9)} - outer_ids
        )
        return pd.DataFrame(
            sorted(participant_keys), columns=["source_id", "participant_id"]
        )

    monkeypatch.setattr("glycoshift.revision_runner.pq.ParquetFile", guarded_parquet)
    monkeypatch.setattr("glycoshift.revision_runner.load_frame", normalized)
    load_crossfit_inner_bundle(prepared, config, outer_fold=0)
    assert observed_paths


@pytest.mark.parametrize("defect", ["role", "missing", "path", "original_partition"])
def test_crossfit_metadata_defects_fail_before_timeline_values(
    tmp_path, monkeypatch, defect
):
    prepared, config = _crossfit(tmp_path)
    roles_path = prepared / "folds/fold-0/participant-roles.private.json"
    manifest_path = prepared / "shared/participant-manifest.private.json"
    path = roles_path if defect in {"role", "missing"} else manifest_path
    payload = json.loads(path.read_text())
    if defect == "role":
        payload["participants"][0]["role"] = "train"
        payload["participants"][0]["outer_fold"] = 4
    elif defect == "missing":
        payload["participants"].pop()
    elif defect == "path":
        payload["participants"][0]["timeline_path"] = (
            payload["participants"][1]["timeline_path"]
        )
    else:
        payload["participants"][0]["partition"] = "external_loop"
    path.write_text(json.dumps(payload))

    def forbidden_values(*args, **kwargs):
        pytest.fail("invalid cross-fitting metadata must fail before timeline reads")

    monkeypatch.setattr("glycoshift.revision_runner.pq.ParquetFile", forbidden_values)
    with pytest.raises(ValueError, match="binding|roles differ"):
        load_crossfit_inner_bundle(prepared, config, outer_fold=0, sequences=False)


def test_crossfit_endpoint_count_binding_rejects_missing_event(tmp_path):
    prepared, config = _crossfit(tmp_path)
    path = prepared / "shared/endpoints/train/events-30.parquet"
    events = pd.read_parquet(path)
    events.iloc[1:].to_parquet(path, index=False)
    # Every original participant is either inner or outer in this fold, so at
    # least one of the strictly separated entrypoints must catch the removal.
    with pytest.raises(ValueError, match="endpoint count"):
        load_crossfit_inner_bundle(prepared, config, outer_fold=0, sequences=False)
        load_crossfit_outer_bundle(prepared, config, outer_fold=0, sequences=False)


def test_crossfit_deep_runner_uses_inner_context_without_outer_values(
    tmp_path, monkeypatch
):
    prepared, fixture_config = _crossfit(tmp_path)
    config = load_config(Path(__file__).parents[1] / "configs/smoke.yaml")
    config["data"].update(fixture_config["data"])
    config["protocol"].update(fixture_config["protocol"])
    config = revision_config(config, seed=17, output=tmp_path / "run")
    expected_data, expected_replays = load_crossfit_inner_bundle(
        prepared, config, outer_fold=0, sequences=False
    )

    def inner_only(*args, **kwargs):
        assert kwargs == {"outer_fold": 0}
        return expected_data, expected_replays

    def train(_config, *, architecture, prepared_data, **callbacks):
        assert architecture == "tcn"
        assert set(prepared_data[1].partition) == {"train", "selection", "calibration"}
        context = callbacks["revision_checkpoint_proxy"].__self__
        assert context.sensitivity_source_policy == "supported_sources"
        return tmp_path / "run"

    monkeypatch.setattr(
        "glycoshift.revision_runner.load_crossfit_inner_bundle", inner_only
    )
    monkeypatch.setattr("glycoshift.deep_training.run_deep_training", train)
    assert run_revision_deep(config, prepared, architecture="tcn", outer_fold=0) == (
        tmp_path / "run"
    )


def test_crossfit_endpoints_require_identity_predicate_before_value_scan(
    tmp_path, monkeypatch
):
    prepared, config = _crossfit(tmp_path)
    role_path = prepared / "folds/fold-0/participant-roles.private.json"
    roles = json.loads(role_path.read_text())["participants"]
    inner_ids = {
        row["participant_id"] for row in roles if row["role"] != "outer_evaluation"
    }
    dataset = ds.dataset
    scans = []

    class GuardedDataset:
        def __init__(self, source):
            self.source = source

        def __getattr__(self, name):
            return getattr(self.source, name)

        def to_table(self, *args, **kwargs):
            assert kwargs.get("filter") is not None
            table = self.source.to_table(*args, **kwargs)
            assert set(table["participant_id"].to_pylist()).issubset(inner_ids)
            scans.append(len(table))
            return table

    def guarded_dataset(*args, **kwargs):
        return GuardedDataset(dataset(*args, **kwargs))

    monkeypatch.setattr("glycoshift.revision_runner.ds.dataset", guarded_dataset)
    load_crossfit_inner_bundle(prepared, config, outer_fold=0, sequences=False)
    assert scans


@pytest.mark.parametrize("loader", ["original", "inner", "outer"])
def test_replay_loaders_validate_corrected_confirmation_masks(tmp_path, loader):
    normalized, legacy, config = crossfit_inputs(tmp_path)
    config["protocol"]["ascertainment_contract"] = "2026-11-09-complete-confirmation"
    prepared = tmp_path / "prepared"
    if loader == "original":
        prepare_replays(normalized, legacy, prepared, config)
        manifest_path = prepared / "participant-manifest.private.json"
        manifest = json.loads(manifest_path.read_text())["participants"]
        selected = manifest[0]
    else:
        prepare_crossfit_replays(normalized, legacy, prepared, config)
        manifest_path = prepared / "shared/participant-manifest.private.json"
        manifest = json.loads(manifest_path.read_text())["participants"]
        role_path = prepared / "folds/fold-0/participant-roles.private.json"
        roles = json.loads(role_path.read_text())["participants"]
        admitted = {
            entry["participant_id"] for entry in roles
            if (entry["role"] == "outer_evaluation") == (loader == "outer")
        }
        selected = next(
            entry for entry in manifest if entry["participant_id"] in admitted
        )
    path = Path(selected["timeline_path"])
    timeline = pd.read_parquet(path)
    timeline.loc[0, "hypo_60_training_mask"] = not bool(
        timeline.loc[0, "hypo_60_training_mask"]
    )
    timeline.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="confirmation/training mask mismatch"):
        if loader == "original":
            load_replay_bundle(prepared, config, sequences=False)
        elif loader == "inner":
            load_crossfit_inner_bundle(prepared, config, outer_fold=0, sequences=False)
        else:
            load_crossfit_outer_bundle(prepared, config, outer_fold=0, sequences=False)
