import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_revision_prepare import inputs

from glycoshift.config import load_config
from glycoshift.revision_prepare import prepare_replays
from glycoshift.revision_runner import (
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
