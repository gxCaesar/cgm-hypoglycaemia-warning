import copy
import json

import pandas as pd
import pytest
from test_revision_inputs import fixture

from glycoshift.revision_prepare import aggregate_denominators, prepare_replays
from glycoshift.windows import build_windows


def inputs(tmp_path):
    raw, config = fixture()
    config = copy.deepcopy(config)
    config["data"].update(
        development_sources=["s"], forbidden_development_sources=["external"]
    )
    normalized, legacy = tmp_path / "normalized", tmp_path / "legacy"
    normalized.mkdir()
    legacy.mkdir()
    for index, partition in enumerate(("train", "selection", "calibration")):
        frame = raw.copy()
        frame["participant_id"] = str(index)
        frame.loc[35:38, "cgm_mg_dl"] = 50
        frame.loc[20, ["cgm_observed", "cgm_interpolated"]] = [False, True]
        frame.to_parquet(normalized / f"part-{index}.parquet", index=False)
        old = build_windows(frame, config)
        old["partition"] = partition
        old.to_parquet(legacy / f"part-{index}.parquet", index=False)
    # An originally excluded participant must not be silently admitted.
    excluded = raw.copy()
    excluded["participant_id"] = "excluded"
    excluded.to_parquet(normalized / "excluded.parquet", index=False)
    return normalized, legacy, config


def test_prepare_keeps_original_partitions_and_immutable_denominators(tmp_path):
    normalized, legacy, config = inputs(tmp_path)
    output = tmp_path / "output"
    result = prepare_replays(normalized, legacy, output, config)
    assert result["status"] == "passed"
    assert result["original_participants"] == 3
    assert result["original_excluded_participants_not_added"] == 1
    counts = result["source_partition_counts"]
    assert all(row["legacy_interpolated_anchor_decisions"] == 1 for row in counts)
    manifest = json.loads((output / "participant-manifest.private.json").read_text())
    assert {r["partition"] for r in manifest["participants"]} == {
        "train",
        "selection",
        "calibration",
    }
    denominators = pd.read_csv(output / "denominator-audit.csv")
    for _, group in denominators.groupby(["partition", "horizon"]):
        assert group.base_events.nunique() == group.base_days.nunique() == 1
        assert group.base_events.iloc[0] == 1
    with pytest.raises(FileExistsError):
        prepare_replays(normalized, legacy, output, config)


def test_mismatched_retained_input_stops_preparation(tmp_path):
    normalized, legacy, config = inputs(tmp_path)
    path = legacy / "part-0.parquet"
    changed = pd.read_parquet(path)
    changed.loc[0, "cgm_last"] += 1
    changed.to_parquet(path, index=False)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="anchor glucose mismatch"):
        prepare_replays(normalized, legacy, output, config)
    failure = json.loads((output / "preparation-failure.json").read_text())
    assert failure["status"] == "failed"
    assert not (output / "input-audit.json").exists()


def test_original_participant_overlap_fails_closed(tmp_path):
    normalized, legacy, config = inputs(tmp_path)
    duplicate = pd.read_parquet(legacy / "part-0.parquet")
    duplicate["partition"] = "selection"
    duplicate.to_parquet(legacy / "overlap.parquet", index=False)
    with pytest.raises(ValueError, match="overlaps partitions"):
        prepare_replays(normalized, legacy, tmp_path / "output", config)


def test_aggregate_does_not_invent_categorical_source_partition_cells():
    rows = pd.DataFrame(
        {
            "source_id": pd.Categorical(["s", "t"], categories=["s", "t", "unused"]),
            "partition": pd.Categorical(
                ["train", "selection"], categories=["train", "selection"]
            ),
            "horizon": [30, 60],
            "cut": [54, 70],
            "base_events": [3, 4],
        }
    )
    result = aggregate_denominators(rows)
    assert len(result) == 2
    assert result.base_events.tolist() == [3, 4]
