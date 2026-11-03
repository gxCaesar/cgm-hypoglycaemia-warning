from pathlib import Path

import pandas as pd
import pytest

from glycoshift.config import load_config
from glycoshift.pipeline import prepare_dataset
from glycoshift.split import participant_split


def _frame() -> pd.DataFrame:
    rows = []
    for source in ("dev", "HUPA-UCM"):
        for participant in range(4):
            rows.append(
                {
                    "source_id": source,
                    "participant_id": f"{source}-{participant}",
                }
            )
    return pd.DataFrame(rows)


def test_external_source_is_never_in_development() -> None:
    partitions = participant_split(
        _frame(),
        development_sources=["dev"],
        external_source="HUPA-UCM",
        forbidden_development_sources=["HUPA-UCM", "OhioT1DM"],
        validation_fraction=0.25,
        seed=42,
    )
    assert all(source == "HUPA-UCM" for source, _ in partitions["external"])
    assert not partitions["train"].intersection(partitions["external"])


def test_forbidden_source_fails_closed() -> None:
    with pytest.raises(ValueError, match="forbidden"):
        participant_split(
            _frame(),
            development_sources=["dev", "HUPA-UCM"],
            external_source="HUPA-UCM",
            forbidden_development_sources=["HUPA-UCM"],
            validation_fraction=0.25,
            seed=42,
        )


def test_three_way_development_split_and_stress_source() -> None:
    frame = _frame()
    partitions = participant_split(
        frame,
        development_sources=["dev"],
        external_source="HUPA-UCM",
        forbidden_development_sources=["HUPA-UCM"],
        train_fraction=0.5,
        selection_fraction=0.25,
        calibration_fraction=0.25,
        seed=42,
    )
    assert len(partitions["train"]) == 2
    assert len(partitions["selection"]) == 1
    assert len(partitions["calibration"]) == 1
    assert len(partitions["external_loop"]) == 4


def test_development_preparation_does_not_load_holdout_rows() -> None:
    config = load_config(
        Path(__file__).parents[1] / "configs" / "smoke.yaml"
    )
    config["data"]["participants_per_source"] = 3
    config["data"]["days_per_participant"] = 1
    frame, windows, _ = prepare_dataset(config, include_holdouts=False)
    assert set(frame["source_id"]) == {"dev_alpha", "dev_beta"}
    assert set(windows["partition"]) == {
        "train",
        "selection",
        "calibration",
    }
