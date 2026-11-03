from pathlib import Path

import pytest
import yaml

from glycoshift.adapters import write_partitioned_frame
from glycoshift.config import load_config
from glycoshift.io import (
    _validate_real_data_provenance,
    load_evaluation_frame,
    load_frame,
)
from glycoshift.synthetic import make_synthetic_cgm


def test_load_frame_restricts_source_scoped_participant_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(Path(__file__).parents[1] / "configs" / "smoke.yaml")
    config["data"]["participants_per_source"] = 3
    config["data"]["days_per_participant"] = 1
    full = make_synthetic_cgm(config, seed=int(config["split"]["seed"]))
    development = full.loc[
        full["source_id"].isin(config["data"]["development_sources"])
    ]
    requested = set(
        development[["source_id", "participant_id"]]
        .drop_duplicates()
        .sort_values(["source_id", "participant_id"])
        .groupby("source_id", sort=True)
        .head(1)
        .itertuples(index=False, name=None)
    )

    observed = load_frame(
        config,
        sources=set(config["data"]["development_sources"]),
        participant_keys=requested,
    )

    observed_keys = set(
        observed[["source_id", "participant_id"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    assert observed_keys == requested

    normalized = write_partitioned_frame(full, tmp_path / "normalized")
    real_config = {
        **config,
        "project": {**config["project"], "synthetic": False},
        "data": {**config["data"], "input_path": str(normalized)},
    }
    monkeypatch.setattr(
        "glycoshift.io._validate_real_data_provenance",
        lambda _config: None,
    )
    pushed_down = load_frame(
        real_config,
        sources=set(config["data"]["development_sources"]),
        participant_keys=requested,
    )
    pushed_down_keys = set(
        pushed_down[["source_id", "participant_id"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    assert pushed_down_keys == requested


def test_real_data_provenance_requires_development_only_manifest(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.yaml"
    source_map_path = tmp_path / "source-map.yaml"
    source_map_path.write_text("source_column: source_file\nrules: []\n")
    manifest = {
        "release_id": "release-1",
        "input_file": "authorized.parquet",
        "sha256": "a" * 64,
        "source_map": source_map_path.name,
        "sources": {
            "dev": {},
        },
        "role": "development",
        "origin": "metabonet-public",
        "glucose_range_audit": {
            "policy": "fail",
            "minimum_mg_dl": 20,
            "maximum_mg_dl": 600,
            "out_of_range_rows": 0,
            "by_source": {},
            "release_sha256_verified": False,
        },
    }
    manifest_path.write_text(yaml.safe_dump(manifest))
    config = {
        "_config_path": str(tmp_path / "config.yaml"),
        "data": {
            "source_manifest": manifest_path.name,
            "source_map": source_map_path.name,
            "development_sources": ["dev"],
            "external_source": "Loop",
            "stress_sources": ["HUPA-UCM"],
            "forbidden_development_sources": ["Loop", "HUPA-UCM"],
            "glucose_range": {
                "policy": "fail",
                "minimum_mg_dl": 20,
                "maximum_mg_dl": 600,
                "maximum_rows": 0,
                "approved_release_id": "release-1",
                "approved_release_sha256": "a" * 64,
            },
        },
    }
    _validate_real_data_provenance(config)
    manifest["sources"]["Loop"] = {}
    manifest_path.write_text(yaml.safe_dump(manifest))
    with pytest.raises(ValueError, match="held-out"):
        _validate_real_data_provenance(config)


def test_real_evaluation_frame_requires_isolated_direct_hupa_manifest(
    tmp_path: Path,
) -> None:
    config = load_config(Path(__file__).parents[1] / "configs" / "smoke.yaml")
    config["project"]["synthetic"] = False
    frame = make_synthetic_cgm(
        {
            **config,
            "project": {**config["project"], "synthetic": True},
        }
    )
    hupa = frame.loc[frame["source_id"] == "HUPA-UCM"].copy()
    evaluation_path = write_partitioned_frame(
        hupa,
        tmp_path / "hupa-normalized",
    )
    development_path = tmp_path / "development-normalized"
    development_path.mkdir()
    manifest_path = tmp_path / "hupa-manifest.yaml"
    manifest = {
        "release_id": "doi:10.17632/3hbcscwz44.1",
        "input_file": "HUPA-UCM Diabetes Dataset.zip",
        "sha256": "a" * 64,
        "source_map": None,
        "role": "stress_hupa_ucm",
        "origin": "mendeley-direct",
        "sources": {"HUPA-UCM": {"participants": 2, "rows": len(hupa)}},
    }
    manifest_path.write_text(yaml.safe_dump(manifest))
    config["data"]["input_path"] = str(development_path)
    config["data"]["evaluation_inputs"] = {
        "stress_hupa_ucm": {
            "input_path": str(evaluation_path),
            "source_manifest": str(manifest_path),
            "expected_source": "HUPA-UCM",
            "required_origin": "mendeley-direct",
        }
    }

    loaded = load_evaluation_frame(config, "stress_hupa_ucm")
    assert set(loaded["source_id"]) == {"HUPA-UCM"}

    manifest["origin"] = "metabonet-public"
    manifest_path.write_text(yaml.safe_dump(manifest))
    with pytest.raises(ValueError, match="origin"):
        load_evaluation_frame(config, "stress_hupa_ucm")
