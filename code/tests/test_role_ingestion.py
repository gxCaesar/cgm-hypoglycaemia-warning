import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
import pyarrow.dataset as ds
import pytest

from glycoshift.adapters import SourceMap, SourceRule, sha256_file
from glycoshift.role_ingest import (
    ingest_hupa_archive,
    ingest_metabonet_role,
)
from glycoshift.streaming import iter_participants


def test_metabonet_role_ingest_excludes_unselected_values_before_normalizing(
    tmp_path: Path,
) -> None:
    timestamps = pd.date_range("2026-01-01", periods=6, freq="5min")
    raw = pd.DataFrame(
        {
            "source_file": ["dev-raw"] * 12 + ["loop-raw"] * 6,
            "id": ["dev-p1"] * 6 + ["dev-p2"] * 6 + ["loop-p1"] * 6,
            "date": [*timestamps, *timestamps, *timestamps],
            "CGM": [
                100,
                101,
                102,
                103,
                104,
                105,
                110,
                111,
                112,
                113,
                114,
                115,
                *([9999] * 6),
            ],
            "basal": [0.1] * 18,
            "bolus": [0.0] * 18,
            "carbs": [0.0] * 18,
            "steps": [0.0] * 18,
        }
    )
    input_path = tmp_path / "release.parquet"
    raw.to_parquet(input_path, index=False)
    source_map = SourceMap(
        source_column="source_file",
        rules=(
            SourceRule(source_id="CTR3", value="dev-raw"),
            SourceRule(
                source_id="Loop Observational Study",
                value="loop-raw",
            ),
        ),
    )
    output_path = tmp_path / "development"
    manifest_path = tmp_path / "development-manifest.json"

    manifest = ingest_metabonet_role(
        input_path,
        output_path,
        manifest_path,
        source_map=source_map,
        canonical_sources={"CTR3"},
        role="development",
        release_id="release-v1",
        batch_size=4,
    )

    normalized = ds.dataset(
        output_path,
        format="parquet",
        partitioning="hive",
    ).to_table().to_pandas()
    assert set(normalized["source_id"]) == {"CTR3"}
    assert normalized["cgm_mg_dl"].max() == 115
    assert manifest["role"] == "development"
    assert manifest["origin"] == "metabonet-public"
    assert manifest["sources"] == {"CTR3": {"participants": 2, "rows": 12}}
    assert json.loads(manifest_path.read_text()) == manifest
    shards = list(
        iter_participants(
            output_path,
            batch_size=2,
            sources={"CTR3"},
        )
    )
    assert {
        str(participant["participant_id"].iloc[0])
        for participant in shards
    } == {"dev-p1", "dev-p2"}


def test_metabonet_range_amendment_is_hash_bound_and_audited(
    tmp_path: Path,
) -> None:
    raw = pd.DataFrame(
        {
            "source_file": ["uom"] * 6,
            "id": ["p1"] * 6,
            "date": pd.date_range("2026-01-01", periods=6, freq="5min"),
            "CGM": [55.0, 1.8018, 50.0, 60.0, 70.0, 80.0],
            "steps": [0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    input_path = tmp_path / "release.parquet"
    raw.to_parquet(input_path, index=False)
    source_map = SourceMap(
        source_column="source_file",
        rules=(SourceRule(source_id="T1D-UOM", value="uom"),),
    )
    output_path = tmp_path / "development"
    manifest_path = tmp_path / "development-manifest.json"

    manifest = ingest_metabonet_role(
        input_path,
        output_path,
        manifest_path,
        source_map=source_map,
        canonical_sources={"T1D-UOM"},
        role="development",
        release_id="v1.1",
        batch_size=3,
        glucose_range_policy="exclude_noninterpolable",
        allowed_invalid_glucose_sources={"T1D-UOM"},
        maximum_invalid_glucose_rows=1,
        approved_invalid_glucose_values_mg_dl=(1.8018,),
        expected_input_sha256=sha256_file(input_path),
        expected_release_id="v1.1",
    )

    normalized = ds.dataset(
        output_path,
        format="parquet",
        partitioning="hive",
    ).to_table().to_pandas()
    assert len(normalized) == 5
    assert 1.8018 not in normalized["cgm_mg_dl"].tolist()
    assert manifest["glucose_range_audit"] == {
        "minimum_mg_dl": 20.0,
        "maximum_mg_dl": 600.0,
        "nonmissing_rows": 6,
        "out_of_range_rows": 1,
        "below_minimum_rows": 1,
        "above_maximum_rows": 0,
        "by_source": {"T1D-UOM": 1},
        "observed_out_of_range_values_mg_dl": [1.8018],
        "policy": "exclude_noninterpolable",
        "action": "excluded_without_interpolation",
        "release_sha256_verified": True,
    }


def test_metabonet_range_failure_happens_before_output_creation(
    tmp_path: Path,
) -> None:
    raw = pd.DataFrame(
        {
            "source_file": ["uom"],
            "id": ["p1"],
            "date": [pd.Timestamp("2026-01-01")],
            "CGM": [1.8018],
        }
    )
    input_path = tmp_path / "release.parquet"
    raw.to_parquet(input_path, index=False)
    source_map = SourceMap(
        source_column="source_file",
        rules=(SourceRule(source_id="T1D-UOM", value="uom"),),
    )
    output_path = tmp_path / "development"
    manifest_path = tmp_path / "development-manifest.json"

    with pytest.raises(ValueError, match="1 values outside 20–600"):
        ingest_metabonet_role(
            input_path,
            output_path,
            manifest_path,
            source_map=source_map,
            canonical_sources={"T1D-UOM"},
            role="development",
            release_id="v1.1",
        )
    assert not output_path.exists()
    assert not manifest_path.exists()


def test_direct_hupa_ingest_converts_release_units_and_marks_ambiguous_zeros(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "hupa.zip"
    csv_text = "\n".join(
        (
            "time;glucose;calories;heart_rate;steps;basal_rate;"
            "bolus_volume_delivered;carb_input",
            "2026-01-01T00:00:00;100;1;80;0;0;0;0",
            "2026-01-01T00:05:00;101;1;80;3;0.05;0;0",
            "2026-01-01T00:10:00;102;1;80;0;0.05;1.5;1",
            "2026-01-01T00:15:00;103;1;80;0;0.05;0;0",
            "2026-01-01T00:20:00;104;1;80;0;0.05;0;0",
            "2026-01-01T00:25:00;105;1;80;0;0.05;0;0",
        )
    )
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(
            "HUPA-UCM Diabetes Dataset/Preprocessed/HUPA0001P.csv",
            csv_text,
        )
    output_path = tmp_path / "hupa-normalized"
    manifest_path = tmp_path / "hupa-manifest.json"

    manifest = ingest_hupa_archive(
        archive_path,
        output_path,
        manifest_path,
        release_id="doi:10.17632/3hbcscwz44.1",
        expected_participant_count=1,
    )

    normalized = ds.dataset(
        output_path,
        format="parquet",
        partitioning="hive",
    ).to_table().to_pandas()
    assert set(normalized["source_id"]) == {"HUPA-UCM"}
    assert set(normalized["participant_id"]) == {"HUPA0001P"}
    assert normalized.loc[2, "carbs_g"] == 10
    assert normalized.loc[1, "basal_insulin_u"] == 0.05
    assert normalized.loc[2, "bolus_insulin_u"] == 1.5
    assert normalized.loc[0, "basal_insulin_u_zero_ambiguous"]
    assert normalized.loc[0, "carbs_g_zero_ambiguous"]
    assert not normalized["activity_value_zero_ambiguous"].any()
    assert normalized["cgm_preprocessed_by_source"].all()
    assert manifest["origin"] == "mendeley-direct"
    assert manifest["role"] == "stress_hupa_ucm"
    assert manifest["sources"] == {
        "HUPA-UCM": {"participants": 1, "rows": 6}
    }


def _two_participant_release(tmp_path: Path):
    """Three participants, four samples each. Two are excluded, one survives, which
    is the shape of the real amendment: two of 845."""
    raw = pd.DataFrame(
        {
            "source_file": ["loop"] * 12,
            "id": ["79"] * 4 + ["1001"] * 4 + ["keep"] * 4,
            "date": list(pd.date_range("2026-01-01", periods=4, freq="5min"))
            + list(pd.date_range("2026-03-01", periods=4, freq="5min"))
            + list(pd.date_range("2026-05-01", periods=4, freq="5min")),
            "CGM": [90.0, 95.0, 100.0, 105.0, 110.0, 115.0,
                    120.0, 125.0, 130.0, 135.0, 140.0, 145.0],
        }
    )
    path = tmp_path / "release.parquet"
    raw.to_parquet(path, index=False)
    return path, SourceMap(
        source_column="source_file",
        rules=(SourceRule(source_id="Loop Observational Study", value="loop"),),
    )


def _ingest_loop(tmp_path: Path, path, source_map, **extra):
    return ingest_metabonet_role(
        path,
        tmp_path / "out",
        tmp_path / "manifest.json",
        source_map=source_map,
        canonical_sources={"Loop Observational Study"},
        role="external_loop",
        release_id="v1.1",
        batch_size=4,
        **extra,
    )


def test_duplicate_participant_exclusion_drops_them_and_is_recorded(
    tmp_path: Path,
) -> None:
    # DATA_QUALITY_AMENDMENT_LOOP_DUPLICATE_PARTICIPANTS.md. The endpoint is an
    # equal-source PARTICIPANT macro, so one record enrolled twice is weighted twice.
    path, source_map = _two_participant_release(tmp_path)

    manifest = _ingest_loop(
        tmp_path, path, source_map,
        excluded_participants=frozenset(
            {("Loop Observational Study", "79"),
             ("Loop Observational Study", "1001")}),
        expected_excluded_rows=8,
    )

    assert manifest["sources"]["Loop Observational Study"]["participants"] == 1
    assert manifest["duplicate_participant_exclusion"] == {
        "participants": ["Loop Observational Study/1001",
                         "Loop Observational Study/79"],
        "rows_excluded": 8,
        "amendment": "DATA_QUALITY_AMENDMENT_LOOP_DUPLICATE_PARTICIPANTS.md",
    }


def test_duplicate_exclusion_refuses_a_different_row_count(tmp_path: Path) -> None:
    # A release whose duplicated rows are not the reviewed ones has not been reviewed.
    path, source_map = _two_participant_release(tmp_path)

    with pytest.raises(ValueError, match="covers 63388 rows, this release has 8"):
        _ingest_loop(
            tmp_path, path, source_map,
            excluded_participants=frozenset(
                {("Loop Observational Study", "79"),
                 ("Loop Observational Study", "1001")}),
            expected_excluded_rows=63388,
        )


def test_duplicate_exclusion_refuses_a_participant_that_is_not_there(
    tmp_path: Path,
) -> None:
    # Approving an identifier the release does not contain means the approval was
    # written against different data, so it does not carry.
    path, source_map = _two_participant_release(tmp_path)

    with pytest.raises(ValueError, match="does not match the data"):
        _ingest_loop(
            tmp_path, path, source_map,
            excluded_participants=frozenset(
                {("Loop Observational Study", "79"),
                 ("Loop Observational Study", "absent-id")}),
            expected_excluded_rows=4,
        )


def test_no_exclusion_leaves_both_participants_in(tmp_path: Path) -> None:
    # The default path is unchanged, which is what makes the exclusion an amendment
    # rather than a silent behaviour change.
    path, source_map = _two_participant_release(tmp_path)

    manifest = _ingest_loop(tmp_path, path, source_map)

    assert manifest["sources"]["Loop Observational Study"]["participants"] == 3
    assert "duplicate_participant_exclusion" not in manifest
