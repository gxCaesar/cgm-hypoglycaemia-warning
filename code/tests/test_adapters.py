import numpy as np
import pandas as pd
import pytest

from glycoshift.adapters import SourceMap, SourceRule, normalize_metabonet


def test_metabonet_adapter_requires_reviewed_source_mapping() -> None:
    raw = pd.DataFrame(
        {
            "source_file": ["study-a"] * 6,
            "id": ["p1"] * 6,
            "date": pd.date_range("2026-01-01", periods=6, freq="5min"),
            "CGM": [100, 101, np.nan, 103, 104, 105],
            "basal": [0.0, 0.1, np.nan, 0.1, 0.1, 0.1],
            "bolus": [np.nan] * 6,
            "carbs": [np.nan] * 6,
        }
    )
    source_map = SourceMap(
        source_column="source_file",
        rules=(SourceRule(source_id="CTR3", value="study-a"),),
    )
    normalized = normalize_metabonet(raw, source_map=source_map)
    assert set(normalized["source_id"]) == {"CTR3"}
    assert normalized["cgm_mg_dl"].isna().sum() == 0
    assert normalized["cgm_interpolated"].sum() == 1
    assert normalized["basal_insulin_u_zero_ambiguous"].iloc[0]
    assert not normalized["timezone_known"].any()


def test_unmapped_source_fails_closed() -> None:
    raw = pd.DataFrame(
        {
            "source_file": ["unexpected"],
            "id": ["p1"],
            "date": ["2026-01-01"],
            "CGM": [100],
        }
    )
    source_map = SourceMap(
        source_column="source_file",
        rules=(SourceRule(source_id="CTR3", value="study-a"),),
    )
    with pytest.raises(ValueError, match="matched 0"):
        normalize_metabonet(raw, source_map=source_map)


def test_out_of_range_cgm_is_never_interpolated_when_excluded() -> None:
    raw = pd.DataFrame(
        {
            "source_file": ["uom"] * 5,
            "id": ["p1"] * 5,
            "date": pd.date_range("2026-01-01", periods=5, freq="5min"),
            "CGM": [55.0, 1.8018, 1.8018, 50.0, 60.0],
        }
    )
    source_map = SourceMap(
        source_column="source_file",
        rules=(SourceRule(source_id="T1D-UOM", value="uom"),),
    )

    with pytest.raises(ValueError, match="2 values outside 20–600"):
        normalize_metabonet(raw, source_map=source_map)

    normalized = normalize_metabonet(
        raw,
        source_map=source_map,
        glucose_range_policy="exclude_noninterpolable",
    )
    assert normalized["cgm_mg_dl"].tolist() == [55.0, 50.0, 60.0]
    assert not normalized["cgm_interpolated"].any()
    assert not normalized["cgm_lower_censored"].any()
    assert (
        normalized["timestamp"].iloc[1] - normalized["timestamp"].iloc[0]
        == pd.Timedelta(minutes=15)
    )


def test_lower_censor_policy_preserves_event_state_without_interpolation() -> None:
    raw = pd.DataFrame(
        {
            "source_file": ["uom"] * 4,
            "id": ["p1"] * 4,
            "date": pd.date_range("2026-01-01", periods=4, freq="5min"),
            "CGM": [55.0, 1.8018, 1.8018, 50.0],
        }
    )
    source_map = SourceMap(
        source_column="source_file",
        rules=(SourceRule(source_id="T1D-UOM", value="uom"),),
    )

    normalized = normalize_metabonet(
        raw,
        source_map=source_map,
        glucose_range_policy="left_censor",
        lower_censor_proxy_mg_dl=39.0,
    )
    assert normalized["cgm_mg_dl"].tolist() == [55.0, 39.0, 39.0, 50.0]
    assert normalized["cgm_lower_censored"].tolist() == [
        False,
        True,
        True,
        False,
    ]
    assert not normalized["cgm_interpolated"].any()
