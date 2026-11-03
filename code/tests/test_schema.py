import numpy as np
import pandas as pd
import pytest

from glycoshift.adapters import SourceMap, SourceRule, normalize_metabonet
from glycoshift.schema import validate_frame


def test_cross_participant_duplicate_cgm_day_fails_closed() -> None:
    sample_count = 96
    raw = pd.DataFrame(
        {
            "source_file": ["study-a"] * sample_count,
            "id": ["p1"] * sample_count,
            "date": pd.date_range(
                "2026-01-01",
                periods=sample_count,
                freq="5min",
            ),
            "CGM": 110 + np.sin(np.arange(sample_count) / 8),
        }
    )
    source_map = SourceMap(
        source_column="source_file",
        rules=(
            SourceRule(source_id="CTR3", value="study-a"),
            SourceRule(source_id="DCLP3", value="study-b"),
        ),
    )
    first = normalize_metabonet(raw, source_map=source_map)
    copied_raw = raw.assign(source_file="study-b", id="p2")
    copied = normalize_metabonet(copied_raw, source_map=source_map)
    with pytest.raises(ValueError, match="duplicate CGM days"):
        validate_frame(pd.concat([first, copied], ignore_index=True))
