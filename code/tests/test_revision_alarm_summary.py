from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from glycoshift.revision_alarm_summary import summarize_alarm_glucose


def _alarms():
    return pd.DataFrame(
        {
            "source_id": ["A", "A", "B"],
            "participant_id": ["private-person", "private-person", "other-person"],
            "context_end": pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC"),
            "cgm_last": [60.0, 100.0, 200.0],
            "base_evaluable": [True, True, False],
            "matched": [True, False, False],
            "unmatched": [False, True, False],
        }
    )


def test_exact_categories_and_quantiles_without_private_output_or_mutation():
    alarms = _alarms()
    before = alarms.copy(deep=True)
    rows = summarize_alarm_glucose(alarms, source_order=("A", "B", "C"))
    lookup = {(r["source_id"], r["alarm_category"]): r for r in rows}
    assert lookup[None, "emitted"]["alarm_count"] == 3
    assert lookup[None, "emitted"]["median"] == 100
    assert lookup[None, "evaluable"]["alarm_count"] == 2
    assert lookup[None, "evaluable"]["q25"] == 70
    assert lookup[None, "matched"]["mean"] == 60
    assert lookup[None, "unmatched"]["mean"] == 100
    assert lookup["B", "emitted"]["alarm_count"] == 1
    assert lookup["B", "unmatched"]["alarm_count"] == 0
    assert lookup["C", "emitted"]["median"] is None
    text = json.dumps(rows, allow_nan=False)
    assert "private-person" not in text and "2026-01" not in text
    pd.testing.assert_frame_equal(before, alarms)


def test_empty_ledger_retains_declared_sources_and_defined_zero_counts():
    rows = summarize_alarm_glucose(_alarms().iloc[:0], source_order=("A",))
    assert len(rows) == 8
    assert all(row["alarm_count"] == 0 and row["mean"] is None for row in rows)
    json.dumps(rows, allow_nan=False)


@pytest.mark.parametrize("defect", ["duplicate", "source", "nonfinite", "status"])
def test_known_bad_ledger_fails(defect):
    alarms = _alarms()
    if defect == "duplicate":
        alarms = pd.concat([alarms, alarms.iloc[[0]]], ignore_index=True)
    elif defect == "source":
        alarms.loc[0, "source_id"] = "undeclared"
    elif defect == "nonfinite":
        alarms.loc[0, "cgm_last"] = np.nan
    else:
        alarms.loc[2, "unmatched"] = True
    with pytest.raises(ValueError):
        summarize_alarm_glucose(alarms, source_order=("A", "B"))
