from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds
import pytest

from glycoshift.adapters import write_partitioned_frame
from glycoshift.config import load_config
from glycoshift.io import write_json
from glycoshift.pipeline import (
    load_audited_development_windows,
    prepare_dataset,
    validate_training_output_isolation,
)
from glycoshift.streaming import (
    WINDOW_CATEGORICAL_COLUMNS,
    load_validated_window_shards,
    prepare_window_shards,
    validate_window_shard_audit,
)
from glycoshift.synthetic import make_synthetic_cgm


def _assert_window_values_equal(
    observed: pd.DataFrame,
    expected: pd.DataFrame,
) -> None:
    categorical = [
        column
        for column in WINDOW_CATEGORICAL_COLUMNS
        if column in observed and column in expected
    ]
    for column in categorical:
        pd.testing.assert_series_equal(
            observed[column].astype("string"),
            expected[column].astype("string"),
            check_names=False,
        )
    pd.testing.assert_frame_equal(
        observed.drop(columns=categorical),
        expected.drop(columns=categorical),
        check_dtype=False,
    )


def test_streaming_prepare_excludes_held_out_sources(tmp_path: Path) -> None:
    config = load_config(Path(__file__).parents[1] / "configs" / "smoke.yaml")
    config["data"]["participants_per_source"] = 3
    config["data"]["days_per_participant"] = 1
    config["protocol"]["context_hours"] = 0.5
    frame = make_synthetic_cgm(config)
    normalized = write_partitioned_frame(frame, tmp_path / "normalized")
    audit = prepare_window_shards(
        normalized,
        tmp_path / "windows",
        config=config,
        include_holdouts=False,
        batch_size=400,
    )
    assert audit["participants_seen"] == 6
    assert audit["participants_written"] == 6
    assert set(audit["development_event_prevalence"]) == {
        "hyper_30",
        "hyper_60",
        "hypo_30",
        "hypo_60",
    }
    window_dataset = ds.dataset(
        tmp_path / "windows",
        format="parquet",
        partitioning="hive",
    )
    partitions = set(
        window_dataset.to_table(columns=["partition"])["partition"].to_pylist()
    )
    assert partitions == {"train", "selection", "calibration"}


def test_streaming_prepare_matches_in_memory_windows(tmp_path: Path) -> None:
    config = load_config(Path(__file__).parents[1] / "configs" / "smoke.yaml")
    config["data"]["participants_per_source"] = 3
    config["data"]["days_per_participant"] = 1
    config["protocol"]["context_hours"] = 0.5
    frame = make_synthetic_cgm(config)
    normalized = write_partitioned_frame(frame, tmp_path / "normalized")

    streaming_audit = prepare_window_shards(
        normalized,
        tmp_path / "windows",
        config=config,
        include_holdouts=False,
        batch_size=400,
    )
    audit_path = tmp_path / "cohort_audit.json"
    write_json(streaming_audit, audit_path)
    _, expected, expected_audit = prepare_dataset(
        config,
        include_holdouts=False,
    )
    observed, validated_audit = load_validated_window_shards(
        tmp_path / "windows",
        audit_path,
        config=config,
    )

    sort_columns = ["source_id", "participant_id", "context_end"]
    expected = expected.sort_values(sort_columns).reset_index(drop=True)
    observed = observed.sort_values(sort_columns).reset_index(drop=True)
    _assert_window_values_equal(observed[expected.columns], expected)
    for column in ("source_id", "participant_id", "partition"):
        assert isinstance(observed[column].dtype, pd.CategoricalDtype)
    assert streaming_audit["windows_written"] == len(expected)
    assert streaming_audit["by_partition"] == {
        str(partition): int(count)
        for partition, count in expected["partition"].value_counts().items()
    }
    assert (
        streaming_audit["development_event_prevalence"]
        == expected_audit["development_event_prevalence"]
    )
    assert validated_audit == streaming_audit
    real_config = {
        **config,
        "project": {**config["project"], "synthetic": False},
        "data": {
            **config["data"],
            "training_window_shards": str(tmp_path / "windows"),
            "training_cohort_audit": str(audit_path),
        },
    }
    loaded, loaded_audit = load_audited_development_windows(real_config)
    _assert_window_values_equal(
        loaded.sort_values(sort_columns).reset_index(drop=True)[
            expected.columns
        ],
        expected,
    )
    assert loaded_audit == streaming_audit
    with pytest.raises(
        ValueError,
        match="must not reuse the preprocessing artifact directory",
    ):
        validate_training_output_isolation(
            real_config,
            audit_path.parent,
        )
    validate_training_output_isolation(
        real_config,
        tmp_path / "training-run",
    )
    validation_summary = validate_window_shard_audit(
        tmp_path / "windows",
        audit_path,
        config=config,
        batch_size=400,
    )
    assert validation_summary["files"] == streaming_audit["participants_written"]
    assert validation_summary["participants"] == streaming_audit[
        "participants_written"
    ]
    assert validation_summary["windows"] == streaming_audit["windows_written"]
    assert validation_summary["by_partition"] == streaming_audit["by_partition"]
    assert (
        validation_summary["development_event_prevalence"]
        == streaming_audit["development_event_prevalence"]
    )

    invalid_audit = dict(streaming_audit)
    invalid_audit["windows_written"] += 1
    invalid_audit_path = tmp_path / "invalid-cohort-audit.json"
    write_json(invalid_audit, invalid_audit_path)
    with pytest.raises(
        ValueError,
        match="window shard row count conflicts with audit",
    ):
        load_validated_window_shards(
            tmp_path / "windows",
            invalid_audit_path,
            config=config,
        )
    with pytest.raises(
        ValueError,
        match="window shard row count conflicts with audit",
    ):
        validate_window_shard_audit(
            tmp_path / "windows",
            invalid_audit_path,
            config=config,
            batch_size=400,
        )


def test_prevalence_mask_selection_matches_full_frame_copy():
    """The eligibility mask is applied per column, not by copying the frame.

    `load_validated_window_shards` used to materialise `windows.loc[mask]` --
    every column of a 27.5M-row frame -- and then read a single label column
    out of it. Selecting the column inside `.loc` removes that duplicate from
    the process peak, and the peak decides how many training lanes fit on a
    host. This asserts the two forms agree exactly on the three statistics the
    audit compares, on a frame carrying the awkward parts of the real one:
    nulls in the labels, a categorical, and mixed dtypes.
    """

    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(20260815)
    rows = 4096
    frame = pd.DataFrame(
        {
            "source_id": pd.Categorical(
                rng.choice(["a", "b", "c"], size=rows)
            ),
            "participant_id": [f"p{index % 97}" for index in range(rows)],
            "hypo_currently_active": rng.random(rows) < 0.31,
            "hyper_currently_active": rng.random(rows) < 0.24,
            "padding_f64": rng.normal(size=rows),
            "padding_i64": rng.integers(0, 1000, size=rows),
        }
    )
    for kind in ("hypo", "hyper"):
        for horizon in (30, 60):
            labels = rng.random(rows)
            labels[rng.random(rows) < 0.17] = np.nan
            frame[f"{kind}_{horizon}_label"] = np.where(
                np.isnan(labels), np.nan, (labels > 0.5).astype(float)
            )

    def statistics(series: pd.Series) -> dict[str, float]:
        cleaned = series.dropna().astype(int)
        return {
            "eligible_windows": int(len(cleaned)),
            "positive_windows": int(cleaned.sum()),
            "prevalence": float(cleaned.mean()),
        }

    for kind in ("hypo", "hyper"):
        mask = ~frame[f"{kind}_currently_active"].astype(bool)
        copied = frame.loc[mask]  # the previous, full-width form
        assert copied.shape[1] == frame.shape[1]
        for horizon in (30, 60):
            column = f"{kind}_{horizon}_label"
            before = statistics(copied[column])
            after = statistics(frame.loc[mask, column])
            assert before == after, (kind, horizon, before, after)
            # identity of the selected series itself, not merely of the summary
            pd.testing.assert_series_equal(
                copied[column], frame.loc[mask, column]
            )


def test_window_audit_derivations_avoid_whole_frame_stringification():
    """Audit derivations read a few columns, not a stringified copy of the frame.

    `load_validated_window_shards` used to compute `set(col.astype(str))` over
    39.3M rows to obtain three partition names, and to call `windows.assign(...)`
    -- a copy of all 51 columns -- to stringify the three the grouping needs.
    Both forms are compared here against the cheap ones on a frame with unused
    categories (what a partition slice leaves behind) and a null, because those
    are the cases where deduplicating first could have diverged.
    """

    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(20260815)
    rows = 8192
    partitions = pd.Categorical(
        rng.choice(["train", "selection", "calibration"], size=rows),
        categories=["train", "selection", "calibration", "unused_partition"],
    )
    sources = pd.Categorical(
        rng.choice(["src_a", "src_b", "src_c"], size=rows),
        categories=["src_a", "src_b", "src_c", "src_unused"],
    )
    frame = pd.DataFrame(
        {
            "source_id": sources,
            "participant_id": pd.Categorical(
                [f"p{index % 53}" for index in range(rows)]
            ),
            "partition": partitions,
        }
    )
    # every window of one participant sits in one partition, as the audit requires
    assignment = {
        participant: rng.choice(["train", "selection", "calibration"])
        for participant in frame["participant_id"].unique()
    }
    frame["partition"] = pd.Categorical(
        [assignment[participant] for participant in frame["participant_id"]],
        categories=["train", "selection", "calibration", "unused_partition"],
    )
    for _ in range(4):
        frame[f"padding_{_}"] = rng.normal(size=rows)

    assert set(map(str, frame["partition"].unique())) == set(
        frame["partition"].astype(str)
    )
    assert set(map(str, frame["source_id"].unique())) == set(
        frame["source_id"].astype(str)
    )

    previous = (
        frame.assign(
            source_id=frame["source_id"].astype(str),
            participant_id=frame["participant_id"].astype(str),
            partition=frame["partition"].astype(str),
        )
        .groupby(["source_id", "participant_id"])["partition"]
        .nunique()
    )
    current = (
        frame[["source_id", "participant_id", "partition"]]
        .astype(str)
        .groupby(["source_id", "participant_id"])["partition"]
        .nunique()
    )
    pd.testing.assert_series_equal(previous, current)
    expected_groups = len(
        frame[["source_id", "participant_id"]].astype(str).drop_duplicates()
    )
    assert int(len(current)) == expected_groups
    # the audit's own check: one partition per participant
    assert not (current != 1).any()


def test_dataset_source_index_order_survives_cheap_unique():
    """`source_to_index` must be identical, because it indexes sources into the model.

    WindowSequenceDataset derived its source ordering with
    `windows["source_id"].astype(str).unique()`, stringifying every window row
    once per dataset. Deduplicating first is only admissible if the sorted
    result is unchanged, including after a partition slice leaves categories
    unused and when the column is not categorical at all.
    """

    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(20260815)
    cases = {
        "categorical_all_used": pd.Series(
            pd.Categorical(rng.choice(["s_b", "s_a", "s_c"], size=4096))
        ),
        "categorical_unused_left_over": pd.Series(
            pd.Categorical(
                rng.choice(["s_b", "s_a"], size=4096),
                categories=["s_a", "s_b", "s_c", "s_zz"],
            )
        ),
        "object_dtype": pd.Series(rng.choice(["s_b", "s_a", "s_c"], size=4096)),
        "single_source": pd.Series(pd.Categorical(["s_only"] * 4096)),
    }
    for name, column in cases.items():
        previous = sorted(column.astype(str).unique())
        current = sorted(map(str, column.unique()))
        assert previous == current, name
        assert {source: index for index, source in enumerate(previous)} == {
            source: index for index, source in enumerate(current)
        }, name
