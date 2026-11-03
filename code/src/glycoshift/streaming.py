from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from .schema import cgm_day_fingerprints, validate_frame
from .split import participant_split
from .windows import build_windows

WINDOW_CATEGORICAL_COLUMNS = (
    "source_id",
    "participant_id",
    "partition",
    "hypo_30_event_id",
    "hypo_60_event_id",
    "severe_hypo_30_event_id",
    "severe_hypo_60_event_id",
    "hyper_30_event_id",
    "hyper_60_event_id",
)


def _compact_window_table(table: pa.Table) -> pd.DataFrame:
    categories = [
        column
        for column in WINDOW_CATEGORICAL_COLUMNS
        if column in table.column_names
        and table[column].null_count < table.num_rows
    ]
    frame = table.to_pandas(
        categories=categories,
        split_blocks=True,
        self_destruct=True,
    )
    for column in categories:
        ordered = sorted(frame[column].cat.categories, key=str)
        frame[column] = frame[column].cat.reorder_categories(ordered)
    return frame


def _key(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["source_id"].astype(str)
        + "\x1f"
        + frame["participant_id"].astype(str)
    )


def participant_index(
    path: str | Path,
    *,
    batch_size: int = 500_000,
    sources: set[str] | None = None,
) -> pd.DataFrame:
    dataset = ds.dataset(Path(path), format="parquet", partitioning="hive")
    filter_expression = (
        ds.field("source_id").isin(sorted(sources))
        if sources is not None
        else None
    )
    keys: set[tuple[str, str]] = set()
    for batch in dataset.to_batches(
        columns=["source_id", "participant_id"],
        batch_size=batch_size,
        filter=filter_expression,
    ):
        frame = batch.to_pandas()
        keys.update(
            zip(  # noqa: B905
                frame["source_id"].astype(str),
                frame["participant_id"].astype(str),
            )
        )
    return pd.DataFrame(sorted(keys), columns=["source_id", "participant_id"])


def iter_participants(
    path: str | Path,
    *,
    batch_size: int = 500_000,
    sources: set[str] | None = None,
) -> Iterator[pd.DataFrame]:
    dataset = ds.dataset(Path(path), format="parquet", partitioning="hive")
    filter_expression = (
        ds.field("source_id").isin(sorted(sources))
        if sources is not None
        else None
    )
    carry = pd.DataFrame()
    completed: set[str] = set()
    for batch in dataset.to_batches(
        batch_size=batch_size,
        filter=filter_expression,
        use_threads=False,
        batch_readahead=1,
        fragment_readahead=1,
    ):
        frame = batch.to_pandas()
        combined = pd.concat((carry, frame), ignore_index=True)
        combined["_participant_key"] = _key(combined)
        last_key = str(combined["_participant_key"].iloc[-1])
        ready = combined.loc[combined["_participant_key"] != last_key]
        carry = combined.loc[combined["_participant_key"] == last_key].drop(
            columns="_participant_key"
        )
        for key, participant in ready.groupby("_participant_key", sort=False):
            string_key = str(key)
            if string_key in completed:
                raise ValueError(
                    "normalized parquet is not participant-contiguous; "
                    f"repeated key {string_key!r}"
                )
            completed.add(string_key)
            yield (
                participant.drop(columns="_participant_key")
                .sort_values("timestamp")
                .reset_index(drop=True)
            )
    if not carry.empty:
        final_key = str(_key(carry).iloc[0])
        if final_key in completed:
            raise ValueError(
                "normalized parquet is not participant-contiguous at final key"
            )
        yield carry.sort_values("timestamp").reset_index(drop=True)


def _partitions(
    index: pd.DataFrame,
    config: dict[str, Any],
    *,
    include_holdouts: bool,
) -> dict[str, set[tuple[str, str]]]:
    split = config["split"]
    return participant_split(
        index,
        development_sources=list(config["data"]["development_sources"]),
        external_source=str(config["data"]["external_source"]),
        forbidden_development_sources=list(
            config["data"]["forbidden_development_sources"]
        ),
        stress_sources=list(config["data"].get("stress_sources", [])),
        train_fraction=float(split["train_fraction"]),
        selection_fraction=float(split["selection_fraction"]),
        calibration_fraction=float(split["calibration_fraction"]),
        include_holdouts=include_holdouts,
        seed=int(split["seed"]),
    )


def prepare_window_shards(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    config: dict[str, Any],
    include_holdouts: bool,
    batch_size: int = 500_000,
) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite window shards: {output}")
    output.mkdir(parents=True, exist_ok=True)
    source_filter = (
        None
        if include_holdouts
        else set(map(str, config["data"]["development_sources"]))
    )
    print(
        json.dumps(
            {
                "event": "window_stage",
                "stage": "participant_index_started",
                "source_count": (
                    len(source_filter) if source_filter is not None else None
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    index = participant_index(
        input_path,
        batch_size=batch_size,
        sources=source_filter,
    )
    print(
        json.dumps(
            {
                "event": "window_stage",
                "stage": "participant_iteration_started",
                "participant_count": len(index),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    partitions = _partitions(
        index,
        config,
        include_holdouts=include_holdouts,
    )
    lookup = {
        key: partition
        for partition, keys in partitions.items()
        for key in keys
    }
    allowed = (
        set(lookup)
        if include_holdouts
        else set().union(
            *(
                keys
                for name, keys in partitions.items()
                if name in {"train", "selection", "calibration"}
            )
        )
    )
    audit: dict[str, Any] = {
        "participants_seen": 0,
        "participants_written": 0,
        "windows_written": 0,
        "by_partition": {},
        "excluded_no_eligible_windows": 0,
    }
    prevalence_counts = {
        f"{kind}_{int(horizon)}": {
            "eligible_windows": 0,
            "positive_windows": 0,
        }
        for kind in ("hypo", "hyper")
        for horizon in config["protocol"]["horizons_minutes"]
    }
    fingerprint_owner: dict[str, tuple[str, str, str]] = {}
    for participant in iter_participants(
        input_path,
        batch_size=batch_size,
        sources=source_filter,
    ):
        audit["participants_seen"] += 1
        if (
            audit["participants_seen"] == 1
            or audit["participants_seen"] % 25 == 0
        ):
            print(
                json.dumps(
                    {
                        "event": "window_progress",
                        "participants_seen": audit["participants_seen"],
                        "participants_written": audit["participants_written"],
                        "windows_written": audit["windows_written"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        validate_frame(participant)
        key = (
            str(participant["source_id"].iloc[0]),
            str(participant["participant_id"].iloc[0]),
        )
        if key not in allowed:
            continue
        for fingerprint, record in cgm_day_fingerprints(participant):
            previous = fingerprint_owner.get(fingerprint)
            if (
                previous is not None
                and previous[:2] != record[:2]
            ):
                raise ValueError(
                    "cross-participant duplicate CGM day detected in "
                    f"streaming input: {previous}, {record}"
                )
            fingerprint_owner[fingerprint] = record
        try:
            windows = build_windows(participant, config)
        except ValueError as error:
            if str(error) != "no eligible windows were generated":
                raise
            audit["excluded_no_eligible_windows"] += 1
            continue
        partition = lookup[key]
        windows["partition"] = partition
        for kind in ("hypo", "hyper"):
            eligible_mask = ~windows[f"{kind}_currently_active"].astype(bool)
            for horizon in config["protocol"]["horizons_minutes"]:
                task = f"{kind}_{int(horizon)}"
                labels = (
                    windows.loc[eligible_mask, f"{task}_label"]
                    .dropna()
                    .astype(int)
                )
                prevalence_counts[task]["eligible_windows"] += len(labels)
                prevalence_counts[task]["positive_windows"] += int(
                    labels.sum()
                )
        digest = hashlib.sha256(
            f"{key[0]}\x1f{key[1]}".encode()
        ).hexdigest()[:20]
        partition_dir = output / f"partition={partition}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.Table.from_pandas(windows, preserve_index=False),
            partition_dir / f"participant-{digest}.parquet",
            compression="zstd",
        )
        audit["participants_written"] += 1
        audit["windows_written"] += len(windows)
        audit["by_partition"][partition] = (
            int(audit["by_partition"].get(partition, 0)) + len(windows)
        )
    minimum_prevalence = float(
        config["evaluation"].get("event_prevalence_minimum", 0.0001)
    )
    maximum_prevalence = float(
        config["evaluation"].get("event_prevalence_maximum", 0.30)
    )
    prevalence: dict[str, dict[str, float | int]] = {}
    for task, counts in prevalence_counts.items():
        eligible_windows = int(counts["eligible_windows"])
        positive_windows = int(counts["positive_windows"])
        value = (
            positive_windows / eligible_windows
            if eligible_windows
            else float("nan")
        )
        prevalence[task] = {
            "eligible_windows": eligible_windows,
            "positive_windows": positive_windows,
            "prevalence": value,
        }
        if (
            not math.isfinite(value)
            or value < minimum_prevalence
            or value > maximum_prevalence
        ):
            raise ValueError(
                "development event prevalence outside frozen bounds for "
                f"{task}: {value}"
            )
    audit["development_event_prevalence"] = prevalence
    return audit


def load_validated_window_shards(
    path: str | Path,
    audit_path: str | Path,
    *,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    shard_path = Path(path).expanduser().resolve()
    resolved_audit_path = Path(audit_path).expanduser().resolve()
    if not shard_path.is_dir():
        raise FileNotFoundError(f"window shard directory is absent: {shard_path}")
    if not resolved_audit_path.is_file():
        raise FileNotFoundError(
            f"window shard audit is absent: {resolved_audit_path}"
        )
    with resolved_audit_path.open("r", encoding="utf-8") as handle:
        audit = json.load(handle)
    if not isinstance(audit, dict):
        raise ValueError("window shard audit must be a JSON mapping")

    dataset = ds.dataset(
        shard_path,
        format="parquet",
        partitioning="hive",
    )
    windows = _compact_window_table(dataset.to_table())
    required_columns = {
        "source_id",
        "participant_id",
        "partition",
        *(
            column
            for kind in ("hypo", "hyper")
            for horizon in config["protocol"]["horizons_minutes"]
            for column in (
                f"{kind}_{int(horizon)}_label",
                f"{kind}_currently_active",
            )
        ),
    }
    missing_columns = sorted(required_columns.difference(windows.columns))
    if missing_columns:
        raise ValueError(
            f"window shards are missing required columns: {missing_columns}"
        )
    if windows.empty:
        raise ValueError("window shard dataset is empty")

    allowed_partitions = {"train", "selection", "calibration"}
    # Taking `.astype(str)` over the whole column materialises one Python
    # string per row -- 39.3M of them -- to build a set of three values. The
    # distinct values are the same whether stringified before or after
    # deduplication, so deduplicate first.
    observed_partitions = set(map(str, windows["partition"].unique()))
    if observed_partitions != allowed_partitions:
        raise ValueError(
            "window shard partition mismatch: "
            f"expected {sorted(allowed_partitions)}, "
            f"observed {sorted(observed_partitions)}"
        )
    development_sources = set(
        map(str, config["data"]["development_sources"])
    )
    forbidden_sources = set(
        map(str, config["data"].get("forbidden_development_sources", []))
    )
    observed_sources = set(map(str, windows["source_id"].unique()))
    unexpected_sources = sorted(observed_sources.difference(development_sources))
    contamination = sorted(observed_sources.intersection(forbidden_sources))
    if unexpected_sources or contamination:
        raise ValueError(
            "window shards contain non-development sources: "
            f"unexpected={unexpected_sources}, forbidden={contamination}"
        )

    # `windows.assign(...)` copied all 51 columns of the 39.3M-row frame just to
    # stringify three of them, and only those three are read afterwards. Taking
    # the three columns first gives the identical grouping on a fraction of the
    # memory; the categorical-to-string conversion is kept so the grouping keys
    # and `nunique` semantics are byte-for-byte what they were.
    participant_partition_counts = (
        windows[["source_id", "participant_id", "partition"]]
        .astype(str)
        .groupby(["source_id", "participant_id"], observed=True)["partition"]
        .nunique()
    )
    if (participant_partition_counts != 1).any():
        raise ValueError("a participant occurs in multiple window partitions")
    participant_count = int(len(participant_partition_counts))
    expected_participants = int(audit.get("participants_written", -1))
    if participant_count != expected_participants:
        raise ValueError(
            "window shard participant count conflicts with audit: "
            f"{participant_count} != {expected_participants}"
        )
    if len(dataset.files) != expected_participants:
        raise ValueError(
            "window shard file count conflicts with audit: "
            f"{len(dataset.files)} != {expected_participants}"
        )

    expected_rows = int(audit.get("windows_written", -1))
    if len(windows) != expected_rows:
        raise ValueError(
            "window shard row count conflicts with audit: "
            f"{len(windows)} != {expected_rows}"
        )
    by_partition = {
        str(partition): int(count)
        for partition, count in windows["partition"].value_counts().items()
    }
    expected_by_partition = {
        str(partition): int(count)
        for partition, count in audit.get("by_partition", {}).items()
    }
    if by_partition != expected_by_partition:
        raise ValueError(
            "window shard partition counts conflict with audit: "
            f"{by_partition} != {expected_by_partition}"
        )

    expected_prevalence = audit.get("development_event_prevalence")
    if not isinstance(expected_prevalence, dict):
        raise ValueError("window shard audit lacks development event prevalence")
    for kind in ("hypo", "hyper"):
        # Selecting rows without naming a column copies all 64 columns of a
        # 27.5M-row frame, and only one label column per task is ever read from
        # the result. Applying the mask per column is value-identical and takes
        # a full-frame duplicate out of the process peak, which is what decides
        # how many training lanes fit on a host.
        eligible_mask = ~windows[f"{kind}_currently_active"].astype(bool)
        for horizon in config["protocol"]["horizons_minutes"]:
            task = f"{kind}_{int(horizon)}"
            labels = (
                windows.loc[eligible_mask, f"{task}_label"]
                .dropna()
                .astype(int)
            )
            observed = {
                "eligible_windows": int(len(labels)),
                "positive_windows": int(labels.sum()),
                "prevalence": float(labels.mean()),
            }
            expected = expected_prevalence.get(task)
            if not isinstance(expected, dict):
                raise ValueError(
                    f"window shard audit lacks prevalence for {task}"
                )
            if (
                observed["eligible_windows"]
                != int(expected.get("eligible_windows", -1))
                or observed["positive_windows"]
                != int(expected.get("positive_windows", -1))
                or not math.isclose(
                    observed["prevalence"],
                    float(expected.get("prevalence", float("nan"))),
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
            ):
                raise ValueError(
                    f"window shard prevalence conflicts with audit for {task}: "
                    f"{observed} != {expected}"
                )
    return windows, audit


def validate_window_shard_audit(
    path: str | Path,
    audit_path: str | Path,
    *,
    config: dict[str, Any],
    batch_size: int = 500_000,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("window audit batch_size must be positive")
    shard_path = Path(path).expanduser().resolve()
    resolved_audit_path = Path(audit_path).expanduser().resolve()
    if not shard_path.is_dir():
        raise FileNotFoundError(f"window shard directory is absent: {shard_path}")
    if not resolved_audit_path.is_file():
        raise FileNotFoundError(
            f"window shard audit is absent: {resolved_audit_path}"
        )
    with resolved_audit_path.open("r", encoding="utf-8") as handle:
        audit = json.load(handle)
    if not isinstance(audit, dict):
        raise ValueError("window shard audit must be a JSON mapping")

    dataset = ds.dataset(
        shard_path,
        format="parquet",
        partitioning="hive",
    )
    task_columns = [
        f"{kind}_{int(horizon)}_label"
        for kind in ("hypo", "hyper")
        for horizon in config["protocol"]["horizons_minutes"]
    ]
    required_columns = {
        "source_id",
        "participant_id",
        "partition",
        "hypo_currently_active",
        "hyper_currently_active",
        *task_columns,
    }
    missing_columns = sorted(required_columns.difference(dataset.schema.names))
    if missing_columns:
        raise ValueError(
            f"window shards are missing required columns: {missing_columns}"
        )

    rows = 0
    by_partition: dict[str, int] = {}
    participant_partitions: dict[tuple[str, str], str] = {}
    observed_sources: set[str] = set()
    prevalence_counts = {
        task: {"eligible_windows": 0, "positive_windows": 0}
        for task in task_columns
    }
    for batch in dataset.to_batches(
        columns=sorted(required_columns),
        batch_size=batch_size,
        use_threads=False,
        batch_readahead=1,
        fragment_readahead=1,
    ):
        frame = batch.to_pandas()
        rows += len(frame)
        partitions = frame["partition"].astype(str)
        for partition, count in partitions.value_counts().items():
            by_partition[str(partition)] = (
                int(by_partition.get(str(partition), 0)) + int(count)
            )
        sources = frame["source_id"].astype(str)
        participants = frame["participant_id"].astype(str)
        observed_sources.update(sources.unique())
        for source, participant, partition in zip(  # noqa: B905
            sources,
            participants,
            partitions,
        ):
            key = (source, participant)
            previous = participant_partitions.get(key)
            if previous is not None and previous != partition:
                raise ValueError(
                    "a participant occurs in multiple window partitions"
                )
            participant_partitions[key] = partition
        for kind in ("hypo", "hyper"):
            eligible = frame.loc[
                ~frame[f"{kind}_currently_active"].astype(bool)
            ]
            for horizon in config["protocol"]["horizons_minutes"]:
                task = f"{kind}_{int(horizon)}_label"
                labels = eligible[task].dropna().astype(int)
                prevalence_counts[task]["eligible_windows"] += len(labels)
                prevalence_counts[task]["positive_windows"] += int(
                    labels.sum()
                )

    if rows == 0:
        raise ValueError("window shard dataset is empty")
    allowed_partitions = {"train", "selection", "calibration"}
    if set(by_partition) != allowed_partitions:
        raise ValueError(
            "window shard partition mismatch: "
            f"expected {sorted(allowed_partitions)}, "
            f"observed {sorted(by_partition)}"
        )
    development_sources = set(
        map(str, config["data"]["development_sources"])
    )
    forbidden_sources = set(
        map(str, config["data"].get("forbidden_development_sources", []))
    )
    unexpected_sources = sorted(observed_sources.difference(development_sources))
    contamination = sorted(observed_sources.intersection(forbidden_sources))
    if unexpected_sources or contamination:
        raise ValueError(
            "window shards contain non-development sources: "
            f"unexpected={unexpected_sources}, forbidden={contamination}"
        )

    expected_participants = int(audit.get("participants_written", -1))
    if len(participant_partitions) != expected_participants:
        raise ValueError(
            "window shard participant count conflicts with audit: "
            f"{len(participant_partitions)} != {expected_participants}"
        )
    if len(dataset.files) != expected_participants:
        raise ValueError(
            "window shard file count conflicts with audit: "
            f"{len(dataset.files)} != {expected_participants}"
        )
    expected_rows = int(audit.get("windows_written", -1))
    if rows != expected_rows:
        raise ValueError(
            "window shard row count conflicts with audit: "
            f"{rows} != {expected_rows}"
        )
    expected_by_partition = {
        str(partition): int(count)
        for partition, count in audit.get("by_partition", {}).items()
    }
    if by_partition != expected_by_partition:
        raise ValueError(
            "window shard partition counts conflict with audit: "
            f"{by_partition} != {expected_by_partition}"
        )

    expected_prevalence = audit.get("development_event_prevalence")
    if not isinstance(expected_prevalence, dict):
        raise ValueError("window shard audit lacks development event prevalence")
    observed_prevalence: dict[str, dict[str, float | int]] = {}
    for task_column, counts in prevalence_counts.items():
        task = task_column.removesuffix("_label")
        eligible_windows = int(counts["eligible_windows"])
        positive_windows = int(counts["positive_windows"])
        prevalence = (
            positive_windows / eligible_windows
            if eligible_windows
            else float("nan")
        )
        observed = {
            "eligible_windows": eligible_windows,
            "positive_windows": positive_windows,
            "prevalence": prevalence,
        }
        expected = expected_prevalence.get(task)
        if not isinstance(expected, dict):
            raise ValueError(
                f"window shard audit lacks prevalence for {task}"
            )
        if (
            eligible_windows != int(expected.get("eligible_windows", -1))
            or positive_windows != int(expected.get("positive_windows", -1))
            or not math.isclose(
                prevalence,
                float(expected.get("prevalence", float("nan"))),
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        ):
            raise ValueError(
                f"window shard prevalence conflicts with audit for {task}: "
                f"{observed} != {expected}"
            )
        observed_prevalence[task] = observed
    return {
        "files": len(dataset.files),
        "participants": len(participant_partitions),
        "windows": rows,
        "by_partition": dict(sorted(by_partition.items())),
        "sources": sorted(observed_sources),
        "development_event_prevalence": observed_prevalence,
    }
