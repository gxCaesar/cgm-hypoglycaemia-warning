from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds


def _read_audit(path: str | Path) -> dict[str, Any]:
    audit_path = Path(path).expanduser().resolve()
    if not audit_path.is_file():
        raise FileNotFoundError(f"cohort audit is absent: {audit_path}")
    with audit_path.open("r", encoding="utf-8") as handle:
        audit = json.load(handle)
    if not isinstance(audit, dict):
        raise ValueError("cohort audit must be a JSON mapping")
    return audit


def _gib(value: int) -> float:
    return value / (1024**3)


def _quantile_from_histogram(
    histogram: np.ndarray,
    quantile: float,
) -> int:
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be between zero and one")
    total = int(histogram.sum())
    if total == 0:
        return 0
    target = max(1, math.ceil(total * quantile))
    return int(np.searchsorted(np.cumsum(histogram), target, side="left"))


def _memory_retrieval_workload(
    dataset: ds.Dataset,
    *,
    partition: str,
    stride_minutes: int,
    d_model: int,
) -> dict[str, int | float]:
    histogram = np.zeros(1, dtype=np.int64)
    query_count = 0
    record_count = 0
    candidate_comparisons = 0
    participant_count = 0
    for fragment in dataset.get_fragments(
        filter=ds.field("partition") == partition
    ):
        frame = fragment.to_table(
            columns=[
                "source_id",
                "participant_id",
                "context_start",
                "context_end",
            ],
            use_threads=False,
        ).to_pandas()
        if frame.empty:
            continue
        grouped = frame.groupby(
            ["source_id", "participant_id"],
            sort=False,
            observed=False,
        )
        for _, participant in grouped:
            participant_count += 1
            context_start = pd.to_datetime(
                participant["context_start"],
                utc=True,
            )
            context_end = pd.to_datetime(
                participant["context_end"],
                utc=True,
            )
            buckets = context_end.dt.floor(f"{stride_minutes}min")
            records = (
                context_end.loc[~buckets.duplicated()]
                .sort_values()
                .astype("int64")
                .to_numpy()
            )
            queries = context_start.astype("int64").to_numpy()
            candidates = np.searchsorted(
                records,
                queries,
                side="left",
            )
            participant_histogram = np.bincount(candidates)
            if len(participant_histogram) > len(histogram):
                histogram = np.pad(
                    histogram,
                    (0, len(participant_histogram) - len(histogram)),
                )
            histogram[: len(participant_histogram)] += participant_histogram
            query_count += len(queries)
            record_count += len(records)
            candidate_comparisons += int(candidates.sum())

    float32_record_bytes = record_count * d_model * 4
    float32_stack_bytes = candidate_comparisons * d_model * 4
    bf16_transfer_bytes = candidate_comparisons * d_model * 2
    contiguous_storage_bytes = (
        float32_record_bytes * 2 + record_count * 8
    )
    return {
        "participants": participant_count,
        "queries": query_count,
        "memory_records": record_count,
        "mean_candidates_per_query": candidate_comparisons
        / max(1, query_count),
        "median_candidates_per_query": _quantile_from_histogram(
            histogram,
            0.50,
        ),
        "p90_candidates_per_query": _quantile_from_histogram(
            histogram,
            0.90,
        ),
        "p99_candidates_per_query": _quantile_from_histogram(
            histogram,
            0.99,
        ),
        "maximum_candidates_per_query": len(histogram) - 1,
        "candidate_comparisons_per_pass": candidate_comparisons,
        "memory_record_payload_gib": _gib(float32_record_bytes),
        "candidate_cpu_stack_tib_per_pass": float32_stack_bytes
        / (1024**4),
        "candidate_gpu_transfer_tib_per_pass": bf16_transfer_bytes
        / (1024**4),
        "contiguous_resident_storage_gib": _gib(
            contiguous_storage_bytes
        ),
        "contiguous_build_transfer_gib": _gib(float32_record_bytes),
        "contiguous_candidate_cpu_stack_tib_per_pass": 0.0,
        "contiguous_candidate_gpu_transfer_tib_per_pass": 0.0,
    }


def audit_training_workload(
    window_shards: str | Path,
    cohort_audit: str | Path,
    *,
    config: dict[str, Any],
    batch_size: int = 500_000,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("workload audit batch_size must be positive")
    shard_path = Path(window_shards).expanduser().resolve()
    if not shard_path.is_dir():
        raise FileNotFoundError(f"window shard directory is absent: {shard_path}")
    audit = _read_audit(cohort_audit)
    dataset = ds.dataset(
        shard_path,
        format="parquet",
        partitioning="hive",
    )

    horizons = [
        int(value) for value in config["protocol"]["horizons_minutes"]
    ]
    label_columns = [
        f"{kind}_{horizon}_label"
        for kind in ("hypo", "hyper")
        for horizon in horizons
    ]
    required_columns = {
        "source_id",
        "participant_id",
        "partition",
        "context_start",
        "context_end",
        *label_columns,
    }
    missing_columns = sorted(required_columns.difference(dataset.schema.names))
    if missing_columns:
        raise ValueError(
            f"window shards are missing workload columns: {missing_columns}"
        )

    source_counts: dict[str, dict[str, int]] = {}
    scanned_train_rows = 0
    for batch in dataset.to_batches(
        columns=["source_id", *label_columns],
        filter=ds.field("partition") == "train",
        batch_size=batch_size,
        use_threads=False,
        batch_readahead=1,
        fragment_readahead=1,
    ):
        frame = batch.to_pandas()
        positive = frame[label_columns].fillna(0).ne(0).any(axis=1)
        work = pd.DataFrame(
            {
                "source_id": frame["source_id"].astype(str),
                "positive": positive,
            }
        )
        scanned_train_rows += len(work)
        for source, source_frame in work.groupby("source_id", sort=False):
            counts = source_counts.setdefault(
                str(source),
                {"positive": 0, "negative": 0},
            )
            positive_count = int(source_frame["positive"].sum())
            counts["positive"] += positive_count
            counts["negative"] += len(source_frame) - positive_count

    expected_by_partition = {
        str(partition): int(count)
        for partition, count in audit.get("by_partition", {}).items()
    }
    expected_partitions = {"train", "selection", "calibration"}
    if set(expected_by_partition) != expected_partitions:
        raise ValueError(
            "cohort audit partition mismatch: "
            f"expected {sorted(expected_partitions)}, "
            f"observed {sorted(expected_by_partition)}"
        )
    if scanned_train_rows != expected_by_partition["train"]:
        raise ValueError(
            "scanned train rows conflict with cohort audit: "
            f"{scanned_train_rows} != {expected_by_partition['train']}"
        )

    development_sources = set(
        map(str, config["data"]["development_sources"])
    )
    observed_sources = set(source_counts)
    unexpected_sources = sorted(observed_sources.difference(development_sources))
    if unexpected_sources:
        raise ValueError(
            "training shards contain non-development sources: "
            f"{unexpected_sources}"
        )

    negative_ratio = float(config["training"]["negative_ratio"])
    sampling_by_source: dict[str, dict[str, int | float]] = {}
    sampled_train_rows = 0
    positive_train_rows = 0
    for source in sorted(source_counts):
        positives = int(source_counts[source]["positive"])
        negatives = int(source_counts[source]["negative"])
        retained_negatives = (
            min(negatives, int(math.ceil(positives * negative_ratio)))
            if positives
            else min(negatives, 1_000)
        )
        retained = positives + retained_negatives
        sampled_train_rows += retained
        positive_train_rows += positives
        sampling_by_source[source] = {
            "positive_windows": positives,
            "negative_windows": negatives,
            "retained_negative_windows": retained_negatives,
            "retained_windows": retained,
            "retention_fraction": retained / max(1, positives + negatives),
        }

    training = config["training"]
    model = config["model"]
    per_device_batch_size = int(training["batch_size"])
    effective_batch_size = int(
        training.get("effective_batch_size", per_device_batch_size)
    )
    accumulation_steps = max(
        1,
        math.ceil(effective_batch_size / per_device_batch_size),
    )
    maximum_epochs = int(training["maximum_epochs"])
    warmup_memory_epochs = int(training["warmup_memory_epochs"])
    memory_refresh_epochs = max(0, maximum_epochs - warmup_memory_epochs)
    train_batches_per_epoch = math.ceil(
        sampled_train_rows / per_device_batch_size
    )
    optimizer_steps_per_epoch = math.ceil(
        train_batches_per_epoch / accumulation_steps
    )
    selection_rows = expected_by_partition["selection"]
    calibration_rows = expected_by_partition["calibration"]
    d_model = 512 if str(model["variant"]) == "base" else 128
    memory_stride_minutes = int(model["memory_stride_minutes"])
    training_embedding_bytes = sampled_train_rows * d_model * 4
    selection_embedding_bytes = selection_rows * d_model * 4
    calibration_embedding_bytes = calibration_rows * d_model * 4
    train_encoder_examples = sampled_train_rows * (
        maximum_epochs + memory_refresh_epochs
    )
    selection_encoder_examples = selection_rows * (maximum_epochs + 1)
    calibration_encoder_examples = calibration_rows

    return {
        "status": "passed",
        "cohort": {
            "participants_seen": int(audit["participants_seen"]),
            "participants_written": int(audit["participants_written"]),
            "excluded_no_eligible_windows": int(
                audit["excluded_no_eligible_windows"]
            ),
            "windows": sum(expected_by_partition.values()),
            "by_partition": expected_by_partition,
        },
        "training_sampling": {
            "policy": "per_source_positive_plus_negative_ratio",
            "negative_ratio": negative_ratio,
            "positive_windows": positive_train_rows,
            "train_windows_before_sampling": scanned_train_rows,
            "train_windows_after_sampling": sampled_train_rows,
            "retention_fraction": sampled_train_rows
            / max(1, scanned_train_rows),
            "by_source": sampling_by_source,
        },
        "optimization_upper_bound": {
            "per_device_batch_size": per_device_batch_size,
            "effective_batch_size": effective_batch_size,
            "gradient_accumulation_steps": accumulation_steps,
            "maximum_epochs": maximum_epochs,
            "warmup_memory_epochs": warmup_memory_epochs,
            "memory_refresh_epochs": memory_refresh_epochs,
            "train_batches_per_epoch": train_batches_per_epoch,
            "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
            "train_encoder_examples": train_encoder_examples,
            "selection_encoder_examples": selection_encoder_examples,
            "calibration_encoder_examples": calibration_encoder_examples,
            "total_encoder_examples": (
                train_encoder_examples
                + selection_encoder_examples
                + calibration_encoder_examples
            ),
        },
        "embedding_buffers": {
            "d_model": d_model,
            "dtype": "float32",
            "training_gib": _gib(training_embedding_bytes),
            "selection_gib": _gib(selection_embedding_bytes),
            "calibration_gib": _gib(calibration_embedding_bytes),
        },
        "strict_past_memory_retrieval": {
            "stride_minutes": memory_stride_minutes,
            "oracle_implementation": "all_prior_candidates_then_top_k",
            "adopted_implementation": (
                "full_window_encoding_contiguous_gpu_bank_grouped_exact_top_k"
            ),
            "anchor_only_encoding_adopted": False,
            "train": _memory_retrieval_workload(
                dataset,
                partition="train",
                stride_minutes=memory_stride_minutes,
                d_model=d_model,
            ),
            "selection": _memory_retrieval_workload(
                dataset,
                partition="selection",
                stride_minutes=memory_stride_minutes,
                d_model=d_model,
            ),
            "calibration": _memory_retrieval_workload(
                dataset,
                partition="calibration",
                stride_minutes=memory_stride_minutes,
                d_model=d_model,
            ),
        },
    }
