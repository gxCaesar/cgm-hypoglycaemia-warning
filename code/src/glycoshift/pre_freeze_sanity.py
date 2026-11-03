from __future__ import annotations

import argparse
import hashlib
import json
import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import yaml
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import load_config, save_config
from .mero_reference import write_immutable_json
from .pipeline import _predict_probability
from .windows import FEATURE_COLUMNS

PROTOCOL_ID = "glycoshift-mero-cf5-pilot-v1.1"
TASK_ORDER = ("hypo_30", "hypo_60", "hyper_30", "hyper_60")
CONTROL_GLOBAL = "constant_global_train_prevalence"
CONTROL_SOURCE = "constant_source_conditional_train_prevalence"
CONTROL_SHUFFLE = "label_shuffle_logistic_seed20260806"
KEY_COLUMNS = ("source_id", "participant_id", "context_end", "partition")


def _sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_mapping(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as handle:
        if resolved.suffix.lower() == ".json":
            payload = json.load(handle)
        else:
            payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"artifact must contain a mapping: {resolved}")
    return payload


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def derive_label_shuffle_seed(task: str, *, root_seed: int = 20260806) -> int:
    if task not in TASK_ORDER:
        raise ValueError(f"unknown label-shuffle task: {task}")
    message = f"{int(root_seed)}|label_shuffle|{task}".encode()
    return int.from_bytes(hashlib.sha256(message).digest()[:4], "big") >> 1


def build_window_manifest(window_root: str | Path) -> dict[str, Any]:
    root = Path(window_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"window shard directory is absent: {root}")
    files = sorted(
        (path for path in root.rglob("*.parquet") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if not files:
        raise ValueError("window shard directory contains no Parquet files")

    digest = hashlib.sha256()
    records: list[dict[str, Any]] = []
    for path in files:
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError("window shard path escapes the shard root") from error
        byte_count = int(resolved.stat().st_size)
        row_count = int(pq.ParquetFile(resolved).metadata.num_rows)
        file_digest = _sha256_file(resolved)
        digest.update(
            (f"{relative}\t{byte_count}\t{row_count}\t{file_digest}\n").encode()
        )
        records.append(
            {
                "relative_path": relative,
                "bytes": byte_count,
                "rows": row_count,
                "sha256": file_digest,
            }
        )
    return {
        "serialization": "relative_path<TAB>bytes<TAB>rows<TAB>sha256<LF>",
        "sha256": digest.hexdigest(),
        "files": len(records),
        "bytes": sum(int(record["bytes"]) for record in records),
        "rows": sum(int(record["rows"]) for record in records),
        "records": records,
    }


def validate_runner_contract(
    contract: Mapping[str, Any],
    *,
    project_root: str | Path,
    contract_path: str | Path,
) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    resolved_contract = Path(contract_path).expanduser().resolve()
    try:
        contract_relative = resolved_contract.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError("runner contract must be inside the project root") from error
    if contract_relative != "code/configs/pre-freeze-sanity.v1.1.yaml":
        raise ValueError("unexpected runner contract path")
    if contract.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("runner contract protocol_id mismatch")
    scope = contract.get("scope")
    if not isinstance(scope, dict):
        raise ValueError("runner contract lacks scope")
    if tuple(scope.get("tasks", ())) != TASK_ORDER:
        raise ValueError("runner contract task order mismatch")
    if tuple(scope.get("scan_partitions", ())) != ("train", "selection"):
        raise ValueError("runner must scan only train and selection")
    if scope.get("fit_partition") != "train":
        raise ValueError("runner fit partition must be train")
    if scope.get("score_partition") != "selection":
        raise ValueError("runner score partition must be selection")
    if scope.get("calibration_rows_loaded") is not False:
        raise ValueError("runner contract must prohibit calibration rows")
    if scope.get("external_rows_loaded") is not False:
        raise ValueError("runner contract must prohibit external rows")
    sources = tuple(map(str, scope.get("source_order", ())))
    if not sources or sources != tuple(sorted(sources)):
        raise ValueError("runner source order must be non-empty and lexical")

    shuffle = contract.get("label_shuffle")
    if not isinstance(shuffle, dict):
        raise ValueError("runner contract lacks label_shuffle")
    if int(shuffle.get("root_seed", -1)) != 20260806:
        raise ValueError("runner label-shuffle root seed mismatch")
    if tuple(shuffle.get("within_source_row_order", ())) != (
        "participant_id_exact_utf8",
        "context_end_ascending",
    ):
        raise ValueError("runner label-shuffle row order mismatch")
    if tuple(shuffle.get("unique_row_key", ())) != (
        "source_id",
        "participant_id",
        "context_end",
    ):
        raise ValueError("runner unique row key mismatch")
    expected_shuffle = {
        "task_seed_message": "20260806|label_shuffle|<task>",
        "task_seed_derivation": (
            "sha256_first_four_bytes_big_endian_then_right_shift_one"
        ),
        "generator": "numpy.random.Generator_with_PCG64",
        "source_iteration": "lexical",
        "permutations": 1,
    }
    for key, expected in expected_shuffle.items():
        if shuffle.get(key) != expected:
            raise ValueError(f"runner label-shuffle contract mismatch: {key}")

    sampling = contract.get("sampling")
    if not isinstance(sampling, dict):
        raise ValueError("runner contract lacks sampling")
    expected_sampling = {
        "negative_ratio": 20,
        "seed": 42,
        "requirement": "full_train_set_retained_with_unit_sampling_weights",
    }
    for key, expected in expected_sampling.items():
        if sampling.get(key) != expected:
            raise ValueError(f"runner sampling contract mismatch: {key}")

    model = contract.get("model")
    if not isinstance(model, dict):
        raise ValueError("runner contract lacks model")
    expected_model = {
        "features": "frozen_FEATURE_COLUMNS_19",
        "scaler": "sklearn_StandardScaler_default",
        "estimator": "sklearn_LogisticRegression",
        "solver": "liblinear",
        "class_weight": "balanced",
        "max_iter": 500,
        "random_state": 42,
        "calibration": "prohibited",
        "threshold_fitting": "prohibited",
    }
    for key, expected in expected_model.items():
        if model.get(key) != expected:
            raise ValueError(f"runner model contract mismatch: {key}")

    bootstrap = contract.get("evaluation", {}).get("bootstrap")
    if not isinstance(bootstrap, dict):
        raise ValueError("runner contract lacks bootstrap")
    evaluation = contract["evaluation"]
    expected_evaluation = {
        "metric": "sklearn_average_precision_score_semantics",
        "aggregation": (
            "recompute_task_auprc_then_source_macro_then_equal_source_mean"
        ),
        "degenerate_source_task": {
            "zero_positive": 0.0,
            "all_positive": 1.0,
            "zero_eligible": "fail",
        },
    }
    for key, expected in expected_evaluation.items():
        if evaluation.get(key) != expected:
            raise ValueError(f"runner evaluation contract mismatch: {key}")
    if int(bootstrap.get("replicates", -1)) != 1000:
        raise ValueError("runner bootstrap replicate count mismatch")
    if int(bootstrap.get("seed", -1)) != 20260724:
        raise ValueError("runner bootstrap seed mismatch")
    expected_bootstrap = {
        "method": "source_stratified_paired_participant_bootstrap",
        "generator": "numpy.random.Generator_with_PCG64",
        "source_iteration": "lexical",
        "participant_order": "exact_utf8",
        "draws": "source_participant_count_with_replacement_within_each_source",
        "pairing": (
            "identical_multiplicities_for_candidate_and_reference_and_all_tasks"
        ),
        "quantile_method": "linear",
    }
    for key, expected in expected_bootstrap.items():
        if bootstrap.get(key) != expected:
            raise ValueError(f"runner bootstrap contract mismatch: {key}")

    decision = contract.get("decision")
    if not isinstance(decision, dict):
        raise ValueError("runner contract lacks decision")
    expected_decision = {
        "reference": CONTROL_SOURCE,
        "candidate": CONTROL_SHUFFLE,
        "point_delta_maximum": 0.02,
        "bootstrap_95_upper_bound_maximum": 0.02,
        "scientific_fail_exit_code": 3,
        "retry_on_scientific_fail": False,
    }
    for key, expected in expected_decision.items():
        if decision.get(key) != expected:
            raise ValueError(f"runner decision contract mismatch: {key}")

    input_binding = contract.get("input_binding")
    if not isinstance(input_binding, dict):
        raise ValueError("runner contract lacks input binding")
    _require_sha256(
        input_binding.get("cohort_audit_sha256"),
        field="input_binding.cohort_audit_sha256",
    )
    _require_sha256(
        input_binding.get("window_manifest_sha256"),
        field="input_binding.window_manifest_sha256",
    )
    for field in ("files", "bytes", "rows"):
        if int(input_binding.get(field, 0)) <= 0:
            raise ValueError(f"input_binding.{field} must be positive")
    partition_rows = input_binding.get("partition_rows")
    partition_participants = input_binding.get("partition_participants")
    expected_partitions = {"train", "selection", "calibration"}
    if (
        not isinstance(partition_rows, dict)
        or set(partition_rows) != expected_partitions
    ):
        raise ValueError("runner partition-row binding mismatch")
    if (
        not isinstance(partition_participants, dict)
        or set(partition_participants) != expected_partitions
    ):
        raise ValueError("runner partition-participant binding mismatch")
    if any(int(value) <= 0 for value in partition_rows.values()):
        raise ValueError("runner partition-row counts must be positive")
    if any(int(value) <= 0 for value in partition_participants.values()):
        raise ValueError("runner partition-participant counts must be positive")
    if sum(map(int, partition_rows.values())) != int(input_binding["rows"]):
        raise ValueError("runner partition rows do not sum to input rows")

    bindings = contract.get("artifact_bindings")
    if not isinstance(bindings, dict):
        raise ValueError("runner contract lacks artifact bindings")
    observed_bindings: dict[str, Any] = {}
    for name in ("protocol_config", "sanity_spec", "split_manifest"):
        binding = bindings.get(name)
        if not isinstance(binding, dict):
            raise ValueError(f"runner contract lacks binding: {name}")
        relative = Path(str(binding.get("path", "")))
        if relative.is_absolute():
            raise ValueError(f"runner binding path must be project-relative: {name}")
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"runner binding escapes project root: {name}") from error
        expected = _require_sha256(binding.get("sha256"), field=f"{name}.sha256")
        observed = _sha256_file(path)
        if observed != expected:
            raise ValueError(f"runner artifact binding drift: {name}")
        observed_bindings[name] = {
            "path": str(path),
            "sha256": observed,
            "bytes": int(path.stat().st_size),
        }
    return {
        "protocol_id": PROTOCOL_ID,
        "source_order": list(sources),
        "runner_contract": {
            "path": str(resolved_contract),
            "sha256": _sha256_file(resolved_contract),
            "bytes": int(resolved_contract.stat().st_size),
        },
        "artifact_bindings": observed_bindings,
    }


def _validate_input_manifest(
    manifest: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> None:
    expected = contract.get("input_binding")
    if not isinstance(expected, dict):
        raise ValueError("runner contract lacks input binding")
    checks = {
        "sha256": "window_manifest_sha256",
        "files": "files",
        "bytes": "bytes",
        "rows": "rows",
    }
    for observed_key, expected_key in checks.items():
        if manifest.get(observed_key) != expected.get(expected_key):
            raise ValueError(
                "window shard manifest mismatch: "
                f"{observed_key}={manifest.get(observed_key)!r}, "
                f"expected={expected.get(expected_key)!r}"
            )


def load_scoped_windows(
    *,
    config: Mapping[str, Any],
    contract: Mapping[str, Any],
    window_root: str | Path,
    cohort_audit_path: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    audit_path = Path(cohort_audit_path).expanduser().resolve()
    if not audit_path.is_file():
        raise FileNotFoundError(f"cohort audit is absent: {audit_path}")
    expected_audit = _require_sha256(
        contract.get("input_binding", {}).get("cohort_audit_sha256"),
        field="input_binding.cohort_audit_sha256",
    )
    observed_audit = _sha256_file(audit_path)
    if observed_audit != expected_audit:
        raise ValueError("cohort audit SHA-256 mismatch")
    with audit_path.open("r", encoding="utf-8") as handle:
        audit = json.load(handle)
    if not isinstance(audit, dict):
        raise ValueError("cohort audit must contain a JSON mapping")

    window_manifest = build_window_manifest(window_root)
    _validate_input_manifest(window_manifest, contract)
    expected_rows = contract["input_binding"]["partition_rows"]
    if audit.get("by_partition") != expected_rows:
        raise ValueError("cohort audit partition rows conflict with contract")

    root = Path(window_root).expanduser().resolve()
    dataset = ds.dataset(root, format="parquet", partitioning="hive")
    label_columns = [f"{task}_label" for task in TASK_ORDER]
    active_columns = sorted(
        {f"{task.split('_')[0]}_currently_active" for task in TASK_ORDER}
    )
    required_columns = {
        "source_id",
        "participant_id",
        "context_end",
        "partition",
        *FEATURE_COLUMNS,
        *label_columns,
        *active_columns,
    }
    missing = sorted(required_columns.difference(dataset.schema.names))
    if missing:
        raise ValueError(f"window shards are missing columns: {missing}")
    scanner = dataset.scanner(
        columns=sorted(required_columns),
        filter=ds.field("partition").isin(["train", "selection"]),
        use_threads=False,
        batch_readahead=1,
        fragment_readahead=1,
    )
    windows = scanner.to_table().to_pandas()
    if windows.empty:
        raise ValueError("scoped window scan is empty")
    observed_partitions = set(windows["partition"].astype(str))
    if observed_partitions != {"train", "selection"}:
        raise ValueError(
            f"scoped window scan exposed forbidden partitions: {observed_partitions}"
        )
    by_partition = {
        str(key): int(value)
        for key, value in windows["partition"].value_counts().items()
    }
    expected_loaded_rows = {
        key: int(expected_rows[key]) for key in ("train", "selection")
    }
    if by_partition != expected_loaded_rows:
        raise ValueError("scoped window row counts conflict with contract")

    source_order = tuple(map(str, contract["scope"]["source_order"]))
    allowed_sources = set(source_order)
    configured_sources = set(map(str, config["data"]["development_sources"]))
    if configured_sources != allowed_sources:
        raise ValueError("configured development sources conflict with contract")
    forbidden_sources = set(
        map(str, config["data"].get("forbidden_development_sources", []))
    )
    observed_sources = set(windows["source_id"].astype(str))
    if observed_sources != allowed_sources:
        raise ValueError("scoped windows do not contain the exact source set")
    contamination = sorted(observed_sources.intersection(forbidden_sources))
    if contamination:
        raise ValueError(f"scoped windows contain forbidden sources: {contamination}")

    keyed = windows.assign(
        source_id=windows["source_id"].astype(str),
        participant_id=windows["participant_id"].astype(str),
        partition=windows["partition"].astype(str),
    )
    partition_counts = keyed.groupby(
        ["source_id", "participant_id"], sort=False, observed=True
    )["partition"].nunique()
    if (partition_counts != 1).any():
        raise ValueError("a participant occurs in both train and selection")
    expected_participants = contract["input_binding"]["partition_participants"]
    for partition in ("train", "selection"):
        observed = int(
            keyed.loc[keyed["partition"] == partition]
            .groupby(["source_id", "participant_id"], sort=False, observed=True)
            .ngroups
        )
        if observed != int(expected_participants[partition]):
            raise ValueError(f"{partition} participant count conflicts with contract")
    if keyed.duplicated(subset=["source_id", "participant_id", "context_end"]).any():
        raise ValueError("window identity is not unique")

    input_manifest = {
        "schema_version": "1.0",
        "protocol_id": PROTOCOL_ID,
        "cohort_audit": {
            "path": str(audit_path),
            "sha256": observed_audit,
            "bytes": int(audit_path.stat().st_size),
        },
        "window_shards": window_manifest,
        "scan": {
            "filter": ["train", "selection"],
            "calibration_rows_loaded": False,
            "external_rows_loaded": False,
            "rows": by_partition,
            "sources": list(source_order),
        },
    }
    return windows, input_manifest


def _task_mask(frame: pd.DataFrame, task: str) -> pd.Series:
    kind = task.split("_")[0]
    return (
        ~frame[f"{kind}_currently_active"].astype(bool) & frame[f"{task}_label"].notna()
    )


def validate_full_train_retention(
    train: pd.DataFrame,
    *,
    source_order: Sequence[str],
    negative_ratio: float,
) -> dict[str, Any]:
    positive = train[[f"{task}_label" for task in TASK_ORDER]].fillna(0).any(axis=1)
    records: dict[str, Any] = {}
    for source in source_order:
        source_mask = train["source_id"].astype(str).eq(source)
        positives = int((source_mask & positive).sum())
        negatives = int((source_mask & ~positive).sum())
        keep_negative = (
            min(negatives, 1000)
            if positives == 0
            else min(negatives, int(math.ceil(positives * negative_ratio)))
        )
        if keep_negative != negatives:
            raise ValueError(
                "frozen full-train retention no longer holds for "
                f"source {source}: {keep_negative} != {negatives}"
            )
        records[source] = {
            "positive_windows": positives,
            "negative_windows": negatives,
            "retained_negative_windows": keep_negative,
            "sampling_weight": 1.0,
        }
    return {
        "negative_ratio": float(negative_ratio),
        "full_train_set_retained": True,
        "sampling_weights_all_one": True,
        "by_source": records,
    }


def train_prevalence_controls(
    train: pd.DataFrame,
    *,
    source_order: Sequence[str],
) -> tuple[dict[str, float], dict[str, dict[str, float]], dict[str, Any]]:
    global_prevalence: dict[str, float] = {}
    source_prevalence: dict[str, dict[str, float]] = {
        source: {} for source in source_order
    }
    denominators: dict[str, Any] = {"global": {}, "by_source": {}}
    for source in source_order:
        denominators["by_source"][source] = {}
    for task in TASK_ORDER:
        mask = _task_mask(train, task)
        labels = train.loc[mask, f"{task}_label"].astype(int)
        if labels.empty or not set(labels.unique()).issubset({0, 1}):
            raise ValueError(f"invalid eligible train labels for {task}")
        prevalence = float(labels.mean())
        if not math.isfinite(prevalence):
            raise ValueError(f"non-finite global train prevalence for {task}")
        global_prevalence[task] = prevalence
        denominators["global"][task] = {
            "eligible_windows": int(len(labels)),
            "positive_windows": int(labels.sum()),
            "weighted_denominator": float(len(labels)),
            "weighted_positive": float(labels.sum()),
            "prevalence": prevalence,
        }
        for source in source_order:
            source_mask = mask & train["source_id"].astype(str).eq(source)
            source_labels = train.loc[source_mask, f"{task}_label"].astype(int)
            if source_labels.empty:
                raise ValueError(
                    f"source {source!r} has no eligible train rows for {task}"
                )
            value = float(source_labels.mean())
            if not math.isfinite(value):
                raise ValueError(
                    f"non-finite source train prevalence for {source}/{task}"
                )
            source_prevalence[source][task] = value
            denominators["by_source"][source][task] = {
                "eligible_windows": int(len(source_labels)),
                "positive_windows": int(source_labels.sum()),
                "weighted_denominator": float(len(source_labels)),
                "weighted_positive": float(source_labels.sum()),
                "prevalence": value,
            }
    return global_prevalence, source_prevalence, denominators


def _validate_numeric_features(frame: pd.DataFrame, *, scope: str) -> None:
    values = frame[list(FEATURE_COLUMNS)].to_numpy(dtype=float, copy=False)
    if not np.isfinite(values).all():
        raise ValueError(f"{scope} contains non-finite frozen summary features")


def _shuffled_labels(
    eligible: pd.DataFrame,
    *,
    task: str,
    source_order: Sequence[str],
    seed: int,
) -> tuple[pd.Series, dict[str, Any]]:
    rng = np.random.Generator(np.random.PCG64(seed))
    output = eligible[f"{task}_label"].astype(int).copy()
    ledger: dict[str, Any] = {}
    for source in source_order:
        source_rows = eligible.loc[
            eligible["source_id"].astype(str).eq(source),
            ["participant_id", "context_end", f"{task}_label"],
        ].copy()
        if source_rows.empty:
            raise ValueError(f"source {source!r} has no eligible rows for {task}")
        source_rows["participant_id"] = source_rows["participant_id"].astype(str)
        ordered = source_rows.sort_values(
            ["participant_id", "context_end"],
            kind="mergesort",
        )
        labels = ordered[f"{task}_label"].astype(int).to_numpy()
        if not set(np.unique(labels)).issubset({0, 1}):
            raise ValueError(f"invalid labels for {source}/{task}")
        permuted = rng.permutation(labels)
        output.loc[ordered.index] = permuted
        ledger[source] = {
            "eligible_windows": int(len(labels)),
            "positive_windows_before": int(labels.sum()),
            "positive_windows_after": int(permuted.sum()),
        }
        if (
            ledger[source]["positive_windows_before"]
            != ledger[source]["positive_windows_after"]
        ):
            raise AssertionError("label shuffle changed stratum prevalence")
    return output, ledger


def fit_label_shuffle_models(
    train: pd.DataFrame,
    selection: pd.DataFrame,
    *,
    source_order: Sequence[str],
    root_seed: int,
    random_state: int,
    max_iter: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    probabilities: dict[str, np.ndarray] = {}
    ledger: dict[str, Any] = {
        "schema_version": "1.0",
        "protocol_id": PROTOCOL_ID,
        "control_id": CONTROL_SHUFFLE,
        "task_order": list(TASK_ORDER),
        "source_order": list(source_order),
        "row_order": ["participant_id_exact_utf8", "context_end_ascending"],
        "tasks": {},
    }
    _validate_numeric_features(selection, scope="selection")
    selection_features = selection[list(FEATURE_COLUMNS)]
    for task in TASK_ORDER:
        mask = _task_mask(train, task)
        eligible = train.loc[
            mask,
            [
                "source_id",
                "participant_id",
                "context_end",
                *FEATURE_COLUMNS,
                f"{task}_label",
            ],
        ]
        _validate_numeric_features(eligible, scope=f"train/{task}")
        seed = derive_label_shuffle_seed(task, root_seed=root_seed)
        shuffled, source_ledger = _shuffled_labels(
            eligible,
            task=task,
            source_order=source_order,
            seed=seed,
        )
        if shuffled.nunique() != 2:
            raise ValueError(f"shuffled training task has one class only: {task}")
        pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=max_iter,
                        random_state=random_state,
                        solver="liblinear",
                    ),
                ),
            ]
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            pipeline.fit(eligible[list(FEATURE_COLUMNS)], shuffled)
        convergence = [
            warning
            for warning in caught
            if issubclass(warning.category, ConvergenceWarning)
        ]
        if convergence:
            raise ValueError(f"label-shuffle logistic did not converge for {task}")
        model = pipeline.named_steps["model"]
        iterations = int(np.max(model.n_iter_))
        if iterations >= max_iter:
            raise ValueError(f"label-shuffle logistic reached max_iter for {task}")
        predicted = _predict_probability(pipeline, selection_features)
        if not np.isfinite(predicted).all():
            raise ValueError(f"non-finite label-shuffle probabilities for {task}")
        probabilities[task] = predicted
        ledger["tasks"][task] = {
            "seed": seed,
            "fit_eligible_windows": int(len(eligible)),
            "fit_positive_windows_before": int(
                eligible[f"{task}_label"].astype(int).sum()
            ),
            "fit_positive_windows_after": int(shuffled.sum()),
            "liblinear_iterations": iterations,
            "coefficient_l2_norm": float(np.linalg.norm(model.coef_)),
            "intercept": float(model.intercept_[0]),
            "by_source": source_ledger,
        }
    return probabilities, ledger


def make_prediction_frame(
    selection: pd.DataFrame,
    *,
    probabilities: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    output = selection[list(KEY_COLUMNS)].copy()
    if set(output["partition"].astype(str)) != {"selection"}:
        raise ValueError("prediction frame must contain selection only")
    for task in TASK_ORDER:
        values = np.asarray(probabilities[task], dtype=float)
        if values.shape != (len(selection),):
            raise ValueError(f"prediction shape mismatch for {task}")
        eligible = _task_mask(selection, task).to_numpy(dtype=bool)
        if not np.isfinite(values[eligible]).all():
            raise ValueError(f"non-finite eligible probabilities for {task}")
        output[f"{task}_label"] = selection[f"{task}_label"].to_numpy()
        output[f"{task}_eligible"] = eligible
        output[f"{task}_probability"] = np.where(eligible, values, np.nan)
    return output


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=float)
    if labels.size == 0:
        raise ValueError("source-task metric has zero eligible rows")
    if not np.isfinite(scores).all():
        raise ValueError("source-task metric has non-finite scores")
    positives = int(labels.sum())
    if positives == 0:
        return 0.0
    if positives == len(labels):
        return 1.0
    order = np.argsort(-scores, kind="mergesort")
    ordered_scores = scores[order]
    ordered_labels = labels[order]
    ends = np.flatnonzero(np.r_[ordered_scores[1:] != ordered_scores[:-1], True])
    true_positive = np.cumsum(ordered_labels, dtype=np.float64)[ends]
    total = ends.astype(np.float64) + 1.0
    increments = np.diff(np.r_[0.0, true_positive])
    return float(np.sum((increments / positives) * (true_positive / total)))


def score_predictions(
    predictions: pd.DataFrame,
    *,
    source_order: Sequence[str],
    status: str,
) -> dict[str, Any]:
    per_source: dict[str, Any] = {}
    pooled: dict[str, float] = {}
    pooled_denominators: dict[str, Any] = {}
    for task in TASK_ORDER:
        eligible = predictions[f"{task}_eligible"].astype(bool)
        labels = predictions.loc[eligible, f"{task}_label"].to_numpy(dtype=int)
        scores = predictions.loc[eligible, f"{task}_probability"].to_numpy(dtype=float)
        pooled[task] = _average_precision(labels, scores)
        pooled_denominators[task] = {
            "eligible_windows": int(len(labels)),
            "positive_windows": int(labels.sum()),
        }
    for source in source_order:
        source_rows = predictions["source_id"].astype(str).eq(source)
        task_values: dict[str, float] = {}
        task_denominators: dict[str, Any] = {}
        for task in TASK_ORDER:
            mask = source_rows & predictions[f"{task}_eligible"].astype(bool)
            labels = predictions.loc[mask, f"{task}_label"].to_numpy(dtype=int)
            scores = predictions.loc[mask, f"{task}_probability"].to_numpy(dtype=float)
            task_values[task] = _average_precision(labels, scores)
            task_denominators[task] = {
                "eligible_windows": int(len(labels)),
                "positive_windows": int(labels.sum()),
            }
        per_source[source] = {
            "task_window_auprc": task_values,
            "macro_window_auprc": float(math.fsum(task_values.values()) / 4),
            "denominators": task_denominators,
        }
    source_macros = [
        float(per_source[source]["macro_window_auprc"]) for source in source_order
    ]
    values = [*pooled.values(), *source_macros]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("non-finite sanity metric")
    return {
        "schema_version": "1.0",
        "protocol_id": PROTOCOL_ID,
        "status": status,
        "partition": "selection",
        "task_order": list(TASK_ORDER),
        "pooled_task_window_auprc": pooled,
        "pooled_four_task_macro_window_auprc": float(math.fsum(pooled.values()) / 4),
        "equal_source_mean_four_task_macro_window_auprc": float(
            math.fsum(source_macros) / len(source_macros)
        ),
        "minimum_source_four_task_macro_window_auprc": float(min(source_macros)),
        "by_source": per_source,
        "pooled_denominators": pooled_denominators,
    }


@dataclass(frozen=True)
class _PreparedTask:
    labels: np.ndarray
    scores: np.ndarray
    participant_codes: np.ndarray
    score_group_ends: np.ndarray
    participant_count: int


def _prepare_bootstrap_task(
    frame: pd.DataFrame,
    *,
    source: str,
    task: str,
    participant_order: Sequence[str],
) -> _PreparedTask:
    mask = frame["source_id"].astype(str).eq(source) & frame[f"{task}_eligible"].astype(
        bool
    )
    task_frame = frame.loc[
        mask,
        ["participant_id", f"{task}_label", f"{task}_probability"],
    ]
    if task_frame.empty:
        raise ValueError(f"zero eligible rows for {source}/{task}")
    mapping = {
        participant: index for index, participant in enumerate(participant_order)
    }
    codes = task_frame["participant_id"].astype(str).map(mapping)
    if codes.isna().any():
        raise ValueError("bootstrap participant mapping is incomplete")
    labels = task_frame[f"{task}_label"].to_numpy(dtype=np.int8)
    scores = task_frame[f"{task}_probability"].to_numpy(dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError(f"non-finite bootstrap scores for {source}/{task}")
    order = np.argsort(-scores, kind="mergesort")
    scores = scores[order]
    labels = labels[order]
    ordered_codes = codes.to_numpy(dtype=np.int64)[order]
    ends = np.flatnonzero(np.r_[scores[1:] != scores[:-1], True])
    return _PreparedTask(
        labels=labels,
        scores=scores,
        participant_codes=ordered_codes,
        score_group_ends=ends,
        participant_count=len(participant_order),
    )


def _weighted_average_precision(
    prepared: _PreparedTask,
    multiplicities: np.ndarray,
) -> float:
    if multiplicities.shape != (prepared.participant_count,):
        raise ValueError("bootstrap multiplicity shape mismatch")
    weights = multiplicities[prepared.participant_codes].astype(float, copy=False)
    total_weight = float(weights.sum())
    if total_weight <= 0:
        raise ValueError("bootstrap source-task has zero eligible rows")
    positive_weight = weights * prepared.labels
    total_positive = float(positive_weight.sum())
    if total_positive == 0:
        return 0.0
    if total_positive == total_weight:
        return 1.0
    true_positive = np.cumsum(positive_weight, dtype=np.float64)[
        prepared.score_group_ends
    ]
    total = np.cumsum(weights, dtype=np.float64)[prepared.score_group_ends]
    increments = np.diff(np.r_[0.0, true_positive])
    precision = np.zeros_like(true_positive, dtype=float)
    np.divide(true_positive, total, out=precision, where=total > 0)
    return float(np.sum((increments / total_positive) * precision))


def source_stratified_paired_bootstrap(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    source_order: Sequence[str],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    alignment = [*KEY_COLUMNS]
    for task in TASK_ORDER:
        alignment.extend([f"{task}_label", f"{task}_eligible"])
    left = candidate[alignment].reset_index(drop=True)
    right = reference[alignment].reset_index(drop=True)
    if not left.equals(right):
        raise ValueError("paired bootstrap prediction rows or outcomes differ")
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")

    participant_orders: dict[str, tuple[str, ...]] = {}
    prepared: dict[str, dict[str, _PreparedTask]] = {}
    for source in source_order:
        participants = tuple(
            sorted(
                candidate.loc[
                    candidate["source_id"].astype(str).eq(source),
                    "participant_id",
                ]
                .astype(str)
                .unique()
            )
        )
        if not participants:
            raise ValueError(f"source {source!r} has no selection participants")
        participant_orders[source] = participants
        prepared[source] = {}
        for task in TASK_ORDER:
            reference_mask = reference["source_id"].astype(str).eq(source) & reference[
                f"{task}_eligible"
            ].astype(bool)
            reference_scores = reference.loc[reference_mask, f"{task}_probability"]
            if reference_scores.nunique(dropna=True) != 1:
                raise ValueError(
                    f"reference is not source-constant for {source}/{task}"
                )
            prepared[source][task] = _prepare_bootstrap_task(
                candidate,
                source=source,
                task=task,
                participant_order=participants,
            )

    def contrast(multiplicities: Mapping[str, np.ndarray]) -> float:
        candidate_sources: list[float] = []
        reference_sources: list[float] = []
        for source in source_order:
            candidate_tasks: list[float] = []
            reference_tasks: list[float] = []
            source_multiplicities = multiplicities[source]
            for task in TASK_ORDER:
                task_prepared = prepared[source][task]
                candidate_tasks.append(
                    _weighted_average_precision(
                        task_prepared,
                        source_multiplicities,
                    )
                )
                weights = source_multiplicities[task_prepared.participant_codes].astype(
                    float, copy=False
                )
                denominator = float(weights.sum())
                if denominator <= 0:
                    raise ValueError("bootstrap source-task has zero eligible rows")
                reference_tasks.append(
                    float(np.dot(weights, task_prepared.labels) / denominator)
                )
            candidate_sources.append(math.fsum(candidate_tasks) / 4)
            reference_sources.append(math.fsum(reference_tasks) / 4)
        return float(
            math.fsum(candidate_sources) / len(candidate_sources)
            - math.fsum(reference_sources) / len(reference_sources)
        )

    ones = {
        source: np.ones(len(participant_orders[source]), dtype=np.int64)
        for source in source_order
    }
    estimate = contrast(ones)
    rng = np.random.Generator(np.random.PCG64(seed))
    differences = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        multiplicities: dict[str, np.ndarray] = {}
        for source in source_order:
            count = len(participant_orders[source])
            draws = rng.integers(0, count, size=count)
            multiplicities[source] = np.bincount(draws, minlength=count)
        differences[replicate] = contrast(multiplicities)
        if (replicate + 1) % 25 == 0 or replicate + 1 == replicates:
            print(
                json.dumps(
                    {
                        "event": "sanity_bootstrap_progress",
                        "replicates_complete": replicate + 1,
                        "replicates_total": replicates,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if not np.isfinite(differences).all() or not math.isfinite(estimate):
        raise ValueError("paired bootstrap produced non-finite values")
    lower, upper = np.quantile(differences, [0.025, 0.975], method="linear")
    return {
        "method": "source_stratified_paired_participant_bootstrap",
        "replicates": int(replicates),
        "seed": int(seed),
        "estimate": float(estimate),
        "ci_low": float(lower),
        "ci_high": float(upper),
        "quantile_method": "linear",
    }


def _write_predictions(frame: pd.DataFrame, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite predictions: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, compression="zstd")


def _write_text_once(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)


def _file_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
    }


def run_pre_freeze_sanity(
    *,
    windows: pd.DataFrame,
    input_manifest: Mapping[str, Any],
    config: Mapping[str, Any],
    contract: Mapping[str, Any],
    output_dir: str | Path,
    result_status: str,
) -> tuple[Path, int]:
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to reuse sanity output directory: {output}")
    output.mkdir(parents=True)
    source_order = tuple(map(str, contract["scope"]["source_order"]))
    train = windows.loc[windows["partition"].astype(str).eq("train")]
    selection = windows.loc[windows["partition"].astype(str).eq("selection")]
    if len(train) != int(contract["input_binding"]["partition_rows"]["train"]):
        raise ValueError("train row count drift")
    if len(selection) != int(contract["input_binding"]["partition_rows"]["selection"]):
        raise ValueError("selection row count drift")

    sampling = validate_full_train_retention(
        train,
        source_order=source_order,
        negative_ratio=float(contract["sampling"]["negative_ratio"]),
    )
    global_prevalence, source_prevalence, denominators = train_prevalence_controls(
        train, source_order=source_order
    )
    denominators["sampling"] = sampling
    denominators["selection_rows"] = int(len(selection))

    global_probabilities = {
        task: np.full(len(selection), global_prevalence[task], dtype=float)
        for task in TASK_ORDER
    }
    source_values = selection["source_id"].astype(str)
    source_probabilities = {
        task: source_values.map(
            {source: source_prevalence[source][task] for source in source_order}
        ).to_numpy(dtype=float)
        for task in TASK_ORDER
    }
    if any(np.isnan(values).any() for values in source_probabilities.values()):
        raise ValueError("source-conditional prevalence is missing a source")

    global_predictions = make_prediction_frame(
        selection,
        probabilities=global_probabilities,
    )
    source_predictions = make_prediction_frame(
        selection,
        probabilities=source_probabilities,
    )
    global_metrics = score_predictions(
        global_predictions,
        source_order=source_order,
        status=result_status,
    )
    source_metrics = score_predictions(
        source_predictions,
        source_order=source_order,
        status=result_status,
    )

    global_dir = output / CONTROL_GLOBAL
    source_dir = output / CONTROL_SOURCE
    shuffle_dir = output / CONTROL_SHUFFLE
    for directory in (global_dir, source_dir, shuffle_dir):
        directory.mkdir()
        write_immutable_json(input_manifest, directory / "input-manifest.json")
    write_immutable_json(global_metrics, global_dir / "metrics.json")
    write_immutable_json(denominators, global_dir / "denominators.json")
    _write_predictions(global_predictions, global_dir / "predictions.parquet")

    write_immutable_json(source_metrics, source_dir / "metrics.json")
    write_immutable_json(
        {
            "schema_version": "1.0",
            "protocol_id": PROTOCOL_ID,
            "source_order": list(source_order),
            "prevalence": source_prevalence,
            "denominators": denominators["by_source"],
        },
        source_dir / "source-task-prevalence.json",
    )
    _write_predictions(source_predictions, source_dir / "predictions.parquet")

    shuffle_probabilities, shuffle_ledger = fit_label_shuffle_models(
        train,
        selection,
        source_order=source_order,
        root_seed=int(contract["label_shuffle"]["root_seed"]),
        random_state=int(contract["model"]["random_state"]),
        max_iter=int(contract["model"]["max_iter"]),
    )
    shuffle_predictions = make_prediction_frame(
        selection,
        probabilities=shuffle_probabilities,
    )
    shuffle_metrics = score_predictions(
        shuffle_predictions,
        source_order=source_order,
        status=result_status,
    )
    bootstrap_contract = contract["evaluation"]["bootstrap"]
    bootstrap = source_stratified_paired_bootstrap(
        shuffle_predictions,
        source_predictions,
        source_order=source_order,
        replicates=int(bootstrap_contract["replicates"]),
        seed=int(bootstrap_contract["seed"]),
    )
    point_delta = float(
        shuffle_metrics["equal_source_mean_four_task_macro_window_auprc"]
        - source_metrics["equal_source_mean_four_task_macro_window_auprc"]
    )
    if not math.isclose(
        point_delta,
        float(bootstrap["estimate"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("bootstrap point estimate disagrees with frozen metric")
    decision_contract = contract["decision"]
    point_pass = point_delta <= float(decision_contract["point_delta_maximum"])
    upper_pass = float(bootstrap["ci_high"]) <= float(
        decision_contract["bootstrap_95_upper_bound_maximum"]
    )
    decision = "pass" if point_pass and upper_pass else "scientific_fail"
    exit_code = (
        0 if decision == "pass" else int(decision_contract["scientific_fail_exit_code"])
    )
    shuffle_metrics["paired_comparison"] = {
        "candidate": CONTROL_SHUFFLE,
        "reference": CONTROL_SOURCE,
        "point_delta": point_delta,
        "bootstrap": bootstrap,
        "point_delta_maximum": float(decision_contract["point_delta_maximum"]),
        "bootstrap_95_upper_bound_maximum": float(
            decision_contract["bootstrap_95_upper_bound_maximum"]
        ),
        "point_rule_pass": point_pass,
        "bootstrap_rule_pass": upper_pass,
        "decision": decision,
        "retry_allowed": False,
    }
    save_config(dict(config), shuffle_dir / "config_resolved.yaml")
    write_immutable_json(shuffle_ledger, shuffle_dir / "shuffle-ledger.json")
    write_immutable_json(shuffle_metrics, shuffle_dir / "metrics.json")
    _write_predictions(shuffle_predictions, shuffle_dir / "predictions.parquet")
    _write_text_once(
        shuffle_dir / "exit.status",
        (f"control={CONTROL_SHUFFLE}\ndecision={decision}\nstatus={exit_code}\n"),
    )

    artifact_paths = sorted(path for path in output.rglob("*") if path.is_file())
    manifest = {
        "schema_version": "1.0",
        "protocol_id": PROTOCOL_ID,
        "decision": decision,
        "exit_code": exit_code,
        "calibration_accessed": False,
        "external_sources_accessed": False,
        "artifacts": [_file_record(path, output) for path in artifact_paths],
    }
    write_immutable_json(manifest, output / "artifact-manifest.json")
    print(
        json.dumps(
            {
                "event": "pre_freeze_sanity_terminal",
                "decision": decision,
                "exit_code": exit_code,
                "output_dir": str(output),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return output, exit_code


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the frozen GlycoShift pre-freeze sanity controls"
    )
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--window-shards", required=True)
    parser.add_argument("--cohort-audit", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    project_root = Path(args.project_root).expanduser().resolve()
    contract_path = Path(args.contract).expanduser().resolve()
    contract = _load_mapping(contract_path)
    validate_runner_contract(
        contract,
        project_root=project_root,
        contract_path=contract_path,
    )
    config = load_config(args.config)
    if bool(config["project"].get("synthetic", False)):
        raise ValueError("real-data sanity CLI refuses a synthetic config")
    windows, input_manifest = load_scoped_windows(
        config=config,
        contract=contract,
        window_root=args.window_shards,
        cohort_audit_path=args.cohort_audit,
    )
    _, exit_code = run_pre_freeze_sanity(
        windows=windows,
        input_manifest=input_manifest,
        config=config,
        contract=contract,
        output_dir=args.output_dir,
        result_status="observed_real_data_selection_only",
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
