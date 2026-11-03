"""One-shot production preparation for the runtime-bound C2 objective.

This module deliberately performs no model selection, calibration, external
evaluation, or training-arm optimisation.  It prepares the two detached C2
references in one fail-closed transaction-like sequence:

1. bind the exact config, frozen full manifest, and live train-shard ledger;
2. fit the frozen weighted 11-source CF5 parent once and retain its selected
   checkpoint states;
3. rebuild train-only participant event streams from the actual Parquet files;
4. bind those streams to an immutable event authority; and
5. replay held-fold OOF logits from the exact selected parent checkpoints.

The first output cannot safely be regenerated from only the serialized parent
artifact because the selected checkpoint states are intentionally not embedded
in that public artifact.  Therefore any pre-existing reference output is a
hard stop rather than a request to refit silently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd
import yaml

from .c2_reference_pipeline import execute_weighted_fit_event_reference
from .c2_training import C2ReferenceVectors, load_c2_reference_vectors
from .config import load_config, resolve_path
from .event_reference_authority import (
    ParticipantStreamInput,
    build_production_event_authority,
    write_production_event_authority,
)
from .event_streams import (
    EVENT_HORIZONS,
    EVENT_PARTITION,
    build_participant_event_stream,
    participant_stream_content_sha256,
)
from .io import load_frame
from .mero_reference import bind_artifact_sha256, write_immutable_json
from .mero_reference_training import (
    ProductionWeightedReferenceFitBundle,
    WeightedProductionReferenceFitConfig,
    fit_production_weighted_reference_bundle,
)
from .mero_weighted_reference import (
    PRODUCTION_SOURCE_ORDER,
    validate_weighted_reference_artifact,
)
from .runtime_provenance import (
    RUNTIME_PROVENANCE_BINDING_ORDER,
    build_runtime_provenance_receipt,
    read_runtime_provenance_receipt,
    runtime_provenance_pins,
    verify_runtime_provenance_receipt_against_paths,
    write_runtime_provenance_receipt,
)
from .shard_manifest import (
    read_window_manifest,
    validate_live_window_partition,
    validate_window_manifest_against_split,
)

PREPARATION_SCHEMA_VERSION = "1.0"
PREPARATION_PROTOCOL_ID = "glycoshift-sota-method-v1.3"
PREPARATION_METHOD = "weighted-cf5-c2-reference-preparation-v1.0"

_ALLOWED_PARTITION_DIRECTORIES = {
    "partition=train",
    "partition=selection",
    "partition=calibration",
}
_OUTPUT_PIN_FIELDS = (
    "c2_base_reference_artifact_sha256",
    "c2_event_authority_artifact_sha256",
    "c2_event_reference_artifact_sha256",
)
_EXPECTED_LOSS_SCORE_CONTRACT: dict[str, Any] = {
    "base_reference": {
        "checkpoint_state_hash": "sorted-tensor-state-sha256-v1",
        "fold_count": 5,
        "loss_denominator": "sampling_weight_sum_v1_2",
        "reference_reuse": "exact_selected_cf5_checkpoint_state",
        "source_order": list(PRODUCTION_SOURCE_ORDER),
    },
    "c2_objective": {
        "base_reference": "detached_weighted_cf5_source_risk",
        "event_reference": (
            "detached_complete_train_participant_oof_event_components"
        ),
        "false_alarm_target_per_patient_day": 0.5,
        "horizon_source_aggregation": (
            "equal_weight_horizon_then_equal_weight_source"
        ),
        "mode": "c2_source_event_regret",
    },
    "event_scores": {
        "complete_trajectory": "all_eligible_train_decisions_per_participant",
        "event_matching": "earliest_unused_event_one_to_one",
        "horizon_order_minutes": list(EVENT_HORIZONS),
        "model_mode": "eval_with_model_gradients_retained",
        "refractory": "soft_30_minutes",
        "score_dtype": "float32",
        "score_transform": "sigmoid_float32",
        "severe_task_order": ["hypo_30", "hypo_60"],
        "zero_event_participants": (
            "exclude_from_sensitivity_include_in_false_alarm_burden"
        ),
    },
    "method": "weighted-cf5-c2-loss-score-contract-v1.1",
    "protocol_id": PREPARATION_PROTOCOL_ID,
    "schema_version": "1.1",
    "scope": {
        "calibration_rows_loaded": False,
        "external_rows_loaded": False,
        "fit_partition": "train",
        "held_partition": "train",
        "selection_rows_loaded": False,
    },
}


def _sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _positive_integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    return value


def _training(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(config.get("training"), field="training")


def _absolute_path(value: str | Path, *, field: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise TypeError(f"{field} must be a path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return Path(os.path.abspath(os.fspath(path)))


def _path_components(path: Path) -> tuple[Path, ...]:
    if not path.is_absolute():
        raise ValueError("path must be absolute after normalization")
    current = Path(path.anchor)
    output = [current]
    for component in path.parts[1:]:
        current = current / component
        output.append(current)
    return tuple(output)


def _existing_without_symlinks(path: Path, *, field: str) -> Path:
    for component in _path_components(path):
        try:
            metadata = component.lstat()
        except FileNotFoundError as error:
            raise FileNotFoundError(f"{field} is absent: {path}") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"{field} must not contain a symlink: {component}")
    return path


def _existing_regular_file(path: Path, *, field: str) -> Path:
    checked = _existing_without_symlinks(path, field=field)
    metadata = checked.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{field} must be a regular file")
    return checked


def _existing_directory(path: Path, *, field: str) -> Path:
    checked = _existing_without_symlinks(path, field=field)
    metadata = checked.lstat()
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{field} must be a directory")
    return checked


def _sha256_file(path: Path, *, field: str) -> tuple[str, int]:
    checked = _existing_regular_file(path, field=field)
    before = checked.lstat()
    digest = hashlib.sha256()
    with checked.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    after = checked.lstat()
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(getattr(before, name) != getattr(after, name) for name in stable_fields):
        raise ValueError(f"{field} changed while hashing")
    return digest.hexdigest(), int(after.st_size)


def _safe_relative_path(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{field} must be a non-empty POSIX relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != value
    ):
        raise ValueError(f"{field} must use normalized safe POSIX spelling")
    return value


def _root_relative_file(
    root: Path,
    relative_path: str,
    *,
    field: str,
) -> Path:
    relative = _safe_relative_path(relative_path, field=field)
    candidate = root / Path(relative)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{field} escapes participant shard root") from error
    return _existing_regular_file(candidate, field=field)


def _path_from_config(
    config: Mapping[str, Any],
    *,
    section: str,
    field: str,
) -> Path:
    values = _mapping(config.get(section), field=section)
    value = values.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{section}.{field} is required")
    resolved = resolve_path(dict(config), value)
    if resolved is None:
        raise ValueError(f"{section}.{field} could not be resolved")
    return _absolute_path(resolved, field=f"{section}.{field}")


@dataclass(frozen=True)
class C2PreparationPaths:
    """All fixed input and output locations for one preparation attempt."""

    resolved_config_path: Path
    input_manifest_path: Path
    split_manifest_path: Path
    preprocessing_contract_path: Path
    loss_score_contract_path: Path
    participant_shard_manifest_path: Path
    participant_shard_root: Path
    runtime_provenance_path: Path
    base_reference_path: Path
    event_authority_path: Path
    event_reference_path: Path
    event_execution_root: Path
    summary_path: Path

    def runtime_kwargs(self) -> dict[str, Any]:
        return {
            "resolved_config_path": self.resolved_config_path,
            "input_manifest_path": self.input_manifest_path,
            "split_manifest_path": self.split_manifest_path,
            "preprocessing_contract_path": self.preprocessing_contract_path,
            "loss_score_contract_path": self.loss_score_contract_path,
            "participant_shard_manifest_path": self.participant_shard_manifest_path,
            "participant_shard_root": self.participant_shard_root,
            "participant_shard_partition": "train",
        }


def resolve_c2_preparation_paths(config: Mapping[str, Any]) -> C2PreparationPaths:
    """Resolve the explicit C2 preparation interface from one loaded config."""

    _training(config)
    paths = C2PreparationPaths(
        resolved_config_path=_path_from_config(
            config,
            section="training",
            field="c2_resolved_config_path",
        ),
        input_manifest_path=_path_from_config(
            config,
            section="training",
            field="c2_input_manifest_path",
        ),
        split_manifest_path=_path_from_config(
            config,
            section="training",
            field="c2_split_manifest_path",
        ),
        preprocessing_contract_path=_path_from_config(
            config,
            section="training",
            field="c2_preprocessing_contract_path",
        ),
        loss_score_contract_path=_path_from_config(
            config,
            section="training",
            field="c2_loss_score_contract_path",
        ),
        participant_shard_manifest_path=_path_from_config(
            config,
            section="training",
            field="c2_participant_shard_manifest_path",
        ),
        participant_shard_root=_path_from_config(
            config,
            section="training",
            field="c2_participant_shard_root",
        ),
        runtime_provenance_path=_path_from_config(
            config,
            section="training",
            field="c2_runtime_provenance_path",
        ),
        base_reference_path=_path_from_config(
            config,
            section="training",
            field="c2_base_reference_path",
        ),
        event_authority_path=_path_from_config(
            config,
            section="training",
            field="c2_event_authority_path",
        ),
        event_reference_path=_path_from_config(
            config,
            section="training",
            field="c2_event_reference_path",
        ),
        event_execution_root=_path_from_config(
            config,
            section="training",
            field="c2_event_execution_root",
        ),
        summary_path=_path_from_config(
            config,
            section="training",
            field="c2_preparation_summary_path",
        ),
    )
    expected_event_reference = _absolute_path(
        paths.event_execution_root / "event-reference.json",
        field="event_reference_path",
    )
    if paths.event_reference_path != expected_event_reference:
        raise ValueError(
            "training.c2_event_reference_path must be "
            "training.c2_event_execution_root/event-reference.json"
        )
    return paths


def _expected_runtime_pins(config: Mapping[str, Any]) -> dict[str, str]:
    supplied = _training(config).get("c2_runtime_expected_pins")
    values = _mapping(supplied, field="training.c2_runtime_expected_pins")
    expected_names = set(RUNTIME_PROVENANCE_BINDING_ORDER).difference(
        {"resolved_config"}
    )
    if set(values) != expected_names:
        raise ValueError(
            "C2 preparation must pin every non-self runtime binding and must "
            "not predeclare resolved_config"
        )
    return {
        str(name): _sha256(
            digest,
            field=f"training.c2_runtime_expected_pins.{name}",
        )
        for name, digest in values.items()
    }


def _assert_config_authority(
    config: Mapping[str, Any],
    paths: C2PreparationPaths,
) -> None:
    configured_path = config.get("_config_path")
    if not isinstance(configured_path, str) or not configured_path:
        raise ValueError("C2 preparation requires a loaded config with _config_path")
    loaded = _absolute_path(configured_path, field="_config_path")
    if loaded != paths.resolved_config_path:
        raise ValueError(
            "C2 preparation config path is not the resolved config authority"
        )
    try:
        live_payload = yaml.safe_load(
            paths.resolved_config_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise ValueError("C2 preparation config authority is not valid YAML") from error
    if not isinstance(live_payload, dict):
        raise ValueError("C2 preparation config authority must be a mapping")
    memory_payload = {
        key: value for key, value in config.items() if key != "_config_path"
    }
    if live_payload != memory_payload:
        raise ValueError("in-memory C2 preparation config differs from live YAML")
    data = _mapping(config.get("data"), field="data")
    source_manifest = _path_from_config(
        config,
        section="data",
        field="source_manifest",
    )
    if source_manifest != paths.input_manifest_path:
        raise ValueError(
            "training.c2_input_manifest_path must equal data.source_manifest"
        )
    if "training_window_shards" not in data:
        raise ValueError("data.training_window_shards is required for C2 preparation")
    window_root = _path_from_config(
        config,
        section="data",
        field="training_window_shards",
    )
    if window_root != paths.participant_shard_root:
        raise ValueError(
            "training.c2_participant_shard_root must equal "
            "data.training_window_shards"
        )
    sources = tuple(map(str, data.get("development_sources", ())))
    if set(sources) != set(PRODUCTION_SOURCE_ORDER):
        raise ValueError("C2 preparation requires exactly the frozen 11 sources")
    if bool(_mapping(config.get("project"), field="project").get("synthetic", False)):
        raise ValueError("C2 production preparation refuses synthetic project configs")


def _validate_loss_score_contract(path: Path) -> None:
    checked = _existing_regular_file(path, field="c2_loss_score_contract_path")
    try:
        payload = json.loads(checked.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("C2 loss-score contract must be valid UTF-8 JSON") from error
    if payload != _EXPECTED_LOSS_SCORE_CONTRACT:
        raise ValueError("C2 loss-score contract differs from the frozen v1.3 schema")


def validate_participant_shard_manifest(
    *,
    participant_shard_manifest_path: str | Path,
    participant_shard_root: str | Path,
    split_manifest_path: str | Path,
    expected_manifest_file_sha256: str,
) -> dict[str, Any]:
    """Bind the frozen full ledger, then verify only its live train partition."""

    manifest_path = _existing_regular_file(
        _absolute_path(
            participant_shard_manifest_path,
            field="participant_shard_manifest_path",
        ),
        field="participant_shard_manifest_path",
    )
    observed_file_sha256, _ = _sha256_file(
        manifest_path,
        field="participant_shard_manifest_path",
    )
    if observed_file_sha256 != _sha256(
        expected_manifest_file_sha256,
        field="expected_manifest_file_sha256",
    ):
        raise ValueError("participant shard manifest file pin mismatch")
    root = _existing_directory(
        _absolute_path(participant_shard_root, field="participant_shard_root"),
        field="participant_shard_root",
    )
    try:
        split_payload = json.loads(
            _existing_regular_file(
                _absolute_path(split_manifest_path, field="split_manifest_path"),
                field="split_manifest_path",
            ).read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("split manifest must be valid UTF-8 JSON") from error
    full_manifest = validate_window_manifest_against_split(
        read_window_manifest(manifest_path),
        _mapping(split_payload, field="split manifest"),
    )
    return validate_live_window_partition(root, full_manifest, partition="train")


def prepare_runtime_provenance_receipt(
    *,
    paths: C2PreparationPaths,
    expected_pins: Mapping[str, str] | None,
) -> dict[str, Any]:
    """Create once or re-verify the receipt against every live input byte."""

    runtime_path = paths.runtime_provenance_path
    if os.path.lexists(runtime_path):
        receipt = read_runtime_provenance_receipt(
            runtime_path,
            expected_pins=expected_pins,
        )
    else:
        receipt = build_runtime_provenance_receipt(
            **paths.runtime_kwargs(),
            expected_pins=expected_pins,
        )
        write_runtime_provenance_receipt(receipt, runtime_path)
    return verify_runtime_provenance_receipt_against_paths(
        receipt,
        **paths.runtime_kwargs(),
        expected_pins=expected_pins,
    )


def _manifest_records(manifest: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("participant shard manifest has no records")
    output: list[Mapping[str, Any]] = []
    relative_paths: set[str] = set()
    for index, value in enumerate(records):
        row = _mapping(value, field=f"participant shard manifest record {index}")
        relative = _safe_relative_path(
            row.get("relative_path"),
            field=f"participant shard manifest record {index}.relative_path",
        )
        if relative in relative_paths:
            raise ValueError("participant shard manifest repeats a relative path")
        relative_paths.add(relative)
        _sha256(
            row.get("sha256"),
            field=f"participant shard manifest record {index}.sha256",
        )
        bytes_value = row.get("bytes")
        if (
            isinstance(bytes_value, bool)
            or not isinstance(bytes_value, int)
            or bytes_value < 1
        ):
            raise ValueError("participant shard manifest record bytes must be positive")
        rows_value = row.get("rows")
        if (
            isinstance(rows_value, bool)
            or not isinstance(rows_value, int)
            or rows_value < 1
        ):
            raise ValueError("participant shard manifest record rows must be positive")
        output.append(row)
    return tuple(output)


def _ordered_participant_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("participant shard must contain rows")
    required = {"context_end", "participant_end_index"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"participant shard is missing ordering fields: {missing}")
    ordered = frame.copy()
    ordered["__c2_context_end"] = pd.to_datetime(
        ordered["context_end"],
        utc=True,
        errors="coerce",
    )
    if ordered["__c2_context_end"].isna().any():
        raise ValueError("participant shard contains invalid context_end")
    return (
        ordered.sort_values(
            ["__c2_context_end", "participant_end_index"],
            kind="mergesort",
        )
        .drop(columns="__c2_context_end")
        .reset_index(drop=True)
    )


def _ordered_train_windows(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "source_id",
        "participant_id",
        "context_end",
        "participant_end_index",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"C2 train windows are missing ordering fields: {missing}")
    ordered = frame.copy()
    ordered["__c2_context_end"] = pd.to_datetime(
        ordered["context_end"],
        utc=True,
        errors="coerce",
    )
    if ordered["__c2_context_end"].isna().any():
        raise ValueError("C2 train windows contain invalid context_end")
    return (
        ordered.sort_values(
            [
                "source_id",
                "participant_id",
                "__c2_context_end",
                "participant_end_index",
            ],
            kind="mergesort",
        )
        .drop(columns="__c2_context_end")
        .reset_index(drop=True)
    )


def _assert_stream_inputs_cover_parent(
    stream_inputs: Sequence[ParticipantStreamInput],
    *,
    parent_base_reference_artifact: Mapping[str, Any],
) -> None:
    parent = validate_weighted_reference_artifact(parent_base_reference_artifact)
    expected = {
        (str(row["source_id"]), str(row["participant_id"]))
        for row in parent["participant_fold_assignment"]
    }
    observed = {
        (item.stream.source_id, item.stream.participant_id)
        for item in stream_inputs
    }
    if len(observed) != len(stream_inputs) or observed != expected:
        raise ValueError(
            "train participant shards do not exactly cover the parent CF5 ledger"
        )


def _load_train_participant_shards(
    *,
    participant_shard_root: str | Path,
    manifest: Mapping[str, Any],
) -> tuple[pd.DataFrame, tuple[ParticipantStreamInput, ...]]:
    """Read only direct train shards and bind each file and participant stream.

    This reads only files physically stored below ``partition=train``.  A
    selection or calibration file is never promoted to the event authority;
    an incorrect in-file partition below the train directory is an error.
    """

    root = _existing_directory(
        _absolute_path(participant_shard_root, field="participant_shard_root"),
        field="participant_shard_root",
    )
    inputs: list[ParticipantStreamInput] = []
    frames: list[pd.DataFrame] = []
    observed_keys: set[tuple[str, str]] = set()
    for index, record in enumerate(_manifest_records(manifest)):
        relative = _safe_relative_path(
            record["relative_path"],
            field=f"participant shard manifest record {index}.relative_path",
        )
        partition_directory = PurePosixPath(relative).parts[0]
        if partition_directory not in _ALLOWED_PARTITION_DIRECTORIES:
            raise ValueError(
                "participant shard manifest contains an unsupported partition path: "
                f"{partition_directory}"
            )
        if partition_directory != "partition=train":
            continue
        path = _root_relative_file(
            root,
            relative,
            field=f"train participant shard {relative}",
        )
        digest, byte_count = _sha256_file(
            path,
            field=f"train participant shard {relative}",
        )
        if digest != record["sha256"] or byte_count != record["bytes"]:
            raise ValueError("participant shard changed after manifest derivation")
        frame = _ordered_participant_frame(pd.read_parquet(path))
        digest_after, byte_count_after = _sha256_file(
            path,
            field=f"train participant shard {relative}",
        )
        if digest_after != digest or byte_count_after != byte_count:
            raise ValueError("participant shard changed while being read")
        if len(frame) != int(record["rows"]):
            raise ValueError("participant shard row count disagrees with manifest")
        partitions = set(frame.get("partition", pd.Series(dtype=str)).astype(str))
        if partitions != {EVENT_PARTITION}:
            raise ValueError(
                "train participant shard contains a non-train partition"
            )
        stream = build_participant_event_stream(
            frame,
            expected_content_sha256=participant_stream_content_sha256(frame),
            window_shard_sha256=digest,
        )
        key = (stream.source_id, stream.participant_id)
        if key in observed_keys:
            raise ValueError("train participant shards repeat a participant")
        observed_keys.add(key)
        frames.append(frame)
        inputs.append(
            ParticipantStreamInput(
                stream=stream,
                shard_relative_path=relative,
            )
        )
    if not inputs:
        raise ValueError("participant shard root contains no train participant shards")
    source_index = {
        source: index for index, source in enumerate(PRODUCTION_SOURCE_ORDER)
    }
    ordered_inputs = tuple(
        sorted(
            inputs,
            key=lambda item: (
                source_index.get(item.stream.source_id, len(source_index)),
                item.stream.participant_id.encode("utf-8"),
            ),
        )
    )
    train_windows = _ordered_train_windows(pd.concat(frames, ignore_index=True))
    return train_windows, ordered_inputs


def build_train_participant_stream_inputs(
    *,
    participant_shard_root: str | Path,
    participant_shard_manifest: Mapping[str, Any],
    parent_base_reference_artifact: Mapping[str, Any] | None = None,
) -> tuple[ParticipantStreamInput, ...]:
    """Rebuild every direct train shard with observed file and stream hashes."""

    root = _existing_directory(
        _absolute_path(participant_shard_root, field="participant_shard_root"),
        field="participant_shard_root",
    )
    manifest = validate_live_window_partition(
        root,
        participant_shard_manifest,
        partition="train",
    )
    _, inputs = _load_train_participant_shards(
        participant_shard_root=root,
        manifest=manifest,
    )
    if parent_base_reference_artifact is not None:
        _assert_stream_inputs_cover_parent(
            inputs,
            parent_base_reference_artifact=parent_base_reference_artifact,
        )
    return inputs


def _assert_train_windows_match_streams(
    train_windows: pd.DataFrame,
    stream_inputs: Sequence[ParticipantStreamInput],
) -> None:
    if not isinstance(train_windows, pd.DataFrame) or train_windows.empty:
        raise ValueError("C2 preparation requires non-empty train windows")
    if "partition" not in train_windows:
        raise ValueError("C2 train windows are missing partition")
    if not train_windows["partition"].astype(str).eq(EVENT_PARTITION).all():
        raise ValueError("C2 reference input includes a non-train window")
    keys = {
        (str(source), str(participant))
        for source, participant in zip(  # noqa: B905
            train_windows["source_id"],
            train_windows["participant_id"],
        )
    }
    expected = {
        (item.stream.source_id, item.stream.participant_id)
        for item in stream_inputs
    }
    if keys != expected:
        raise ValueError(
            "loaded train windows do not exactly match direct train shard streams"
        )
    for item in stream_inputs:
        source, participant = item.stream.source_id, item.stream.participant_id
        current = _ordered_participant_frame(
            train_windows.loc[
                train_windows["source_id"].astype(str).eq(source)
                & train_windows["participant_id"].astype(str).eq(participant)
            ]
        )
        if participant_stream_content_sha256(current) != item.stream.content_sha256:
            raise ValueError(
                "loaded train participant windows disagree with direct shard bytes"
            )


def _load_train_only_frame(
    config: dict[str, Any],
    *,
    participant_keys: set[tuple[str, str]],
) -> pd.DataFrame:
    return load_frame(
        config,
        sources=set(PRODUCTION_SOURCE_ORDER),
        participant_keys=participant_keys,
    )


def _assert_outputs_unclaimed(paths: C2PreparationPaths) -> None:
    outputs = (
        paths.base_reference_path,
        paths.event_authority_path,
        paths.event_reference_path,
        paths.summary_path,
    )
    if len({str(path) for path in outputs}) != len(outputs):
        raise ValueError("C2 preparation output paths must be distinct")
    root = paths.event_execution_root
    if os.path.lexists(root):
        root_metadata = root.lstat()
        if stat.S_ISLNK(root_metadata.st_mode):
            raise ValueError("c2_event_execution_root must not be a symlink")
        if not stat.S_ISDIR(root_metadata.st_mode):
            raise ValueError("c2_event_execution_root must be a directory")
        if any(root.iterdir()):
            raise FileExistsError(
                "refusing to reuse a non-empty C2 event execution root"
            )
    for path in outputs:
        if os.path.lexists(path):
            raise FileExistsError(f"refusing to reuse C2 preparation output: {path}")
    for path in (
        paths.base_reference_path,
        paths.event_authority_path,
        paths.runtime_provenance_path,
        paths.summary_path,
    ):
        try:
            path.relative_to(root)
        except ValueError:
            continue
        raise ValueError(
            "non-execution C2 artifact path must be outside execution root"
        )


def _assert_no_premature_output_pins(config: Mapping[str, Any]) -> None:
    training = _training(config)
    premature = [field for field in _OUTPUT_PIN_FIELDS if field in training]
    if premature:
        raise ValueError(
            "preparation config must not predeclare output artifact pins: "
            f"{sorted(premature)}"
        )


def _fit_config(config: Mapping[str, Any]) -> WeightedProductionReferenceFitConfig:
    training = _training(config)
    value = training.get("c2_reference_device", "cpu")
    if not isinstance(value, str) or not value:
        raise ValueError("training.c2_reference_device must be a non-empty string")
    resolved = WeightedProductionReferenceFitConfig(device=value)
    reference_seed = training.get(
        "c2_reference_seed",
        resolved.reference_seed,
    )
    if reference_seed != resolved.reference_seed:
        raise ValueError("training.c2_reference_seed differs from the frozen seed")
    return resolved


def _event_chunk_size(config: Mapping[str, Any]) -> int:
    return _positive_integer(
        _training(config).get("c2_event_chunk_size"),
        field="training.c2_event_chunk_size",
    )


@dataclass(frozen=True)
class C2PreparationResult:
    """Canonical pointers and hashes produced by a completed preparation."""

    summary_path: Path
    runtime_provenance_receipt: Mapping[str, Any]
    base_reference_artifact_sha256: str
    event_authority_artifact_sha256: str
    event_reference_artifact_sha256: str
    reference: C2ReferenceVectors

    def as_dict(self) -> dict[str, Any]:
        return {
            "summary_path": str(self.summary_path),
            "runtime_provenance_artifact_sha256": self.runtime_provenance_receipt[
                "artifact_sha256"
            ],
            "base_reference_artifact_sha256": self.base_reference_artifact_sha256,
            "event_authority_artifact_sha256": self.event_authority_artifact_sha256,
            "event_reference_artifact_sha256": self.event_reference_artifact_sha256,
            "reference": self.reference.receipt(),
        }


def _summary_payload(
    *,
    paths: C2PreparationPaths,
    runtime_receipt: Mapping[str, Any],
    base_artifact: Mapping[str, Any],
    authority: Mapping[str, Any],
    event_artifact: Mapping[str, Any],
    reference: C2ReferenceVectors,
) -> dict[str, Any]:
    pins = runtime_provenance_pins(runtime_receipt)
    return bind_artifact_sha256(
        {
            "schema_version": PREPARATION_SCHEMA_VERSION,
            "protocol_id": PREPARATION_PROTOCOL_ID,
            "method": PREPARATION_METHOD,
            "scope": {
                "fit_partition": EVENT_PARTITION,
                "held_partition": EVENT_PARTITION,
                "selection_rows_loaded": False,
                "calibration_rows_loaded": False,
                "external_rows_loaded": False,
            },
            "paths": {
                "runtime_provenance": str(paths.runtime_provenance_path),
                "base_reference": str(paths.base_reference_path),
                "event_authority": str(paths.event_authority_path),
                "event_reference": str(paths.event_reference_path),
                "event_execution_root": str(paths.event_execution_root),
            },
            "runtime_pins": {
                name: pins[name] for name in RUNTIME_PROVENANCE_BINDING_ORDER
            },
            "artifacts": {
                "runtime_provenance_artifact_sha256": runtime_receipt[
                    "artifact_sha256"
                ],
                "base_reference_artifact_sha256": base_artifact["artifact_sha256"],
                "event_authority_artifact_sha256": authority["artifact_sha256"],
                "event_reference_artifact_sha256": event_artifact["artifact_sha256"],
            },
            "post_preparation_training_pins": {
                "c2_base_reference_artifact_sha256": base_artifact[
                    "artifact_sha256"
                ],
                "c2_base_loss_contract_sha256": pins["loss_score_contract"],
                "c2_event_authority_artifact_sha256": authority["artifact_sha256"],
                "c2_event_reference_artifact_sha256": event_artifact[
                    "artifact_sha256"
                ],
            },
            "receipt_roles": {
                "preparation_runtime_receipt": (
                    "binds_preparation_config_only_not_valid_for_c2_training"
                ),
                "required_next_step": (
                    "build_fully_pinned_training_config_then_run_"
                    "glycoshift.c2_finalize"
                ),
            },
            "reference_receipt": reference.receipt(),
        }
    )


def prepare_c2_references(config: dict[str, Any]) -> C2PreparationResult:
    """Produce the immutable weighted CF5 base and C2 event OOF references.

    Callers must pass a loaded, non-synthetic production configuration.  This
    entry point is intentionally unsuitable for reruns after an output exists:
    preserving the exact selected states is more important than convenience.
    """

    paths = resolve_c2_preparation_paths(config)
    _assert_config_authority(config, paths)
    _assert_no_premature_output_pins(config)
    _validate_loss_score_contract(paths.loss_score_contract_path)
    _assert_outputs_unclaimed(paths)
    expected_pins = _expected_runtime_pins(config)
    runtime_receipt = prepare_runtime_provenance_receipt(
        paths=paths,
        expected_pins=expected_pins,
    )
    runtime_pins = runtime_provenance_pins(runtime_receipt)
    manifest = validate_participant_shard_manifest(
        participant_shard_manifest_path=paths.participant_shard_manifest_path,
        participant_shard_root=paths.participant_shard_root,
        split_manifest_path=paths.split_manifest_path,
        expected_manifest_file_sha256=runtime_pins[
            "participant_shard_manifest"
        ],
    )
    if _training(config).get("c2_base_loss_contract_sha256") is not None:
        configured = _sha256(
            _training(config).get("c2_base_loss_contract_sha256"),
            field="training.c2_base_loss_contract_sha256",
        )
        if configured != runtime_pins["loss_score_contract"]:
            raise ValueError(
                "training.c2_base_loss_contract_sha256 disagrees with the "
                "live loss-score contract"
            )

    train_windows, stream_inputs = _load_train_participant_shards(
        participant_shard_root=paths.participant_shard_root,
        manifest=manifest,
    )
    participant_keys = {
        (str(source), str(participant))
        for source, participant in zip(  # noqa: B905
            train_windows["source_id"],
            train_windows["participant_id"],
        )
    }
    frame = _load_train_only_frame(
        config,
        participant_keys=participant_keys,
    )
    _assert_train_windows_match_streams(train_windows, stream_inputs)
    fit_config = _fit_config(config)
    first_reference_objective = True

    def emit_reference_progress(payload: Mapping[str, Any]) -> None:
        nonlocal first_reference_objective
        record = dict(payload)
        if first_reference_objective:
            record["event"] = "first_finite_reference_objective"
            first_reference_objective = False
        print(json.dumps(record, sort_keys=True), flush=True)

    bundle: ProductionWeightedReferenceFitBundle = (
        fit_production_weighted_reference_bundle(
            frame,
            train_windows,
            artifact_path=paths.base_reference_path,
            input_manifest_sha256=runtime_pins["input_manifest"],
            window_manifest_sha256=manifest["sha256"],
            reference_config_sha256=runtime_pins["resolved_config"],
            loss_contract_sha256=runtime_pins["loss_score_contract"],
            config=fit_config,
            progress_callback=emit_reference_progress,
        )
    )
    base_artifact = validate_weighted_reference_artifact(bundle.artifact)
    _assert_stream_inputs_cover_parent(
        stream_inputs,
        parent_base_reference_artifact=base_artifact,
    )
    authority = build_production_event_authority(
        parent_base_reference_artifact=base_artifact,
        shard_root=paths.participant_shard_root,
        participant_shard_manifest=read_window_manifest(
            paths.participant_shard_manifest_path
        ),
        stream_inputs=stream_inputs,
        split_manifest_sha256=runtime_pins["split_manifest"],
        window_manifest_sha256=manifest["sha256"],
        preprocessing_config_sha256=runtime_pins["preprocessing_contract"],
    )
    write_production_event_authority(authority, paths.event_authority_path)
    event_execution = execute_weighted_fit_event_reference(
        bundle=bundle,
        fit_config=fit_config,
        frame=frame,
        train_windows=train_windows,
        event_authority=authority,
        expected_authority_sha256=authority["artifact_sha256"],
        shard_root=paths.participant_shard_root,
        execution_root=paths.event_execution_root,
        runtime_provenance_receipt=runtime_receipt,
        resolved_config_path=paths.resolved_config_path,
        input_manifest_path=paths.input_manifest_path,
        split_manifest_path=paths.split_manifest_path,
        preprocessing_contract_path=paths.preprocessing_contract_path,
        loss_score_contract_path=paths.loss_score_contract_path,
        participant_shard_manifest_path=paths.participant_shard_manifest_path,
        participant_shard_manifest=read_window_manifest(
            paths.participant_shard_manifest_path
        ),
        expected_runtime_pins=expected_pins,
        inference_chunk_size=_event_chunk_size(config),
    )
    if _absolute_path(
        event_execution.event_reference_path,
        field="event_reference_path",
    ) != paths.event_reference_path:
        raise ValueError("event executor returned an unexpected event reference path")
    event_artifact = event_execution.event_reference_artifact
    reference = load_c2_reference_vectors(
        base_reference_path=paths.base_reference_path,
        event_reference_path=paths.event_reference_path,
        expected_sha256={
            "base_artifact_sha256": base_artifact["artifact_sha256"],
            "event_artifact_sha256": event_artifact["artifact_sha256"],
            "event_authority_sha256": authority["artifact_sha256"],
            "split_manifest_sha256": runtime_pins["split_manifest"],
            "window_manifest_sha256": manifest["sha256"],
            "preprocessing_config_sha256": runtime_pins[
                "preprocessing_contract"
            ],
            "base_input_manifest_sha256": runtime_pins["input_manifest"],
            "base_loss_contract_sha256": runtime_pins["loss_score_contract"],
            "event_loss_score_contract_sha256": runtime_pins[
                "loss_score_contract"
            ],
        },
    )
    summary = _summary_payload(
        paths=paths,
        runtime_receipt=runtime_receipt,
        base_artifact=base_artifact,
        authority=authority,
        event_artifact=event_artifact,
        reference=reference,
    )
    write_immutable_json(summary, paths.summary_path)
    return C2PreparationResult(
        summary_path=paths.summary_path,
        runtime_provenance_receipt=runtime_receipt,
        base_reference_artifact_sha256=base_artifact["artifact_sha256"],
        event_authority_artifact_sha256=authority["artifact_sha256"],
        event_reference_artifact_sha256=event_artifact["artifact_sha256"],
        reference=reference,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare immutable weighted-CF5 C2 reference artifacts."
    )
    parser.add_argument("--config", required=True, help="resolved YAML config path")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = prepare_c2_references(load_config(args.config))
    print(json.dumps(result.as_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
