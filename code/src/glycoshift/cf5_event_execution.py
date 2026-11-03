"""Fail-closed execution boundary for one complete C2 CF5 event reference.

Fit and held-stream model inference stay behind explicit callbacks.  This
module owns the scope, artifact layout, score transform, soft refractory/event
semantics, immutable bytes, hashes, and receipt construction.  A callback can
return only raw severe-event logits; it cannot supply reportable numerators.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd
import torch

from .event_reference import EVENT_REFERENCE_HORIZONS
from .event_reference_authority import (
    PREDICTION_SCHEMA_VERSION,
    CheckpointReceipt,
    PredictionReceipt,
    base_fold_scope_sha256,
    build_production_event_reference_artifact,
    checkpoint_receipt_sha256,
    validate_production_event_authority_against_shards,
    write_production_event_reference_artifact,
)
from .event_regret import participant_event_surrogate
from .event_streams import ParticipantEventStream, build_participant_event_stream
from .mero_reference import canonical_json_bytes
from .mero_weighted_reference import (
    PRODUCTION_SOURCE_ORDER,
    validate_weighted_reference_artifact,
)
from .runtime_provenance import (
    runtime_provenance_pins,
    verify_runtime_provenance_receipt_against_paths,
)

EXECUTION_SCHEMA_VERSION = "1.1"
EXECUTION_PROTOCOL_ID = "glycoshift-sota-method-v1.3"
EXECUTION_METHOD = "callback-driven-checkpoint-bound-cf5-event-execution-v1.1"
REFERENCE_SPLIT = "train"
PRODUCTION_FOLD_COUNT = 5
SCORE_TRANSFORM_ID = "sigmoid-float32-soft-refractory-earliest-unused-v1"
SCORE_BINARY_DTYPE = "float32-little-endian"

_FORBIDDEN_IDENTIFIER_CHARACTERS = {"\x00", "\r", "\n", "|"}


def _identifier(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    if any(character in value for character in _FORBIDDEN_IDENTIFIER_CHARACTERS):
        raise ValueError(f"{field} contains a forbidden protocol delimiter")
    return value


def _sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _integer(value: Any, *, field: str, minimum: int = 0) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError(f"{field} must be an integer")
    resolved = int(value)
    if resolved < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return resolved


def _finite(
    value: Any,
    *,
    field: str,
    minimum: float | None = 0.0,
) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{field} must be numeric")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{field} must be finite")
    if minimum is not None and resolved < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return resolved


def _require_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    return value


def _require_sequence(value: Any, *, field: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{field} must be a sequence")
    return value


def _safe_relative_path(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{field} must be a non-empty POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{field} must be a safe relative path")
    normalized = path.as_posix()
    if normalized != value:
        raise ValueError(f"{field} must use normalized POSIX spelling")
    return normalized


def _resolve_output(root: Path, relative_path: str, *, field: str) -> Path:
    safe = _safe_relative_path(relative_path, field=field)
    candidate = (root / Path(safe)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{field} escapes execution root") from error
    return candidate


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class CallbackScope:
    """A callback's explicit access declaration, restricted to train rows."""

    fit_partition: str
    held_partition: str
    selection_rows_loaded: bool
    calibration_rows_loaded: bool
    external_rows_loaded: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "fit_partition": self.fit_partition,
            "held_partition": self.held_partition,
            "selection_rows_loaded": self.selection_rows_loaded,
            "calibration_rows_loaded": self.calibration_rows_loaded,
            "external_rows_loaded": self.external_rows_loaded,
        }


@dataclass(frozen=True)
class CF5EventRecipe:
    """One fixed source/fold recipe derived from the validated parent ledger."""

    source_id: str
    fold_id: int
    fit_participant_ids: tuple[str, ...]
    held_participant_ids: tuple[str, ...]
    base_fold_sha256: str
    expected_checkpoint_epoch: int
    expected_checkpoint_state_sha256: str
    config_relative_path: str
    checkpoint_relative_path: str
    score_relative_path: str
    prediction_relative_path: str


@dataclass(frozen=True)
class FitCallbackResult:
    """Opaque callback material; checkpoint bytes are never interpreted here."""

    model_config: Mapping[str, Any]
    checkpoint_bytes: bytes
    checkpoint_epoch: int
    checkpoint_state_sha256: str
    scope: CallbackScope


@dataclass(frozen=True)
class PredictionCallbackRequest:
    """Only train-held scope and immutable receipt data reach prediction code."""

    recipe: CF5EventRecipe
    checkpoint_receipt: CheckpointReceipt
    checkpoint_receipt_sha256: str
    expected_authority_sha256: str
    loss_score_contract_sha256: str


@dataclass(frozen=True)
class ParticipantSevereLogits:
    """Raw ordered held-stream logits for both severe-event horizons."""

    source_id: str
    participant_id: str
    raw_logits_by_horizon: Mapping[int, Sequence[float]]


@dataclass(frozen=True)
class PredictionCallbackResult:
    """Complete raw held-stream scores emitted by one prediction callback."""

    participant_logits: Sequence[ParticipantSevereLogits]
    scope: CallbackScope


@dataclass(frozen=True)
class CF5EventExecutionResult:
    """Receipt and immutable artifact outputs from one complete execution."""

    recipes: tuple[CF5EventRecipe, ...]
    checkpoint_receipts: tuple[CheckpointReceipt, ...]
    prediction_receipts: tuple[PredictionReceipt, ...]
    event_reference_artifact: Mapping[str, Any]
    event_reference_path: Path


FitCallback = Callable[[CF5EventRecipe], FitCallbackResult]
PredictionCallback = Callable[[PredictionCallbackRequest], PredictionCallbackResult]


def _validate_scope(scope: Any, *, field: str) -> CallbackScope:
    if not isinstance(scope, CallbackScope):
        raise TypeError(f"{field} must be CallbackScope")
    if (
        scope.fit_partition != REFERENCE_SPLIT
        or scope.held_partition != REFERENCE_SPLIT
    ):
        raise ValueError(f"{field} may access only the train partition")
    for value, name in (
        (scope.selection_rows_loaded, "selection_rows_loaded"),
        (scope.calibration_rows_loaded, "calibration_rows_loaded"),
        (scope.external_rows_loaded, "external_rows_loaded"),
    ):
        if not isinstance(value, bool) or value:
            raise ValueError(f"{field}.{name} must be false")
    return scope


def _canonical_json(payload: Mapping[str, Any], *, field: str) -> bytes:
    try:
        encoded = canonical_json_bytes(payload)
        parsed = json.loads(encoded.decode("utf-8"))
    except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{field} must be canonical JSON serializable") from error
    if not isinstance(parsed, dict) or encoded != canonical_json_bytes(parsed):
        raise ValueError(f"{field} is not canonical JSON")
    return encoded


def _write_bytes(
    root: Path,
    relative_path: str,
    value: bytes,
    *,
    field: str,
    canonical_json: bool,
) -> tuple[Path, str]:
    if not isinstance(value, bytes) or not value:
        raise ValueError(f"{field} bytes must be non-empty bytes")
    path = _resolve_output(root, relative_path, field=field)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = path.read_bytes()
        if canonical_json:
            try:
                parsed = json.loads(existing.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"{field} existing bytes are not canonical JSON"
                ) from error
            if not isinstance(parsed, dict) or existing != canonical_json_bytes(parsed):
                raise ValueError(f"{field} existing bytes are not canonical JSON")
        if existing != value:
            raise FileExistsError(
                f"refusing to overwrite non-identical {field}: {path}"
            )
    else:
        with path.open("xb") as handle:
            handle.write(value)
    observed = path.read_bytes()
    if observed != value:
        raise ValueError(f"{field} bytes mutated during write")
    return path, _sha256_bytes(observed)


def _assignment_ledger(parent: Mapping[str, Any]) -> dict[tuple[str, str], int]:
    assignments: dict[tuple[str, str], int] = {}
    for row in _require_sequence(
        parent.get("participant_fold_assignment"),
        field="parent participant_fold_assignment",
    ):
        mapping = _require_mapping(row, field="parent participant_fold_assignment")
        source = _identifier(mapping.get("source_id"), field="source_id")
        participant = _identifier(
            mapping.get("participant_id"),
            field="participant_id",
        )
        fold_id = _integer(mapping.get("fold_id"), field="fold_id", minimum=0)
        if fold_id >= PRODUCTION_FOLD_COUNT:
            raise ValueError("parent assignment has an invalid CF5 fold")
        key = (source, participant)
        if key in assignments:
            raise ValueError("parent assignment repeats a source-scoped participant")
        assignments[key] = fold_id
    for source in PRODUCTION_SOURCE_ORDER:
        folds = {
            fold for (current, _), fold in assignments.items() if current == source
        }
        if folds != set(range(PRODUCTION_FOLD_COUNT)):
            raise ValueError("parent assignment misses a source-local CF5 fold")
    return assignments


def build_cf5_event_recipes(
    parent_base_reference_artifact: Mapping[str, Any],
) -> tuple[CF5EventRecipe, ...]:
    """Derive the fixed 11-source×5-fold fit/held recipes from the parent."""

    parent = validate_weighted_reference_artifact(parent_base_reference_artifact)
    if tuple(parent["lexical_source_order"]) != PRODUCTION_SOURCE_ORDER:
        raise ValueError("parent source order is not the frozen production order")
    if parent["fold_count"] != PRODUCTION_FOLD_COUNT:
        raise ValueError("parent artifact must use exactly CF5")
    assignments = _assignment_ledger(parent)
    parent_folds = {
        (str(row["source_id"]), int(row["fold_id"])): row
        for row in parent["folds"]
    }
    recipes: list[CF5EventRecipe] = []
    for source_index, source in enumerate(PRODUCTION_SOURCE_ORDER):
        participants = tuple(
            sorted(
                (
                    participant
                    for current_source, participant in assignments
                    if current_source == source
                ),
                key=str.encode,
            )
        )
        for fold_id in range(PRODUCTION_FOLD_COUNT):
            parent_fold = parent_folds[(source, fold_id)]
            held = tuple(
                participant
                for participant in participants
                if assignments[(source, participant)] == fold_id
            )
            fit = tuple(
                participant for participant in participants if participant not in held
            )
            if not fit or not held:
                raise ValueError(
                    "each production CF5 recipe requires fit and held sets"
                )
            name = f"{source_index:02d}-{fold_id}"
            recipes.append(
                CF5EventRecipe(
                    source_id=source,
                    fold_id=fold_id,
                    fit_participant_ids=fit,
                    held_participant_ids=held,
                    base_fold_sha256=base_fold_scope_sha256(
                        source_id=source,
                        fold_id=fold_id,
                        fit_participant_ids=fit,
                        held_participant_ids=held,
                    ),
                    expected_checkpoint_epoch=_integer(
                        parent_fold["checkpoint_epoch"],
                        field="checkpoint_epoch",
                        minimum=1,
                    ),
                    expected_checkpoint_state_sha256=_sha256(
                        parent_fold["checkpoint_state_sha256"],
                        field="checkpoint_state_sha256",
                    ),
                    config_relative_path=f"configs/{name}.json",
                    checkpoint_relative_path=f"checkpoints/{name}.bin",
                    score_relative_path=f"predictions/{name}.f32le",
                    prediction_relative_path=f"predictions/{name}.json",
                )
            )
    if len(recipes) != len(PRODUCTION_SOURCE_ORDER) * PRODUCTION_FOLD_COUNT:
        raise ValueError("CF5 recipe derivation did not produce all 55 folds")
    return tuple(recipes)


def _execution_config_payload(
    recipe: CF5EventRecipe,
    result: FitCallbackResult,
    *,
    runtime_provenance_artifact_sha256: str,
) -> dict[str, Any]:
    if not isinstance(result.model_config, Mapping):
        raise TypeError("fit callback model_config must be a mapping")
    if "execution_scope" in result.model_config:
        raise ValueError("fit callback model_config may not override execution_scope")
    scope = _validate_scope(result.scope, field="fit callback scope")
    return {
        "schema_version": EXECUTION_SCHEMA_VERSION,
        "protocol_id": EXECUTION_PROTOCOL_ID,
        "method": EXECUTION_METHOD,
        "source_id": recipe.source_id,
        "fold_id": recipe.fold_id,
        "fit_participant_ids": list(recipe.fit_participant_ids),
        "held_participant_ids": list(recipe.held_participant_ids),
        "base_fold_sha256": recipe.base_fold_sha256,
        "expected_checkpoint_epoch": recipe.expected_checkpoint_epoch,
        "expected_checkpoint_state_sha256": (
            recipe.expected_checkpoint_state_sha256
        ),
        "runtime_provenance_artifact_sha256": _sha256(
            runtime_provenance_artifact_sha256,
            field="runtime_provenance_artifact_sha256",
        ),
        "execution_scope": scope.as_dict(),
        "fit_callback_config": dict(result.model_config),
    }


def _stream_by_key(
    authority: Mapping[str, Any],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    rows = _require_sequence(
        authority.get("participant_streams"),
        field="participant_streams",
    )
    output: dict[tuple[str, str], Mapping[str, Any]] = {}
    for value in rows:
        row = _require_mapping(value, field="participant_stream")
        key = (
            _identifier(row.get("source_id"), field="source_id"),
            _identifier(row.get("participant_id"), field="participant_id"),
        )
        if key in output:
            raise ValueError("authority repeats a participant stream")
        output[key] = row
    return output


def _materialize_streams(
    authority: Mapping[str, Any],
    *,
    shard_root: str | Path,
) -> dict[tuple[str, str], ParticipantEventStream]:
    root = Path(shard_root).expanduser().resolve()
    output: dict[tuple[str, str], ParticipantEventStream] = {}
    for key, metadata in _stream_by_key(authority).items():
        relative_path = _safe_relative_path(
            metadata["shard_relative_path"],
            field="authority shard_relative_path",
        )
        if not relative_path.startswith("partition=train/"):
            raise ValueError("authority shard path must be train-only")
        shard_path = _resolve_output(
            root,
            relative_path,
            field="authority shard_relative_path",
        )
        stream = build_participant_event_stream(
            pd.read_parquet(shard_path),
            expected_content_sha256=metadata["stream_content_sha256"],
            window_shard_sha256=metadata["window_shard_sha256"],
        )
        if key != (stream.source_id, stream.participant_id):
            raise ValueError("materialized stream identity disagrees with authority")
        output[key] = stream
    return output


def _float32_logits(
    values: Any,
    *,
    expected_count: int,
    field: str,
) -> np.ndarray:
    supplied = _require_sequence(values, field=field)
    if len(supplied) != expected_count:
        raise ValueError(f"{field} does not cover every eligible decision")
    resolved = np.asarray(
        [
            _finite(value, field=field, minimum=None)
            for value in supplied
        ],
        dtype=np.float64,
    )
    if bool((np.abs(resolved) > np.finfo(np.float32).max).any()):
        raise ValueError(f"{field} cannot be represented as float32")
    output = resolved.astype("<f4")
    if not np.isfinite(output).all():
        raise ValueError(f"{field} became non-finite in float32")
    return output


def _derived_prediction_outputs(
    result: PredictionCallbackResult,
    *,
    recipe: CF5EventRecipe,
    checkpoint_digest: str,
    streams: Mapping[tuple[str, str], ParticipantEventStream],
) -> tuple[list[dict[str, Any]], bytes, list[dict[str, Any]]]:
    if not isinstance(result, PredictionCallbackResult):
        raise TypeError("prediction callback must return PredictionCallbackResult")
    _validate_scope(result.scope, field="prediction callback scope")
    supplied = _require_sequence(
        result.participant_logits,
        field="participant_logits",
    )
    expected = {
        (recipe.source_id, participant)
        for participant in recipe.held_participant_ids
    }
    by_key: dict[tuple[str, str], ParticipantSevereLogits] = {}
    for row in supplied:
        if not isinstance(row, ParticipantSevereLogits):
            raise TypeError("participant_logits must contain ParticipantSevereLogits")
        key = (
            _identifier(row.source_id, field="source_id"),
            _identifier(row.participant_id, field="participant_id"),
        )
        if key not in expected or key in by_key:
            raise ValueError(
                "prediction callback does not exactly cover held participants"
            )
        by_key[key] = row
    if set(by_key) != expected:
        raise ValueError(
            "prediction callback must cover every held participant exactly once"
        )

    prediction_rows: list[dict[str, Any]] = []
    score_ledger: list[dict[str, Any]] = []
    score_chunks: list[bytes] = []
    offset = 0
    for source, participant in sorted(by_key, key=lambda key: key[1].encode()):
        row = by_key[(source, participant)]
        if set(row.raw_logits_by_horizon) != set(EVENT_REFERENCE_HORIZONS):
            raise ValueError("raw logits must cover both frozen horizons")
        stream = streams.get((source, participant))
        if stream is None:
            raise ValueError("prediction callback participant is absent from authority")
        for horizon in EVENT_REFERENCE_HORIZONS:
            logits = _float32_logits(
                row.raw_logits_by_horizon[horizon],
                expected_count=stream.decision_count,
                field=f"raw_logits[{source}|{participant}|{horizon}]",
            )
            binary = logits.tobytes(order="C")
            score_chunks.append(binary)
            probabilities = torch.sigmoid(
                torch.tensor(logits.tolist(), dtype=torch.float32)
            )
            with torch.no_grad():
                surrogate = participant_event_surrogate(
                    probabilities,
                    stream.decision_minutes,
                    stream.event_onset_minutes(horizon),
                    horizon_minutes=horizon,
                    patient_days=stream.patient_days,
                )
            event_credit_sum = float(surrogate.event_credit.sum())
            false_alarm_mass = float(surrogate.false_alarm_mass)
            prediction_rows.append(
                {
                    "source_id": source,
                    "participant_id": participant,
                    "fold_id": recipe.fold_id,
                    "horizon_minutes": horizon,
                    "decision_count": stream.decision_count,
                    "patient_days": stream.patient_days,
                    "event_count": stream.event_count(horizon),
                    "event_credit_sum": event_credit_sum,
                    "false_alarm_mass": false_alarm_mass,
                    "stream_content_sha256": stream.content_sha256,
                    "window_shard_sha256": stream.window_shard_sha256,
                    "checkpoint_receipt_sha256": checkpoint_digest,
                }
            )
            score_ledger.append(
                {
                    "source_id": source,
                    "participant_id": participant,
                    "fold_id": recipe.fold_id,
                    "horizon_minutes": horizon,
                    "decision_count": stream.decision_count,
                    "byte_offset": offset,
                    "byte_count": len(binary),
                    "stream_content_sha256": stream.content_sha256,
                    "window_shard_sha256": stream.window_shard_sha256,
                    "checkpoint_receipt_sha256": checkpoint_digest,
                }
            )
            offset += len(binary)
    score_bytes = b"".join(score_chunks)
    if not score_bytes or offset != len(score_bytes):
        raise AssertionError("decision-score binary assembly drift")
    return prediction_rows, score_bytes, score_ledger


def _validated_execution_inputs(
    *,
    parent_base_reference_artifact: Mapping[str, Any],
    event_authority: Mapping[str, Any],
    expected_authority_sha256: str,
    shard_root: str | Path,
    participant_shard_manifest: Mapping[str, Any],
    split_manifest_sha256: str,
    preprocessing_config_sha256: str,
    loss_score_contract_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    parent = validate_weighted_reference_artifact(parent_base_reference_artifact)
    authority = validate_production_event_authority_against_shards(
        event_authority,
        parent_base_reference_artifact=parent,
        shard_root=shard_root,
        participant_shard_manifest=participant_shard_manifest,
    )
    expected_digest = _sha256(
        expected_authority_sha256,
        field="expected_authority_sha256",
    )
    if authority["artifact_sha256"] != expected_digest:
        raise ValueError(
            "expected authority SHA-256 does not match event authority"
        )
    split_digest = _sha256(split_manifest_sha256, field="split_manifest_sha256")
    preprocessing_digest = _sha256(
        preprocessing_config_sha256,
        field="preprocessing_config_sha256",
    )
    if (
        authority["bindings"]["split_manifest_sha256"] != split_digest
        or authority["bindings"]["preprocessing_config_sha256"]
        != preprocessing_digest
    ):
        raise ValueError("execution bindings disagree with event authority")
    contract_digest = _sha256(
        loss_score_contract_sha256,
        field="loss_score_contract_sha256",
    )
    return parent, authority, expected_digest, contract_digest


def execute_cf5_event_reference(
    *,
    parent_base_reference_artifact: Mapping[str, Any],
    event_authority: Mapping[str, Any],
    expected_authority_sha256: str,
    shard_root: str | Path,
    execution_root: str | Path,
    runtime_provenance_receipt: Mapping[str, Any],
    resolved_config_path: str | Path,
    input_manifest_path: str | Path,
    split_manifest_path: str | Path,
    preprocessing_contract_path: str | Path,
    loss_score_contract_path: str | Path,
    participant_shard_manifest_path: str | Path,
    participant_shard_manifest: Mapping[str, Any],
    expected_runtime_pins: Mapping[str, str] | None,
    fit_callback: FitCallback,
    prediction_callback: PredictionCallback,
) -> CF5EventExecutionResult:
    """Run all 55 callback recipes and build one immutable event reference.

    Callbacks receive recipe metadata only.  The executor derives all scope,
    writes canonical JSON itself, re-hashes every output, and delegates the
    final parent/shard/receipt revalidation to the production authority builder.
    """

    if not callable(fit_callback) or not callable(prediction_callback):
        raise TypeError("fit_callback and prediction_callback must be callable")
    runtime_receipt = verify_runtime_provenance_receipt_against_paths(
        runtime_provenance_receipt,
        resolved_config_path=resolved_config_path,
        input_manifest_path=input_manifest_path,
        split_manifest_path=split_manifest_path,
        preprocessing_contract_path=preprocessing_contract_path,
        loss_score_contract_path=loss_score_contract_path,
        participant_shard_manifest_path=participant_shard_manifest_path,
        participant_shard_root=shard_root,
        participant_shard_partition="train",
        expected_pins=expected_runtime_pins,
    )
    runtime_pins = runtime_provenance_pins(runtime_receipt)
    split_manifest_sha256 = runtime_pins["split_manifest"]
    preprocessing_config_sha256 = runtime_pins["preprocessing_contract"]
    loss_score_contract_sha256 = runtime_pins["loss_score_contract"]
    parent, authority, expected_digest, contract_digest = _validated_execution_inputs(
        parent_base_reference_artifact=parent_base_reference_artifact,
        event_authority=event_authority,
        expected_authority_sha256=expected_authority_sha256,
        shard_root=shard_root,
        participant_shard_manifest=participant_shard_manifest,
        split_manifest_sha256=split_manifest_sha256,
        preprocessing_config_sha256=preprocessing_config_sha256,
        loss_score_contract_sha256=loss_score_contract_sha256,
    )
    recipes = build_cf5_event_recipes(parent)
    root = Path(execution_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    streams = _materialize_streams(authority, shard_root=shard_root)

    checkpoints: list[CheckpointReceipt] = []
    predictions: list[PredictionReceipt] = []
    for recipe in recipes:
        fit_result = fit_callback(recipe)
        if not isinstance(fit_result, FitCallbackResult):
            raise TypeError("fit_callback must return FitCallbackResult")
        config_bytes = _canonical_json(
            _execution_config_payload(
                recipe,
                fit_result,
                runtime_provenance_artifact_sha256=runtime_receipt[
                    "artifact_sha256"
                ],
            ),
            field="callback config",
        )
        _, config_digest = _write_bytes(
            root,
            recipe.config_relative_path,
            config_bytes,
            field="callback config",
            canonical_json=True,
        )
        epoch = _integer(
            fit_result.checkpoint_epoch,
            field="checkpoint_epoch",
            minimum=1,
        )
        if epoch > 30:
            raise ValueError("checkpoint_epoch exceeds the frozen CF5 maximum")
        if epoch != recipe.expected_checkpoint_epoch:
            raise ValueError("callback checkpoint epoch disagrees with parent fold")
        state_digest = _sha256(
            fit_result.checkpoint_state_sha256,
            field="fit callback checkpoint_state_sha256",
        )
        if state_digest != recipe.expected_checkpoint_state_sha256:
            raise ValueError("callback checkpoint state disagrees with parent fold")
        _, checkpoint_digest = _write_bytes(
            root,
            recipe.checkpoint_relative_path,
            fit_result.checkpoint_bytes,
            field="callback checkpoint",
            canonical_json=False,
        )
        checkpoint_receipt = CheckpointReceipt(
            source_id=recipe.source_id,
            fold_id=recipe.fold_id,
            fit_participant_ids=recipe.fit_participant_ids,
            held_participant_ids=recipe.held_participant_ids,
            parent_base_reference_artifact_sha256=parent["artifact_sha256"],
            base_fold_sha256=recipe.base_fold_sha256,
            config_relative_path=recipe.config_relative_path,
            config_sha256=config_digest,
            checkpoint_relative_path=recipe.checkpoint_relative_path,
            checkpoint_sha256=checkpoint_digest,
            checkpoint_epoch=epoch,
            checkpoint_state_sha256=state_digest,
        )
        checkpoint_digest = checkpoint_receipt_sha256(checkpoint_receipt)
        request = PredictionCallbackRequest(
            recipe=recipe,
            checkpoint_receipt=checkpoint_receipt,
            checkpoint_receipt_sha256=checkpoint_digest,
            expected_authority_sha256=expected_digest,
            loss_score_contract_sha256=contract_digest,
        )
        prediction_result = prediction_callback(request)
        prediction_rows, score_bytes, score_ledger = _derived_prediction_outputs(
            prediction_result,
            recipe=recipe,
            checkpoint_digest=checkpoint_digest,
            streams=streams,
        )
        _, score_digest = _write_bytes(
            root,
            recipe.score_relative_path,
            score_bytes,
            field="callback decision scores",
            canonical_json=False,
        )
        prediction_payload = {
            "schema_version": PREDICTION_SCHEMA_VERSION,
            "source_id": recipe.source_id,
            "fold_id": recipe.fold_id,
            "expected_authority_sha256": expected_digest,
            "checkpoint_receipt_sha256": checkpoint_digest,
            "loss_score_contract_sha256": contract_digest,
            "runtime_provenance_artifact_sha256": runtime_receipt[
                "artifact_sha256"
            ],
            "execution_scope": _validate_scope(
                prediction_result.scope,
                field="prediction callback scope",
            ).as_dict(),
            "decision_score_contract": {
                "transform": SCORE_TRANSFORM_ID,
                "binary_dtype": SCORE_BINARY_DTYPE,
                "binary_relative_path": recipe.score_relative_path,
                "binary_sha256": score_digest,
                "record_order": (
                    "participant-utf8_then-horizon-30-60_then-decision"
                ),
            },
            "decision_score_ledger": score_ledger,
            "prediction_records": prediction_rows,
        }
        prediction_bytes = _canonical_json(
            prediction_payload,
            field="callback prediction",
        )
        _, prediction_digest = _write_bytes(
            root,
            recipe.prediction_relative_path,
            prediction_bytes,
            field="callback prediction",
            canonical_json=True,
        )
        checkpoints.append(checkpoint_receipt)
        predictions.append(
            PredictionReceipt(
                source_id=recipe.source_id,
                fold_id=recipe.fold_id,
                expected_authority_sha256=expected_digest,
                checkpoint_receipt_sha256=checkpoint_digest,
                loss_score_contract_sha256=contract_digest,
                prediction_relative_path=recipe.prediction_relative_path,
                prediction_sha256=prediction_digest,
            )
        )
    if len(checkpoints) != len(PRODUCTION_SOURCE_ORDER) * PRODUCTION_FOLD_COUNT:
        raise ValueError("execution did not produce all 55 checkpoint receipts")
    if len(predictions) != len(PRODUCTION_SOURCE_ORDER) * PRODUCTION_FOLD_COUNT:
        raise ValueError("execution did not produce all 55 prediction receipts")

    artifact = build_production_event_reference_artifact(
        parent_base_reference_artifact=parent,
        expected_authority=authority,
        expected_authority_sha256=expected_digest,
        shard_root=shard_root,
        participant_shard_manifest=participant_shard_manifest,
        receipt_root=root,
        checkpoint_receipts=checkpoints,
        prediction_receipts=predictions,
        split_manifest_sha256=split_manifest_sha256,
        preprocessing_config_sha256=preprocessing_config_sha256,
        loss_score_contract_sha256=contract_digest,
    )
    artifact_path = root / "event-reference.json"
    write_production_event_reference_artifact(artifact, artifact_path)
    if artifact_path.read_bytes() != canonical_json_bytes(artifact):
        raise ValueError("event reference bytes are not canonical after write")
    return CF5EventExecutionResult(
        recipes=recipes,
        checkpoint_receipts=tuple(checkpoints),
        prediction_receipts=tuple(predictions),
        event_reference_artifact=artifact,
        event_reference_path=artifact_path,
    )
