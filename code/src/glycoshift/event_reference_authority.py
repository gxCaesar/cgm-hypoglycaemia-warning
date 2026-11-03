"""Fail-closed production authority for C2 event OOF references.

``event_reference`` remains a deliberately small synthetic ledger.  This module
is the production boundary: it anchors C2 OOF measurements to a validated CF5
base reference, byte-verified participant shards, and structured fold receipts.
It is intentionally not a model-fitting runner.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd

from .event_reference import EVENT_REFERENCE_HORIZONS
from .event_streams import ParticipantEventStream, build_participant_event_stream
from .mero_reference import (
    bind_artifact_sha256,
    canonical_json_bytes,
    read_canonical_json,
    verify_artifact_payload,
    write_immutable_json,
)
from .mero_weighted_reference import (
    PRODUCTION_SOURCE_ORDER,
    validate_weighted_reference_artifact,
)
from .shard_manifest import validate_live_window_partition, validate_window_manifest

AUTHORITY_SCHEMA_VERSION = "1.1"
AUTHORITY_PROTOCOL_ID = "glycoshift-sota-method-v1.3"
AUTHORITY_METHOD = "participant-cf5-train-ledger-event-oof-authority-v1.1"
REFERENCE_SCHEMA_VERSION = "1.1"
REFERENCE_METHOD = "participant-cf5-c2-checkpoint-bound-event-oof-reference-v1.1"
PREDICTION_SCHEMA_VERSION = "1.1"
REFERENCE_SPLIT = "train"
PRODUCTION_FOLD_COUNT = 5

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


def _finite(value: Any, *, field: str, minimum: float = 0.0) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{field} must be numeric")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < minimum:
        raise ValueError(f"{field} must be finite and at least {minimum}")
    return resolved


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


def _resolve_relative(root: Path, relative_path: str, *, field: str) -> Path:
    safe = _safe_relative_path(relative_path, field=field)
    resolved_root = root.expanduser().resolve()
    if not resolved_root.is_dir():
        raise FileNotFoundError(f"authority root is absent: {resolved_root}")
    candidate = (resolved_root / Path(safe)).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"{field} escapes its authority root") from error
    if not candidate.is_file():
        raise FileNotFoundError(f"authority file is absent: {candidate}")
    return candidate


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    return value


def _require_sequence(value: Any, *, field: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{field} must be a sequence")
    return value


@dataclass(frozen=True)
class ParticipantStreamInput:
    """A rebuilt participant stream plus its safe path below the shard root."""

    stream: ParticipantEventStream
    shard_relative_path: str


@dataclass(frozen=True)
class CheckpointReceipt:
    """A fold-local checkpoint receipt with explicit fit/held scope."""

    source_id: str
    fold_id: int
    fit_participant_ids: tuple[str, ...]
    held_participant_ids: tuple[str, ...]
    parent_base_reference_artifact_sha256: str
    base_fold_sha256: str
    config_relative_path: str
    config_sha256: str
    checkpoint_relative_path: str
    checkpoint_sha256: str
    checkpoint_epoch: int
    checkpoint_state_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "fold_id": self.fold_id,
            "fit_participant_ids": list(self.fit_participant_ids),
            "held_participant_ids": list(self.held_participant_ids),
            "parent_base_reference_artifact_sha256": (
                self.parent_base_reference_artifact_sha256
            ),
            "base_fold_sha256": self.base_fold_sha256,
            "config_relative_path": self.config_relative_path,
            "config_sha256": self.config_sha256,
            "checkpoint_relative_path": self.checkpoint_relative_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_epoch": self.checkpoint_epoch,
            "checkpoint_state_sha256": self.checkpoint_state_sha256,
        }


@dataclass(frozen=True)
class PredictionReceipt:
    """A canonical fold prediction file bound to checkpoint and authority."""

    source_id: str
    fold_id: int
    expected_authority_sha256: str
    checkpoint_receipt_sha256: str
    loss_score_contract_sha256: str
    prediction_relative_path: str
    prediction_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "fold_id": self.fold_id,
            "expected_authority_sha256": self.expected_authority_sha256,
            "checkpoint_receipt_sha256": self.checkpoint_receipt_sha256,
            "loss_score_contract_sha256": self.loss_score_contract_sha256,
            "prediction_relative_path": self.prediction_relative_path,
            "prediction_sha256": self.prediction_sha256,
        }


@dataclass(frozen=True)
class ProductionEventOOFRecord:
    """One receipt-derived held-fold OOF numerator ledger entry."""

    source_id: str
    participant_id: str
    fold_id: int
    horizon_minutes: int
    decision_count: int
    patient_days: int
    event_count: int
    event_credit_sum: float
    false_alarm_mass: float
    stream_content_sha256: str
    window_shard_sha256: str
    checkpoint_receipt_sha256: str
    prediction_receipt_sha256: str

    @property
    def event_miss(self) -> float:
        if self.event_count == 0:
            return 0.0
        return 1.0 - self.event_credit_sum / self.event_count

    @property
    def false_alarms_per_patient_day(self) -> float:
        return self.false_alarm_mass / self.patient_days

    def as_prediction_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "participant_id": self.participant_id,
            "fold_id": self.fold_id,
            "horizon_minutes": self.horizon_minutes,
            "decision_count": self.decision_count,
            "patient_days": self.patient_days,
            "event_count": self.event_count,
            "event_credit_sum": self.event_credit_sum,
            "false_alarm_mass": self.false_alarm_mass,
            "stream_content_sha256": self.stream_content_sha256,
            "window_shard_sha256": self.window_shard_sha256,
            "checkpoint_receipt_sha256": self.checkpoint_receipt_sha256,
        }

    def as_dict(self) -> dict[str, Any]:
        payload = self.as_prediction_dict()
        payload.update(
            {
                "prediction_receipt_sha256": self.prediction_receipt_sha256,
                "event_miss": self.event_miss,
                "false_alarms_per_patient_day": self.false_alarms_per_patient_day,
            }
        )
        return payload


def _normalized_participants(values: Iterable[Any], *, field: str) -> tuple[str, ...]:
    participants = tuple(_identifier(value, field=field) for value in values)
    if not participants or len(set(participants)) != len(participants):
        raise ValueError(f"{field} must be non-empty and unique")
    return tuple(sorted(participants, key=str.encode))


def base_fold_scope_sha256(
    *,
    source_id: str,
    fold_id: int,
    fit_participant_ids: Iterable[str],
    held_participant_ids: Iterable[str],
) -> str:
    """Hash the exact ordered source/fold fit and held participant sets."""

    payload = {
        "source_id": _identifier(source_id, field="source_id"),
        "fold_id": _integer(fold_id, field="fold_id", minimum=0),
        "fit_participant_ids": list(
            _normalized_participants(fit_participant_ids, field="fit_participant_id")
        ),
        "held_participant_ids": list(
            _normalized_participants(
                held_participant_ids,
                field="held_participant_id",
            )
        ),
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _normalize_checkpoint_receipt(receipt: CheckpointReceipt) -> CheckpointReceipt:
    if not isinstance(receipt, CheckpointReceipt):
        raise TypeError("checkpoint_receipts must contain CheckpointReceipt values")
    fold_id = _integer(receipt.fold_id, field="fold_id", minimum=0)
    if fold_id >= PRODUCTION_FOLD_COUNT:
        raise ValueError("fold_id must be less than the production fold count")
    epoch = _integer(receipt.checkpoint_epoch, field="checkpoint_epoch", minimum=1)
    if epoch > 30:
        raise ValueError("checkpoint_epoch exceeds the frozen CF5 maximum")
    return CheckpointReceipt(
        source_id=_identifier(receipt.source_id, field="source_id"),
        fold_id=fold_id,
        fit_participant_ids=_normalized_participants(
            receipt.fit_participant_ids,
            field="fit_participant_id",
        ),
        held_participant_ids=_normalized_participants(
            receipt.held_participant_ids,
            field="held_participant_id",
        ),
        parent_base_reference_artifact_sha256=_sha256(
            receipt.parent_base_reference_artifact_sha256,
            field="parent_base_reference_artifact_sha256",
        ),
        base_fold_sha256=_sha256(receipt.base_fold_sha256, field="base_fold_sha256"),
        config_relative_path=_safe_relative_path(
            receipt.config_relative_path,
            field="config_relative_path",
        ),
        config_sha256=_sha256(receipt.config_sha256, field="config_sha256"),
        checkpoint_relative_path=_safe_relative_path(
            receipt.checkpoint_relative_path,
            field="checkpoint_relative_path",
        ),
        checkpoint_sha256=_sha256(
            receipt.checkpoint_sha256,
            field="checkpoint_sha256",
        ),
        checkpoint_epoch=epoch,
        checkpoint_state_sha256=_sha256(
            receipt.checkpoint_state_sha256,
            field="checkpoint_state_sha256",
        ),
    )


def checkpoint_receipt_sha256(receipt: CheckpointReceipt) -> str:
    normalized = _normalize_checkpoint_receipt(receipt)
    return hashlib.sha256(canonical_json_bytes(normalized.as_dict())).hexdigest()


def _normalize_prediction_receipt(receipt: PredictionReceipt) -> PredictionReceipt:
    if not isinstance(receipt, PredictionReceipt):
        raise TypeError("prediction_receipts must contain PredictionReceipt values")
    fold_id = _integer(receipt.fold_id, field="fold_id", minimum=0)
    if fold_id >= PRODUCTION_FOLD_COUNT:
        raise ValueError("fold_id must be less than the production fold count")
    return PredictionReceipt(
        source_id=_identifier(receipt.source_id, field="source_id"),
        fold_id=fold_id,
        expected_authority_sha256=_sha256(
            receipt.expected_authority_sha256,
            field="expected_authority_sha256",
        ),
        checkpoint_receipt_sha256=_sha256(
            receipt.checkpoint_receipt_sha256,
            field="checkpoint_receipt_sha256",
        ),
        loss_score_contract_sha256=_sha256(
            receipt.loss_score_contract_sha256,
            field="loss_score_contract_sha256",
        ),
        prediction_relative_path=_safe_relative_path(
            receipt.prediction_relative_path,
            field="prediction_relative_path",
        ),
        prediction_sha256=_sha256(
            receipt.prediction_sha256,
            field="prediction_sha256",
        ),
    )


def prediction_receipt_sha256(receipt: PredictionReceipt) -> str:
    normalized = _normalize_prediction_receipt(receipt)
    return hashlib.sha256(canonical_json_bytes(normalized.as_dict())).hexdigest()


def _normalize_record(
    record: ProductionEventOOFRecord,
    *,
    require_prediction_receipt: bool,
) -> ProductionEventOOFRecord:
    if not isinstance(record, ProductionEventOOFRecord):
        raise TypeError(
            "prediction records must contain ProductionEventOOFRecord values"
        )
    fold_id = _integer(record.fold_id, field="fold_id", minimum=0)
    if fold_id >= PRODUCTION_FOLD_COUNT:
        raise ValueError("fold_id must be less than the production fold count")
    horizon = _integer(record.horizon_minutes, field="horizon_minutes", minimum=1)
    if horizon not in EVENT_REFERENCE_HORIZONS:
        raise ValueError("horizon_minutes must cover the frozen 30/60 horizons")
    event_count = _integer(record.event_count, field="event_count", minimum=0)
    credit = _finite(record.event_credit_sum, field="event_credit_sum")
    if credit > event_count:
        raise ValueError("event_credit_sum cannot exceed event_count")
    if event_count == 0 and credit != 0.0:
        raise ValueError("zero-event records require event_credit_sum=0")
    prediction_digest = _sha256(
        record.prediction_receipt_sha256,
        field="prediction_receipt_sha256",
    )
    if not require_prediction_receipt:
        prediction_digest = "0" * 64
    return ProductionEventOOFRecord(
        source_id=_identifier(record.source_id, field="source_id"),
        participant_id=_identifier(record.participant_id, field="participant_id"),
        fold_id=fold_id,
        horizon_minutes=horizon,
        decision_count=_integer(
            record.decision_count,
            field="decision_count",
            minimum=1,
        ),
        patient_days=_integer(record.patient_days, field="patient_days", minimum=1),
        event_count=event_count,
        event_credit_sum=credit,
        false_alarm_mass=_finite(record.false_alarm_mass, field="false_alarm_mass"),
        stream_content_sha256=_sha256(
            record.stream_content_sha256,
            field="stream_content_sha256",
        ),
        window_shard_sha256=_sha256(
            record.window_shard_sha256,
            field="window_shard_sha256",
        ),
        checkpoint_receipt_sha256=_sha256(
            record.checkpoint_receipt_sha256,
            field="checkpoint_receipt_sha256",
        ),
        prediction_receipt_sha256=prediction_digest,
    )


def _record_from_prediction_payload(
    row: Mapping[str, Any],
    *,
    prediction_receipt_digest: str,
) -> ProductionEventOOFRecord:
    required = {
        "source_id",
        "participant_id",
        "fold_id",
        "horizon_minutes",
        "decision_count",
        "patient_days",
        "event_count",
        "event_credit_sum",
        "false_alarm_mass",
        "stream_content_sha256",
        "window_shard_sha256",
        "checkpoint_receipt_sha256",
    }
    missing = sorted(required.difference(row))
    extras = sorted(set(row).difference(required))
    if missing or extras:
        raise ValueError(
            "prediction record fields differ from the frozen schema: "
            f"missing={missing}, extras={extras}"
        )
    return _normalize_record(
        ProductionEventOOFRecord(
            source_id=row["source_id"],
            participant_id=row["participant_id"],
            fold_id=row["fold_id"],
            horizon_minutes=row["horizon_minutes"],
            decision_count=row["decision_count"],
            patient_days=row["patient_days"],
            event_count=row["event_count"],
            event_credit_sum=row["event_credit_sum"],
            false_alarm_mass=row["false_alarm_mass"],
            stream_content_sha256=row["stream_content_sha256"],
            window_shard_sha256=row["window_shard_sha256"],
            checkpoint_receipt_sha256=row["checkpoint_receipt_sha256"],
            prediction_receipt_sha256=prediction_receipt_digest,
        ),
        require_prediction_receipt=True,
    )


def _record_from_artifact_payload(row: Mapping[str, Any]) -> ProductionEventOOFRecord:
    required = set(ProductionEventOOFRecord.__dataclass_fields__) | {
        "event_miss",
        "false_alarms_per_patient_day",
    }
    missing = sorted(required.difference(row))
    extras = sorted(set(row).difference(required))
    if missing or extras:
        raise ValueError(
            "production OOF record fields differ from the frozen schema: "
            f"missing={missing}, extras={extras}"
        )
    record = _normalize_record(
        ProductionEventOOFRecord(
            source_id=row["source_id"],
            participant_id=row["participant_id"],
            fold_id=row["fold_id"],
            horizon_minutes=row["horizon_minutes"],
            decision_count=row["decision_count"],
            patient_days=row["patient_days"],
            event_count=row["event_count"],
            event_credit_sum=row["event_credit_sum"],
            false_alarm_mass=row["false_alarm_mass"],
            stream_content_sha256=row["stream_content_sha256"],
            window_shard_sha256=row["window_shard_sha256"],
            checkpoint_receipt_sha256=row["checkpoint_receipt_sha256"],
            prediction_receipt_sha256=row["prediction_receipt_sha256"],
        ),
        require_prediction_receipt=True,
    )
    for field, expected in (
        ("event_miss", record.event_miss),
        ("false_alarms_per_patient_day", record.false_alarms_per_patient_day),
    ):
        supplied = _finite(row.get(field), field=field)
        if not math.isclose(supplied, expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"production OOF record {field} disagrees with numerators")
    return record


def _validated_parent(
    parent_base_reference_artifact: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[tuple[str, str], int]]:
    parent = validate_weighted_reference_artifact(parent_base_reference_artifact)
    if tuple(parent["lexical_source_order"]) != PRODUCTION_SOURCE_ORDER:
        raise ValueError(
            "parent base reference source order is not the frozen production order"
        )
    if parent["fold_count"] != PRODUCTION_FOLD_COUNT:
        raise ValueError("parent base reference must use exactly CF5")
    expected: dict[tuple[str, str], int] = {}
    for row in parent["participant_fold_assignment"]:
        source = _identifier(row["source_id"], field="source_id")
        participant = _identifier(row["participant_id"], field="participant_id")
        fold_id = _integer(row["fold_id"], field="fold_id", minimum=0)
        if fold_id >= PRODUCTION_FOLD_COUNT:
            raise ValueError("parent assignment has an invalid CF5 fold")
        key = (source, participant)
        if key in expected:
            raise ValueError("parent assignment repeats a source-scoped participant")
        expected[key] = fold_id
    for source in PRODUCTION_SOURCE_ORDER:
        folds = {fold for (current, _), fold in expected.items() if current == source}
        if folds != set(range(PRODUCTION_FOLD_COUNT)):
            raise ValueError("parent assignment does not cover every source-local fold")
    return parent, expected


def _ordered_assignments(
    expected: Mapping[tuple[str, str], int],
) -> list[dict[str, Any]]:
    source_to_index = {
        source: index for index, source in enumerate(PRODUCTION_SOURCE_ORDER)
    }
    return [
        {"source_id": source, "participant_id": participant, "fold_id": fold_id}
        for (source, participant), fold_id in sorted(
            expected.items(),
            key=lambda item: (
                source_to_index[item[0][0]],
                item[0][1].encode("utf-8"),
            ),
        )
    ]


def _stream_metadata_from_input(
    item: ParticipantStreamInput,
    *,
    shard_root: Path,
) -> dict[str, Any]:
    if not isinstance(item, ParticipantStreamInput):
        raise TypeError("stream_inputs must contain ParticipantStreamInput values")
    if not isinstance(item.stream, ParticipantEventStream):
        raise TypeError(
            "ParticipantStreamInput.stream must be a ParticipantEventStream"
        )
    stream = item.stream
    relative_path = _safe_relative_path(
        item.shard_relative_path,
        field="shard_relative_path",
    )
    if not relative_path.startswith("partition=train/"):
        raise ValueError("participant stream shard path must be train-only")
    path = _resolve_relative(
        shard_root,
        relative_path,
        field="shard_relative_path",
    )
    observed_shard = _sha256_file(path)
    if observed_shard != _sha256(
        stream.window_shard_sha256,
        field="window_shard_sha256",
    ):
        raise ValueError("participant stream shard SHA-256 does not match shard bytes")
    frame = pd.read_parquet(path)
    rebuilt = build_participant_event_stream(
        frame,
        expected_content_sha256=_sha256(stream.content_sha256, field="content_sha256"),
        window_shard_sha256=observed_shard,
    )
    if (
        rebuilt.source_id != stream.source_id
        or rebuilt.participant_id != stream.participant_id
        or rebuilt.partition != stream.partition
        or rebuilt.decision_count != stream.decision_count
        or rebuilt.patient_days != stream.patient_days
        or tuple(rebuilt.decision_minutes.tolist())
        != tuple(stream.decision_minutes.tolist())
    ):
        raise ValueError("supplied participant stream disagrees with its shard")
    for horizon in EVENT_REFERENCE_HORIZONS:
        if tuple(rebuilt.event_onset_minutes(horizon).tolist()) != tuple(
            stream.event_onset_minutes(horizon).tolist()
        ):
            raise ValueError(
                "supplied participant event onsets disagree with its shard"
            )
    if rebuilt.partition != REFERENCE_SPLIT:
        raise ValueError("participant stream must be train-only")
    return {
        "source_id": rebuilt.source_id,
        "participant_id": rebuilt.participant_id,
        "partition": rebuilt.partition,
        "shard_relative_path": relative_path,
        "window_shard_sha256": observed_shard,
        "stream_content_sha256": rebuilt.content_sha256,
        "decision_count": rebuilt.decision_count,
        "patient_days": rebuilt.patient_days,
        "event_counts": {
            str(horizon): rebuilt.event_count(horizon)
            for horizon in EVENT_REFERENCE_HORIZONS
        },
    }


def _normalize_stream_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "source_id",
        "participant_id",
        "fold_id",
        "partition",
        "shard_relative_path",
        "window_shard_sha256",
        "stream_content_sha256",
        "decision_count",
        "patient_days",
        "event_counts",
    }
    missing = sorted(required.difference(row))
    if missing:
        raise ValueError(
            f"participant stream authority row is missing fields: {missing}"
        )
    counts = _require_mapping(row["event_counts"], field="event_counts")
    if set(counts) != {str(value) for value in EVENT_REFERENCE_HORIZONS}:
        raise ValueError("participant stream authority must cover both frozen horizons")
    shard_relative_path = _safe_relative_path(
        row["shard_relative_path"],
        field="shard_relative_path",
    )
    if not shard_relative_path.startswith("partition=train/"):
        raise ValueError("authority shard path must be train-only")
    return {
        "source_id": _identifier(row["source_id"], field="source_id"),
        "participant_id": _identifier(row["participant_id"], field="participant_id"),
        "fold_id": _integer(row["fold_id"], field="fold_id", minimum=0),
        "partition": _identifier(row["partition"], field="partition"),
        "shard_relative_path": shard_relative_path,
        "window_shard_sha256": _sha256(
            row["window_shard_sha256"],
            field="window_shard_sha256",
        ),
        "stream_content_sha256": _sha256(
            row["stream_content_sha256"],
            field="stream_content_sha256",
        ),
        "decision_count": _integer(
            row["decision_count"],
            field="decision_count",
            minimum=1,
        ),
        "patient_days": _integer(row["patient_days"], field="patient_days", minimum=1),
        "event_counts": {
            str(horizon): _integer(
                counts[str(horizon)],
                field=f"event_counts[{horizon}]",
                minimum=0,
            )
            for horizon in EVENT_REFERENCE_HORIZONS
        },
    }


def _authority_payload(
    *,
    parent: Mapping[str, Any],
    expected: Mapping[tuple[str, str], int],
    stream_rows: Sequence[Mapping[str, Any]],
    split_manifest_sha256: str,
    window_manifest_sha256: str,
    preprocessing_config_sha256: str,
) -> dict[str, Any]:
    normalized_streams = [_normalize_stream_metadata(row) for row in stream_rows]
    stream_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    paths: set[str] = set()
    for row in normalized_streams:
        key = (row["source_id"], row["participant_id"])
        if key in stream_by_key:
            raise ValueError("participant stream authority repeats a participant")
        if row["shard_relative_path"] in paths:
            raise ValueError("participant stream authority repeats a shard path")
        paths.add(row["shard_relative_path"])
        if row["partition"] != REFERENCE_SPLIT:
            raise ValueError("participant stream authority must use train-only streams")
        if row["fold_id"] != expected.get(key):
            raise ValueError("participant stream fold disagrees with parent assignment")
        stream_by_key[key] = row
    if set(stream_by_key) != set(expected):
        raise ValueError(
            "participant stream authority does not exactly cover parent ledger"
        )
    for source in PRODUCTION_SOURCE_ORDER:
        for horizon in EVENT_REFERENCE_HORIZONS:
            if not any(
                row["event_counts"][str(horizon)] > 0
                for row in stream_by_key.values()
                if row["source_id"] == source
            ):
                raise ValueError(
                    "every source/horizon requires an event-positive stream"
                )
    source_to_index = {
        source: index for index, source in enumerate(PRODUCTION_SOURCE_ORDER)
    }
    ordered_streams = sorted(
        stream_by_key.values(),
        key=lambda row: (
            source_to_index[row["source_id"]],
            row["participant_id"].encode("utf-8"),
        ),
    )
    return {
        "schema_version": AUTHORITY_SCHEMA_VERSION,
        "protocol_id": AUTHORITY_PROTOCOL_ID,
        "method": AUTHORITY_METHOD,
        "split": REFERENCE_SPLIT,
        "lexical_source_order": list(PRODUCTION_SOURCE_ORDER),
        "fold_count": PRODUCTION_FOLD_COUNT,
        "horizon_order_minutes": list(EVENT_REFERENCE_HORIZONS),
        "bindings": {
            "parent_base_reference_artifact_sha256": _sha256(
                parent["artifact_sha256"],
                field="parent_base_reference_artifact_sha256",
            ),
            "parent_base_fold_ledger_sha256": _sha256(
                parent["fold_ledger_sha256"],
                field="parent_base_fold_ledger_sha256",
            ),
            "split_manifest_sha256": _sha256(
                split_manifest_sha256,
                field="split_manifest_sha256",
            ),
            "window_manifest_sha256": _sha256(
                window_manifest_sha256,
                field="window_manifest_sha256",
            ),
            "preprocessing_config_sha256": _sha256(
                preprocessing_config_sha256,
                field="preprocessing_config_sha256",
            ),
        },
        "participant_fold_assignment": _ordered_assignments(expected),
        "participant_streams": ordered_streams,
    }


def build_production_event_authority(
    *,
    parent_base_reference_artifact: Mapping[str, Any],
    shard_root: str | Path,
    participant_shard_manifest: Mapping[str, Any],
    stream_inputs: Sequence[ParticipantStreamInput],
    split_manifest_sha256: str,
    window_manifest_sha256: str,
    preprocessing_config_sha256: str,
) -> dict[str, Any]:
    """Build a self-hashed, CF5-complete authority from real participant shards."""

    parent, expected = _validated_parent(parent_base_reference_artifact)
    root = Path(shard_root)
    observed_manifest = validate_live_window_partition(
        root,
        validate_window_manifest(participant_shard_manifest),
        partition="train",
    )
    if observed_manifest["sha256"] != _sha256(
        window_manifest_sha256,
        field="window_manifest_sha256",
    ):
        raise ValueError("window shard manifest SHA-256 does not match shard root")
    rows: list[dict[str, Any]] = []
    for item in stream_inputs:
        row = _stream_metadata_from_input(item, shard_root=root)
        key = (row["source_id"], row["participant_id"])
        if key not in expected:
            raise ValueError("participant stream is absent from the parent base ledger")
        row["fold_id"] = expected[key]
        rows.append(row)
    return bind_artifact_sha256(
        _authority_payload(
            parent=parent,
            expected=expected,
            stream_rows=rows,
            split_manifest_sha256=split_manifest_sha256,
            window_manifest_sha256=window_manifest_sha256,
            preprocessing_config_sha256=preprocessing_config_sha256,
        )
    )


def validate_production_event_authority(
    payload: Mapping[str, Any],
    *,
    parent_base_reference_artifact: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate canonical authority structure and optionally its parent binding."""

    if not isinstance(payload, Mapping):
        raise TypeError("production event authority must be a mapping")
    verify_artifact_payload(payload)
    if payload.get("schema_version") != AUTHORITY_SCHEMA_VERSION:
        raise ValueError("unsupported event authority schema_version")
    if payload.get("protocol_id") != AUTHORITY_PROTOCOL_ID:
        raise ValueError("event authority protocol_id is incompatible")
    if payload.get("method") != AUTHORITY_METHOD:
        raise ValueError("event authority method is incompatible")
    if payload.get("split") != REFERENCE_SPLIT:
        raise ValueError("event authority split must be train")
    if tuple(payload.get("lexical_source_order", ())) != PRODUCTION_SOURCE_ORDER:
        raise ValueError("event authority source order is incompatible")
    if payload.get("fold_count") != PRODUCTION_FOLD_COUNT:
        raise ValueError("event authority must use exactly CF5")
    if tuple(payload.get("horizon_order_minutes", ())) != EVENT_REFERENCE_HORIZONS:
        raise ValueError("event authority horizons are incompatible")
    bindings = _require_mapping(payload.get("bindings"), field="bindings")
    for field in (
        "parent_base_reference_artifact_sha256",
        "parent_base_fold_ledger_sha256",
        "split_manifest_sha256",
        "window_manifest_sha256",
        "preprocessing_config_sha256",
    ):
        _sha256(bindings.get(field), field=f"bindings.{field}")

    assignments: dict[tuple[str, str], int] = {}
    for row in _require_sequence(
        payload.get("participant_fold_assignment"),
        field="participant_fold_assignment",
    ):
        mapping = _require_mapping(row, field="participant_fold_assignment row")
        key = (
            _identifier(mapping.get("source_id"), field="source_id"),
            _identifier(mapping.get("participant_id"), field="participant_id"),
        )
        fold_id = _integer(mapping.get("fold_id"), field="fold_id", minimum=0)
        if fold_id >= PRODUCTION_FOLD_COUNT or key in assignments:
            raise ValueError("event authority assignment ledger is invalid")
        assignments[key] = fold_id
    for source in PRODUCTION_SOURCE_ORDER:
        if {
            fold for (current, _), fold in assignments.items() if current == source
        } != set(range(PRODUCTION_FOLD_COUNT)):
            raise ValueError(
                "event authority assignment ledger misses a source-local fold"
            )

    streams = [
        _normalize_stream_metadata(
            _require_mapping(row, field="participant_stream row")
        )
        for row in _require_sequence(
            payload.get("participant_streams"),
            field="participant_streams",
        )
    ]
    if parent_base_reference_artifact is None:
        expected = assignments
        parent = None
    else:
        parent, expected = _validated_parent(parent_base_reference_artifact)
        if (
            bindings["parent_base_reference_artifact_sha256"]
            != parent["artifact_sha256"]
        ):
            raise ValueError("event authority parent artifact binding disagrees")
        if bindings["parent_base_fold_ledger_sha256"] != parent["fold_ledger_sha256"]:
            raise ValueError("event authority parent fold ledger binding disagrees")
        if assignments != expected:
            raise ValueError("event authority assignment ledger disagrees with parent")
    rebuilt = bind_artifact_sha256(
        _authority_payload(
            parent=(
                parent
                if parent is not None
                else {
                    "artifact_sha256": bindings[
                        "parent_base_reference_artifact_sha256"
                    ],
                    "fold_ledger_sha256": bindings["parent_base_fold_ledger_sha256"],
                }
            ),
            expected=expected,
            stream_rows=streams,
            split_manifest_sha256=bindings["split_manifest_sha256"],
            window_manifest_sha256=bindings["window_manifest_sha256"],
            preprocessing_config_sha256=bindings["preprocessing_config_sha256"],
        )
    )
    if rebuilt != dict(payload):
        raise ValueError("event authority contains derived-field drift")
    return rebuilt


def validate_production_event_authority_against_shards(
    payload: Mapping[str, Any],
    *,
    parent_base_reference_artifact: Mapping[str, Any],
    shard_root: str | Path,
    participant_shard_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Re-open every bound shard and fail on byte, stream, or count drift."""

    authority = validate_production_event_authority(
        payload,
        parent_base_reference_artifact=parent_base_reference_artifact,
    )
    root = Path(shard_root)
    observed_manifest = validate_live_window_partition(
        root,
        validate_window_manifest(participant_shard_manifest),
        partition="train",
    )
    if observed_manifest["sha256"] != authority["bindings"]["window_manifest_sha256"]:
        raise ValueError(
            "authority window manifest SHA-256 no longer matches shard root"
        )
    for metadata in authority["participant_streams"]:
        if not metadata["shard_relative_path"].startswith("partition=train/"):
            raise ValueError("authority shard path must be train-only")
        path = _resolve_relative(
            root,
            metadata["shard_relative_path"],
            field="shard_relative_path",
        )
        if _sha256_file(path) != metadata["window_shard_sha256"]:
            raise ValueError("authority shard SHA-256 no longer matches shard bytes")
        frame = pd.read_parquet(path)
        rebuilt = build_participant_event_stream(
            frame,
            expected_content_sha256=metadata["stream_content_sha256"],
            window_shard_sha256=metadata["window_shard_sha256"],
        )
        if (
            rebuilt.source_id != metadata["source_id"]
            or rebuilt.participant_id != metadata["participant_id"]
            or rebuilt.partition != REFERENCE_SPLIT
            or rebuilt.decision_count != metadata["decision_count"]
            or rebuilt.patient_days != metadata["patient_days"]
        ):
            raise ValueError("authority stream metadata no longer matches shard")
        for horizon in EVENT_REFERENCE_HORIZONS:
            if rebuilt.event_count(horizon) != metadata["event_counts"][str(horizon)]:
                raise ValueError("authority event denominator no longer matches shard")
    return authority


def read_production_event_authority(path: str | Path) -> dict[str, Any]:
    return validate_production_event_authority(read_canonical_json(path))


def write_production_event_authority(
    payload: Mapping[str, Any],
    path: str | Path,
) -> str:
    return write_immutable_json(validate_production_event_authority(payload), path)


def _expected_fold_sets(
    expected: Mapping[tuple[str, str], int],
    *,
    source_id: str,
    fold_id: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    source_participants = tuple(
        sorted(
            (
                participant
                for (source, participant) in expected
                if source == source_id
            ),
            key=str.encode,
        )
    )
    held = tuple(
        participant
        for participant in source_participants
        if expected[(source_id, participant)] == fold_id
    )
    fit = tuple(
        participant for participant in source_participants if participant not in held
    )
    if not fit or not held:
        raise ValueError("parent CF5 fold has an empty fit or held participant set")
    return fit, held


def _validate_checkpoint_receipts(
    receipts: Sequence[CheckpointReceipt],
    *,
    expected: Mapping[tuple[str, str], int],
    parent: Mapping[str, Any],
    receipt_root: Path,
) -> dict[tuple[str, int], tuple[CheckpointReceipt, str]]:
    expected_keys = {
        (source, fold_id)
        for source in PRODUCTION_SOURCE_ORDER
        for fold_id in range(PRODUCTION_FOLD_COUNT)
    }
    output: dict[tuple[str, int], tuple[CheckpointReceipt, str]] = {}
    parent_folds = {
        (str(row["source_id"]), int(row["fold_id"])): row
        for row in parent["folds"]
    }
    if set(parent_folds) != expected_keys:
        raise ValueError("parent weighted reference fold ledger is incomplete")
    for supplied in receipts:
        receipt = _normalize_checkpoint_receipt(supplied)
        key = (receipt.source_id, receipt.fold_id)
        if key not in expected_keys or key in output:
            raise ValueError(
                "checkpoint receipts must be unique and cover production source/folds"
            )
        fit, held = _expected_fold_sets(
            expected,
            source_id=receipt.source_id,
            fold_id=receipt.fold_id,
        )
        if receipt.fit_participant_ids != fit or receipt.held_participant_ids != held:
            raise ValueError(
                "checkpoint receipt fit/held sets disagree with parent CF5 ledger"
            )
        source_participants = {
            participant
            for (source, participant) in expected
            if source == receipt.source_id
        }
        if set(fit).intersection(held) or set(fit).union(held) != source_participants:
            raise ValueError(
                "checkpoint receipt fit/held sets are not disjoint complements"
            )
        expected_fold_hash = base_fold_scope_sha256(
            source_id=receipt.source_id,
            fold_id=receipt.fold_id,
            fit_participant_ids=fit,
            held_participant_ids=held,
        )
        if receipt.base_fold_sha256 != expected_fold_hash:
            raise ValueError(
                "checkpoint receipt base-fold hash disagrees with CF5 ledger"
            )
        if receipt.parent_base_reference_artifact_sha256 != parent["artifact_sha256"]:
            raise ValueError("checkpoint receipt parent artifact binding disagrees")
        parent_fold = parent_folds[key]
        if receipt.checkpoint_epoch != parent_fold["checkpoint_epoch"]:
            raise ValueError("checkpoint receipt epoch disagrees with parent fold")
        if (
            receipt.checkpoint_state_sha256
            != parent_fold["checkpoint_state_sha256"]
        ):
            raise ValueError("checkpoint receipt state disagrees with parent fold")
        for relative, digest, field in (
            (receipt.config_relative_path, receipt.config_sha256, "config"),
            (receipt.checkpoint_relative_path, receipt.checkpoint_sha256, "checkpoint"),
        ):
            path = _resolve_relative(
                receipt_root,
                relative,
                field=f"{field}_relative_path",
            )
            if _sha256_file(path) != digest:
                raise ValueError(
                    f"checkpoint receipt {field} SHA-256 does not match bytes"
                )
        output[key] = (receipt, checkpoint_receipt_sha256(receipt))
    if set(output) != expected_keys:
        raise ValueError(
            "checkpoint receipts must cover all 55 production source/folds"
        )
    return output


def _read_canonical_prediction(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("prediction receipt file is not UTF-8 JSON") from error
    if not isinstance(payload, dict) or raw != canonical_json_bytes(payload):
        raise ValueError("prediction receipt file is not canonical JSON")
    return payload


def _validate_prediction_receipts(
    receipts: Sequence[PredictionReceipt],
    *,
    authority: Mapping[str, Any],
    expected: Mapping[tuple[str, str], int],
    checkpoint_by_key: Mapping[tuple[str, int], tuple[CheckpointReceipt, str]],
    expected_authority_sha256: str,
    loss_score_contract_sha256: str,
    receipt_root: Path,
) -> tuple[
    dict[tuple[str, int], tuple[PredictionReceipt, str]],
    tuple[ProductionEventOOFRecord, ...],
]:
    expected_keys = set(checkpoint_by_key)
    stream_by_key = {
        (row["source_id"], row["participant_id"]): row
        for row in authority["participant_streams"]
    }
    output: dict[tuple[str, int], tuple[PredictionReceipt, str]] = {}
    records: list[ProductionEventOOFRecord] = []
    for supplied in receipts:
        receipt = _normalize_prediction_receipt(supplied)
        key = (receipt.source_id, receipt.fold_id)
        if key not in expected_keys or key in output:
            raise ValueError(
                "prediction receipts must be unique and cover production source/folds"
            )
        checkpoint, checkpoint_digest = checkpoint_by_key[key]
        if receipt.expected_authority_sha256 != expected_authority_sha256:
            raise ValueError("prediction receipt authority binding disagrees")
        if receipt.checkpoint_receipt_sha256 != checkpoint_digest:
            raise ValueError("prediction receipt checkpoint binding disagrees")
        if receipt.loss_score_contract_sha256 != loss_score_contract_sha256:
            raise ValueError("prediction receipt loss/score contract binding disagrees")
        path = _resolve_relative(
            receipt_root,
            receipt.prediction_relative_path,
            field="prediction_relative_path",
        )
        if _sha256_file(path) != receipt.prediction_sha256:
            raise ValueError("prediction receipt SHA-256 does not match bytes")
        payload = _read_canonical_prediction(path)
        header = {
            "schema_version": PREDICTION_SCHEMA_VERSION,
            "source_id": receipt.source_id,
            "fold_id": receipt.fold_id,
            "expected_authority_sha256": expected_authority_sha256,
            "checkpoint_receipt_sha256": checkpoint_digest,
            "loss_score_contract_sha256": loss_score_contract_sha256,
        }
        for field, value in header.items():
            if payload.get(field) != value:
                raise ValueError(f"prediction payload {field} disagrees with receipt")
        rows = _require_sequence(
            payload.get("prediction_records"),
            field="prediction_records",
        )
        receipt_digest = prediction_receipt_sha256(receipt)
        parsed = tuple(
            _record_from_prediction_payload(
                _require_mapping(row, field="prediction_record"),
                prediction_receipt_digest=receipt_digest,
            )
            for row in rows
        )
        expected_records = {
            (receipt.source_id, participant, horizon)
            for participant in checkpoint.held_participant_ids
            for horizon in EVENT_REFERENCE_HORIZONS
        }
        observed_records = {
            (row.source_id, row.participant_id, row.horizon_minutes) for row in parsed
        }
        if len(observed_records) != len(parsed) or observed_records != expected_records:
            raise ValueError(
                "prediction payload does not exactly cover held participants×horizons"
            )
        for row in parsed:
            stream = stream_by_key[(row.source_id, row.participant_id)]
            if (
                row.fold_id != expected[(row.source_id, row.participant_id)]
                or row.checkpoint_receipt_sha256 != checkpoint_digest
                or row.decision_count != stream["decision_count"]
                or row.patient_days != stream["patient_days"]
                or row.event_count != stream["event_counts"][str(row.horizon_minutes)]
                or row.stream_content_sha256 != stream["stream_content_sha256"]
                or row.window_shard_sha256 != stream["window_shard_sha256"]
            ):
                raise ValueError(
                    "prediction denominator or stream binding disagrees with authority"
                )
        output[key] = (receipt, receipt_digest)
        records.extend(parsed)
    if set(output) != expected_keys:
        raise ValueError(
            "prediction receipts must cover all 55 production source/folds"
        )
    return output, tuple(records)


def _derive_components(
    records: Sequence[ProductionEventOOFRecord],
) -> dict[str, dict[str, dict[str, float | int]]]:
    components: dict[str, dict[str, dict[str, float | int]]] = {}
    for source in PRODUCTION_SOURCE_ORDER:
        by_horizon: dict[str, dict[str, float | int]] = {}
        for horizon in EVENT_REFERENCE_HORIZONS:
            rows = tuple(
                row
                for row in records
                if row.source_id == source and row.horizon_minutes == horizon
            )
            positive = tuple(row for row in rows if row.event_count > 0)
            if not rows or not positive:
                raise ValueError(
                    "production event components require complete positive denominators"
                )
            by_horizon[str(horizon)] = {
                "event_miss": (
                    math.fsum(row.event_miss for row in positive) / len(positive)
                ),
                "false_alarms_per_patient_day": math.fsum(
                    row.false_alarms_per_patient_day for row in rows
                )
                / len(rows),
                "participant_count": len(rows),
                "event_positive_participant_count": len(positive),
                "event_count": sum(row.event_count for row in rows),
                "event_credit_sum": math.fsum(row.event_credit_sum for row in rows),
                "false_alarm_mass": math.fsum(row.false_alarm_mass for row in rows),
                "decision_count": sum(row.decision_count for row in rows),
                "patient_days": sum(row.patient_days for row in rows),
            }
        components[source] = by_horizon
    return components


def _ordered_records(
    records: Sequence[ProductionEventOOFRecord],
) -> list[ProductionEventOOFRecord]:
    source_to_index = {
        source: index for index, source in enumerate(PRODUCTION_SOURCE_ORDER)
    }
    return sorted(
        records,
        key=lambda row: (
            source_to_index[row.source_id],
            row.participant_id.encode("utf-8"),
            row.horizon_minutes,
        ),
    )


def build_production_event_reference_artifact(
    *,
    parent_base_reference_artifact: Mapping[str, Any],
    expected_authority: Mapping[str, Any],
    expected_authority_sha256: str,
    shard_root: str | Path,
    participant_shard_manifest: Mapping[str, Any],
    receipt_root: str | Path,
    checkpoint_receipts: Sequence[CheckpointReceipt],
    prediction_receipts: Sequence[PredictionReceipt],
    split_manifest_sha256: str,
    preprocessing_config_sha256: str,
    loss_score_contract_sha256: str,
) -> dict[str, Any]:
    """Build a production C2 OOF artifact only from complete verified receipts."""

    expected_digest = _sha256(
        expected_authority_sha256,
        field="expected_authority_sha256",
    )
    authority = validate_production_event_authority_against_shards(
        expected_authority,
        parent_base_reference_artifact=parent_base_reference_artifact,
        shard_root=shard_root,
        participant_shard_manifest=participant_shard_manifest,
    )
    if authority["artifact_sha256"] != expected_digest:
        raise ValueError("expected authority SHA-256 does not match supplied authority")
    parent, expected = _validated_parent(parent_base_reference_artifact)
    bindings = authority["bindings"]
    if (
        bindings["split_manifest_sha256"]
        != _sha256(split_manifest_sha256, field="split_manifest_sha256")
        or bindings["preprocessing_config_sha256"]
        != _sha256(preprocessing_config_sha256, field="preprocessing_config_sha256")
    ):
        raise ValueError(
            "production reference bindings disagree with expected authority"
        )
    contract_digest = _sha256(
        loss_score_contract_sha256,
        field="loss_score_contract_sha256",
    )
    root = Path(receipt_root)
    checkpoint_by_key = _validate_checkpoint_receipts(
        checkpoint_receipts,
        expected=expected,
        parent=parent,
        receipt_root=root,
    )
    prediction_by_key, records = _validate_prediction_receipts(
        prediction_receipts,
        authority=authority,
        expected=expected,
        checkpoint_by_key=checkpoint_by_key,
        expected_authority_sha256=expected_digest,
        loss_score_contract_sha256=contract_digest,
        receipt_root=root,
    )
    expected_record_keys = {
        (source, participant, horizon)
        for source, participant in expected
        for horizon in EVENT_REFERENCE_HORIZONS
    }
    observed_record_keys = {
        (row.source_id, row.participant_id, row.horizon_minutes) for row in records
    }
    if (
        len(observed_record_keys) != len(records)
        or observed_record_keys != expected_record_keys
    ):
        raise ValueError(
            "production OOF records do not exactly cover authority participants"
        )
    ordered_records = _ordered_records(records)
    payload: dict[str, Any] = {
        "schema_version": REFERENCE_SCHEMA_VERSION,
        "protocol_id": AUTHORITY_PROTOCOL_ID,
        "method": REFERENCE_METHOD,
        "split": REFERENCE_SPLIT,
        "lexical_source_order": list(PRODUCTION_SOURCE_ORDER),
        "fold_count": PRODUCTION_FOLD_COUNT,
        "horizon_order_minutes": list(EVENT_REFERENCE_HORIZONS),
        "bindings": {
            "expected_authority_sha256": expected_digest,
            "parent_base_reference_artifact_sha256": parent["artifact_sha256"],
            "split_manifest_sha256": bindings["split_manifest_sha256"],
            "window_manifest_sha256": bindings["window_manifest_sha256"],
            "preprocessing_config_sha256": bindings["preprocessing_config_sha256"],
            "loss_score_contract_sha256": contract_digest,
        },
        "checkpoint_receipts": [
            {**receipt.as_dict(), "receipt_sha256": digest}
            for _, (receipt, digest) in sorted(checkpoint_by_key.items())
        ],
        "prediction_receipts": [
            {**receipt.as_dict(), "receipt_sha256": digest}
            for _, (receipt, digest) in sorted(prediction_by_key.items())
        ],
        "participant_oof_records": [row.as_dict() for row in ordered_records],
        "reference_components": _derive_components(ordered_records),
    }
    return bind_artifact_sha256(payload)


def _exact_fields(
    row: Mapping[str, Any],
    *,
    fields: set[str],
    field: str,
) -> None:
    missing = sorted(fields.difference(row))
    extras = sorted(set(row).difference(fields))
    if missing or extras:
        raise ValueError(
            f"{field} fields differ from the frozen schema: "
            f"missing={missing}, extras={extras}"
        )


def _checkpoint_receipt_from_artifact_payload(
    row: Mapping[str, Any],
) -> tuple[CheckpointReceipt, str]:
    fields = set(CheckpointReceipt.__dataclass_fields__) | {"receipt_sha256"}
    _exact_fields(row, fields=fields, field="checkpoint receipt")
    receipt = _normalize_checkpoint_receipt(
        CheckpointReceipt(
            source_id=row["source_id"],
            fold_id=row["fold_id"],
            fit_participant_ids=tuple(
                _require_sequence(
                    row["fit_participant_ids"],
                    field="fit_participant_ids",
                )
            ),
            held_participant_ids=tuple(
                _require_sequence(
                    row["held_participant_ids"],
                    field="held_participant_ids",
                )
            ),
            parent_base_reference_artifact_sha256=row[
                "parent_base_reference_artifact_sha256"
            ],
            base_fold_sha256=row["base_fold_sha256"],
            config_relative_path=row["config_relative_path"],
            config_sha256=row["config_sha256"],
            checkpoint_relative_path=row["checkpoint_relative_path"],
            checkpoint_sha256=row["checkpoint_sha256"],
            checkpoint_epoch=row["checkpoint_epoch"],
            checkpoint_state_sha256=row["checkpoint_state_sha256"],
        )
    )
    digest = _sha256(row["receipt_sha256"], field="checkpoint receipt_sha256")
    if digest != checkpoint_receipt_sha256(receipt):
        raise ValueError("checkpoint receipt SHA-256 disagrees with receipt contents")
    return receipt, digest


def _prediction_receipt_from_artifact_payload(
    row: Mapping[str, Any],
) -> tuple[PredictionReceipt, str]:
    fields = set(PredictionReceipt.__dataclass_fields__) | {"receipt_sha256"}
    _exact_fields(row, fields=fields, field="prediction receipt")
    receipt = _normalize_prediction_receipt(
        PredictionReceipt(
            source_id=row["source_id"],
            fold_id=row["fold_id"],
            expected_authority_sha256=row["expected_authority_sha256"],
            checkpoint_receipt_sha256=row["checkpoint_receipt_sha256"],
            loss_score_contract_sha256=row["loss_score_contract_sha256"],
            prediction_relative_path=row["prediction_relative_path"],
            prediction_sha256=row["prediction_sha256"],
        )
    )
    digest = _sha256(row["receipt_sha256"], field="prediction receipt_sha256")
    if digest != prediction_receipt_sha256(receipt):
        raise ValueError("prediction receipt SHA-256 disagrees with receipt contents")
    return receipt, digest


def _receipt_keys() -> set[tuple[str, int]]:
    return {
        (source, fold_id)
        for source in PRODUCTION_SOURCE_ORDER
        for fold_id in range(PRODUCTION_FOLD_COUNT)
    }


def _validate_embedded_receipts(
    payload: Mapping[str, Any],
    *,
    bindings: Mapping[str, Any],
    records: Sequence[ProductionEventOOFRecord],
) -> None:
    expected_keys = _receipt_keys()
    checkpoint_by_key: dict[tuple[str, int], tuple[CheckpointReceipt, str]] = {}
    checkpoint_rows = _require_sequence(
        payload.get("checkpoint_receipts"),
        field="checkpoint_receipts",
    )
    for value in checkpoint_rows:
        receipt, digest = _checkpoint_receipt_from_artifact_payload(
            _require_mapping(value, field="checkpoint receipt")
        )
        key = (receipt.source_id, receipt.fold_id)
        if key not in expected_keys or key in checkpoint_by_key:
            raise ValueError(
                "checkpoint receipts must be unique production source/folds"
            )
        if (
            receipt.parent_base_reference_artifact_sha256
            != bindings["parent_base_reference_artifact_sha256"]
        ):
            raise ValueError("checkpoint receipt parent artifact binding disagrees")
        checkpoint_by_key[key] = (receipt, digest)
    if set(checkpoint_by_key) != expected_keys:
        raise ValueError(
            "checkpoint receipts must cover all 55 production source/folds"
        )

    held_by_key: dict[tuple[str, int], tuple[str, ...]] = {}
    for source in PRODUCTION_SOURCE_ORDER:
        source_held_sets: list[set[str]] = []
        for fold_id in range(PRODUCTION_FOLD_COUNT):
            receipt, _ = checkpoint_by_key[(source, fold_id)]
            held = set(receipt.held_participant_ids)
            if any(held.intersection(existing) for existing in source_held_sets):
                raise ValueError(
                    "checkpoint held participant sets overlap across folds"
                )
            source_held_sets.append(held)
            held_by_key[(source, fold_id)] = receipt.held_participant_ids
        source_participants = set().union(*source_held_sets)
        for fold_id in range(PRODUCTION_FOLD_COUNT):
            receipt, _ = checkpoint_by_key[(source, fold_id)]
            held = set(receipt.held_participant_ids)
            expected_fit = tuple(sorted(source_participants - held, key=str.encode))
            if receipt.fit_participant_ids != expected_fit:
                raise ValueError(
                    "checkpoint receipt fit/held sets are not disjoint complements"
                )
            expected_hash = base_fold_scope_sha256(
                source_id=source,
                fold_id=fold_id,
                fit_participant_ids=expected_fit,
                held_participant_ids=receipt.held_participant_ids,
            )
            if receipt.base_fold_sha256 != expected_hash:
                raise ValueError(
                    "checkpoint receipt base-fold hash disagrees with CF5 ledger"
                )

    prediction_by_key: dict[tuple[str, int], tuple[PredictionReceipt, str]] = {}
    prediction_rows = _require_sequence(
        payload.get("prediction_receipts"),
        field="prediction_receipts",
    )
    for value in prediction_rows:
        receipt, digest = _prediction_receipt_from_artifact_payload(
            _require_mapping(value, field="prediction receipt")
        )
        key = (receipt.source_id, receipt.fold_id)
        if key not in expected_keys or key in prediction_by_key:
            raise ValueError(
                "prediction receipts must be unique production source/folds"
            )
        _, checkpoint_digest = checkpoint_by_key[key]
        if receipt.expected_authority_sha256 != bindings["expected_authority_sha256"]:
            raise ValueError("prediction receipt authority binding disagrees")
        if receipt.checkpoint_receipt_sha256 != checkpoint_digest:
            raise ValueError("prediction receipt checkpoint binding disagrees")
        if receipt.loss_score_contract_sha256 != bindings["loss_score_contract_sha256"]:
            raise ValueError("prediction receipt loss/score contract binding disagrees")
        prediction_by_key[key] = (receipt, digest)
    if set(prediction_by_key) != expected_keys:
        raise ValueError(
            "prediction receipts must cover all 55 production source/folds"
        )

    records_by_key: dict[tuple[str, int], list[ProductionEventOOFRecord]] = {
        key: [] for key in expected_keys
    }
    by_participant: dict[tuple[str, str], list[ProductionEventOOFRecord]] = {}
    for record in records:
        key = (record.source_id, record.fold_id)
        if key not in records_by_key:
            raise ValueError(
                "production OOF record source/fold is absent from receipts"
            )
        _, checkpoint_digest = checkpoint_by_key[key]
        _, prediction_digest = prediction_by_key[key]
        if record.checkpoint_receipt_sha256 != checkpoint_digest:
            raise ValueError("production OOF record checkpoint receipt disagrees")
        if record.prediction_receipt_sha256 != prediction_digest:
            raise ValueError("production OOF record prediction receipt disagrees")
        records_by_key[key].append(record)
        by_participant.setdefault(
            (record.source_id, record.participant_id), []
        ).append(record)

    for key, rows in records_by_key.items():
        source, _ = key
        expected_records = {
            (source, participant, horizon)
            for participant in held_by_key[key]
            for horizon in EVENT_REFERENCE_HORIZONS
        }
        observed_records = {
            (row.source_id, row.participant_id, row.horizon_minutes)
            for row in rows
        }
        if len(observed_records) != len(rows) or observed_records != expected_records:
            raise ValueError(
                "production OOF records do not exactly cover held "
                "participants×horizons"
            )

    for participant_rows in by_participant.values():
        if {row.horizon_minutes for row in participant_rows} != set(
            EVENT_REFERENCE_HORIZONS
        ):
            raise ValueError("production OOF participant is missing a frozen horizon")
        shared = {
            (
                row.fold_id,
                row.decision_count,
                row.patient_days,
                row.stream_content_sha256,
                row.window_shard_sha256,
                row.checkpoint_receipt_sha256,
                row.prediction_receipt_sha256,
            )
            for row in participant_rows
        }
        if len(shared) != 1:
            raise ValueError(
                "production OOF participant stream bindings disagree across horizons"
            )


def validate_production_event_reference_artifact(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the immutable reference payload and all numerator derivations."""

    if not isinstance(payload, Mapping):
        raise TypeError("production event reference artifact must be a mapping")
    _exact_fields(
        payload,
        fields={
            "schema_version",
            "protocol_id",
            "method",
            "split",
            "lexical_source_order",
            "fold_count",
            "horizon_order_minutes",
            "bindings",
            "checkpoint_receipts",
            "prediction_receipts",
            "participant_oof_records",
            "reference_components",
            "artifact_sha256",
        },
        field="production event reference artifact",
    )
    verify_artifact_payload(payload)
    if (
        payload.get("schema_version") != REFERENCE_SCHEMA_VERSION
        or payload.get("protocol_id") != AUTHORITY_PROTOCOL_ID
        or payload.get("method") != REFERENCE_METHOD
        or payload.get("split") != REFERENCE_SPLIT
        or tuple(payload.get("lexical_source_order", ())) != PRODUCTION_SOURCE_ORDER
        or payload.get("fold_count") != PRODUCTION_FOLD_COUNT
        or tuple(payload.get("horizon_order_minutes", ())) != EVENT_REFERENCE_HORIZONS
    ):
        raise ValueError("production event reference protocol drift")
    bindings = _require_mapping(payload.get("bindings"), field="bindings")
    _exact_fields(
        bindings,
        fields={
            "expected_authority_sha256",
            "parent_base_reference_artifact_sha256",
            "split_manifest_sha256",
            "window_manifest_sha256",
            "preprocessing_config_sha256",
            "loss_score_contract_sha256",
        },
        field="bindings",
    )
    for field in (
        "expected_authority_sha256",
        "parent_base_reference_artifact_sha256",
        "split_manifest_sha256",
        "window_manifest_sha256",
        "preprocessing_config_sha256",
        "loss_score_contract_sha256",
    ):
        _sha256(bindings.get(field), field=f"bindings.{field}")
    records = tuple(
        _record_from_artifact_payload(
            _require_mapping(row, field="participant_oof_record")
        )
        for row in _require_sequence(
            payload.get("participant_oof_records"),
            field="participant_oof_records",
        )
    )
    keys = {(row.source_id, row.participant_id, row.horizon_minutes) for row in records}
    if len(keys) != len(records):
        raise ValueError("production event reference repeats an OOF record")
    components = _derive_components(records)
    if payload.get("reference_components") != components:
        raise ValueError("production event reference contains derived-field drift")
    _validate_embedded_receipts(
        payload,
        bindings=bindings,
        records=records,
    )
    rebuilt = dict(payload)
    rebuilt.pop("artifact_sha256", None)
    return bind_artifact_sha256(rebuilt)


def read_production_event_reference_artifact(path: str | Path) -> dict[str, Any]:
    return validate_production_event_reference_artifact(read_canonical_json(path))


def write_production_event_reference_artifact(
    payload: Mapping[str, Any],
    path: str | Path,
) -> str:
    return write_immutable_json(
        validate_production_event_reference_artifact(payload),
        path,
    )
