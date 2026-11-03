"""Immutable train-only C2 severe-event reference artifacts.

This module is deliberately limited to the event-reference ledger.  It does
not fit models, read cohort rows, or perform threshold selection.  A caller
supplies the complete held-fold OOF participant measurements and fold receipts;
the builder validates their train-only source-scoped structure, derives the
participant-macro reference components, and freezes the canonical payload.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from .mero_reference import (
    bind_artifact_sha256,
    canonical_json_bytes,
    read_canonical_json,
    verify_artifact_payload,
    write_immutable_json,
)

EVENT_REFERENCE_SCHEMA_VERSION = "1.0"
EVENT_REFERENCE_PROTOCOL_ID = "glycoshift-sota-method-v1.2"
EVENT_REFERENCE_METHOD = "participant-cross-fitted-c2-event-regret-v1.2"
EVENT_REFERENCE_SPLIT = "train"
EVENT_REFERENCE_HORIZONS = (30, 60)

_FORBIDDEN_IDENTIFIER_CHARACTERS = {"\x00", "\r", "\n", "|"}


def _validate_identifier(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    if any(character in value for character in _FORBIDDEN_IDENTIFIER_CHARACTERS):
        raise ValueError(f"{field} contains a forbidden protocol delimiter")
    return value


def _validate_integer(value: Any, *, field: str, minimum: int = 0) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError(f"{field} must be an integer")
    resolved = int(value)
    if resolved < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return resolved


def _validate_finite_float(value: Any, *, field: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{field} must be numeric")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{field} must be finite")
    return resolved


def _validate_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _validate_source_order(source_order: Sequence[str]) -> tuple[str, ...]:
    sources = tuple(
        _validate_identifier(source, field="source_id") for source in source_order
    )
    if not sources or len(set(sources)) != len(sources):
        raise ValueError("source_order must be non-empty and unique")
    if sources != tuple(sorted(sources)):
        raise ValueError("source_order must use lexical order")
    return sources


def _require_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    return value


def _require_sequence(value: Any, *, field: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{field} must be a sequence")
    return value


@dataclass(frozen=True)
class ParticipantEventOOFRecord:
    """One complete held-fold event measurement for a source-scoped participant."""

    source_id: str
    participant_id: str
    fold_id: int
    horizon_minutes: int
    decision_count: int
    patient_days: int
    event_count: int
    event_miss: float
    false_alarms_per_patient_day: float

    @property
    def fapd(self) -> float:
        """Short alias for the frozen false-alarms-per-patient-day component."""

        return self.false_alarms_per_patient_day

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "participant_id": self.participant_id,
            "fold_id": self.fold_id,
            "horizon_minutes": self.horizon_minutes,
            "decision_count": self.decision_count,
            "patient_days": self.patient_days,
            "event_count": self.event_count,
            "event_miss": self.event_miss,
            "false_alarms_per_patient_day": self.false_alarms_per_patient_day,
        }


# The shorter name is useful at the call site and keeps the public data model
# readable without weakening the event-specific serialized field names.
ParticipantOOFRecord = ParticipantEventOOFRecord


@dataclass(frozen=True)
class EventReferenceFoldReceipt:
    """A source/fold receipt binding temporary checkpoint and OOF predictions."""

    source_id: str
    fold_id: int
    checkpoint_receipt_sha256: str
    held_participant_set_sha256: str
    held_participant_count: int
    prediction_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "fold_id": self.fold_id,
            "checkpoint_receipt_sha256": self.checkpoint_receipt_sha256,
            "held_participant_set_sha256": self.held_participant_set_sha256,
            "held_participant_count": self.held_participant_count,
            "prediction_sha256": self.prediction_sha256,
        }


def source_scoped_participant_set_sha256(
    *,
    source_id: str,
    participant_ids: Iterable[str],
) -> str:
    """Hash one source-scoped held-participant set in lexical identifier order."""

    source = _validate_identifier(source_id, field="source_id")
    participants = tuple(
        _validate_identifier(participant_id, field="participant_id")
        for participant_id in participant_ids
    )
    if not participants:
        raise ValueError("held participant set must be non-empty")
    if len(set(participants)) != len(participants):
        raise ValueError("held participant set must not repeat participants")
    payload = {
        "source_id": source,
        "participant_ids": sorted(participants, key=str.encode),
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _normalize_participant_record(
    record: ParticipantEventOOFRecord,
    *,
    fold_count: int,
) -> ParticipantEventOOFRecord:
    if not isinstance(record, ParticipantEventOOFRecord):
        raise TypeError(
            "participant_records must contain ParticipantEventOOFRecord values"
        )
    source = _validate_identifier(record.source_id, field="source_id")
    participant = _validate_identifier(record.participant_id, field="participant_id")
    fold_id = _validate_integer(record.fold_id, field="fold_id")
    if fold_id >= fold_count:
        raise ValueError("fold_id must be less than fold_count")
    horizon = _validate_integer(record.horizon_minutes, field="horizon_minutes")
    if horizon not in EVENT_REFERENCE_HORIZONS:
        raise ValueError("horizon_minutes must be one of (30, 60)")
    decision_count = _validate_integer(
        record.decision_count,
        field="decision_count",
        minimum=1,
    )
    patient_days = _validate_integer(
        record.patient_days,
        field="patient_days",
        minimum=1,
    )
    event_count = _validate_integer(
        record.event_count,
        field="event_count",
    )
    event_miss = _validate_finite_float(record.event_miss, field="event_miss")
    if not 0.0 <= event_miss <= 1.0:
        raise ValueError("event_miss must lie in [0, 1]")
    if event_count == 0 and event_miss != 0.0:
        raise ValueError("zero-event records must use event_miss=0")
    fapd = _validate_finite_float(
        record.false_alarms_per_patient_day,
        field="false_alarms_per_patient_day",
    )
    if fapd < 0.0:
        raise ValueError("false_alarms_per_patient_day must be non-negative")
    return ParticipantEventOOFRecord(
        source_id=source,
        participant_id=participant,
        fold_id=fold_id,
        horizon_minutes=horizon,
        decision_count=decision_count,
        patient_days=patient_days,
        event_count=event_count,
        event_miss=event_miss,
        false_alarms_per_patient_day=fapd,
    )


def _normalize_fold_receipt(
    receipt: EventReferenceFoldReceipt,
    *,
    fold_count: int,
) -> EventReferenceFoldReceipt:
    if not isinstance(receipt, EventReferenceFoldReceipt):
        raise TypeError("fold_receipts must contain EventReferenceFoldReceipt values")
    source = _validate_identifier(receipt.source_id, field="source_id")
    fold_id = _validate_integer(receipt.fold_id, field="fold_id")
    if fold_id >= fold_count:
        raise ValueError("fold_id must be less than fold_count")
    return EventReferenceFoldReceipt(
        source_id=source,
        fold_id=fold_id,
        checkpoint_receipt_sha256=_validate_sha256(
            receipt.checkpoint_receipt_sha256,
            field="checkpoint_receipt_sha256",
        ),
        held_participant_set_sha256=_validate_sha256(
            receipt.held_participant_set_sha256,
            field="held_participant_set_sha256",
        ),
        held_participant_count=_validate_integer(
            receipt.held_participant_count,
            field="held_participant_count",
            minimum=1,
        ),
        prediction_sha256=_validate_sha256(
            receipt.prediction_sha256,
            field="prediction_sha256",
        ),
    )


def _ordered_records(
    records: Iterable[ParticipantEventOOFRecord],
    *,
    source_order: tuple[str, ...],
) -> tuple[ParticipantEventOOFRecord, ...]:
    source_to_index = {source: index for index, source in enumerate(source_order)}
    return tuple(
        sorted(
            records,
            key=lambda row: (
                source_to_index[row.source_id],
                row.participant_id.encode("utf-8"),
                row.horizon_minutes,
            ),
        )
    )


def _validate_record_ledger(
    records: tuple[ParticipantEventOOFRecord, ...],
    *,
    source_order: tuple[str, ...],
) -> None:
    if {row.source_id for row in records} != set(source_order):
        raise ValueError("participant record sources must exactly match source_order")

    by_participant: dict[
        tuple[str, str], list[ParticipantEventOOFRecord]
    ] = defaultdict(list)
    seen_keys: set[tuple[str, str, int]] = set()
    for row in records:
        key = (row.source_id, row.participant_id, row.horizon_minutes)
        if key in seen_keys:
            raise ValueError(
                "participant OOF records must be unique per source/participant/horizon"
            )
        seen_keys.add(key)
        by_participant[(row.source_id, row.participant_id)].append(row)

    for participant_key, participant_rows in by_participant.items():
        horizons = {row.horizon_minutes for row in participant_rows}
        if horizons != set(EVENT_REFERENCE_HORIZONS):
            raise ValueError(
                "every source-scoped participant requires exactly one record per "
                "horizon"
            )
        ledgers = {
            (
                row.fold_id,
                row.decision_count,
                row.patient_days,
            )
            for row in participant_rows
        }
        if len(ledgers) != 1:
            source, participant = participant_key
            raise ValueError(
                "participant fold and denominators disagree across horizons: "
                f"{source}/{participant}"
            )

    for source in source_order:
        for horizon in EVENT_REFERENCE_HORIZONS:
            source_horizon_rows = [
                row
                for row in records
                if row.source_id == source and row.horizon_minutes == horizon
            ]
            if not source_horizon_rows:
                raise ValueError("every source requires records for every horizon")
            if not any(row.event_count > 0 for row in source_horizon_rows):
                raise ValueError(
                    "every source/horizon requires an event-positive participant"
                )


def _validate_fold_receipts(
    receipts: tuple[EventReferenceFoldReceipt, ...],
    *,
    source_order: tuple[str, ...],
    fold_count: int,
    records: tuple[ParticipantEventOOFRecord, ...],
) -> None:
    expected_keys = {
        (source, fold_id)
        for source in source_order
        for fold_id in range(fold_count)
    }
    receipt_keys = {(receipt.source_id, receipt.fold_id) for receipt in receipts}
    if len(receipt_keys) != len(receipts):
        raise ValueError("fold receipts must be unique per source/fold")
    if receipt_keys != expected_keys:
        raise ValueError("fold receipts must cover every source/fold pair")

    receipts_by_key = {(row.source_id, row.fold_id): row for row in receipts}
    for source, fold_id in sorted(
        expected_keys,
        key=lambda item: (source_order.index(item[0]), item[1]),
    ):
        held_participants = sorted(
            {
                row.participant_id
                for row in records
                if row.source_id == source and row.fold_id == fold_id
            },
            key=str.encode,
        )
        if not held_participants:
            raise ValueError(
                "participant records leave a source/fold without held data"
            )
        receipt = receipts_by_key[(source, fold_id)]
        if receipt.held_participant_count != len(held_participants):
            raise ValueError(
                "held participant count disagrees with participant records"
            )
        expected_hash = source_scoped_participant_set_sha256(
            source_id=source,
            participant_ids=held_participants,
        )
        if receipt.held_participant_set_sha256 != expected_hash:
            raise ValueError(
                "held participant set hash disagrees with participant records"
            )


def _derive_reference_components(
    records: tuple[ParticipantEventOOFRecord, ...],
    *,
    source_order: tuple[str, ...],
) -> dict[str, dict[str, dict[str, float | int]]]:
    components: dict[str, dict[str, dict[str, float | int]]] = {}
    for source in source_order:
        by_horizon: dict[str, dict[str, float | int]] = {}
        for horizon in EVENT_REFERENCE_HORIZONS:
            rows = tuple(
                row
                for row in records
                if row.source_id == source and row.horizon_minutes == horizon
            )
            positive_rows = tuple(row for row in rows if row.event_count > 0)
            # _validate_record_ledger guarantees the non-empty denominator.
            by_horizon[str(horizon)] = {
                "event_miss": math.fsum(row.event_miss for row in positive_rows)
                / len(positive_rows),
                "false_alarms_per_patient_day": math.fsum(
                    row.false_alarms_per_patient_day for row in rows
                )
                / len(rows),
                "participant_count": len(rows),
                "event_positive_participant_count": len(positive_rows),
                "event_count": sum(row.event_count for row in rows),
                "decision_count": sum(row.decision_count for row in rows),
                "patient_days": sum(row.patient_days for row in rows),
            }
        components[source] = by_horizon
    return components


def build_event_reference_artifact(
    *,
    source_order: Sequence[str],
    participant_records: Sequence[ParticipantEventOOFRecord],
    fold_receipts: Sequence[EventReferenceFoldReceipt],
    fold_count: int,
    parent_base_reference_artifact_sha256: str,
    split_manifest_sha256: str,
    preprocessing_config_sha256: str,
    loss_score_contract_sha256: str,
    split: str = EVENT_REFERENCE_SPLIT,
) -> dict[str, Any]:
    """Build a self-hashed C2 OOF event-reference artifact.

    ``split`` is an explicit guard solely so callers cannot accidentally use
    the artifact builder for selection, calibration, or external OOF rows.
    """

    sources = _validate_source_order(source_order)
    resolved_fold_count = _validate_integer(
        fold_count,
        field="fold_count",
        minimum=2,
    )
    if split != EVENT_REFERENCE_SPLIT:
        raise ValueError("event reference split must be train")
    bindings = {
        "parent_base_reference_artifact_sha256": _validate_sha256(
            parent_base_reference_artifact_sha256,
            field="parent_base_reference_artifact_sha256",
        ),
        "split_manifest_sha256": _validate_sha256(
            split_manifest_sha256,
            field="split_manifest_sha256",
        ),
        "preprocessing_config_sha256": _validate_sha256(
            preprocessing_config_sha256,
            field="preprocessing_config_sha256",
        ),
        "loss_score_contract_sha256": _validate_sha256(
            loss_score_contract_sha256,
            field="loss_score_contract_sha256",
        ),
    }
    normalized_records = tuple(
        _normalize_participant_record(record, fold_count=resolved_fold_count)
        for record in participant_records
    )
    normalized_receipts = tuple(
        _normalize_fold_receipt(receipt, fold_count=resolved_fold_count)
        for receipt in fold_receipts
    )
    if not normalized_records:
        raise ValueError("participant_records must be non-empty")
    _validate_record_ledger(normalized_records, source_order=sources)
    _validate_fold_receipts(
        normalized_receipts,
        source_order=sources,
        fold_count=resolved_fold_count,
        records=normalized_records,
    )
    ordered_records = _ordered_records(normalized_records, source_order=sources)
    source_to_index = {source: index for index, source in enumerate(sources)}
    ordered_receipts = tuple(
        sorted(
            normalized_receipts,
            key=lambda row: (source_to_index[row.source_id], row.fold_id),
        )
    )
    payload: dict[str, Any] = {
        "schema_version": EVENT_REFERENCE_SCHEMA_VERSION,
        "protocol_id": EVENT_REFERENCE_PROTOCOL_ID,
        "method": EVENT_REFERENCE_METHOD,
        "split": EVENT_REFERENCE_SPLIT,
        "lexical_source_order": list(sources),
        "source_to_index": source_to_index,
        "horizon_order_minutes": list(EVENT_REFERENCE_HORIZONS),
        "fold_count": resolved_fold_count,
        "bindings": bindings,
        "fold_receipts": [receipt.as_dict() for receipt in ordered_receipts],
        "participant_oof_records": [record.as_dict() for record in ordered_records],
        "reference_components": _derive_reference_components(
            ordered_records,
            source_order=sources,
        ),
    }
    return bind_artifact_sha256(payload)


def _participant_record_from_payload(
    row: Mapping[str, Any],
) -> ParticipantEventOOFRecord:
    try:
        return ParticipantEventOOFRecord(
            source_id=row["source_id"],
            participant_id=row["participant_id"],
            fold_id=row["fold_id"],
            horizon_minutes=row["horizon_minutes"],
            decision_count=row["decision_count"],
            patient_days=row["patient_days"],
            event_count=row["event_count"],
            event_miss=row["event_miss"],
            false_alarms_per_patient_day=row["false_alarms_per_patient_day"],
        )
    except KeyError as error:
        raise ValueError(
            f"participant OOF record is missing {error.args[0]}"
        ) from error


def _fold_receipt_from_payload(row: Mapping[str, Any]) -> EventReferenceFoldReceipt:
    try:
        return EventReferenceFoldReceipt(
            source_id=row["source_id"],
            fold_id=row["fold_id"],
            checkpoint_receipt_sha256=row["checkpoint_receipt_sha256"],
            held_participant_set_sha256=row["held_participant_set_sha256"],
            held_participant_count=row["held_participant_count"],
            prediction_sha256=row["prediction_sha256"],
        )
    except KeyError as error:
        raise ValueError(f"fold receipt is missing {error.args[0]}") from error


def validate_event_reference_artifact(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate self-hash, structural scope, and all rebuilt derivations."""

    if not isinstance(payload, Mapping):
        raise TypeError("event reference artifact must be a mapping")
    verify_artifact_payload(payload)
    if payload.get("schema_version") != EVENT_REFERENCE_SCHEMA_VERSION:
        raise ValueError("unsupported event-reference schema_version")
    if payload.get("protocol_id") != EVENT_REFERENCE_PROTOCOL_ID:
        raise ValueError("event-reference protocol_id is incompatible")
    if payload.get("method") != EVENT_REFERENCE_METHOD:
        raise ValueError("event-reference method is incompatible")
    if payload.get("split") != EVENT_REFERENCE_SPLIT:
        raise ValueError("event reference split must be train")
    if tuple(payload.get("horizon_order_minutes", ())) != EVENT_REFERENCE_HORIZONS:
        raise ValueError("event-reference horizons are incompatible")

    try:
        sources = _validate_source_order(
            _require_sequence(
                payload["lexical_source_order"],
                field="lexical_source_order",
            )
        )
        source_to_index = _require_mapping(
            payload["source_to_index"],
            field="source_to_index",
        )
        bindings = _require_mapping(payload["bindings"], field="bindings")
        records = tuple(
            _participant_record_from_payload(
                _require_mapping(row, field="participant_oof_record")
            )
            for row in _require_sequence(
                payload["participant_oof_records"],
                field="participant_oof_records",
            )
        )
        receipts = tuple(
            _fold_receipt_from_payload(_require_mapping(row, field="fold_receipt"))
            for row in _require_sequence(
                payload["fold_receipts"],
                field="fold_receipts",
            )
        )
        components = _require_mapping(
            payload["reference_components"],
            field="reference_components",
        )
    except KeyError as error:
        raise ValueError(
            f"event reference artifact is missing {error.args[0]}"
        ) from error

    if source_to_index != {source: index for index, source in enumerate(sources)}:
        raise ValueError("event-reference source_to_index disagrees with source order")
    rebuilt = build_event_reference_artifact(
        source_order=sources,
        participant_records=records,
        fold_receipts=receipts,
        fold_count=_validate_integer(
            payload["fold_count"],
            field="fold_count",
            minimum=2,
        ),
        parent_base_reference_artifact_sha256=bindings.get(
            "parent_base_reference_artifact_sha256"
        ),
        split_manifest_sha256=bindings.get("split_manifest_sha256"),
        preprocessing_config_sha256=bindings.get("preprocessing_config_sha256"),
        loss_score_contract_sha256=bindings.get("loss_score_contract_sha256"),
        split=payload["split"],
    )
    if components != rebuilt["reference_components"]:
        raise ValueError("event-reference artifact contains derived-field drift")
    if rebuilt != dict(payload):
        raise ValueError("event-reference artifact contains derived-field drift")
    return rebuilt


def read_event_reference_artifact(path: str | Path) -> dict[str, Any]:
    """Read canonical bytes and rebuild the C2 reference artifact fail-closed."""

    return validate_event_reference_artifact(read_canonical_json(path))


def write_event_reference_artifact(
    payload: Mapping[str, Any],
    path: str | Path,
) -> str:
    """Write the canonical payload once, permitting only an identical retry."""

    return write_immutable_json(validate_event_reference_artifact(payload), path)
