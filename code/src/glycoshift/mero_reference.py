from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any

PROTOCOL_ID = "glycoshift-mero-cf5-pilot-v1.1"
REFERENCE_METHOD = "participant-cross-fitted-mero-style-v1.1"
REFERENCE_SPLIT = "train"
TASK_ORDER = ("hypo_30", "hypo_60", "hyper_30", "hyper_60")
_FORBIDDEN_IDENTIFIER_CHARACTERS = {"\x00", "\r", "\n", "|"}


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


def _validate_sha256(value: str, *, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _validate_identifier(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    if any(character in value for character in _FORBIDDEN_IDENTIFIER_CHARACTERS):
        raise ValueError(f"{field} contains a forbidden protocol delimiter")
    return value


def _validate_source_order(source_order: Sequence[str]) -> tuple[str, ...]:
    resolved = tuple(
        _validate_identifier(source, field="source_id") for source in source_order
    )
    if not resolved or len(set(resolved)) != len(resolved):
        raise ValueError("source_order must be non-empty and unique")
    if resolved != tuple(sorted(resolved)):
        raise ValueError("source_order must use lexical order")
    return resolved


def canonical_json_bytes(payload: Any) -> bytes:
    """Return the exact canonical JSON encoding frozen by protocol v1.1."""

    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def artifact_payload_sha256(payload: Mapping[str, Any]) -> str:
    """Hash a complete artifact after omitting its self-hash field."""

    hash_input = dict(payload)
    hash_input.pop("artifact_sha256", None)
    return hashlib.sha256(canonical_json_bytes(hash_input)).hexdigest()


def bind_artifact_sha256(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy with a verified or newly inserted artifact self-hash."""

    bound = dict(payload)
    expected = artifact_payload_sha256(bound)
    supplied = bound.get("artifact_sha256")
    if supplied is not None and supplied != expected:
        raise ValueError("artifact_sha256 does not match the canonical payload")
    bound["artifact_sha256"] = expected
    return bound


def verify_artifact_payload(payload: Mapping[str, Any]) -> str:
    supplied = _validate_sha256(
        payload.get("artifact_sha256"),
        field="artifact_sha256",
    )
    expected = artifact_payload_sha256(payload)
    if supplied != expected:
        raise ValueError("artifact_sha256 does not match the canonical payload")
    return supplied


def write_immutable_json(payload: Mapping[str, Any], path: str | Path) -> str:
    """Write canonical bytes once, allowing only an identical idempotent write."""

    bound = bind_artifact_sha256(payload)
    serialized = canonical_json_bytes(bound)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if output.read_bytes() != serialized:
            raise FileExistsError(
                f"refusing to overwrite non-identical artifact: {output}"
            )
        return str(bound["artifact_sha256"])
    with output.open("xb") as handle:
        handle.write(serialized)
    return str(bound["artifact_sha256"])


def read_canonical_json(path: str | Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("artifact root must be a mapping")
    verify_artifact_payload(payload)
    if raw != canonical_json_bytes(payload):
        raise ValueError("artifact bytes are not canonical protocol-v1.1 JSON")
    return payload


def derive_reference_model_seed(
    *,
    reference_seed: int,
    source_id: str,
    fold_id: int,
) -> int:
    reference_seed = _validate_integer(reference_seed, field="reference_seed")
    source = _validate_identifier(source_id, field="source_id")
    fold_id = _validate_integer(fold_id, field="fold_id")
    message = f"{reference_seed}|{source}|{fold_id}|model".encode()
    digest = hashlib.sha256(message).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False) >> 1


@dataclass(frozen=True)
class ParticipantFoldAssignment:
    source_id: str
    participant_id: str
    fold_id: int
    assignment_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "participant_id": self.participant_id,
            "fold_id": self.fold_id,
            "assignment_sha256": self.assignment_sha256,
        }


def assign_source_scoped_folds(
    participants_by_source: Mapping[str, Iterable[str]],
    *,
    source_order: Sequence[str],
    fold_count: int = 5,
    reference_seed: int = 20260805,
) -> tuple[ParticipantFoldAssignment, ...]:
    """Assign deterministic participant folds independently within each source."""

    sources = _validate_source_order(source_order)
    fold_count = _validate_integer(fold_count, field="fold_count", minimum=2)
    reference_seed = _validate_integer(reference_seed, field="reference_seed")
    if set(participants_by_source) != set(sources):
        raise ValueError("participants_by_source must exactly match source_order")

    assignments: list[ParticipantFoldAssignment] = []
    for source in sources:
        participants = tuple(
            _validate_identifier(participant, field="participant_id")
            for participant in participants_by_source[source]
        )
        if len(participants) < fold_count:
            raise ValueError(
                f"source {source!r} has fewer participants than fold_count"
            )
        if len(set(participants)) != len(participants):
            raise ValueError(f"source {source!r} contains duplicate participants")

        hashed: list[tuple[str, bytes, str]] = []
        for participant in participants:
            message = f"{reference_seed}|{source}|{participant}".encode()
            digest = hashlib.sha256(message).hexdigest()
            hashed.append((digest, participant.encode("utf-8"), participant))
        hashed.sort(key=lambda item: (item[0], item[1]))
        for ordinal, (digest, _, participant) in enumerate(hashed):
            assignments.append(
                ParticipantFoldAssignment(
                    source_id=source,
                    participant_id=participant,
                    fold_id=ordinal % fold_count,
                    assignment_sha256=digest,
                )
            )
    return tuple(assignments)


def assert_reference_scope(
    rows: Iterable[Mapping[str, Any]],
    *,
    source_order: Sequence[str],
    forbidden_sources: Iterable[str] = (),
) -> None:
    """Fail closed on any non-train or out-of-protocol synthetic row."""

    sources = set(_validate_source_order(source_order))
    forbidden = {
        _validate_identifier(source, field="forbidden_source")
        for source in forbidden_sources
    }
    for row_number, row in enumerate(rows):
        if row.get("partition") != REFERENCE_SPLIT:
            raise ValueError(
                f"reference row {row_number} is not in the train partition"
            )
        source = _validate_identifier(row.get("source_id"), field="source_id")
        _validate_identifier(row.get("participant_id"), field="participant_id")
        if source in forbidden or source not in sources:
            raise ValueError(
                f"reference row {row_number} uses a forbidden source: {source}"
            )


def select_reference_checkpoint(
    epoch_objectives: Iterable[tuple[int, float]],
) -> tuple[int, float]:
    """Select the lowest deterministic objective, breaking exact ties early."""

    values = tuple(
        (
            _validate_integer(epoch, field="checkpoint_epoch", minimum=1),
            _validate_finite_float(value, field="checkpoint_objective"),
        )
        for epoch, value in epoch_objectives
    )
    if not values:
        raise ValueError("at least one epoch objective is required")
    epochs = [epoch for epoch, _ in values]
    if len(set(epochs)) != len(epochs):
        raise ValueError("checkpoint epochs must be unique positive integers")
    return min(values, key=lambda item: (item[1], item[0]))


@dataclass(frozen=True)
class ReferenceFoldRecord:
    source_id: str
    fold_id: int
    fit_participant_count: int
    held_participant_count: int
    fit_raw_window_count: int
    fit_sampled_window_count: int
    held_raw_window_count: int
    held_sampled_window_count: int
    derived_model_seed: int
    checkpoint_epoch: int
    checkpoint_objective: float
    epoch_objectives: tuple[float, ...]
    fold_oof_loss: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "fold_id": self.fold_id,
            "fit_participant_count": self.fit_participant_count,
            "held_participant_count": self.held_participant_count,
            "fit_raw_window_count": self.fit_raw_window_count,
            "fit_sampled_window_count": self.fit_sampled_window_count,
            "held_raw_window_count": self.held_raw_window_count,
            "held_sampled_window_count": self.held_sampled_window_count,
            "derived_model_seed": self.derived_model_seed,
            "checkpoint_epoch": self.checkpoint_epoch,
            "checkpoint_objective": self.checkpoint_objective,
            "epoch_objectives": list(self.epoch_objectives),
            "fold_oof_loss": self.fold_oof_loss,
        }


def build_reference_artifact(
    *,
    source_order: Sequence[str],
    assignments: Sequence[ParticipantFoldAssignment],
    fold_records: Sequence[ReferenceFoldRecord],
    per_source_reference_risks: Mapping[str, float],
    reference_seed: int,
    data_sampling_seed: int,
    fold_count: int,
    input_manifest_sha256: str,
    window_manifest_sha256: str,
    reference_config_sha256: str,
    global_class_positive_weights: Sequence[float],
    loss_contract_sha256: str,
) -> dict[str, Any]:
    """Build and validate the complete immutable v1.1 reference payload."""

    sources = _validate_source_order(source_order)
    reference_seed = _validate_integer(reference_seed, field="reference_seed")
    data_sampling_seed = _validate_integer(
        data_sampling_seed,
        field="data_sampling_seed",
    )
    fold_count = _validate_integer(fold_count, field="fold_count", minimum=2)
    for field, digest in (
        ("input_manifest_sha256", input_manifest_sha256),
        ("window_manifest_sha256", window_manifest_sha256),
        ("reference_config_sha256", reference_config_sha256),
        ("loss_contract_sha256", loss_contract_sha256),
    ):
        _validate_sha256(digest, field=field)
    positive_weights = tuple(
        _validate_finite_float(value, field="global_class_positive_weight")
        for value in global_class_positive_weights
    )
    if len(positive_weights) != len(TASK_ORDER) or min(positive_weights) <= 0:
        raise ValueError("one positive global class weight is required per task")

    assignment_rows = tuple(assignments)
    fold_rows = tuple(fold_records)
    if set(per_source_reference_risks) != set(sources):
        raise ValueError("one reference risk is required for every source")
    if {row.source_id for row in assignment_rows} != set(sources):
        raise ValueError("assignment sources must exactly match source_order")
    if {row.source_id for row in fold_rows} != set(sources):
        raise ValueError("fold-record sources must exactly match source_order")
    expected_assignments = assign_source_scoped_folds(
        {
            source: [
                row.participant_id for row in assignment_rows if row.source_id == source
            ]
            for source in sources
        },
        source_order=sources,
        fold_count=fold_count,
        reference_seed=reference_seed,
    )
    if set(expected_assignments) != set(assignment_rows):
        raise ValueError("participant fold assignment disagrees with protocol")

    for source in sources:
        source_assignments = [
            assignment
            for assignment in assignment_rows
            if assignment.source_id == source
        ]
        if len(source_assignments) < fold_count:
            raise ValueError(f"source {source!r} has an incomplete fold ledger")
        participant_ids = [row.participant_id for row in source_assignments]
        if len(set(participant_ids)) != len(participant_ids):
            raise ValueError(f"source {source!r} repeats a participant")
        if {row.fold_id for row in source_assignments} != set(range(fold_count)):
            raise ValueError(f"source {source!r} does not cover every fold")

        source_folds = [row for row in fold_rows if row.source_id == source]
        if len(source_folds) != fold_count or {
            row.fold_id for row in source_folds
        } != set(range(fold_count)):
            raise ValueError(f"source {source!r} has incomplete fold records")
        total_participants = len(source_assignments)
        weighted_losses: list[float] = []
        held_windows = 0
        for row in source_folds:
            fit_participant_count = _validate_integer(
                row.fit_participant_count,
                field="fit_participant_count",
                minimum=1,
            )
            held_participant_count = _validate_integer(
                row.held_participant_count,
                field="held_participant_count",
                minimum=1,
            )
            fit_raw_window_count = _validate_integer(
                row.fit_raw_window_count,
                field="fit_raw_window_count",
                minimum=1,
            )
            fit_sampled_window_count = _validate_integer(
                row.fit_sampled_window_count,
                field="fit_sampled_window_count",
                minimum=1,
            )
            held_raw_window_count = _validate_integer(
                row.held_raw_window_count,
                field="held_raw_window_count",
                minimum=1,
            )
            held_sampled_window_count = _validate_integer(
                row.held_sampled_window_count,
                field="held_sampled_window_count",
                minimum=1,
            )
            checkpoint_epoch = _validate_integer(
                row.checkpoint_epoch,
                field="checkpoint_epoch",
                minimum=1,
            )
            derived_model_seed = _validate_integer(
                row.derived_model_seed,
                field="derived_model_seed",
            )
            if (
                min(
                    fit_participant_count,
                    held_participant_count,
                    fit_raw_window_count,
                    fit_sampled_window_count,
                    held_raw_window_count,
                    held_sampled_window_count,
                    checkpoint_epoch,
                )
                <= 0
            ):
                raise ValueError("reference fold counts and epoch must be positive")
            if checkpoint_epoch > 30:
                raise ValueError("reference checkpoint epoch exceeds the protocol")
            checkpoint_objective = _validate_finite_float(
                row.checkpoint_objective,
                field="checkpoint_objective",
            )
            if checkpoint_objective < 0:
                raise ValueError("checkpoint objective must be non-negative")
            epoch_objectives = tuple(
                _validate_finite_float(
                    value,
                    field="epoch_objective",
                )
                for value in row.epoch_objectives
            )
            if len(epoch_objectives) != 30 or min(epoch_objectives) < 0:
                raise ValueError(
                    "reference folds require 30 non-negative epoch objectives"
                )
            selected_epoch, selected_objective = select_reference_checkpoint(
                enumerate(epoch_objectives, start=1)
            )
            if (
                checkpoint_epoch != selected_epoch
                or checkpoint_objective != selected_objective
            ):
                raise ValueError("checkpoint selection disagrees with epoch ledger")
            if fit_sampled_window_count > fit_raw_window_count:
                raise ValueError("sampled fit windows exceed raw fit windows")
            if held_sampled_window_count > held_raw_window_count:
                raise ValueError("sampled held windows exceed raw held windows")
            expected_held = sum(
                assignment.fold_id == row.fold_id for assignment in source_assignments
            )
            if held_participant_count != expected_held:
                raise ValueError("held participant count disagrees with fold ledger")
            if fit_participant_count != total_participants - expected_held:
                raise ValueError("fit participant count disagrees with fold ledger")
            expected_seed = derive_reference_model_seed(
                reference_seed=reference_seed,
                source_id=source,
                fold_id=row.fold_id,
            )
            if derived_model_seed != expected_seed:
                raise ValueError("derived model seed disagrees with protocol")
            fold_oof_loss = _validate_finite_float(
                row.fold_oof_loss,
                field="fold_oof_loss",
            )
            if fold_oof_loss < 0:
                raise ValueError("fold OOF losses must be finite and non-negative")
            weighted_losses.append(fold_oof_loss * held_sampled_window_count)
            held_windows += held_sampled_window_count

        expected_risk = math.fsum(weighted_losses) / held_windows
        supplied_risk = _validate_finite_float(
            per_source_reference_risks[source],
            field="reference_risk",
        )
        if supplied_risk < 0:
            raise ValueError("reference risks must be finite and non-negative")
        if not math.isclose(expected_risk, supplied_risk, rel_tol=0, abs_tol=1e-12):
            raise ValueError("reference risk disagrees with held-window OOF mean")

    ordered_assignments = sorted(
        assignment_rows,
        key=lambda row: (
            sources.index(row.source_id),
            row.assignment_sha256,
            row.participant_id.encode("utf-8"),
        ),
    )
    ordered_folds = sorted(
        fold_rows,
        key=lambda row: (sources.index(row.source_id), row.fold_id),
    )
    assignment_payload = [row.as_dict() for row in ordered_assignments]
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "protocol_id": PROTOCOL_ID,
        "method": REFERENCE_METHOD,
        "split": REFERENCE_SPLIT,
        "lexical_source_order": list(sources),
        "source_to_index": {source: index for index, source in enumerate(sources)},
        "fold_count": fold_count,
        "reference_seed": reference_seed,
        "data_sampling_seed": data_sampling_seed,
        "participant_fold_assignment": assignment_payload,
        "folds": [row.as_dict() for row in ordered_folds],
        "per_source_reference_risks": {
            source: float(per_source_reference_risks[source]) for source in sources
        },
        "task_order": list(TASK_ORDER),
        "global_class_positive_weights": list(positive_weights),
        "input_manifest_sha256": input_manifest_sha256,
        "window_manifest_sha256": window_manifest_sha256,
        "reference_config_sha256": reference_config_sha256,
        "loss_contract_sha256": loss_contract_sha256,
        "source_order_sha256": hashlib.sha256(
            canonical_json_bytes(list(sources))
        ).hexdigest(),
        "fold_ledger_sha256": hashlib.sha256(
            canonical_json_bytes(assignment_payload)
        ).hexdigest(),
    }
    return bind_artifact_sha256(payload)


def validate_reference_artifact(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate structure, derivations, and the canonical self-hash."""

    verify_artifact_payload(payload)
    if payload.get("schema_version") != "1.0":
        raise ValueError("unsupported reference schema_version")
    if payload.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("reference protocol_id is incompatible")
    if payload.get("method") != REFERENCE_METHOD:
        raise ValueError("reference method is incompatible")
    if payload.get("split") != REFERENCE_SPLIT:
        raise ValueError("reference split must be train")
    if tuple(payload.get("task_order", ())) != TASK_ORDER:
        raise ValueError("reference task order is incompatible")

    assignments = tuple(
        ParticipantFoldAssignment(
            source_id=_validate_identifier(row["source_id"], field="source_id"),
            participant_id=_validate_identifier(
                row["participant_id"],
                field="participant_id",
            ),
            fold_id=_validate_integer(row["fold_id"], field="fold_id"),
            assignment_sha256=_validate_sha256(
                row["assignment_sha256"],
                field="assignment_sha256",
            ),
        )
        for row in payload["participant_fold_assignment"]
    )
    for row in assignments:
        _validate_identifier(row.source_id, field="source_id")
        _validate_identifier(row.participant_id, field="participant_id")
        _validate_sha256(row.assignment_sha256, field="assignment_sha256")
    folds = tuple(
        ReferenceFoldRecord(
            source_id=_validate_identifier(row["source_id"], field="source_id"),
            fold_id=_validate_integer(row["fold_id"], field="fold_id"),
            fit_participant_count=_validate_integer(
                row["fit_participant_count"],
                field="fit_participant_count",
                minimum=1,
            ),
            held_participant_count=_validate_integer(
                row["held_participant_count"],
                field="held_participant_count",
                minimum=1,
            ),
            fit_raw_window_count=_validate_integer(
                row["fit_raw_window_count"],
                field="fit_raw_window_count",
                minimum=1,
            ),
            fit_sampled_window_count=_validate_integer(
                row["fit_sampled_window_count"],
                field="fit_sampled_window_count",
                minimum=1,
            ),
            held_raw_window_count=_validate_integer(
                row["held_raw_window_count"],
                field="held_raw_window_count",
                minimum=1,
            ),
            held_sampled_window_count=_validate_integer(
                row["held_sampled_window_count"],
                field="held_sampled_window_count",
                minimum=1,
            ),
            derived_model_seed=_validate_integer(
                row["derived_model_seed"],
                field="derived_model_seed",
            ),
            checkpoint_epoch=_validate_integer(
                row["checkpoint_epoch"],
                field="checkpoint_epoch",
                minimum=1,
            ),
            checkpoint_objective=_validate_finite_float(
                row["checkpoint_objective"],
                field="checkpoint_objective",
            ),
            epoch_objectives=tuple(
                _validate_finite_float(value, field="epoch_objective")
                for value in row["epoch_objectives"]
            ),
            fold_oof_loss=_validate_finite_float(
                row["fold_oof_loss"],
                field="fold_oof_loss",
            ),
        )
        for row in payload["folds"]
    )
    rebuilt = build_reference_artifact(
        source_order=tuple(payload["lexical_source_order"]),
        assignments=assignments,
        fold_records=folds,
        per_source_reference_risks=payload["per_source_reference_risks"],
        reference_seed=_validate_integer(
            payload["reference_seed"],
            field="reference_seed",
        ),
        data_sampling_seed=_validate_integer(
            payload["data_sampling_seed"],
            field="data_sampling_seed",
        ),
        fold_count=_validate_integer(
            payload["fold_count"],
            field="fold_count",
            minimum=2,
        ),
        input_manifest_sha256=payload["input_manifest_sha256"],
        window_manifest_sha256=payload["window_manifest_sha256"],
        reference_config_sha256=payload["reference_config_sha256"],
        global_class_positive_weights=payload["global_class_positive_weights"],
        loss_contract_sha256=payload["loss_contract_sha256"],
    )
    if rebuilt != dict(payload):
        raise ValueError("reference artifact contains derived-field drift")
    return rebuilt


def read_reference_artifact(path: str | Path) -> dict[str, Any]:
    return validate_reference_artifact(read_canonical_json(path))


def write_reference_artifact(
    payload: Mapping[str, Any],
    path: str | Path,
) -> str:
    return write_immutable_json(validate_reference_artifact(payload), path)
