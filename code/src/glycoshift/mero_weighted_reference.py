"""Immutable CF5 successor artifacts for the weighted v1.2 estimand.

This module deliberately does not alter the frozen v1.1 artifact schema in
``mero_reference``.  It records the sampling-weight denominator used by every
fit and held-fold evaluation so an OOF risk can be reconstructed as the mean
over the represented eligible-window population.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from .mero_reference import (
    TASK_ORDER,
    ParticipantFoldAssignment,
    assign_source_scoped_folds,
    bind_artifact_sha256,
    canonical_json_bytes,
    derive_reference_model_seed,
    read_canonical_json,
    select_reference_checkpoint,
    verify_artifact_payload,
    write_immutable_json,
)

SCHEMA_VERSION = "2.1"
PROTOCOL_ID = "glycoshift-mero-cf5-sampling-weight-sum-v1.2"
REFERENCE_METHOD = "participant-cross-fitted-mero-style-sampling-weight-sum-v1.2"
REFERENCE_SPLIT = "train"
SAMPLING_WEIGHT_SUM_V1_2 = "sampling_weight_sum_v1_2"
CHECKPOINT_STATE_HASH_METHOD = "sorted-tensor-state-sha256-v1"
FOLD_COUNT = 5
MAXIMUM_EPOCHS = 30
PRODUCTION_SOURCE_ORDER = (
    "AZT1D",
    "BrisT1D",
    "CTR3",
    "DCLP3",
    "DCLP5",
    "Flair",
    "IOBP2",
    "PEDAP",
    "ReplaceBG",
    "Shanghai T1DM",
    "T1D-UOM",
)

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


def _validate_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _validate_identifier(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    if any(character in value for character in _FORBIDDEN_IDENTIFIER_CHARACTERS):
        raise ValueError(f"{field} contains a forbidden protocol delimiter")
    return value


def _validate_source_order(source_order: Sequence[str]) -> tuple[str, ...]:
    resolved = tuple(
        _validate_identifier(source, field="source_id") for source in source_order
    )
    if resolved != PRODUCTION_SOURCE_ORDER:
        raise ValueError(
            "weighted reference source_order must be the frozen production order"
        )
    return resolved


def _validate_sampling_weight_denominator(
    value: Any,
    *,
    field: str,
    raw_window_count: int,
) -> float:
    denominator = _validate_finite_float(value, field=field)
    if denominator <= 0:
        raise ValueError(f"{field} must be positive")
    if not math.isclose(
        denominator,
        float(raw_window_count),
        rel_tol=1e-6,
        abs_tol=1e-6,
    ):
        raise ValueError(f"{field} disagrees with the represented raw-window count")
    return denominator


@dataclass(frozen=True)
class WeightedReferenceFoldRecord:
    """One source/fold ledger under the sampling-weight-sum v1.2 contract."""

    source_id: str
    fold_id: int
    fit_participant_count: int
    held_participant_count: int
    fit_raw_window_count: int
    fit_sampled_window_count: int
    held_raw_window_count: int
    held_sampled_window_count: int
    fit_sampling_weight_denominator: float
    held_sampling_weight_denominator: float
    loss_denominator_contract: str
    derived_model_seed: int
    checkpoint_epoch: int
    checkpoint_state_sha256: str
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
            "fit_sampling_weight_denominator": self.fit_sampling_weight_denominator,
            "held_sampling_weight_denominator": self.held_sampling_weight_denominator,
            "loss_denominator_contract": self.loss_denominator_contract,
            "derived_model_seed": self.derived_model_seed,
            "checkpoint_epoch": self.checkpoint_epoch,
            "checkpoint_state_sha256": self.checkpoint_state_sha256,
            "checkpoint_objective": self.checkpoint_objective,
            "epoch_objectives": list(self.epoch_objectives),
            "fold_oof_loss": self.fold_oof_loss,
        }


def _validate_assignment_ledger(
    assignments: Sequence[ParticipantFoldAssignment],
    *,
    source_order: tuple[str, ...],
    reference_seed: int,
) -> tuple[ParticipantFoldAssignment, ...]:
    rows = tuple(assignments)
    if {row.source_id for row in rows} != set(source_order):
        raise ValueError("assignment sources must exactly match source_order")
    if len(set(rows)) != len(rows):
        raise ValueError("assignment ledger contains duplicate rows")
    participants_by_source: dict[str, list[str]] = {
        source: [] for source in source_order
    }
    for row in rows:
        source = _validate_identifier(row.source_id, field="assignment.source_id")
        participant = _validate_identifier(
            row.participant_id,
            field="assignment.participant_id",
        )
        _validate_integer(row.fold_id, field="assignment.fold_id", minimum=0)
        _validate_sha256(row.assignment_sha256, field="assignment.assignment_sha256")
        if row.fold_id >= FOLD_COUNT:
            raise ValueError("assignment.fold_id is outside the frozen fold range")
        participants_by_source[source].append(participant)
    expected = assign_source_scoped_folds(
        participants_by_source,
        source_order=source_order,
        fold_count=FOLD_COUNT,
        reference_seed=reference_seed,
    )
    if set(expected) != set(rows):
        raise ValueError("participant fold assignment disagrees with protocol")
    return rows


def _validate_fold_record(
    row: WeightedReferenceFoldRecord,
    *,
    source_id: str,
    reference_seed: int,
    source_participant_count: int,
    assignment_rows: Sequence[ParticipantFoldAssignment],
) -> None:
    if _validate_identifier(row.source_id, field="fold.source_id") != source_id:
        raise ValueError("fold record source does not match its ledger")
    fold_id = _validate_integer(row.fold_id, field="fold.fold_id", minimum=0)
    if fold_id >= FOLD_COUNT:
        raise ValueError("fold.fold_id is outside the frozen fold range")
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
    if fit_sampled_window_count > fit_raw_window_count:
        raise ValueError("sampled fit windows exceed raw fit windows")
    if held_sampled_window_count > held_raw_window_count:
        raise ValueError("sampled held windows exceed raw held windows")
    _validate_sampling_weight_denominator(
        row.fit_sampling_weight_denominator,
        field="fit_sampling_weight_denominator",
        raw_window_count=fit_raw_window_count,
    )
    _validate_sampling_weight_denominator(
        row.held_sampling_weight_denominator,
        field="held_sampling_weight_denominator",
        raw_window_count=held_raw_window_count,
    )
    if row.loss_denominator_contract != SAMPLING_WEIGHT_SUM_V1_2:
        raise ValueError(
            "fold loss_denominator_contract must be sampling_weight_sum_v1_2"
        )
    expected_held = sum(
        assignment.fold_id == fold_id for assignment in assignment_rows
    )
    if held_participant_count != expected_held:
        raise ValueError("held participant count disagrees with fold ledger")
    if fit_participant_count != source_participant_count - expected_held:
        raise ValueError("fit participant count disagrees with fold ledger")
    expected_seed = derive_reference_model_seed(
        reference_seed=reference_seed,
        source_id=source_id,
        fold_id=fold_id,
    )
    if (
        _validate_integer(row.derived_model_seed, field="derived_model_seed")
        != expected_seed
    ):
        raise ValueError("derived model seed disagrees with protocol")
    checkpoint_epoch = _validate_integer(
        row.checkpoint_epoch,
        field="checkpoint_epoch",
        minimum=1,
    )
    _validate_sha256(
        row.checkpoint_state_sha256,
        field="checkpoint_state_sha256",
    )
    checkpoint_objective = _validate_finite_float(
        row.checkpoint_objective,
        field="checkpoint_objective",
    )
    if checkpoint_objective < 0:
        raise ValueError("checkpoint objective must be non-negative")
    epoch_objectives = tuple(
        _validate_finite_float(value, field="epoch_objective")
        for value in row.epoch_objectives
    )
    if len(epoch_objectives) != MAXIMUM_EPOCHS or min(epoch_objectives) < 0:
        raise ValueError("reference folds require 30 non-negative epoch objectives")
    selected_epoch, selected_objective = select_reference_checkpoint(
        enumerate(epoch_objectives, start=1)
    )
    if checkpoint_epoch != selected_epoch or checkpoint_objective != selected_objective:
        raise ValueError("checkpoint selection disagrees with epoch ledger")
    fold_oof_loss = _validate_finite_float(row.fold_oof_loss, field="fold_oof_loss")
    if fold_oof_loss < 0:
        raise ValueError("fold OOF loss must be non-negative")


def build_weighted_reference_artifact(
    *,
    source_order: Sequence[str],
    assignments: Sequence[ParticipantFoldAssignment],
    fold_records: Sequence[WeightedReferenceFoldRecord],
    per_source_reference_risks: Mapping[str, float],
    reference_seed: int,
    data_sampling_seed: int,
    input_manifest_sha256: str,
    window_manifest_sha256: str,
    reference_config_sha256: str,
    loss_contract_sha256: str,
    global_class_positive_weights: Sequence[float],
) -> dict[str, Any]:
    """Build a self-validating weighted CF5 successor artifact.

    This is intentionally a lower-level builder: it verifies a completed
    ledger but does not fit a model or read a cohort.
    """

    sources = _validate_source_order(source_order)
    reference_seed = _validate_integer(reference_seed, field="reference_seed")
    data_sampling_seed = _validate_integer(
        data_sampling_seed,
        field="data_sampling_seed",
    )
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
    if set(per_source_reference_risks) != set(sources):
        raise ValueError("one reference risk is required for every source")

    assignment_rows = _validate_assignment_ledger(
        assignments,
        source_order=sources,
        reference_seed=reference_seed,
    )
    fold_rows = tuple(fold_records)
    if {row.source_id for row in fold_rows} != set(sources):
        raise ValueError("fold-record sources must exactly match source_order")

    for source in sources:
        source_assignments = tuple(
            row for row in assignment_rows if row.source_id == source
        )
        source_folds = tuple(row for row in fold_rows if row.source_id == source)
        if len(source_folds) != FOLD_COUNT or {
            row.fold_id for row in source_folds
        } != set(range(FOLD_COUNT)):
            raise ValueError(f"source {source!r} has incomplete fold records")
        for row in source_folds:
            _validate_fold_record(
                row,
                source_id=source,
                reference_seed=reference_seed,
                source_participant_count=len(source_assignments),
                assignment_rows=source_assignments,
            )
        held_denominator = math.fsum(
            float(row.held_sampling_weight_denominator) for row in source_folds
        )
        weighted_losses = math.fsum(
            float(row.fold_oof_loss) * float(row.held_sampling_weight_denominator)
            for row in source_folds
        )
        expected_risk = weighted_losses / held_denominator
        supplied_risk = _validate_finite_float(
            per_source_reference_risks[source],
            field="reference_risk",
        )
        if supplied_risk < 0:
            raise ValueError("reference risks must be non-negative")
        if not math.isclose(expected_risk, supplied_risk, rel_tol=0, abs_tol=1e-12):
            raise ValueError(
                "reference risk disagrees with held sampling-weight OOF mean"
            )

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
    fold_payload = [row.as_dict() for row in ordered_folds]
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "method": REFERENCE_METHOD,
        "split": REFERENCE_SPLIT,
        "loss_denominator_contract": SAMPLING_WEIGHT_SUM_V1_2,
        "checkpoint_state_hash_method": CHECKPOINT_STATE_HASH_METHOD,
        "lexical_source_order": list(sources),
        "source_to_index": {source: index for index, source in enumerate(sources)},
        "fold_count": FOLD_COUNT,
        "reference_seed": reference_seed,
        "data_sampling_seed": data_sampling_seed,
        "participant_fold_assignment": assignment_payload,
        "folds": fold_payload,
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
        "assignment_ledger_sha256": hashlib.sha256(
            canonical_json_bytes(assignment_payload)
        ).hexdigest(),
        "fold_ledger_sha256": hashlib.sha256(
            canonical_json_bytes(fold_payload)
        ).hexdigest(),
    }
    return bind_artifact_sha256(payload)


def validate_weighted_reference_artifact(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the successor schema, derivations, and canonical self-hash."""

    verify_artifact_payload(payload)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported weighted reference schema_version")
    if payload.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("weighted reference protocol_id is incompatible")
    if payload.get("method") != REFERENCE_METHOD:
        raise ValueError("weighted reference method is incompatible")
    if payload.get("split") != REFERENCE_SPLIT:
        raise ValueError("weighted reference split must be train")
    if payload.get("loss_denominator_contract") != SAMPLING_WEIGHT_SUM_V1_2:
        raise ValueError("weighted reference requires sampling_weight_sum_v1_2")
    if payload.get("checkpoint_state_hash_method") != CHECKPOINT_STATE_HASH_METHOD:
        raise ValueError("weighted reference checkpoint-state hash is incompatible")
    if tuple(payload.get("task_order", ())) != TASK_ORDER:
        raise ValueError("weighted reference task order is incompatible")
    if payload.get("fold_count") != FOLD_COUNT:
        raise ValueError("weighted reference fold_count must be 5")

    assignments = tuple(
        ParticipantFoldAssignment(
            source_id=_validate_identifier(row["source_id"], field="source_id"),
            participant_id=_validate_identifier(
                row["participant_id"],
                field="participant_id",
            ),
            fold_id=_validate_integer(row["fold_id"], field="fold_id", minimum=0),
            assignment_sha256=_validate_sha256(
                row["assignment_sha256"],
                field="assignment_sha256",
            ),
        )
        for row in payload["participant_fold_assignment"]
    )
    folds = tuple(
        WeightedReferenceFoldRecord(
            source_id=_validate_identifier(row["source_id"], field="source_id"),
            fold_id=_validate_integer(row["fold_id"], field="fold_id", minimum=0),
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
            fit_sampling_weight_denominator=_validate_finite_float(
                row["fit_sampling_weight_denominator"],
                field="fit_sampling_weight_denominator",
            ),
            held_sampling_weight_denominator=_validate_finite_float(
                row["held_sampling_weight_denominator"],
                field="held_sampling_weight_denominator",
            ),
            loss_denominator_contract=row["loss_denominator_contract"],
            derived_model_seed=_validate_integer(
                row["derived_model_seed"],
                field="derived_model_seed",
            ),
            checkpoint_epoch=_validate_integer(
                row["checkpoint_epoch"],
                field="checkpoint_epoch",
                minimum=1,
            ),
            checkpoint_state_sha256=_validate_sha256(
                row["checkpoint_state_sha256"],
                field="checkpoint_state_sha256",
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
    rebuilt = build_weighted_reference_artifact(
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
        input_manifest_sha256=payload["input_manifest_sha256"],
        window_manifest_sha256=payload["window_manifest_sha256"],
        reference_config_sha256=payload["reference_config_sha256"],
        loss_contract_sha256=payload["loss_contract_sha256"],
        global_class_positive_weights=payload["global_class_positive_weights"],
    )
    if rebuilt != dict(payload):
        raise ValueError("weighted reference artifact contains derived-field drift")
    return rebuilt


def read_weighted_reference_artifact(path: str | Path) -> dict[str, Any]:
    """Read a canonical, validated weighted successor artifact."""

    return validate_weighted_reference_artifact(read_canonical_json(path))


def write_weighted_reference_artifact(
    payload: Mapping[str, Any],
    path: str | Path,
) -> str:
    """Write the weighted successor artifact immutably and canonically."""

    return write_immutable_json(validate_weighted_reference_artifact(payload), path)
