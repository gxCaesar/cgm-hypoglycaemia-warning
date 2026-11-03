from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from torch import nn

from .c2_training import (
    C2ObjectiveCoordinator,
    C2ReferenceVectors,
    IndexedDecisionSequenceSource,
    load_c2_reference_vectors,
)
from .config import load_config, resolve_path
from .event_reference_authority import (
    read_production_event_authority,
    validate_production_event_authority_against_shards,
)
from .event_streams import (
    EVENT_HORIZONS,
    AllSourceParticipantPackSampler,
    ParticipantEventStream,
    build_participant_event_stream,
)
from .mero_weighted_reference import read_weighted_reference_artifact
from .runtime_provenance import (
    RUNTIME_PROVENANCE_BINDING_ORDER,
    read_runtime_provenance_receipt,
    runtime_provenance_pins,
    verify_runtime_provenance_receipt_against_paths,
)
from .sequences import (
    ParticipantSequenceStore,
    SequenceScaler,
    WindowSequenceDataset,
)
from .shard_manifest import read_window_manifest


def _sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _required_path(config: dict[str, Any], field: str) -> Path:
    value = config["training"].get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"C2 requires training.{field}")
    path = resolve_path(config, value)
    if path is None:
        raise ValueError(f"C2 path could not be resolved: training.{field}")
    return path


def _runtime_paths(config: dict[str, Any]) -> dict[str, Path]:
    return {
        argument: _required_path(config, config_field)
        for argument, config_field in (
            ("resolved_config_path", "c2_resolved_config_path"),
            ("input_manifest_path", "c2_input_manifest_path"),
            ("split_manifest_path", "c2_split_manifest_path"),
            ("preprocessing_contract_path", "c2_preprocessing_contract_path"),
            ("loss_score_contract_path", "c2_loss_score_contract_path"),
            (
                "participant_shard_manifest_path",
                "c2_participant_shard_manifest_path",
            ),
            ("participant_shard_root", "c2_participant_shard_root"),
        )
    }


def _expected_runtime_pins(config: dict[str, Any]) -> dict[str, str]:
    supplied = config["training"].get("c2_runtime_expected_pins")
    if not isinstance(supplied, Mapping):
        raise TypeError("training.c2_runtime_expected_pins must be a mapping")
    expected_names = set(RUNTIME_PROVENANCE_BINDING_ORDER).difference(
        {"resolved_config"}
    )
    if set(supplied) != expected_names:
        raise ValueError(
            "C2 runtime must pin every non-self provenance binding and must "
            "not predeclare resolved_config"
        )
    return {
        str(field): _sha256(
            digest,
            field=f"training.c2_runtime_expected_pins.{field}",
        )
        for field, digest in supplied.items()
    }


def _load_runtime_receipt(
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Path], dict[str, str]]:
    expected_pins = _expected_runtime_pins(config)
    paths = _runtime_paths(config)
    loaded_config_path = Path(config["_config_path"]).resolve()
    if loaded_config_path != paths["resolved_config_path"].resolve():
        raise ValueError("C2 runtime config path is not the loaded config authority")
    reloaded = load_config(paths["resolved_config_path"])
    current_payload = {
        key: value for key, value in config.items() if key != "_config_path"
    }
    reloaded_payload = {
        key: value for key, value in reloaded.items() if key != "_config_path"
    }
    if current_payload != reloaded_payload:
        raise ValueError("in-memory C2 config differs from the live resolved config")
    from .c2_finalize import finalize_c2_training_runtime

    finalized = finalize_c2_training_runtime(
        config,
        preparation_summary_path=_required_path(
            config, "c2_preparation_summary_path"
        ),
    )
    receipt = read_runtime_provenance_receipt(
        _required_path(config, "c2_runtime_provenance_path"),
        expected_pins=expected_pins,
    )
    receipt = verify_runtime_provenance_receipt_against_paths(
        receipt,
        **paths,
        participant_shard_partition="train",
        expected_pins=expected_pins,
    )
    if receipt != finalized:
        raise ValueError("C2 runtime receipt differs from finalization output")
    return receipt, paths, runtime_provenance_pins(receipt)


def _event_sources(
    *,
    frame: pd.DataFrame,
    train_windows: pd.DataFrame,
    scaler: SequenceScaler,
    context_samples: int,
    source_order: tuple[str, ...],
    authority: Mapping[str, Any],
    participant_store: ParticipantSequenceStore | None = None,
) -> tuple[
    dict[tuple[str, str], ParticipantEventStream],
    dict[tuple[str, str], IndexedDecisionSequenceSource],
]:
    source_to_index = {
        source: index for index, source in enumerate(source_order)
    }
    dataset = WindowSequenceDataset(
        frame,
        train_windows,
        scaler=scaler,
        context_samples=context_samples,
        source_to_index=source_to_index,
        participant_store=participant_store,
        copy_windows=False,
    )
    positions_by_key = {
        (str(source), str(participant)): tuple(map(int, positions))
        for (source, participant), positions in dataset.windows.groupby(
            ["source_id", "participant_id"],
            sort=False,
        ).indices.items()
    }
    metadata_by_key = {
        (str(row["source_id"]), str(row["participant_id"])): row
        for row in authority["participant_streams"]
    }
    if set(positions_by_key) != set(metadata_by_key):
        raise ValueError(
            "loaded train windows do not exactly cover C2 authority participants"
        )

    streams: dict[tuple[str, str], ParticipantEventStream] = {}
    sources: dict[tuple[str, str], IndexedDecisionSequenceSource] = {}
    for key in sorted(metadata_by_key):
        metadata = metadata_by_key[key]
        positions = np.asarray(positions_by_key[key], dtype=np.int64)
        participant_windows = dataset.windows.iloc[positions].reset_index(drop=True)
        stream = build_participant_event_stream(
            participant_windows,
            expected_content_sha256=metadata["stream_content_sha256"],
            window_shard_sha256=metadata["window_shard_sha256"],
        )
        if (
            stream.decision_count != metadata["decision_count"]
            or stream.patient_days != metadata["patient_days"]
            or any(
                stream.event_count(horizon)
                != metadata["event_counts"][str(horizon)]
                for horizon in EVENT_HORIZONS
            )
        ):
            raise ValueError("loaded C2 stream disagrees with authority metadata")
        eligible_positions = tuple(
            int(positions[index]) for index in stream.eligible_row_indices
        )
        streams[key] = stream
        sources[key] = IndexedDecisionSequenceSource(
            dataset,
            indices=eligible_positions,
        )
    return streams, sources


@dataclass(frozen=True)
class C2RuntimeContext:
    coordinator: C2ObjectiveCoordinator
    reference: C2ReferenceVectors
    runtime_provenance_receipt: Mapping[str, Any]
    event_authority: Mapping[str, Any]

    def receipt(self) -> dict[str, Any]:
        return {
            "runtime_provenance_artifact_sha256": self.runtime_provenance_receipt[
                "artifact_sha256"
            ],
            "event_authority_artifact_sha256": self.event_authority[
                "artifact_sha256"
            ],
            "reference": self.reference.receipt(),
            "complete_stream_contract": {
                "event_scale": self.coordinator.event_scale,
                "dual_updates_enabled": self.coordinator.update_dual,
                "event_pack_seed": self.coordinator.sampler.seed,
                "updates_per_epoch": len(self.coordinator.sampler),
                "target_fapd": self.coordinator.target_fapd,
                "chunk_size": self.coordinator.chunk_size,
                "gradient_checkpointing": self.coordinator.use_checkpoint,
            },
        }


def build_c2_runtime_context(
    config: dict[str, Any],
    *,
    model: nn.Module,
    frame: pd.DataFrame,
    train_windows: pd.DataFrame,
    scaler: SequenceScaler,
    context_samples: int,
    updates_per_epoch: int,
    update_dual: bool = True,
    participant_store: ParticipantSequenceStore | None = None,
) -> C2RuntimeContext:
    """Build the live-byte-bound C1/C2 stream path before any optimizer step."""

    receipt, runtime_paths, runtime_pins = _load_runtime_receipt(config)
    base_reference_path = _required_path(config, "c2_base_reference_path")
    event_reference_path = _required_path(config, "c2_event_reference_path")
    authority_path = _required_path(config, "c2_event_authority_path")
    authority = validate_production_event_authority_against_shards(
        read_production_event_authority(authority_path),
        parent_base_reference_artifact=read_weighted_reference_artifact(
            base_reference_path
        ),
        shard_root=runtime_paths["participant_shard_root"],
        participant_shard_manifest=read_window_manifest(
            runtime_paths["participant_shard_manifest_path"]
        ),
    )
    expected_authority = _sha256(
        config["training"].get("c2_event_authority_artifact_sha256"),
        field="training.c2_event_authority_artifact_sha256",
    )
    if authority["artifact_sha256"] != expected_authority:
        raise ValueError("C2 event authority artifact pin mismatch")
    if (
        authority["bindings"]["split_manifest_sha256"]
        != runtime_pins["split_manifest"]
        or authority["bindings"]["preprocessing_config_sha256"]
        != runtime_pins["preprocessing_contract"]
    ):
        raise ValueError("C2 event authority disagrees with live runtime inputs")

    reference = load_c2_reference_vectors(
        base_reference_path=base_reference_path,
        event_reference_path=event_reference_path,
        expected_sha256={
            "base_artifact_sha256": _sha256(
                config["training"].get("c2_base_reference_artifact_sha256"),
                field="training.c2_base_reference_artifact_sha256",
            ),
            "event_artifact_sha256": _sha256(
                config["training"].get("c2_event_reference_artifact_sha256"),
                field="training.c2_event_reference_artifact_sha256",
            ),
            "event_authority_sha256": authority["artifact_sha256"],
            "split_manifest_sha256": runtime_pins["split_manifest"],
            "window_manifest_sha256": authority["bindings"][
                "window_manifest_sha256"
            ],
            "preprocessing_config_sha256": runtime_pins[
                "preprocessing_contract"
            ],
            "base_input_manifest_sha256": runtime_pins["input_manifest"],
            "base_loss_contract_sha256": _sha256(
                config["training"].get("c2_base_loss_contract_sha256"),
                field="training.c2_base_loss_contract_sha256",
            ),
            "event_loss_score_contract_sha256": runtime_pins[
                "loss_score_contract"
            ],
        },
    )
    if reference.source_order != tuple(authority["lexical_source_order"]):
        raise ValueError("C2 reference and authority source orders disagree")
    streams, sequence_sources = _event_sources(
        frame=frame,
        train_windows=train_windows,
        scaler=scaler,
        context_samples=context_samples,
        source_order=reference.source_order,
        authority=authority,
        participant_store=participant_store,
    )
    all_participants = {
        source: tuple(
            sorted(
                participant
                for current_source, participant in streams
                if current_source == source
            )
        )
        for source in reference.source_order
    }
    event_positive = {
        horizon: {
            source: tuple(
                participant
                for participant in all_participants[source]
                if streams[(source, participant)].event_count(horizon) > 0
            )
            for source in reference.source_order
        }
        for horizon in EVENT_HORIZONS
    }
    sampler = AllSourceParticipantPackSampler(
        source_order=reference.source_order,
        all_participants=all_participants,
        event_positive_participants=event_positive,
        updates_per_epoch=updates_per_epoch,
        seed=int(config["training"]["c2_event_pack_seed"]),
    )
    coordinator = C2ObjectiveCoordinator(
        model=model,
        reference=reference,
        sampler=sampler,
        streams=streams,
        sequence_provider=lambda source, participant: sequence_sources[
            (source, participant)
        ],
        event_scale=float(config["training"]["c2_event_scale"]),
        target_fapd=float(config["training"].get("c2_target_fapd", 0.5)),
        chunk_size=int(config["training"]["c2_event_chunk_size"]),
        use_checkpoint=bool(
            config["training"].get("c2_event_gradient_checkpointing", True)
        ),
        initial_dual_value=float(
            config["training"].get("c2_initial_dual_value", 0.0)
        ),
        update_dual=update_dual,
    )
    return C2RuntimeContext(
        coordinator=coordinator,
        reference=reference,
        runtime_provenance_receipt=receipt,
        event_authority=authority,
    )
