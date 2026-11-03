from __future__ import annotations

import io
import math
from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from .baselines import build_temporal_baseline
from .c2_training import IndexedDecisionSequenceSource, chunked_severe_logits
from .cf5_event_execution import (
    CallbackScope,
    CF5EventExecutionResult,
    FitCallbackResult,
    ParticipantSevereLogits,
    PredictionCallbackRequest,
    PredictionCallbackResult,
    execute_cf5_event_reference,
)
from .checkpoint_state import checkpoint_state_sha256
from .event_reference_authority import (
    validate_production_event_authority_against_shards,
)
from .event_streams import build_participant_event_stream
from .mero_reference_training import (
    ProductionWeightedReferenceFitBundle,
    SyntheticReferenceFoldResult,
    WeightedProductionReferenceFitConfig,
)
from .mero_weighted_reference import validate_weighted_reference_artifact
from .models import count_parameters
from .sequences import WindowSequenceDataset

_EXPECTED_LSTM_PARAMETER_COUNT = 268_166
_CHECKPOINT_SCHEMA_VERSION = "1.1"
_CHECKPOINT_METHOD = "weighted-cf5-selected-lstm-state-v1.3"


def _train_scope() -> CallbackScope:
    return CallbackScope(
        fit_partition="train",
        held_partition="train",
        selection_rows_loaded=False,
        calibration_rows_loaded=False,
        external_rows_loaded=False,
    )


def _fold_by_key(
    bundle: ProductionWeightedReferenceFitBundle,
) -> dict[tuple[str, int], SyntheticReferenceFoldResult]:
    parent = validate_weighted_reference_artifact(bundle.artifact)
    output = {
        (fold.source_id, fold.fold_id): fold
        for fold in bundle.fit_result.folds
    }
    expected = {
        (source, fold_id)
        for source in parent["lexical_source_order"]
        for fold_id in range(int(parent["fold_count"]))
    }
    if set(output) != expected:
        raise ValueError("weighted fit bundle does not cover every source-fold")
    artifact_folds = {
        (str(row["source_id"]), int(row["fold_id"])): row
        for row in parent["folds"]
    }
    for key, fold in output.items():
        row = artifact_folds[key]
        exact_pairs = (
            (len(fold.fit_participant_ids), row["fit_participant_count"]),
            (len(fold.held_participant_ids), row["held_participant_count"]),
            (fold.fit_raw_window_count, row["fit_raw_window_count"]),
            (fold.fit_sampled_window_count, row["fit_sampled_window_count"]),
            (fold.held_raw_window_count, row["held_raw_window_count"]),
            (fold.held_sampled_window_count, row["held_sampled_window_count"]),
            (fold.loss_denominator_contract, row["loss_denominator_contract"]),
            (fold.derived_model_seed, row["derived_model_seed"]),
            (fold.checkpoint_epoch, row["checkpoint_epoch"]),
            (tuple(fold.epoch_objectives), tuple(row["epoch_objectives"])),
        )
        if any(observed != expected_value for observed, expected_value in exact_pairs):
            raise ValueError("in-memory CF5 fold disagrees with parent fold ledger")
        float_pairs = (
            (
                fold.fit_evaluation_denominator,
                row["fit_sampling_weight_denominator"],
            ),
            (
                fold.held_evaluation_denominator,
                row["held_sampling_weight_denominator"],
            ),
            (fold.checkpoint_objective, row["checkpoint_objective"]),
            (fold.fold_oof_loss, row["fold_oof_loss"]),
        )
        if any(
            not math.isclose(
                float(observed),
                float(expected_value),
                rel_tol=0,
                abs_tol=0,
            )
            for observed, expected_value in float_pairs
        ):
            raise ValueError("in-memory CF5 fold numerics disagree with parent ledger")
        if checkpoint_state_sha256(fold.checkpoint_state) != row[
            "checkpoint_state_sha256"
        ]:
            raise ValueError("in-memory CF5 state disagrees with parent fold ledger")
    return output


def _checkpoint_bytes(
    fold: SyntheticReferenceFoldResult,
    *,
    config: WeightedProductionReferenceFitConfig,
) -> bytes:
    payload = {
        "schema_version": _CHECKPOINT_SCHEMA_VERSION,
        "method": _CHECKPOINT_METHOD,
        "source_id": fold.source_id,
        "fold_id": fold.fold_id,
        "fit_participant_ids": list(fold.fit_participant_ids),
        "held_participant_ids": list(fold.held_participant_ids),
        "derived_model_seed": fold.derived_model_seed,
        "checkpoint_epoch": fold.checkpoint_epoch,
        "checkpoint_state_sha256": checkpoint_state_sha256(
            fold.checkpoint_state
        ),
        "checkpoint_objective": fold.checkpoint_objective,
        "model_state": dict(fold.checkpoint_state),
        "scaler": fold.scaler.as_dict(),
        "fit_config": asdict(config),
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def _loaded_checkpoint(
    value: bytes,
    *,
    source_id: str,
    fold_id: int,
    expected_checkpoint_epoch: int,
    expected_checkpoint_state_sha256: str,
    device: torch.device,
) -> dict[str, Any]:
    payload = torch.load(
        io.BytesIO(value),
        map_location=device,
        weights_only=False,
    )
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != _CHECKPOINT_SCHEMA_VERSION
        or payload.get("method") != _CHECKPOINT_METHOD
        or payload.get("source_id") != source_id
        or payload.get("fold_id") != fold_id
        or payload.get("checkpoint_epoch") != expected_checkpoint_epoch
        or payload.get("checkpoint_state_sha256")
        != expected_checkpoint_state_sha256
    ):
        raise ValueError("serialized CF5 checkpoint contract mismatch")
    state = payload.get("model_state")
    if (
        not isinstance(state, Mapping)
        or checkpoint_state_sha256(state) != expected_checkpoint_state_sha256
    ):
        raise ValueError("serialized CF5 checkpoint state digest mismatch")
    return payload


def _held_sequence_sources(
    *,
    frame: pd.DataFrame,
    train_windows: pd.DataFrame,
    fold: SyntheticReferenceFoldResult,
    authority_streams: Mapping[tuple[str, str], Mapping[str, Any]],
    context_samples: int,
) -> dict[tuple[str, str], IndexedDecisionSequenceSource]:
    source_frame = frame.loc[frame["source_id"].astype(str).eq(fold.source_id)]
    held_windows = train_windows.loc[
        train_windows["source_id"].astype(str).eq(fold.source_id)
        & train_windows["participant_id"].astype(str).isin(
            fold.held_participant_ids
        )
    ].copy()
    dataset = WindowSequenceDataset(
        source_frame,
        held_windows,
        scaler=fold.scaler,
        context_samples=context_samples,
        source_to_index={fold.source_id: 0},
    )
    positions_by_key = {
        (str(source), str(participant)): tuple(map(int, positions))
        for (source, participant), positions in dataset.windows.groupby(
            ["source_id", "participant_id"],
            sort=False,
        ).indices.items()
    }
    expected = {
        (fold.source_id, participant) for participant in fold.held_participant_ids
    }
    if set(positions_by_key) != expected:
        raise ValueError("held inference windows do not cover the fold recipe")
    output: dict[tuple[str, str], IndexedDecisionSequenceSource] = {}
    for key in sorted(expected):
        metadata = authority_streams.get(key)
        if metadata is None:
            raise ValueError("held participant is absent from event authority")
        positions = np.asarray(positions_by_key[key], dtype=np.int64)
        stream = build_participant_event_stream(
            dataset.windows.iloc[positions].reset_index(drop=True),
            expected_content_sha256=metadata["stream_content_sha256"],
            window_shard_sha256=metadata["window_shard_sha256"],
        )
        output[key] = IndexedDecisionSequenceSource(
            dataset,
            indices=tuple(
                int(positions[index]) for index in stream.eligible_row_indices
            ),
        )
    return output


def execute_weighted_fit_event_reference(
    *,
    bundle: ProductionWeightedReferenceFitBundle,
    fit_config: WeightedProductionReferenceFitConfig,
    frame: pd.DataFrame,
    train_windows: pd.DataFrame,
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
    inference_chunk_size: int,
    model_factory: Callable[[], nn.Module] | None = None,
) -> CF5EventExecutionResult:
    """Emit event OOF scores from the exact states used by the base artifact."""

    authority = validate_production_event_authority_against_shards(
        event_authority,
        parent_base_reference_artifact=bundle.artifact,
        shard_root=shard_root,
        participant_shard_manifest=participant_shard_manifest,
    )
    folds = _fold_by_key(bundle)
    checkpoint_bytes = {
        key: _checkpoint_bytes(fold, config=fit_config)
        for key, fold in folds.items()
    }
    authority_streams = {
        (row["source_id"], row["participant_id"]): row
        for row in authority["participant_streams"]
    }
    factory = model_factory or (lambda: build_temporal_baseline("lstm"))

    def fit_callback(recipe: Any) -> FitCallbackResult:
        fold = folds[(recipe.source_id, recipe.fold_id)]
        if (
            recipe.fit_participant_ids != fold.fit_participant_ids
            or recipe.held_participant_ids != fold.held_participant_ids
        ):
            raise ValueError("CF5 execution recipe disagrees with fitted fold")
        checkpoint_value = checkpoint_bytes[(recipe.source_id, recipe.fold_id)]
        state_digest = checkpoint_state_sha256(fold.checkpoint_state)
        _loaded_checkpoint(
            checkpoint_value,
            source_id=fold.source_id,
            fold_id=fold.fold_id,
            expected_checkpoint_epoch=fold.checkpoint_epoch,
            expected_checkpoint_state_sha256=state_digest,
            device=torch.device("cpu"),
        )
        return FitCallbackResult(
            model_config={
                "architecture": "lstm",
                "parameters": _EXPECTED_LSTM_PARAMETER_COUNT,
                "derived_model_seed": fold.derived_model_seed,
                "loss_denominator_contract": fold.loss_denominator_contract,
                "reference_artifact_sha256": bundle.artifact["artifact_sha256"],
            },
            checkpoint_bytes=checkpoint_value,
            checkpoint_epoch=fold.checkpoint_epoch,
            checkpoint_state_sha256=state_digest,
            scope=_train_scope(),
        )

    def prediction_callback(
        request: PredictionCallbackRequest,
    ) -> PredictionCallbackResult:
        key = (request.recipe.source_id, request.recipe.fold_id)
        fold = folds[key]
        device = torch.device(fit_config.device)
        checkpoint = _loaded_checkpoint(
            checkpoint_bytes[key],
            source_id=key[0],
            fold_id=key[1],
            expected_checkpoint_epoch=fold.checkpoint_epoch,
            expected_checkpoint_state_sha256=checkpoint_state_sha256(
                fold.checkpoint_state
            ),
            device=device,
        )
        model = factory().to(device)
        if count_parameters(model) != _EXPECTED_LSTM_PARAMETER_COUNT:
            raise ValueError("event OOF model is not the frozen production LSTM")
        model.load_state_dict(checkpoint["model_state"], strict=True)
        sources = _held_sequence_sources(
            frame=frame,
            train_windows=train_windows,
            fold=fold,
            authority_streams=authority_streams,
            context_samples=fit_config.context_samples,
        )
        rows: list[ParticipantSevereLogits] = []
        for participant in fold.held_participant_ids:
            with torch.no_grad():
                logits = chunked_severe_logits(
                    model,
                    sources[(fold.source_id, participant)],
                    chunk_size=inference_chunk_size,
                    use_checkpoint=False,
                )
            rows.append(
                ParticipantSevereLogits(
                    source_id=fold.source_id,
                    participant_id=participant,
                    raw_logits_by_horizon={
                        horizon: tuple(
                            float(value) for value in scores.detach().cpu().tolist()
                        )
                        for horizon, scores in logits.items()
                    },
                )
            )
        return PredictionCallbackResult(
            participant_logits=rows,
            scope=_train_scope(),
        )

    return execute_cf5_event_reference(
        parent_base_reference_artifact=bundle.artifact,
        event_authority=authority,
        expected_authority_sha256=expected_authority_sha256,
        shard_root=shard_root,
        execution_root=execution_root,
        runtime_provenance_receipt=runtime_provenance_receipt,
        resolved_config_path=resolved_config_path,
        input_manifest_path=input_manifest_path,
        split_manifest_path=split_manifest_path,
        preprocessing_contract_path=preprocessing_contract_path,
        loss_score_contract_path=loss_score_contract_path,
        participant_shard_manifest_path=participant_shard_manifest_path,
        participant_shard_manifest=participant_shard_manifest,
        expected_runtime_pins=expected_runtime_pins,
        fit_callback=fit_callback,
        prediction_callback=prediction_callback,
    )
