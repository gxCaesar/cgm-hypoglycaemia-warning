from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from .baselines import build_temporal_baseline
from .c2_runtime import C2RuntimeContext, build_c2_runtime_context
from .c2_training import C2EventUpdateResult
from .config import resolve_path, save_config
from .deep_training import (
    _action_dropout_probabilities,
    _append_log,
    _device,
    _drop_action_groups,
    _load_development_training_data,
    _positive_weights,
    _predict_dataset,
    _restore_regression_scale,
    _seed_everything,
    _selection_score,
    _unweighted_multitask_loss,
)
from .forecast_training import (
    _calibrate_operational_endpoint,
    _calibrate_ordinary_tasks,
    _checkpoint_proxy,
)
from .io import write_json
from .mero_objectives import AllSourceLossAccumulator, ObjectiveMode
from .mero_reference import read_reference_artifact
from .mero_reference_training import (
    SAMPLED_RECORD_COUNT_V1_1,
    SAMPLING_WEIGHT_SUM_V1_2,
)
from .mero_weighted_reference import read_weighted_reference_artifact
from .models import count_parameters
from .pipeline import evaluate_predictions, validate_training_output_isolation
from .sampling import AllSourceBatchSampler
from .schema import ACTION_FIELDS
from .sequences import (
    ParticipantSequenceStore,
    SequenceScaler,
    WindowSequenceDataset,
    sample_training_windows,
)

_OBJECTIVE_MODES = frozenset(
    {
        "all_source_erm",
        "hardmax_zero",
        "mero",
        "c1_cf_mero",
        "c2_source_event_regret",
    }
)


def _required_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _all_source_loader(
    dataset: WindowSequenceDataset,
    *,
    source_order: tuple[str, ...],
    epoch_draws: int,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> tuple[DataLoader, AllSourceBatchSampler]:
    source_indices = dataset.windows["source_id"].map(dataset.source_to_index)
    if source_indices.isna().any():
        raise ValueError("MERO training contains an unknown source")
    sampler = AllSourceBatchSampler(
        source_indices.to_numpy(dtype=np.int64, copy=False),
        source_order=source_order,
        epoch_draws=epoch_draws,
        batch_size=batch_size,
        seed=seed,
    )
    return (
        DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        sampler,
    )


def _reference_vector(
    config: dict[str, Any],
    *,
    mode: ObjectiveMode,
    source_order: tuple[str, ...],
    positive_weights: torch.Tensor,
    data_sampling_seed: int,
) -> tuple[torch.Tensor | None, dict[str, Any] | None]:
    if mode == "all_source_erm":
        return None, None
    if mode == "hardmax_zero":
        return torch.zeros(len(source_order), dtype=torch.float32), {
            "mode": "explicit_zero_vector",
            "source_order": list(source_order),
        }

    reference_value = config["training"].get("mero_reference_path")
    if not isinstance(reference_value, str) or not reference_value:
        raise ValueError("MERO mode requires training.mero_reference_path")
    reference_path = resolve_path(config, reference_value)
    if reference_path is None:
        raise ValueError("MERO reference path could not be resolved")
    reference_contract = config["training"].get("mero_reference_contract")
    if reference_contract == SAMPLED_RECORD_COUNT_V1_1:
        artifact = read_reference_artifact(reference_path)
        receipt_mode = "immutable_participant_cf5_legacy_v1_1"
    elif reference_contract == SAMPLING_WEIGHT_SUM_V1_2:
        artifact = read_weighted_reference_artifact(reference_path)
        receipt_mode = "immutable_participant_cf5_weighted_v1_2"
    else:
        raise ValueError(
            "MERO mode requires an explicit supported "
            "training.mero_reference_contract"
        )
    expected_artifact_sha256 = _required_sha256(
        config["training"].get("mero_reference_artifact_sha256"),
        field="training.mero_reference_artifact_sha256",
    )
    if artifact["artifact_sha256"] != expected_artifact_sha256:
        raise ValueError("MERO reference artifact hash does not match training")
    expected_loss_contract_sha256 = _required_sha256(
        config["training"].get("mero_reference_loss_contract_sha256"),
        field="training.mero_reference_loss_contract_sha256",
    )
    if artifact["loss_contract_sha256"] != expected_loss_contract_sha256:
        raise ValueError("MERO reference loss contract does not match training")
    provenance_bindings: dict[str, str] = {}
    for artifact_field, config_field in (
        ("input_manifest_sha256", "mero_reference_input_manifest_sha256"),
        ("window_manifest_sha256", "mero_reference_window_manifest_sha256"),
        ("reference_config_sha256", "mero_reference_config_sha256"),
    ):
        expected = _required_sha256(
            config["training"].get(config_field),
            field=f"training.{config_field}",
        )
        if artifact[artifact_field] != expected:
            raise ValueError(
                f"MERO reference {artifact_field} does not match training"
            )
        provenance_bindings[artifact_field] = str(artifact[artifact_field])
    if tuple(artifact["lexical_source_order"]) != source_order:
        raise ValueError("MERO reference source order does not match training")
    if int(artifact["data_sampling_seed"]) != data_sampling_seed:
        raise ValueError("MERO reference data-sampling seed does not match training")
    if not bool(config["project"].get("synthetic", False)):
        observed_weights = np.asarray(
            artifact["global_class_positive_weights"],
            dtype=float,
        )
        current_weights = positive_weights.detach().cpu().numpy().astype(float)
        if not np.array_equal(observed_weights, current_weights):
            raise ValueError("MERO reference positive weights do not match training")
    values = torch.tensor(
        [
            float(artifact["per_source_reference_risks"][source])
            for source in source_order
        ],
        dtype=torch.float32,
    )
    return values, {
        "mode": receipt_mode,
        "path": str(reference_path),
        "artifact_sha256": str(artifact["artifact_sha256"]),
        "schema_version": str(artifact["schema_version"]),
        "protocol_id": str(artifact["protocol_id"]),
        "method": str(artifact["method"]),
        "reference_seed": int(artifact["reference_seed"]),
        "fold_count": int(artifact["fold_count"]),
        "data_sampling_seed": int(artifact["data_sampling_seed"]),
        "loss_denominator_contract": reference_contract,
        "loss_contract_sha256": str(artifact["loss_contract_sha256"]),
        **provenance_bindings,
        "source_order": list(source_order),
    }


def _attach_severe_scores(predictions: pd.DataFrame) -> None:
    for horizon in (30, 60):
        logit = f"hypo_{horizon}_logit"
        if logit not in predictions:
            raise ValueError(f"prediction frame is missing {logit}")
        values = pd.to_numeric(predictions[logit], errors="coerce").to_numpy(
            dtype=float
        )
        if not np.isfinite(values).all():
            raise ValueError("severe-hypoglycemia checkpoint scores are non-finite")
        predictions[f"severe_hypo_{horizon}_raw_logit"] = values
        predictions[f"severe_hypo_{horizon}_raw_probability"] = 1.0 / (
            1.0 + np.exp(-np.clip(values, -50.0, 50.0))
        )


def _selection_candidate_count(config: dict[str, Any]) -> int | None:
    value = config["evaluation"].get("selection_operational_candidate_count")
    if value is None:
        return None
    if not bool(config["project"].get("synthetic", False)):
        raise ValueError(
            "real-data checkpoint proxy requires all unique selection scores"
        )
    resolved = int(value)
    if resolved < 2:
        raise ValueError("selection operational candidate count must be at least two")
    return resolved


def _epoch_draws(
    config: dict[str, Any],
    *,
    default: int,
    batch_size: int,
    source_count: int,
) -> int:
    supplied = config["training"].get("mero_epoch_draws")
    if supplied is None:
        if not bool(config["project"].get("synthetic", False)):
            raise ValueError("real-data MERO training requires mero_epoch_draws")
        resolved = int(default)
    else:
        resolved = int(supplied)
    if resolved <= 0:
        raise ValueError("MERO epoch draws must be positive")
    final_size = resolved % batch_size
    if final_size and final_size < source_count:
        if not bool(config["project"].get("synthetic", False)):
            raise ValueError("MERO final physical batch would omit a source")
        resolved += source_count - final_size
    return resolved


def _c2_event_update_schedule(
    *,
    base_updates_per_epoch: int,
    event_updates_per_epoch: int,
) -> tuple[int, ...]:
    if base_updates_per_epoch <= 0 or event_updates_per_epoch <= 0:
        raise ValueError("C2 update counts must be positive")
    if event_updates_per_epoch > base_updates_per_epoch:
        raise ValueError("C2 event updates cannot exceed base updates per epoch")
    if event_updates_per_epoch == 1:
        return (0,)
    schedule = tuple(
        (index * (base_updates_per_epoch - 1)) // (event_updates_per_epoch - 1)
        for index in range(event_updates_per_epoch)
    )
    if len(set(schedule)) != event_updates_per_epoch or schedule[0] != 0:
        raise AssertionError("C2 event-update schedule is not unique and anchored")
    return schedule


def _rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "state": numpy_state[1].tolist(),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
    }


def _restore_rng_state(payload: dict[str, Any]) -> None:
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if set(payload) != required:
        raise ValueError("resume RNG-state fields differ from the frozen schema")
    random.setstate(payload["python"])
    numpy_state = payload["numpy"]
    if set(numpy_state) != {
        "bit_generator",
        "state",
        "position",
        "has_gauss",
        "cached_gaussian",
    }:
        raise ValueError("resume NumPy RNG-state fields differ from the schema")
    np.random.set_state(
        (
            str(numpy_state["bit_generator"]),
            np.asarray(numpy_state["state"], dtype=np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(payload["torch_cpu"])
    if torch.cuda.is_available():
        cuda_state = payload["torch_cuda"]
        if len(cuda_state) != torch.cuda.device_count():
            raise ValueError("resume CUDA RNG state does not match visible devices")
        torch.cuda.set_rng_state_all(cuda_state)


def _optimizer_and_scheduler(
    model: nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
    total_steps: int,
) -> tuple[AdamW, LambdaLR]:
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    warmup_steps = max(1, int(round(total_steps * 0.05)))

    def lr_multiplier(step: int) -> float:
        if step < warmup_steps:
            return max(1e-8, step / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return optimizer, LambdaLR(optimizer, lr_lambda=lr_multiplier)


def run_mero_training(
    config: dict[str, Any],
    *,
    objective_mode: str,
) -> Path:
    """Train one information-matched all-source LSTM objective arm.

    The registered arms share every data, model, optimizer, checkpoint, and
    evaluation path. C2 adds only its frozen train-stream event path.
    """

    if objective_mode not in _OBJECTIVE_MODES:
        raise ValueError(f"unsupported MERO objective mode: {objective_mode}")
    is_c2 = objective_mode == "c2_source_event_regret"
    is_stream_matched_c1 = objective_mode == "c1_cf_mero"
    uses_complete_stream_adapter = is_c2 or is_stream_matched_c1
    base_objective_mode = cast(
        ObjectiveMode,
        "mero" if uses_complete_stream_adapter else objective_mode,
    )
    c2_fields = {
        field for field in config["training"] if field.startswith("c2_")
    }
    if not uses_complete_stream_adapter and c2_fields:
        raise ValueError(
            "objectives without the complete-stream adapter must not receive "
            "C2 training fields"
        )
    training = config["training"]
    if uses_complete_stream_adapter:
        event_scale = float(training.get("c2_event_scale", float("nan")))
        if not math.isfinite(event_scale):
            raise ValueError("stream-matched C1/C2 requires finite c2_event_scale")
        if is_stream_matched_c1 and event_scale != 0.0:
            raise ValueError("stream-matched C1 requires c2_event_scale=0")
        if is_c2 and event_scale <= 0.0:
            raise ValueError("C2 requires a positive c2_event_scale")
    resume_from_last = uses_complete_stream_adapter and bool(
        training.get("c2_resume_from_last", False)
    )
    seed = int(training.get("seed", config["split"]["seed"]))
    data_sampling_seed = int(training.get("data_sampling_seed", seed))
    _seed_everything(seed)

    run_dir = resolve_path(config, config["output"]["run_dir"])
    if run_dir is None:
        raise ValueError("output.run_dir is required")
    completed_artifacts = (
        run_dir / "model_summary.json",
        run_dir / "predictions.parquet",
    )
    if not bool(config["project"].get("synthetic", False)):
        if any(path.exists() for path in completed_artifacts):
            raise FileExistsError(
                f"refusing to overwrite a completed real-data run: {run_dir}"
            )
        existing_checkpoints = (
            run_dir / "ckpts" / "best.pt",
            run_dir / "ckpts" / "last.pt",
        )
        if (
            any(path.exists() for path in existing_checkpoints)
            and not resume_from_last
        ):
            raise FileExistsError(
                f"real-data run has checkpoints but resume is disabled: {run_dir}"
            )
        if resume_from_last and any(
            path.exists() for path in existing_checkpoints
        ) != all(path.exists() for path in existing_checkpoints):
            raise FileNotFoundError(
                "complete-stream resume requires both retained checkpoints"
            )
    validate_training_output_isolation(config, run_dir)
    frame, windows, audit = _load_development_training_data(config)

    train_windows = windows.loc[windows["partition"] == "train"].copy()
    selection_windows = windows.loc[windows["partition"] == "selection"].copy()
    calibration_windows = windows.loc[
        windows["partition"] == "calibration"
    ].copy()
    if any(
        part.empty
        for part in (train_windows, selection_windows, calibration_windows)
    ):
        raise ValueError(
            "MERO training requires train, selection, and calibration windows"
        )
    window_counts = {
        "train_windows_before_sampling": int(len(train_windows)),
        "selection_windows": int(len(selection_windows)),
        "calibration_windows": int(len(calibration_windows)),
    }
    del windows
    if not train_windows["partition"].eq("train").all():
        raise ValueError("MERO model fitting received a non-train row")

    source_order = tuple(sorted(map(str, config["data"]["development_sources"])))
    observed_sources = tuple(sorted(train_windows["source_id"].astype(str).unique()))
    if observed_sources != source_order:
        raise ValueError("MERO train split is missing a development source")
    source_to_index = {
        source: index for index, source in enumerate(source_order)
    }

    train_keys = set(
        zip(  # noqa: B905
            train_windows["source_id"].astype(str),
            train_windows["participant_id"].astype(str),
        )
    )
    scaler = SequenceScaler.fit(frame, train_keys)
    dropout_probabilities = _action_dropout_probabilities(
        frame,
        train_keys,
        maximum=float(config["training"].get("feature_group_dropout_max", 0.5)),
    )
    sampled_train = sample_training_windows(
        train_windows,
        negative_ratio=float(config["training"]["negative_ratio"]),
        seed=data_sampling_seed,
    )
    window_counts["train_windows_after_sampling"] = int(len(sampled_train))
    positive_weights = _positive_weights(train_windows)

    sample_minutes = int(config["protocol"]["sample_minutes"])
    context_samples = int(
        float(config["protocol"]["context_hours"]) * 60 / sample_minutes
    )
    participant_store = ParticipantSequenceStore.from_frame(frame)
    train_dataset = WindowSequenceDataset(
        frame,
        sampled_train,
        scaler=scaler,
        context_samples=context_samples,
        source_to_index=source_to_index,
        participant_store=participant_store,
        copy_windows=False,
    )
    selection_dataset = WindowSequenceDataset(
        frame,
        selection_windows,
        scaler=scaler,
        context_samples=context_samples,
        participant_store=participant_store,
        copy_windows=False,
    )
    calibration_dataset = WindowSequenceDataset(
        frame,
        calibration_windows,
        scaler=scaler,
        context_samples=context_samples,
        participant_store=participant_store,
        copy_windows=False,
    )

    batch_size = int(training["batch_size"])
    effective_batch_size = int(training.get("effective_batch_size", batch_size))
    if effective_batch_size < batch_size or effective_batch_size % batch_size:
        raise ValueError(
            "MERO effective_batch_size must be a positive batch-size multiple"
        )
    accumulation_steps = effective_batch_size // batch_size
    draws = _epoch_draws(
        config,
        default=len(train_dataset),
        batch_size=batch_size,
        source_count=len(source_order),
    )
    loader, batch_sampler = _all_source_loader(
        train_dataset,
        source_order=source_order,
        epoch_draws=draws,
        batch_size=batch_size,
        num_workers=int(training.get("num_workers", 0)),
        seed=data_sampling_seed,
    )

    device = _device(config)
    model = build_temporal_baseline("lstm").to(device)
    positive_weights = positive_weights.to(device)
    reference_risks, reference_receipt = _reference_vector(
        config,
        mode=base_objective_mode,
        source_order=source_order,
        positive_weights=positive_weights,
        data_sampling_seed=data_sampling_seed,
    )
    maximum_epochs = int(training["maximum_epochs"])
    total_updates = max(
        1,
        math.ceil(len(loader) / accumulation_steps) * maximum_epochs,
    )
    base_updates_per_epoch = math.ceil(len(loader) / accumulation_steps)
    c2_context: C2RuntimeContext | None = None
    c2_schedule: tuple[int, ...] = ()
    if uses_complete_stream_adapter:
        event_updates_per_epoch = int(training["c2_event_updates_per_epoch"])
        c2_schedule = _c2_event_update_schedule(
            base_updates_per_epoch=base_updates_per_epoch,
            event_updates_per_epoch=event_updates_per_epoch,
        )
        c2_context = build_c2_runtime_context(
            config,
            model=model,
            frame=frame,
            train_windows=train_windows,
            scaler=scaler,
            context_samples=context_samples,
            updates_per_epoch=event_updates_per_epoch,
            update_dual=is_c2,
            participant_store=participant_store,
        )
        if reference_receipt is None or reference_risks is None:
            raise AssertionError(
                "stream-matched C1/C2 requires a validated weighted base reference"
            )
        if (
            c2_context.reference.base_artifact_sha256
            != reference_receipt["artifact_sha256"]
        ):
            raise ValueError("stream adapter and C1 base artifacts disagree")
        expected_base = torch.tensor(
            c2_context.reference.base_source_risks,
            dtype=reference_risks.dtype,
        )
        if not torch.equal(reference_risks.cpu(), expected_base):
            raise ValueError("stream adapter and C1 base vectors disagree")
    del (
        frame,
        train_windows,
        sampled_train,
        selection_windows,
        calibration_windows,
    )
    optimizer, scheduler = _optimizer_and_scheduler(
        model,
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        total_steps=total_updates,
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / "ckpts"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train.jsonl"
    save_config(config, run_dir / "config_resolved.yaml")
    write_json(audit, run_dir / "cohort_audit.json")
    write_json(scaler.as_dict(), run_dir / "scaler.json")
    write_json(dropout_probabilities, run_dir / "feature_group_dropout.json")
    write_json(
        reference_receipt or {"mode": "none", "source_order": list(source_order)},
        run_dir / "reference_receipt.json",
    )
    if c2_context is not None:
        receipt_name = (
            "c2_runtime_receipt.json"
            if is_c2
            else "c1_stream_adapter_receipt.json"
        )
        write_json(c2_context.receipt(), run_dir / receipt_name)

    best_key = (-float("inf"), -float("inf"), -float("inf"), -float("inf"))
    best_epoch = -1
    patience = 0
    update_step = 0
    first_loss_logged = False
    start_epoch = 0
    selection_candidate_count = _selection_candidate_count(config)
    resume_path = checkpoint_dir / "last.pt"
    if resume_from_last and resume_path.exists():
        resume = torch.load(resume_path, map_location=device)
        if (
            resume.get("architecture") != "lstm"
            or resume.get("objective_mode") != objective_mode
            or tuple(resume.get("source_order", ())) != source_order
            or resume.get("reference_receipt") != reference_receipt
            or c2_context is None
            or resume.get("complete_stream_adapter_receipt")
            != c2_context.receipt()
            or (
                is_c2
                and resume.get("c2_runtime_receipt")
                != c2_context.receipt()
            )
        ):
            raise ValueError("complete-stream resume checkpoint contract mismatch")
        if not (checkpoint_dir / "best.pt").exists():
            raise FileNotFoundError(
                "complete-stream resume requires the retained best checkpoint"
            )
        model.load_state_dict(resume["model_state"])
        optimizer.load_state_dict(resume["optimizer_state"])
        scheduler.load_state_dict(resume["scheduler_state"])
        adapter_state = resume.get("complete_stream_adapter_state")
        if is_c2 and resume.get("c2_state") != adapter_state:
            raise ValueError("C2-specific and shared resume states disagree")
        c2_context.coordinator.load_state_dict(adapter_state)
        _restore_rng_state(resume["rng_state"])
        completed_epoch = int(resume["epoch"])
        if c2_context.coordinator.epoch != completed_epoch + 1:
            raise ValueError(
                "complete-stream resume coordinator epoch disagrees with checkpoint"
            )
        start_epoch = completed_epoch + 1
        update_step = int(resume["update_step"])
        best_key = tuple(map(float, resume["best_key"]))
        if len(best_key) != 4:
            raise ValueError("complete-stream resume best-key shape drift")
        best_epoch = int(resume["best_epoch"])
        patience = int(resume["patience"])
        first_loss_logged = bool(resume["first_loss_logged"])
        _append_log(
            log_path,
            {
                "event": "resumed_completed_epoch",
                "completed_epoch": completed_epoch,
                "next_epoch": start_epoch,
                "update_step": update_step,
            },
        )

    for epoch in range(start_epoch, maximum_epochs):
        batch_sampler.set_epoch(epoch)
        model.train()
        epoch_objectives: list[tuple[float, int]] = []
        epoch_source_risks: list[np.ndarray] = []
        epoch_active_weights: list[np.ndarray] = []
        epoch_stream_objectives: list[float] = []
        epoch_stream_pack_fapd: list[float] = []
        physical_group: list[dict[str, Any]] = []
        epoch_base_update_index = 0
        c2_schedule_set = set(c2_schedule)
        optimizer.zero_grad(set_to_none=True)

        def complete_effective_update(
            *,
            current_epoch: int = epoch,
            objective_ledger: list[tuple[float, int]] = epoch_objectives,
            source_risk_ledger: list[np.ndarray] = epoch_source_risks,
            active_weight_ledger: list[np.ndarray] = epoch_active_weights,
            stream_objective_ledger: list[float] = epoch_stream_objectives,
            stream_fapd_ledger: list[float] = epoch_stream_pack_fapd,
            event_schedule: set[int] = c2_schedule_set,
        ) -> None:
            nonlocal update_step, first_loss_logged, physical_group
            nonlocal epoch_base_update_index
            expected_records = sum(
                int(batch["labels"].shape[0]) for batch in physical_group
            )
            accumulator = AllSourceLossAccumulator(
                source_order=source_order,
                mode=base_objective_mode,
                reference_risks=reference_risks,
                expected_records=expected_records,
            )
            for stored_batch in physical_group:
                sequence = _drop_action_groups(
                    stored_batch["sequence"],
                    probabilities=tuple(
                        dropout_probabilities[field] for field in ACTION_FIELDS
                    ),
                )
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    outputs = model(sequence)
                    per_example = _unweighted_multitask_loss(
                        outputs,
                        stored_batch,
                        positive_weights=positive_weights,
                        auxiliary_weight=float(
                            training["auxiliary_regression_weight"]
                        ),
                    )
                accumulator.add(
                    per_example,
                    stored_batch["source_index"],
                    sampling_weights=stored_batch["sampling_weight"],
                )
            result = accumulator.finalize()
            stream_update: C2EventUpdateResult | None = None
            training_objective = result.objective
            active_weights = result.active_source_weights
            if (
                uses_complete_stream_adapter
                and epoch_base_update_index in event_schedule
            ):
                if c2_context is None:
                    raise AssertionError("complete-stream context is absent")
                stream_update = c2_context.coordinator.compute_update(
                    pack=c2_context.coordinator.expected_pack(),
                    base_source_risks=result.raw_source_risks,
                )
                training_objective = stream_update.objective.objective
                active_weights = stream_update.objective.active_source_weights
            if not bool(torch.isfinite(training_objective)):
                raise FloatingPointError(
                    "non-finite MERO objective at "
                    f"epoch={current_epoch} update={update_step}"
                )
            if not first_loss_logged:
                payload = {
                    "event": "first_finite_loss",
                    "epoch": current_epoch,
                    "update": update_step,
                    "loss": float(training_objective.detach().cpu()),
                    "objective_mode": objective_mode,
                    "records": result.record_count,
                    "complete_stream_update": stream_update is not None,
                    "c2_event_update": is_c2 and stream_update is not None,
                }
                _append_log(log_path, payload)
                print(json.dumps(payload, sort_keys=True), flush=True)
                first_loss_logged = True
            training_objective.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(),
                float(training["gradient_clip_norm"]),
                error_if_nonfinite=True,
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            if stream_update is not None:
                if c2_context is None:
                    raise AssertionError("complete-stream context is absent")
                c2_context.coordinator.commit_update(stream_update)
                stream_objective_ledger.append(
                    float(training_objective.detach().cpu())
                )
                stream_fapd_ledger.append(
                    float(stream_update.equal_source_fapd.detach().cpu())
                )
            update_step += 1
            epoch_base_update_index += 1
            objective_ledger.append(
                (float(training_objective.detach().cpu()), result.record_count)
            )
            source_risk_ledger.append(
                result.raw_source_risks.detach().cpu().numpy()
            )
            active_weight_ledger.append(
                active_weights.detach().cpu().numpy()
            )
            physical_group = []

        for batch in loader:
            physical_group.append(
                {
                    key: value.to(device) if isinstance(value, torch.Tensor) else value
                    for key, value in batch.items()
                }
            )
            if len(physical_group) == accumulation_steps:
                complete_effective_update()
        if physical_group:
            complete_effective_update()
        if epoch_base_update_index != base_updates_per_epoch:
            raise AssertionError("base optimizer-update count drifted within epoch")
        stream_dual_value: float | None = None
        complete_train_fapd: float | None = None
        if uses_complete_stream_adapter:
            if c2_context is None:
                raise AssertionError("complete-stream context is absent")
            if len(epoch_stream_objectives) != len(c2_schedule):
                raise AssertionError(
                    "complete-stream update schedule was not fully consumed"
                )
            stream_dual_value = c2_context.coordinator.finish_epoch()
            complete_train_fapd = (
                c2_context.coordinator.last_complete_train_fapd
            )
            if complete_train_fapd is None:
                raise AssertionError("complete-train FAPD measurement is absent")

        model.eval()
        selection_predictions = _predict_dataset(
            model,
            selection_dataset,
            device=device,
            batch_size=batch_size,
            num_workers=int(training.get("num_workers", 0)),
            memory_stride_minutes=int(config["model"]["memory_stride_minutes"]),
        )
        _attach_severe_scores(selection_predictions)
        operational_score, tie_fapd, proxy = _checkpoint_proxy(
            selection_predictions,
            source_order=source_order,
            candidate_count=selection_candidate_count,
        )
        window_score = _selection_score(selection_predictions)
        key = (operational_score, window_score, -tie_fapd, -float(epoch))
        record_total = sum(records for _, records in epoch_objectives)
        train_objective = math.fsum(
            value * records for value, records in epoch_objectives
        ) / record_total
        source_risk_mean = np.mean(np.stack(epoch_source_risks), axis=0)
        active_weight_mean = np.mean(np.stack(epoch_active_weights), axis=0)
        epoch_payload = {
            "epoch": epoch,
            "train_effective_update_weighted_objective": float(train_objective),
            "selection_operational_auc": operational_score,
            "selection_macro_window_auprc": window_score,
            "selection_proxy_tie_fapd": tie_fapd,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "source_risk_mean": source_risk_mean.tolist(),
            "active_source_weight_mean": active_weight_mean.tolist(),
            "update_step": update_step,
            "complete_stream_update_count": len(epoch_stream_objectives),
            "complete_stream_objective_mean": (
                None
                if not epoch_stream_objectives
                else float(np.mean(epoch_stream_objectives))
            ),
            "complete_stream_pack_fapd_mean": (
                None
                if not epoch_stream_pack_fapd
                else float(np.mean(epoch_stream_pack_fapd))
            ),
            "complete_train_equal_source_fapd": complete_train_fapd,
            "complete_stream_dual_value": stream_dual_value,
            "c2_event_update_count": (
                len(epoch_stream_objectives) if is_c2 else 0
            ),
            "c2_event_objective_mean": (
                float(np.mean(epoch_stream_objectives))
                if is_c2 and epoch_stream_objectives
                else None
            ),
            "c2_equal_source_fapd_mean": (
                float(np.mean(epoch_stream_pack_fapd))
                if is_c2 and epoch_stream_pack_fapd
                else None
            ),
            "c2_dual_value": stream_dual_value if is_c2 else None,
        }
        _append_log(log_path, epoch_payload)
        improved = key > best_key
        if improved:
            best_key = key
            best_epoch = epoch
            patience = 0
        else:
            patience += 1
        checkpoint = {
            "epoch": epoch,
            "checkpoint_boundary": "completed_epoch",
            "update_step": update_step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "architecture": "lstm",
            "objective_mode": objective_mode,
            "source_order": list(source_order),
            "reference_receipt": reference_receipt,
            "selection_operational_score": operational_score,
            "selection_macro_window_auprc": window_score,
            "selection_proxy_tie_fapd": tie_fapd,
            "selection_operational_proxy": proxy,
            "best_key": list(best_key),
            "best_epoch": best_epoch,
            "patience": patience,
            "first_loss_logged": first_loss_logged,
            "rng_state": _rng_state(),
            "c2_state": (
                None
                if not is_c2 or c2_context is None
                else c2_context.coordinator.state_dict()
            ),
            "complete_stream_adapter_state": (
                None
                if c2_context is None
                else c2_context.coordinator.state_dict()
            ),
            "c2_runtime_receipt": (
                None if not is_c2 or c2_context is None else c2_context.receipt()
            ),
            "complete_stream_adapter_receipt": (
                None if c2_context is None else c2_context.receipt()
            ),
        }
        torch.save(checkpoint, checkpoint_dir / "last.pt")
        if improved:
            best_checkpoint = dict(checkpoint)
            best_checkpoint.pop("optimizer_state")
            best_checkpoint.pop("scheduler_state")
            best_checkpoint.pop("rng_state")
            torch.save(best_checkpoint, checkpoint_dir / "best.pt")
        elif patience >= int(training["early_stopping_patience"]):
            break

    checkpoint = torch.load(checkpoint_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    prediction_kwargs = {
        "device": device,
        "batch_size": batch_size,
        "num_workers": int(training.get("num_workers", 0)),
        "memory_stride_minutes": int(config["model"]["memory_stride_minutes"]),
    }
    selection_predictions = _predict_dataset(
        model,
        selection_dataset,
        **prediction_kwargs,
    )
    calibration_predictions = _predict_dataset(
        model,
        calibration_dataset,
        **prediction_kwargs,
    )
    for predictions in (selection_predictions, calibration_predictions):
        _attach_severe_scores(predictions)
        _restore_regression_scale(predictions, scaler)
    task_summary = _calibrate_ordinary_tasks(
        selection_predictions,
        calibration_predictions,
        config=config,
    )
    operational = _calibrate_operational_endpoint(
        selection_predictions,
        calibration_predictions,
        config=config,
        source_order=source_order,
    )
    predictions = pd.concat(
        (
            selection_predictions.assign(partition="selection"),
            calibration_predictions.assign(partition="calibration"),
        ),
        ignore_index=True,
    )
    predictions.to_parquet(run_dir / "predictions.parquet", index=False)

    model_summary = {
        "status": (
            "synthetic_mero_arm_smoke"
            if bool(config["project"].get("synthetic", False))
            else "development_only_external_untouched"
        ),
        "variant": objective_mode,
        "architecture": "lstm",
        "parameters": int(count_parameters(model)),
        "best_epoch": int(best_epoch),
        "best_selection_operational_auc": float(best_key[0]),
        "best_selection_macro_window_auprc": float(best_key[1]),
        "training_seed": seed,
        "split_seed": int(config["split"]["seed"]),
        "tasks": task_summary,
        "source_order": list(source_order),
        "reference": reference_receipt,
        "c2_runtime": (
            None if not is_c2 or c2_context is None else c2_context.receipt()
        ),
        "complete_stream_adapter": (
            None if c2_context is None else c2_context.receipt()
        ),
        "checkpoint_contract": (
            "equal-source participant-macro severe-event sensitivity-FAPD "
            "AUC over the configured frozen threshold sweep"
        ),
    }
    write_json(model_summary, run_dir / "model_summary.json")
    write_json(operational, run_dir / "operational_metrics.json")
    write_json(
        checkpoint["selection_operational_proxy"],
        run_dir / "selection_operational_proxy.json",
    )
    metrics = evaluate_predictions(
        predictions,
        config=config,
        model_summary=model_summary,
    )
    metrics["operational_v1_2"] = operational
    metrics["legacy_pooled_evaluator_reportable"] = False
    write_json(metrics, run_dir / "metrics.json")
    write_json(
        {
            "device": str(device),
            "objective_mode": objective_mode,
            "training_seed": seed,
            "data_sampling_seed": data_sampling_seed,
            "split_seed": int(config["split"]["seed"]),
            "parameters": int(count_parameters(model)),
            "best_epoch": int(best_epoch),
            "best_operational_score": float(best_key[0]),
            "epoch_draws": draws,
            "physical_batch_size": batch_size,
            "effective_batch_size": effective_batch_size,
            **window_counts,
            "source_order": list(source_order),
            "reference": reference_receipt,
            "c2_event_updates_per_epoch": len(c2_schedule) if is_c2 else 0,
            "c2_event_update_schedule": (
                list(c2_schedule) if is_c2 else []
            ),
            "best_checkpoint_c2_state": checkpoint.get("c2_state"),
            "complete_stream_updates_per_epoch": (
                len(c2_schedule) if c2_context is not None else 0
            ),
            "complete_stream_update_schedule": (
                list(c2_schedule) if c2_context is not None else []
            ),
            "best_checkpoint_complete_stream_state": checkpoint.get(
                "complete_stream_adapter_state"
            ),
            "c2_runtime": (
                None if not is_c2 or c2_context is None else c2_context.receipt()
            ),
            "complete_stream_adapter": (
                None if c2_context is None else c2_context.receipt()
            ),
            "feature_group_dropout_probabilities": dropout_probabilities,
            "external_evaluated": False,
        },
        run_dir / "training_summary.json",
    )
    return run_dir
