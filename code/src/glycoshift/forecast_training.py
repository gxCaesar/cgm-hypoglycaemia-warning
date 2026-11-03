from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from .calibration import apply_temperature, fit_conformal_threshold, fit_temperature
from .config import resolve_path, save_config
from .deep_training import (
    _action_dropout_probabilities,
    _append_log,
    _device,
    _drop_action_groups,
    _load_development_training_data,
    _seed_everything,
    _selection_score,
)
from .io import write_json
from .metrics import select_alarm_threshold
from .operational_metrics import (
    OperationalEndpointSpec,
    fit_operational_alarm_policy,
    operational_endpoint_metrics,
    selection_operational_curve,
)
from .pipeline import evaluate_predictions, validate_training_output_isolation
from .sampling import AllSourceBatchSampler
from .schema import ACTION_FIELDS
from .sequences import (
    REGRESSION_HORIZONS,
    TASKS,
    ForecastWindowSequenceDataset,
    ParticipantSequenceStore,
    SequenceScaler,
    sample_training_windows,
)
from .taskaware_baselines import (
    PatchMLPForecaster,
    PatchMLPForecasterConfig,
    forecast_task_probabilities,
    glimmer_weighted_mae_per_example,
    inverse_standardized_cgm,
)

PATCHMLP_COMMIT = "b36bbc92ecfc4732acaabb6d5e8c4ff487876f5d"
GLIMMER_COMMIT = "d807553d64f6275db3dedb5cfa579ae0deaf1969"


def _patchmlp_config(
    config: dict[str, Any],
    *,
    context_samples: int,
    prediction_samples: int,
) -> PatchMLPForecasterConfig:
    values = config["model"].get("taskaware_forecast", {})
    if not isinstance(values, dict):
        raise ValueError("model.taskaware_forecast must be a mapping")
    return PatchMLPForecasterConfig(
        context_samples=context_samples,
        prediction_samples=prediction_samples,
        d_model=int(values.get("d_model", 1024)),
        encoder_layers=int(values.get("encoder_layers", 1)),
        moving_average_kernel=int(values.get("moving_average_kernel", 13)),
        dropout=float(values.get("dropout", 0.1)),
        patch_lengths=tuple(
            int(value) for value in values.get("patch_lengths", (48, 24, 12, 6))
        ),
        use_instance_normalization=bool(
            values.get("use_instance_normalization", True)
        ),
    )


def _all_source_loader(
    dataset: ForecastWindowSequenceDataset,
    *,
    source_order: tuple[str, ...],
    batch_size: int,
    num_workers: int,
    seed: int,
) -> tuple[DataLoader, AllSourceBatchSampler]:
    if batch_size < len(source_order):
        raise ValueError("forecast batch_size must include every source")
    source_indices = dataset.windows["source_id"].map(dataset.source_to_index)
    if source_indices.isna().any():
        raise ValueError("forecast training contains an unknown source")
    epoch_draws = len(dataset)
    final_size = epoch_draws % batch_size
    if final_size and final_size < len(source_order):
        epoch_draws += len(source_order) - final_size
    sampler = AllSourceBatchSampler(
        source_indices.to_numpy(dtype=np.int64, copy=False),
        source_order=source_order,
        epoch_draws=epoch_draws,
        batch_size=batch_size,
        seed=seed,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, sampler


def _source_balanced_loss(
    per_example: torch.Tensor,
    sampling_weight: torch.Tensor,
    source_index: torch.Tensor,
    *,
    source_count: int,
) -> tuple[torch.Tensor, dict[int, float]]:
    if per_example.ndim != 1 or per_example.shape != sampling_weight.shape:
        raise ValueError("forecast loss and sampling weights must be vectors")
    source_losses: list[torch.Tensor] = []
    observed: dict[int, float] = {}
    for source in range(source_count):
        mask = source_index == source
        if not mask.any():
            raise ValueError("all-source forecast batch omitted a source")
        weights = sampling_weight[mask]
        denominator = weights.sum()
        if not torch.isfinite(denominator) or denominator <= 0:
            raise ValueError("forecast sampling weights must be finite and positive")
        value = torch.sum(per_example[mask] * weights) / denominator
        source_losses.append(value)
        observed[source] = float(value.detach().cpu())
    return torch.stack(source_losses).mean(), observed


def _forecast_loss_per_example(
    *,
    loss_name: str,
    forecast_standardized: torch.Tensor,
    forecast_mg_dl: torch.Tensor,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    mask = batch["forecast_mask"]
    if loss_name == "mse":
        available = mask.sum(dim=1)
        if (available == 0).any():
            raise ValueError("every forecast example requires an available target")
        errors = functional.mse_loss(
            forecast_standardized,
            batch["forecast_targets"],
            reduction="none",
        )
        errors = torch.where(mask, errors, torch.zeros_like(errors))
        return errors.sum(dim=1) / available.to(errors.dtype)
    if loss_name == "glimmer":
        return glimmer_weighted_mae_per_example(
            forecast_mg_dl,
            batch["forecast_targets_mg_dl"],
            mask,
        )
    raise ValueError(f"unsupported forecast loss: {loss_name}")


def _probability_to_logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def _predict_forecast_dataset(
    model: PatchMLPForecaster,
    dataset: ForecastWindowSequenceDataset,
    *,
    scaler: SequenceScaler,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    sample_minutes: int,
    severe_hypoglycemia_threshold: float,
) -> pd.DataFrame:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    probabilities = {
        task: np.empty(len(dataset), dtype=np.float32)
        for task in TASKS
    }
    severe_probabilities = {
        horizon: np.empty(len(dataset), dtype=np.float32)
        for horizon in REGRESSION_HORIZONS
    }
    point_forecasts = {
        horizon: np.empty(len(dataset), dtype=np.float32)
        for horizon in REGRESSION_HORIZONS
    }
    scale = scaler.values["cgm_mg_dl"]
    model.eval()
    with torch.no_grad():
        for batch in loader:
            indices = batch["dataset_index"].long()
            sequence = batch["sequence"].to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                output = model(sequence)
            forecast_standardized = output["forecast"].float()
            forecast_mg_dl = inverse_standardized_cgm(
                forecast_standardized,
                mean_mg_dl=scale.mean,
                std_mg_dl=scale.std,
            )
            task_values = forecast_task_probabilities(
                forecast_mg_dl,
                sample_minutes=sample_minutes,
            )
            severe_values = forecast_task_probabilities(
                forecast_mg_dl,
                sample_minutes=sample_minutes,
                hypoglycemia_threshold=severe_hypoglycemia_threshold,
            )
            for task in TASKS:
                probabilities[task][indices] = (
                    task_values[task].float().cpu().numpy()
                )
            for horizon in REGRESSION_HORIZONS:
                severe_probabilities[horizon][indices] = (
                    severe_values[f"hypo_{horizon}"].float().cpu().numpy()
                )
                step = horizon // sample_minutes - 1
                point_forecasts[horizon][indices] = (
                    forecast_mg_dl[:, step].float().cpu().numpy()
                )

    predictions = dataset.windows.copy()
    for task in TASKS:
        probability = probabilities[task]
        predictions[f"{task}_raw_probability"] = probability
        predictions[f"{task}_probability"] = probability
        predictions[f"{task}_logit"] = _probability_to_logit(probability)
    for horizon in REGRESSION_HORIZONS:
        probability = severe_probabilities[horizon]
        predictions[f"severe_hypo_{horizon}_raw_probability"] = probability
        predictions[f"severe_hypo_{horizon}_raw_logit"] = _probability_to_logit(
            probability
        )
        predictions[f"cgm_{horizon}_prediction"] = point_forecasts[horizon]
    return predictions


def _operational_spec(
    *,
    horizon: int,
    score_column: str,
    source_order: tuple[str, ...],
) -> OperationalEndpointSpec:
    return OperationalEndpointSpec(
        score_column=score_column,
        label_column=f"severe_hypo_{horizon}_label",
        event_id_column=f"severe_hypo_{horizon}_event_id",
        event_onset_column=f"severe_hypo_{horizon}_event_onset",
        active_column="severe_hypo_currently_active",
        horizon_minutes=horizon,
        source_order=source_order,
    )


def _frontier_at_budget(curve: dict[str, Any], budget: float) -> float:
    """Best feasible sensitivity on the stored frontier at FAPD <= budget.

    The frontier is already built; this reads it at one abscissa instead of
    integrating it, so it is cheaper than the area it replaces.
    """

    best = 0.0
    for point in curve["frontier"]:
        if float(point["achieved_fapd"]) <= budget:
            best = max(best, float(point["best_feasible_event_sensitivity"]))
    return best


def _checkpoint_proxy(
    predictions: pd.DataFrame,
    *,
    source_order: tuple[str, ...],
    candidate_count: int | None,
    selection_rule: str = "area",
    budget: float = 0.5,
) -> tuple[float, float, dict[str, Any]]:
    """Score a checkpoint on the selection partition.

    `area`, the default and the frozen behaviour, integrates the
    sensitivity-versus-FAPD frontier over FAPD 0 to 2.

    `budget` reads the same frontier at the endpoint's own operating point. The
    two are not interchangeable and the difference is measured rather than
    supposed: across five completed arms the area orders them as the frozen
    primary does **one time in five**, while the frontier read at FAPD 0.5
    orders them **exactly**, five of five. M0's best sensitivity within the
    integrated range occurs at FAPD 1.4365, which the frozen protocol never
    reads.

    The default is unchanged because thirteen completed arms were selected under
    it, and a silent switch would make every one of them incomparable with
    anything run afterwards.
    """

    if selection_rule not in {"area", "budget"}:
        raise ValueError(
            f"checkpoint selection_rule must be area or budget, got {selection_rule!r}"
        )
    horizons: dict[str, Any] = {}
    areas: list[float] = []
    tie_fapd: list[float] = []
    for horizon in REGRESSION_HORIZONS:
        # The benchmark freezes every unique selection logit as a threshold.
        # A sigmoid (especially after float32 conversion or clipping) can merge
        # distinct logits and silently change the checkpoint frontier.
        score_column = f"severe_hypo_{horizon}_raw_logit"
        curve = selection_operational_curve(
            predictions,
            spec=_operational_spec(
                horizon=horizon,
                score_column=score_column,
                source_order=source_order,
            ),
            candidate_count=candidate_count,
        )
        horizons[str(horizon)] = curve
        areas.append(
            float(curve["raw_auc"])
            if selection_rule == "area"
            else _frontier_at_budget(curve, budget)
        )
        tie_fapd.append(float(curve["fapd_at_best_sensitivity"]))
    return (
        float(np.mean(areas)),
        float(np.mean(tie_fapd)),
        {"horizons": horizons, "selection_rule": selection_rule, "budget": budget},
    )


def _calibrate_ordinary_tasks(
    selection_predictions: pd.DataFrame,
    calibration_predictions: pd.DataFrame,
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    task_summary: dict[str, Any] = {}
    for task in TASKS:
        kind, horizon_text = task.split("_")
        horizon = int(horizon_text)
        calibration = calibration_predictions.loc[
            ~calibration_predictions[f"{kind}_currently_active"].astype(bool)
        ]
        temperature = fit_temperature(
            calibration[f"{task}_label"].to_numpy(dtype=int),
            calibration[f"{task}_logit"].to_numpy(dtype=float),
        )
        for predictions in (selection_predictions, calibration_predictions):
            predictions[f"{task}_probability"] = apply_temperature(
                predictions[f"{task}_logit"].to_numpy(dtype=float),
                temperature,
            )
        calibrated = calibration_predictions.loc[
            ~calibration_predictions[f"{kind}_currently_active"].astype(bool)
        ]
        conformal_threshold = fit_conformal_threshold(
            calibrated[f"{task}_label"].to_numpy(dtype=int),
            calibrated[f"{task}_probability"].to_numpy(dtype=float),
            alpha=0.1,
        )
        thresholds: dict[str, float] = {}
        for target in config["evaluation"]["false_alarms_per_patient_day"]:
            thresholds[str(target)] = select_alarm_threshold(
                calibrated,
                probability_column=f"{task}_probability",
                label_column=f"{task}_label",
                event_id_column=f"{task}_event_id",
                event_onset_column=f"{task}_event_onset",
                horizon_minutes=horizon,
                target_false_alarms_per_day=float(target),
                candidate_count=int(
                    config["evaluation"].get("threshold_candidate_count", 2001)
                ),
                workers=int(
                    config["evaluation"].get("alarm_sweep_workers", 1)
                ),
            )
        task_summary[task] = {
            "temperature": float(temperature),
            "conformal_nonconformity_threshold": float(conformal_threshold),
            "threshold_partition": "calibration",
            "thresholds_from_calibration": thresholds,
            "thresholds_from_validation": thresholds,
        }
    return task_summary


def _calibrate_operational_endpoint(
    selection_predictions: pd.DataFrame,
    calibration_predictions: pd.DataFrame,
    *,
    config: dict[str, Any],
    source_order: tuple[str, ...],
) -> dict[str, Any]:
    output: dict[str, Any] = {"horizons": {}}
    sensitivities: list[float] = []
    fapd_values: list[float] = []
    candidate_count = int(
        config["evaluation"].get("threshold_candidate_count", 2001)
    )
    for horizon in REGRESSION_HORIZONS:
        active = "severe_hypo_currently_active"
        label = f"severe_hypo_{horizon}_label"
        raw_logit = f"severe_hypo_{horizon}_raw_logit"
        probability = f"severe_hypo_{horizon}_probability"
        calibration = calibration_predictions.loc[
            ~calibration_predictions[active].astype(bool)
        ]
        temperature = fit_temperature(
            calibration[label].to_numpy(dtype=int),
            calibration[raw_logit].to_numpy(dtype=float),
        )
        for predictions in (selection_predictions, calibration_predictions):
            predictions[probability] = apply_temperature(
                predictions[raw_logit].to_numpy(dtype=float),
                temperature,
            )
        spec = _operational_spec(
            horizon=horizon,
            score_column=probability,
            source_order=source_order,
        )
        policy = fit_operational_alarm_policy(
            calibration_predictions,
            spec=spec,
            nominal_fapd=0.5,
            candidate_count=candidate_count,
            score_transform=f"temperature_scaling:{temperature:.17g}",
        )
        metrics = operational_endpoint_metrics(
            selection_predictions,
            spec=spec,
            threshold=policy.threshold,
        )
        sensitivity = float(
            metrics["equal_source_participant_macro_event_sensitivity"]
        )
        achieved_fapd = float(
            metrics[
                "equal_source_participant_macro_false_alarms_per_patient_day"
            ]
        )
        sensitivities.append(sensitivity)
        fapd_values.append(achieved_fapd)
        output["horizons"][str(horizon)] = {
            "temperature": float(temperature),
            "policy": policy.as_dict(),
            "selection_metrics": metrics,
        }
    output.update(
        {
            "primary_equal_horizon_mean_event_sensitivity": float(
                np.mean(sensitivities)
            ),
            "equal_horizon_mean_achieved_fapd": float(np.mean(fapd_values)),
            "nominal_fapd": 0.5,
            "threshold_partition": "calibration",
            "evaluation_partition": "selection",
        }
    )
    return output


def run_forecast_training(
    config: dict[str, Any],
    *,
    loss_name: str,
) -> Path:
    """Train the audited PatchMLP forecasting comparator on development only."""

    if loss_name not in {"mse", "glimmer"}:
        raise ValueError("forecast loss must be mse or glimmer")
    seed = int(config["training"].get("seed", config["split"]["seed"]))
    _seed_everything(seed)
    run_dir = resolve_path(config, config["output"]["run_dir"])
    if run_dir is None:
        raise ValueError("output.run_dir is required")
    protected = (
        run_dir / "model_summary.json",
        run_dir / "predictions.parquet",
        run_dir / "ckpts" / "best.pt",
    )
    if not bool(config["project"].get("synthetic", False)) and any(
        path.exists() for path in protected
    ):
        raise FileExistsError(
            f"refusing to overwrite an existing real-data run: {run_dir}"
        )
    validate_training_output_isolation(config, run_dir)
    frame, windows, audit = _load_development_training_data(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / "ckpts"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train.jsonl"
    save_config(config, run_dir / "config_resolved.yaml")
    write_json(audit, run_dir / "cohort_audit.json")

    train_windows = windows.loc[windows["partition"] == "train"].copy()
    selection_windows = windows.loc[windows["partition"] == "selection"].copy()
    calibration_windows = windows.loc[windows["partition"] == "calibration"].copy()
    if any(
        part.empty
        for part in (train_windows, selection_windows, calibration_windows)
    ):
        raise ValueError(
            "forecast training requires train, selection, and calibration windows"
        )
    window_counts = {
        "train_windows_before_sampling": int(len(train_windows)),
        "selection_windows": int(len(selection_windows)),
        "calibration_windows": int(len(calibration_windows)),
    }
    del windows
    source_order = tuple(sorted(map(str, config["data"]["development_sources"])))
    observed_train_sources = tuple(
        sorted(train_windows["source_id"].astype(str).unique())
    )
    if observed_train_sources != source_order:
        raise ValueError("forecast training split is missing a development source")
    train_keys = set(
        zip(  # noqa: B905
            train_windows["source_id"].astype(str),
            train_windows["participant_id"].astype(str),
        )
    )
    scaler = SequenceScaler.fit(frame, train_keys)
    write_json(scaler.as_dict(), run_dir / "scaler.json")
    dropout_probabilities = _action_dropout_probabilities(
        frame,
        train_keys,
        maximum=float(config["training"].get("feature_group_dropout_max", 0.5)),
    )
    write_json(dropout_probabilities, run_dir / "feature_group_dropout.json")
    sampled_train = sample_training_windows(
        train_windows,
        negative_ratio=float(config["training"]["negative_ratio"]),
        seed=int(config["training"].get("data_sampling_seed", seed)),
    )
    window_counts["train_windows_after_sampling"] = int(len(sampled_train))

    sample_minutes = int(config["protocol"]["sample_minutes"])
    context_samples = int(
        float(config["protocol"]["context_hours"]) * 60 / sample_minutes
    )
    prediction_samples = max(REGRESSION_HORIZONS) // sample_minutes
    source_to_index = {
        source: index for index, source in enumerate(source_order)
    }
    dataset_kwargs = {
        "scaler": scaler,
        "context_samples": context_samples,
        "prediction_samples": prediction_samples,
        "sample_minutes": sample_minutes,
    }
    participant_store = ParticipantSequenceStore.from_frame(frame)
    train_dataset = ForecastWindowSequenceDataset(
        frame,
        sampled_train,
        source_to_index=source_to_index,
        participant_store=participant_store,
        copy_windows=False,
        **dataset_kwargs,
    )
    selection_dataset = ForecastWindowSequenceDataset(
        frame,
        selection_windows,
        participant_store=participant_store,
        copy_windows=False,
        **dataset_kwargs,
    )
    calibration_dataset = ForecastWindowSequenceDataset(
        frame,
        calibration_windows,
        participant_store=participant_store,
        copy_windows=False,
        **dataset_kwargs,
    )
    del (
        frame,
        train_windows,
        sampled_train,
        selection_windows,
        calibration_windows,
    )

    model_config = _patchmlp_config(
        config,
        context_samples=context_samples,
        prediction_samples=prediction_samples,
    )
    model = PatchMLPForecaster(model_config)
    device = _device(config)
    model = model.to(device)
    training = config["training"]
    batch_size = int(training["batch_size"])
    effective_batch_size = int(training.get("effective_batch_size", batch_size))
    if effective_batch_size % batch_size:
        raise ValueError(
            "forecast effective_batch_size must be a batch_size multiple"
        )
    accumulation_steps = effective_batch_size // batch_size
    num_workers = int(training.get("num_workers", 0))
    loader, batch_sampler = _all_source_loader(
        train_dataset,
        source_order=source_order,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=int(training.get("data_sampling_seed", seed)),
    )
    optimizer = Adam(
        model.parameters(),
        lr=float(training.get("forecast_learning_rate", training["learning_rate"])),
        weight_decay=float(training.get("forecast_weight_decay", 0.0)),
    )
    maximum_epochs = int(training["maximum_epochs"])
    total_steps = max(
        1,
        math.ceil(len(loader) / accumulation_steps) * maximum_epochs,
    )
    warmup_steps = max(1, int(round(total_steps * 0.05)))

    def lr_multiplier(step: int) -> float:
        if step < warmup_steps:
            return max(1e-8, step / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = LambdaLR(optimizer, lr_lambda=lr_multiplier)
    best_key = (-float("inf"), -float("inf"), -float("inf"), -float("inf"))
    best_epoch = -1
    patience = 0
    update_step = 0
    first_loss_logged = False
    selection_candidate_count = config["evaluation"].get(
        "selection_operational_candidate_count"
    )
    if selection_candidate_count is not None:
        if not bool(config["project"].get("synthetic", False)):
            raise ValueError(
                "real-data checkpoint proxy requires all unique selection scores"
            )
        selection_candidate_count = int(selection_candidate_count)
    scale = scaler.values["cgm_mg_dl"]

    for epoch in range(maximum_epochs):
        batch_sampler.set_epoch(epoch)
        model.train()
        epoch_losses: list[float] = []
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(loader):
            batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            sequence = _drop_action_groups(
                batch["sequence"],
                probabilities=tuple(
                    dropout_probabilities[field] for field in ACTION_FIELDS
                ),
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                forecast_standardized = model(sequence)["forecast"]
                forecast_mg_dl = inverse_standardized_cgm(
                    forecast_standardized,
                    mean_mg_dl=scale.mean,
                    std_mg_dl=scale.std,
                )
                per_example = _forecast_loss_per_example(
                    loss_name=loss_name,
                    forecast_standardized=forecast_standardized,
                    forecast_mg_dl=forecast_mg_dl,
                    batch=batch,
                )
                loss, source_losses = _source_balanced_loss(
                    per_example,
                    batch["sampling_weight"],
                    batch["source_index"],
                    source_count=len(source_order),
                )
                group_start = (batch_index // accumulation_steps) * accumulation_steps
                group_size = min(accumulation_steps, len(loader) - group_start)
                scaled_loss = loss / group_size
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite forecast loss at epoch={epoch} batch={batch_index}"
                )
            if not first_loss_logged:
                payload = {
                    "event": "first_finite_loss",
                    "epoch": epoch,
                    "batch": batch_index,
                    "loss": float(loss.detach().cpu()),
                    "loss_name": loss_name,
                }
                _append_log(log_path, payload)
                print(json.dumps(payload, sort_keys=True), flush=True)
                first_loss_logged = True
            scaled_loss.backward()
            epoch_losses.append(float(loss.detach().cpu()))
            should_step = (
                (batch_index + 1) % accumulation_steps == 0
                or batch_index + 1 == len(loader)
            )
            if should_step:
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(training["gradient_clip_norm"]),
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                update_step += 1

        selection_predictions = _predict_forecast_dataset(
            model,
            selection_dataset,
            scaler=scaler,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            sample_minutes=sample_minutes,
            severe_hypoglycemia_threshold=float(
                config["protocol"]["thresholds_mg_dl"]["severe_hypoglycemia"]
            ),
        )
        operational_score, tie_fapd, proxy = _checkpoint_proxy(
            selection_predictions,
            source_order=source_order,
            candidate_count=selection_candidate_count,
        )
        window_score = _selection_score(selection_predictions)
        key = (operational_score, window_score, -tie_fapd, -float(epoch))
        epoch_payload = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)),
            "selection_operational_auc": operational_score,
            "selection_macro_window_auprc": window_score,
            "selection_proxy_tie_fapd": tie_fapd,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "last_batch_source_losses": source_losses,
            "update_step": update_step,
        }
        _append_log(log_path, epoch_payload)
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "model_config": model_config.__dict__,
            "architecture": "patchmlp_forecast",
            "loss_name": loss_name,
            "selection_operational_score": operational_score,
            "selection_macro_window_auprc": window_score,
            "selection_proxy_tie_fapd": tie_fapd,
            "selection_operational_proxy": proxy,
            "patchmlp_commit": PATCHMLP_COMMIT,
            "glimmer_commit": GLIMMER_COMMIT,
            "forecast_output_units": "SequenceScaler standardized CGM",
            "clinical_loss_and_alarm_units": "mg/dL after explicit inverse",
        }
        torch.save(checkpoint, checkpoint_dir / "last.pt")
        if key > best_key:
            best_key = key
            best_epoch = epoch
            patience = 0
            best_checkpoint = dict(checkpoint)
            best_checkpoint.pop("optimizer_state")
            best_checkpoint.pop("scheduler_state")
            torch.save(best_checkpoint, checkpoint_dir / "best.pt")
        else:
            patience += 1
            if patience >= int(training["early_stopping_patience"]):
                break

    checkpoint = torch.load(checkpoint_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    selection_predictions = _predict_forecast_dataset(
        model,
        selection_dataset,
        scaler=scaler,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        sample_minutes=sample_minutes,
        severe_hypoglycemia_threshold=float(
            config["protocol"]["thresholds_mg_dl"]["severe_hypoglycemia"]
        ),
    )
    calibration_predictions = _predict_forecast_dataset(
        model,
        calibration_dataset,
        scaler=scaler,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        sample_minutes=sample_minutes,
        severe_hypoglycemia_threshold=float(
            config["protocol"]["thresholds_mg_dl"]["severe_hypoglycemia"]
        ),
    )
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
    parameters = sum(parameter.numel() for parameter in model.parameters())
    model_summary = {
        "status": (
            "synthetic_taskaware_comparator_smoke"
            if bool(config["project"].get("synthetic", False))
            else "development_only_external_untouched"
        ),
        "variant": "taskaware_forecast",
        "architecture": "patchmlp_forecast",
        "forecast_loss": loss_name,
        "parameters": int(parameters),
        "best_epoch": int(best_epoch),
        "best_selection_operational_auc": float(best_key[0]),
        "best_selection_macro_window_auprc": float(best_key[1]),
        "training_seed": seed,
        "split_seed": int(config["split"]["seed"]),
        "tasks": task_summary,
        "upstream": {
            "patchmlp_commit": PATCHMLP_COMMIT,
            "glimmer_commit": GLIMMER_COMMIT,
        },
        "compatibility_status": "information_matched_adapted_comparator",
        "deviations": [
            "frozen 24-channel six-hour input replaces GLIMMER preprocessing",
            "frozen participant split replaces both upstream split procedures",
            "event-refractory alarm adapter replaces flattened point classification",
            "negative subsampling is corrected by within-source sampling weights",
        ],
        "unit_contract": {
            "model_output": "SequenceScaler standardized CGM",
            "loss_and_alarm": "mg/dL after explicit inverse_standardized_cgm",
        },
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
    write_json(metrics, run_dir / "metrics.json")
    write_json(
        {
            "device": str(device),
            "training_seed": seed,
            "split_seed": int(config["split"]["seed"]),
            "parameters": int(parameters),
            "best_epoch": int(best_epoch),
            "best_operational_score": float(best_key[0]),
            **window_counts,
            "source_order": list(source_order),
            "feature_group_dropout_probabilities": dropout_probabilities,
            "external_evaluated": False,
        },
        run_dir / "training_summary.json",
    )
    return run_dir
