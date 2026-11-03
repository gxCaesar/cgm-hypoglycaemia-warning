from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .calibration import apply_temperature
from .io import write_json
from .pipeline import (
    _continuous_baseline_forecast,
    _predict_probability,
    _risk_score,
    evaluate_predictions,
    prepare_evaluation_dataset,
)


def _gate(
    windows: pd.DataFrame,
    *,
    minimum_participants: int,
    minimum_events: int,
) -> dict[str, int]:
    payload = {
        "participants": int(
            windows[["source_id", "participant_id"]].drop_duplicates().shape[0]
        ),
        "hypo_events": int(windows["hypo_60_event_id"].dropna().nunique()),
        "hyper_events": int(windows["hyper_60_event_id"].dropna().nunique()),
    }
    if payload["participants"] < minimum_participants:
        raise ValueError("external participant gate failed")
    if min(payload["hypo_events"], payload["hyper_events"]) < minimum_events:
        raise ValueError("external event gate failed")
    return payload


def run_frozen_baseline_evaluation(
    config: dict[str, Any],
    *,
    run_dir: str | Path,
    partition: str,
) -> Path:
    if partition not in {"external_loop", "stress_hupa_ucm"}:
        raise ValueError("unsupported frozen partition")
    output_dir = Path(run_dir).expanduser().resolve()
    predictions_path = output_dir / f"predictions_{partition}.parquet"
    metrics_path = output_dir / f"metrics_{partition}.json"
    if predictions_path.exists() or metrics_path.exists():
        raise FileExistsError(
            f"refusing to repeat or overwrite frozen evaluation: {partition}"
        )
    model_path = output_dir / "baseline_models.joblib"
    summary_path = output_dir / "model_summary.json"
    if not model_path.exists() or not summary_path.exists():
        raise FileNotFoundError("baseline run is missing frozen model artifacts")
    models = joblib.load(model_path)
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    _, partition_windows, audit = prepare_evaluation_dataset(
        config,
        partition=partition,
    )
    if partition_windows.empty:
        raise ValueError(f"no eligible windows for {partition}")
    if partition == "external_loop":
        gate = _gate(
            partition_windows,
            minimum_participants=int(
                config["evaluation"]["external_minimum_participants"]
            ),
            minimum_events=int(
                config["evaluation"]["external_minimum_events_per_class"]
            ),
        )
    else:
        gate = {
            "participants": int(
                partition_windows[
                    ["source_id", "participant_id"]
                ].drop_duplicates().shape[0]
            ),
            "hypo_events": int(
                partition_windows["hypo_60_event_id"].dropna().nunique()
            ),
            "hyper_events": int(
                partition_windows["hyper_60_event_id"].dropna().nunique()
            ),
        }

    task_frames: list[pd.DataFrame] = []
    sample_minutes = int(config["protocol"]["sample_minutes"])
    for task, artifact in models.items():
        kind, horizon_text = task.split("_")
        horizon = int(horizon_text)
        eligible = partition_windows.loc[
            ~partition_windows[f"{kind}_currently_active"]
        ].copy()
        baseline = str(artifact["baseline"])
        model = artifact["model"]
        if baseline in {"persistence", "linear"}:
            features = pd.DataFrame(
                {
                    "risk_score": _risk_score(
                        eligible,
                        baseline=baseline,
                        kind=kind,
                        horizon=horizon,
                        sample_minutes=sample_minutes,
                    )
                }
            )
        else:
            features = eligible[list(artifact["features"])]
        if isinstance(model, Pipeline):
            probability = _predict_probability(model, features)
        else:
            probability = model.predict_proba(features)[:, 1]
        probability = np.clip(probability, 1e-6, 1 - 1e-6)
        logit = np.log(probability / (1 - probability))
        eligible[f"{task}_probability"] = apply_temperature(
            logit,
            float(summary["tasks"][task]["temperature"]),
        )
        columns = [
            "source_id",
            "participant_id",
            "context_start",
            "context_end",
            "patient_day",
            "partition",
            f"{task}_label",
            f"{task}_event_id",
            f"{task}_event_onset",
            f"{task}_probability",
        ]
        task_frames.append(eligible[columns])

    key_columns = [
        "source_id",
        "participant_id",
        "context_start",
        "context_end",
        "patient_day",
        "partition",
    ]
    predictions = task_frames[0]
    for task_frame in task_frames[1:]:
        values = [column for column in task_frame if column not in key_columns]
        predictions = predictions.merge(
            task_frame[key_columns + values],
            on=key_columns,
            how="outer",
            validate="one_to_one",
        )
    horizons = [
        int(value)
        for value in config["protocol"]["horizons_minutes"]
    ]
    secondary_columns = [
        *key_columns,
        "severe_hypo_currently_active",
        *(
            column
            for horizon in horizons
            for column in (
                f"severe_hypo_{horizon}_label",
                f"severe_hypo_{horizon}_event_id",
                f"severe_hypo_{horizon}_event_onset",
            )
        ),
    ]
    predictions = predictions.merge(
        partition_windows[
            secondary_columns
        ].drop_duplicates(key_columns),
        on=key_columns,
        how="left",
        validate="one_to_one",
    )
    baseline_name = str(summary["baseline"])
    if baseline_name in {"persistence", "linear"}:
        regression_columns = [
            *key_columns,
            "cgm_last",
            "cgm_slope_30",
            *(
                column
                for horizon in horizons
                for column in (
                    f"cgm_{horizon}_target",
                    f"cgm_{horizon}_target_available",
                )
            ),
        ]
        regression = partition_windows[
            regression_columns
        ].drop_duplicates(key_columns)
        predictions = predictions.merge(
            regression,
            on=key_columns,
            how="left",
            validate="one_to_one",
        )
        for horizon in horizons:
            predictions[f"cgm_{horizon}_prediction"] = (
                _continuous_baseline_forecast(
                    predictions,
                    baseline=baseline_name,
                    horizon=horizon,
                    sample_minutes=sample_minutes,
                )
            )
    predictions.to_parquet(predictions_path, index=False)
    metrics = evaluate_predictions(
        predictions,
        config=config,
        model_summary=summary,
    )
    metrics["external_gate"] = gate
    metrics["cohort_audit"] = audit
    write_json(metrics, metrics_path)
    return metrics_path
