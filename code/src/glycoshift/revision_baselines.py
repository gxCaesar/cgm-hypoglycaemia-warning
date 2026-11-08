"""Matched CGM/mask/clock classical controls for the prospective revision."""

from __future__ import annotations

import argparse
import copy
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .calibration import apply_temperature
from .config import load_config, resolve_path, save_config
from .io import write_json
from .pipeline import (
    _continuous_baseline_forecast,
    _predict_probability,
    _risk_score,
    validate_training_output_isolation,
)
from .revision_inputs import _slope
from .revision_runner import load_replay_bundle, revision_config
from .revision_training import RevisionTrainingContext, _causal, _target_mask
from .sequences import REVISION_CONTRACT, TASKS

BASELINES = ("persistence", "linear", "summary_logistic", "xgboost")
CGM_SUMMARY_FEATURES = (
    "cgm_last",
    "cgm_mean",
    "cgm_std",
    "cgm_min",
    "cgm_max",
    "cgm_slope",
    "cgm_slope_30",
)
REVISION_SUMMARY_FEATURES = (
    *CGM_SUMMARY_FEATURES,
    "cgm_observed_fraction",
    "time_of_day_sin",
    "time_of_day_cos",
)


def _boolean(frame: pd.DataFrame, column: str) -> np.ndarray:
    if (
        column not in frame
        or frame[column].isna().any()
        or not (pd.api.types.is_bool_dtype(frame[column].dtype))
    ):
        raise ValueError(f"normalized {column} must be complete boolean provenance")
    return frame[column].to_numpy(dtype=bool)


def build_revision_features(
    normalized: pd.DataFrame,
    windows: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Derive admitted features solely from exact, past-inclusive contexts.

    Action values, action provenance and source availability are never read.
    Clock semantics are identical to the neural input: recorded local clock when
    present, otherwise the existing timestamp clock, with no inferred timezone.
    """
    required = {
        "source_id",
        "participant_id",
        "timestamp",
        "cgm_mg_dl",
        "cgm_observed",
        "cgm_interpolated",
        "cgm_lower_censored",
    }
    if not required.issubset(normalized):
        missing = sorted(required - set(normalized))
        raise ValueError(f"normalized feature context is missing {missing}")
    window_required = {
        "source_id",
        "participant_id",
        "context_start",
        "context_end",
        "participant_end_index",
        *CGM_SUMMARY_FEATURES,
    }
    if windows.empty or not window_required.issubset(windows):
        raise ValueError("revision windows lack complete context bindings or summaries")
    sample = int(config["protocol"]["sample_minutes"])
    context_samples = int(float(config["protocol"]["context_hours"]) * 60 / sample)
    if sample <= 0 or context_samples < 2:
        raise ValueError("invalid classical sample/context specification")
    clock_column = "timestamp_local" if "timestamp_local" in normalized else "timestamp"
    admitted = normalized[sorted(required | {clock_column})]
    normalized_parts = {
        tuple(map(str, key)): part.sort_values("timestamp").reset_index(drop=True)
        for key, part in admitted.groupby(
            ["source_id", "participant_id"], observed=True
        )
    }
    window_groups = windows.groupby(
        ["source_id", "participant_id"], observed=True
    ).indices
    if set(normalized_parts) != {tuple(map(str, key)) for key in window_groups}:
        raise ValueError(
            "normalized contexts and revision participant population differ"
        )
    result = np.full((len(windows), len(REVISION_SUMMARY_FEATURES)), np.nan)
    step_ns = sample * 60_000_000_000
    for key, positions in window_groups.items():
        part = normalized_parts[tuple(map(str, key))]
        times = pd.DatetimeIndex(
            pd.to_datetime(part["timestamp"], utc=True, errors="coerce")
        )
        if times.hasnans or times.has_duplicates:
            raise ValueError("normalized context timestamps must be valid and unique")
        rows = windows.iloc[positions]
        ends = times.get_indexer(pd.to_datetime(rows["context_end"], utc=True))
        supplied = pd.to_numeric(
            rows["participant_end_index"], errors="coerce"
        ).to_numpy()
        if (ends < 0).any() or not np.array_equal(ends, supplied):
            raise ValueError(
                "participant_end_index conflicts with the normalized anchor time"
            )
        starts = ends - context_samples + 1
        if (starts < 0).any() or not np.array_equal(
            times[starts].asi8,
            pd.DatetimeIndex(pd.to_datetime(rows["context_start"], utc=True)).asi8,
        ):
            raise ValueError(
                "context_start differs from the complete normalized context"
            )
        discontinuities = np.r_[0, np.diff(times.asi8) != step_ns].cumsum()
        if np.any(discontinuities[ends] != discontinuities[starts]):
            raise ValueError("classical feature context crosses a time discontinuity")
        observed = _boolean(part, "cgm_observed")
        interpolated = _boolean(part, "cgm_interpolated")
        censored = _boolean(part, "cgm_lower_censored")
        if np.any(observed & interpolated):
            raise ValueError("normalized observed and interpolated flags conflict")
        if not np.all(observed[ends] & ~interpolated[ends] & ~censored[ends]):
            raise ValueError(
                "classical anchors must be observed, noninterpolated and uncensored"
            )
        censored_prefix = np.r_[0, censored.astype(int).cumsum()]
        if np.any(censored_prefix[ends + 1] != censored_prefix[starts]):
            raise ValueError("classical input context contains censored CGM")
        glucose = pd.to_numeric(part["cgm_mg_dl"], errors="coerce").to_numpy(
            dtype=float
        )
        if not np.isfinite(glucose).all():
            raise ValueError("normalized CGM contexts must be finite")
        series = pd.Series(glucose)
        rolling = series.rolling(context_samples)
        summaries = np.column_stack(
            [
                glucose,
                rolling.mean(),
                rolling.std(ddof=0),
                rolling.min(),
                rolling.max(),
                _slope(glucose, context_samples),
                _slope(glucose, min(6, context_samples)),
            ]
        )[ends]
        cached = rows[list(CGM_SUMMARY_FEATURES)].to_numpy(dtype=float)
        if not np.allclose(summaries, cached, rtol=1e-7, atol=1e-8):
            raise ValueError(
                "replay CGM summaries differ from their normalized contexts"
            )
        observed_prefix = np.r_[0, observed.astype(int).cumsum()]
        fraction = (
            observed_prefix[ends + 1] - observed_prefix[starts]
        ) / context_samples
        clock = pd.to_datetime(part[clock_column], errors="coerce")
        if clock.isna().any():
            raise ValueError("recorded context clock is incomplete or invalid")
        minute = clock.dt.hour.to_numpy()[ends] * 60 + clock.dt.minute.to_numpy()[ends]
        phase = 2 * np.pi * minute / 1440
        result[positions] = np.column_stack(
            [summaries, fraction, np.sin(phase), np.cos(phase)]
        )
    if not np.isfinite(result).all():
        raise ValueError(
            "revision classical feature matrix is non-finite or incomplete"
        )
    features = pd.DataFrame(
        result, index=windows.index, columns=REVISION_SUMMARY_FEATURES
    )
    features.attrs["clock_column"] = clock_column
    return features


def _fit_estimator(
    features: pd.DataFrame,
    labels: np.ndarray,
    *,
    baseline: str,
    seed: int,
    workers: int,
) -> tuple[Any, dict[str, Any]]:
    if len(np.unique(labels)) != 2:
        raise ValueError(
            "classical training task requires both actual retained target classes"
        )
    if baseline == "xgboost":
        try:
            from xgboost import XGBClassifier
            from xgboost import __version__ as xgboost_version
        except ImportError as error:
            raise RuntimeError(
                "xgboost baseline requested but xgboost is not installed"
            ) from error
        parameters = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": seed,
            "n_jobs": workers,
        }
        model = XGBClassifier(**parameters)
        model.fit(features, labels)
        rounds = int(model.get_booster().num_boosted_rounds())
        if rounds != 300:
            raise ValueError("XGBoost did not complete its original 300-tree budget")
        return model, {
            "estimator": "XGBClassifier",
            "parameters": parameters,
            "xgboost_version": xgboost_version,
            "fit_status": "fixed_tree_budget_completed",
            "boosted_rounds": rounds,
            "feature_importances": model.feature_importances_.astype(float).tolist(),
        }
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=500,
                    random_state=seed,
                    solver="liblinear",
                ),
            ),
        ]
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(features, labels)
    if any(issubclass(warning.category, ConvergenceWarning) for warning in caught):
        raise ValueError("revision logistic regression failed to converge")
    fitted = model.named_steps["model"]
    iterations = int(np.max(fitted.n_iter_))
    if iterations >= 500:
        raise ValueError(
            "revision logistic regression reached its original iteration limit"
        )
    return model, {
        "estimator": "StandardScaler+LogisticRegression",
        "parameters": {
            "class_weight": "balanced",
            "max_iter": 500,
            "solver": "liblinear",
            "random_state": seed,
        },
        "fit_status": "converged",
        "iterations": iterations,
        "coefficients": fitted.coef_[0].astype(float).tolist(),
        "intercept": float(fitted.intercept_[0]),
        "scaler_mean": model.named_steps["scale"].mean_.astype(float).tolist(),
        "scaler_scale": model.named_steps["scale"].scale_.astype(float).tolist(),
    }


def _task_features(
    windows: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    baseline: str,
    task: str,
    sample_minutes: int,
) -> pd.DataFrame:
    if baseline in {"persistence", "linear"}:
        kind, horizon = task.split("_")
        return pd.DataFrame(
            {
                "risk_score": _risk_score(
                    summary,
                    baseline=baseline,
                    kind=kind,
                    horizon=int(horizon),
                    sample_minutes=sample_minutes,
                )
            },
            index=windows.index,
        )
    return summary


def run_revision_baseline(config: dict, prepared: Path, *, baseline: str) -> Path:
    """Fit only train, calibrate only calibration, and report full causal timelines."""
    if baseline not in BASELINES:
        raise ValueError(f"unsupported revision classical baseline: {baseline}")
    RevisionTrainingContext._config(config)
    if config["model"].get("input_feature_mode") != "cgm_mask_clock":
        raise ValueError(
            "revision baselines require the matched CGM/mask/clock information budget"
        )
    seed = int(config["training"]["seed"])
    if seed not in (17, 23, 42, 71, 101):
        raise ValueError("classical model seed is outside the prospective contract")
    run_dir = resolve_path(config, config["output"]["run_dir"])
    if run_dir is None:
        raise ValueError("output.run_dir is required")
    protected = ("baseline_models.joblib", "model_summary.json", "predictions.parquet")
    if any((run_dir / name).exists() for name in protected):
        raise FileExistsError(
            f"refusing to overwrite an existing revision fit: {run_dir}"
        )
    validate_training_output_isolation(config, run_dir)
    (frame, windows, audit), replays = load_replay_bundle(
        Path(prepared), config, sequences=True
    )
    if frame is None:
        raise ValueError("classical features require the bound normalized contexts")
    source_order = tuple(sorted(map(str, config["data"]["development_sources"])))
    context = RevisionTrainingContext(
        replays,
        source_order=source_order,
        output_dir=run_dir / "fixed-frame",
    )
    _causal(replays["train"].timeline)
    summaries = build_revision_features(frame, windows, config)
    feature_contract = {
        "revision_contract": REVISION_CONTRACT,
        "input_feature_mode": "cgm_mask_clock",
        "allowed_summary_features": list(REVISION_SUMMARY_FEATURES),
        "context_hours": float(config["protocol"]["context_hours"]),
        "sample_minutes": int(config["protocol"]["sample_minutes"]),
        "clock_column": summaries.attrs["clock_column"],
        "clock_semantics": "recorded_clock_without_inferred_timezone",
        "observation_mask": "mean_cgm_observed_in_the_complete_past_inclusive_context",
        "risk_controls": (
            "original_forecast_risk_mapped_by_train_fitted_scaler_and_logistic"
        ),
        "raw_score_transform": "predict_probability_clip_1e-6_then_logit",
        "fit_partition": "train",
        "calibration_partition": "calibration",
        "training_sampling": (
            "all_original_revision_rows_retained_by_each_actual_task_mask"
        ),
        "action_features_used": False,
        "source_availability_features_used": False,
    }
    effective = copy.deepcopy(config)
    effective["model"].update(
        baseline=baseline, class_weight="balanced", maximum_iterations=500
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(effective, run_dir / "config_resolved.yaml")
    write_json(audit, run_dir / "cohort_audit.json")
    write_json(feature_contract, run_dir / "feature_contract.json")
    predictions = {
        partition: replays[partition].timeline.copy(deep=True)
        for partition in ("selection", "calibration")
    }
    fitted_models, fitted_tasks, target_audit = {}, {}, {}
    train_mask = windows["partition"].eq("train").to_numpy()
    train = windows.loc[train_mask]
    if not len(train):
        raise ValueError("revision classical training partition is empty")
    for task in TASKS:
        labels, retained = _target_mask(train, task)
        train_features = _task_features(
            train,
            summaries.loc[train_mask],
            baseline=baseline,
            task=task,
            sample_minutes=int(config["protocol"]["sample_minutes"]),
        ).loc[retained]
        estimator, fit_details = _fit_estimator(
            train_features,
            labels,
            baseline=baseline,
            seed=seed,
            workers=int(config["training"].get("num_workers", 1)),
        )
        columns = train_features.columns.tolist()
        target_audit[task] = {
            "fit_partition": "train",
            "rows_before_task_mask": len(train),
            "retained_count": int(retained.sum()),
            "positive_count": int(labels.sum()),
            "negative_count": int(len(labels) - labels.sum()),
            "training_target_column": f"{task}_training_target",
            "mask_column": f"{task}_training_mask",
            "target_column": f"severe_hypo_{task.split('_')[1]}_label"
            if task.startswith("hypo_")
            else f"{task}_label",
            "by_source": {
                source: {
                    "retained_count": int(
                        (train.loc[retained, "source_id"].astype(str) == source).sum()
                    ),
                    "positive_count": int(
                        labels[
                            train.loc[retained, "source_id"]
                            .astype(str)
                            .eq(source)
                            .to_numpy()
                        ].sum()
                    ),
                }
                for source in source_order
            },
        }
        fitted_models[task] = {
            "model": estimator,
            "features": columns,
            "baseline": baseline,
        }
        fitted_tasks[task] = {"features": columns, **fit_details}
        for partition, prediction in predictions.items():
            part_mask = windows["partition"].eq(partition).to_numpy()
            features = _task_features(
                prediction,
                summaries.loc[part_mask],
                baseline=baseline,
                task=task,
                sample_minutes=int(config["protocol"]["sample_minutes"]),
            )
            probability = (
                _predict_probability(estimator, features)
                if isinstance(estimator, Pipeline)
                else estimator.predict_proba(features)[:, 1]
            )
            clipped = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
            logit = np.log(clipped / (1 - clipped))
            if not np.isfinite(logit).all():
                raise ValueError("classical model produced non-finite scores")
            prediction[f"{task}_logit"] = logit
            prediction[f"{task}_raw_probability"] = clipped
            prediction[f"{task}_probability"] = apply_temperature(logit, 1)
    continuous = baseline in {"persistence", "linear"}
    for prediction in predictions.values():
        for horizon in (30, 60):
            logit = prediction[f"hypo_{horizon}_logit"].to_numpy(dtype=float)
            prediction[f"severe_hypo_{horizon}_raw_logit"] = logit
            prediction[f"severe_hypo_{horizon}_raw_probability"] = apply_temperature(
                logit, 1
            )
            if continuous:
                prediction[f"cgm_{horizon}_prediction"] = _continuous_baseline_forecast(
                    prediction,
                    baseline=baseline,
                    horizon=horizon,
                    sample_minutes=int(config["protocol"]["sample_minutes"]),
                )
    tasks, operational = context.calibrate(
        predictions["selection"],
        predictions["calibration"],
        config=effective,
        source_order=source_order,
    )
    for task in TASKS:
        tasks[task].update(fitted_tasks[task])
    training_targets = {
        "revision_contract": REVISION_CONTRACT,
        "fit_partition": "train",
        "training_sampling": "all_masked_rows_no_negative_subsampling",
        "tasks": target_audit,
    }
    model_summary = {
        "status": "synthetic_plumbing_only"
        if config["project"].get("synthetic", False)
        else "observed_development_revision",
        "baseline": baseline,
        "training_seed": seed,
        "split_seed": int(config["split"]["seed"]),
        "revision_contract": REVISION_CONTRACT,
        "tasks": tasks,
        "continuous_prediction_available": continuous,
        "input_feature_mode": "cgm_mask_clock",
        "feature_contract": feature_contract,
        "training_targets": training_targets,
        "sklearn_version": sklearn.__version__,
        "operational_checkpoint_contract": "not_applicable_no_epochs",
    }
    combined = pd.concat(predictions.values(), ignore_index=True)
    metrics = context.report(combined, config=effective, model_summary=model_summary)
    joblib.dump(fitted_models, run_dir / "baseline_models.joblib")
    combined.to_parquet(run_dir / "predictions.parquet", index=False)
    write_json(model_summary, run_dir / "model_summary.json")
    write_json(training_targets, run_dir / "training_target_audit.json")
    write_json(operational, run_dir / "operational_metrics.json")
    write_json(metrics, run_dir / "metrics.json")
    write_json(
        {
            "status": model_summary["status"],
            "baseline": baseline,
            "training_seed": seed,
            "revision_contract": REVISION_CONTRACT,
            "training_targets": training_targets,
            "training_rows_before_task_masks": len(train),
            "all_original_train_rows_considered": True,
            "negative_subsampling": False,
            "external_evaluated": False,
            "continuous_prediction_available": continuous,
        },
        run_dir / "training_summary.json",
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--baseline", choices=BASELINES, required=True)
    args = parser.parse_args()
    config = revision_config(
        load_config(args.config), seed=args.seed, output=args.output
    )
    run_revision_baseline(config, args.prepared, baseline=args.baseline)


if __name__ == "__main__":
    main()
