from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .calibration import apply_temperature, fit_conformal_threshold, fit_temperature
from .config import resolve_path, save_config
from .io import load_evaluation_frame, load_frame, write_json
from .metrics import (
    alarm_metrics,
    conformal_classification_metrics,
    event_average_precision,
    probability_metrics,
    regression_metrics,
    select_alarm_threshold,
    selective_risk_curve,
)
from .schema import validate_frame
from .split import participant_split
from .streaming import load_validated_window_shards
from .windows import FEATURE_COLUMNS, build_windows


def _predict_probability(
    pipeline: Pipeline,
    features: pd.DataFrame,
) -> np.ndarray:
    """Compute stable binary probabilities without depending on sparse dispatch."""
    transformed = np.ascontiguousarray(
        pipeline.named_steps["scale"].transform(features),
        dtype=np.float64,
    )
    model = pipeline.named_steps["model"]
    coefficients = np.ascontiguousarray(model.coef_[0], dtype=np.float64)
    logits = np.sum(
        transformed * coefficients[np.newaxis, :],
        axis=1,
        dtype=np.float64,
    )
    logits += float(model.intercept_[0])
    if not np.isfinite(logits).all():
        raise ValueError("non-finite logistic-regression logits")
    logits = np.clip(logits, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-logits))


def _partition_windows(
    windows: pd.DataFrame,
    partitions: dict[str, set[tuple[str, str]]],
) -> pd.DataFrame:
    lookup = {
        key: partition
        for partition, keys in partitions.items()
        for key in keys
    }
    result = windows.copy()
    result["partition"] = [
        lookup.get((str(source), str(participant)), "excluded")
        for source, participant in zip(  # noqa: B905
            result["source_id"], result["participant_id"]
        )
    ]
    return result.loc[result["partition"] != "excluded"].reset_index(drop=True)


def build_audit(
    frame: pd.DataFrame,
    windows: pd.DataFrame | None,
    partitions: dict[str, set[tuple[str, str]]] | None,
) -> dict[str, Any]:
    frame_audit = validate_frame(frame).as_dict()
    payload: dict[str, Any] = {"frame": frame_audit}
    if windows is not None:
        payload["windows"] = {
            "rows": len(windows),
            "by_source": {
                str(key): int(value)
                for key, value in windows["source_id"].value_counts().items()
            },
        }
    if partitions is not None:
        payload["partitions"] = {
            name: {
                "participants": len(keys),
                "sources": sorted({source for source, _ in keys}),
            }
            for name, keys in partitions.items()
        }
    return payload


def prepare_dataset(
    config: dict[str, Any],
    *,
    include_holdouts: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    synthetic = bool(config["project"].get("synthetic", False))
    if include_holdouts and not synthetic:
        raise ValueError(
            "real held-out sources require the isolated frozen-evaluation path"
        )
    development_sources = set(
        map(str, config["data"]["development_sources"])
    )
    frame = load_frame(
        config,
        sources=None if include_holdouts and synthetic else development_sources,
    )
    split_config = config["split"]
    split_kwargs: dict[str, Any]
    if "validation_fraction" in split_config:
        split_kwargs = {
            "validation_fraction": float(split_config["validation_fraction"]),
        }
    else:
        split_kwargs = {
            "train_fraction": float(split_config["train_fraction"]),
            "selection_fraction": float(split_config["selection_fraction"]),
            "calibration_fraction": float(split_config["calibration_fraction"]),
        }
    partitions = participant_split(
        frame,
        development_sources=list(config["data"]["development_sources"]),
        external_source=str(config["data"]["external_source"]),
        forbidden_development_sources=list(
            config["data"]["forbidden_development_sources"]
        ),
        stress_sources=list(config["data"].get("stress_sources", [])),
        include_holdouts=include_holdouts,
        seed=int(config["split"]["seed"]),
        **split_kwargs,
    )
    window_frame = frame
    if not include_holdouts:
        development_keys = set().union(
            *(
                keys
                for name, keys in partitions.items()
                if name in {"train", "selection", "calibration", "validation"}
            )
        )
        row_keys = list(
            zip(  # noqa: B905
                frame["source_id"].astype(str),
                frame["participant_id"].astype(str),
            )
        )
        window_frame = frame.loc[
            [key in development_keys for key in row_keys]
        ].copy()
    windows = _partition_windows(build_windows(window_frame, config), partitions)
    audit = build_audit(frame, windows, partitions)
    development = windows.loc[
        windows["partition"].isin(
            {"train", "selection", "calibration", "validation"}
        )
    ]
    prevalence: dict[str, dict[str, float | int]] = {}
    minimum_prevalence = float(
        config["evaluation"].get("event_prevalence_minimum", 0.0001)
    )
    maximum_prevalence = float(
        config["evaluation"].get("event_prevalence_maximum", 0.30)
    )
    for kind in ("hypo", "hyper"):
        eligible = development.loc[
            ~development[f"{kind}_currently_active"].astype(bool)
        ]
        for horizon in config["protocol"]["horizons_minutes"]:
            task = f"{kind}_{int(horizon)}"
            labels = eligible[f"{task}_label"].dropna().astype(int)
            value = float(labels.mean()) if len(labels) else float("nan")
            prevalence[task] = {
                "eligible_windows": int(len(labels)),
                "positive_windows": int(labels.sum()),
                "prevalence": value,
            }
            if (
                not np.isfinite(value)
                or value < minimum_prevalence
                or value > maximum_prevalence
            ):
                raise ValueError(
                    f"development event prevalence outside frozen bounds for "
                    f"{task}: {value}"
                )
    audit["development_event_prevalence"] = prevalence
    return frame, windows, audit


def load_audited_development_windows(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if bool(config["project"].get("synthetic", False)):
        _, windows, audit = prepare_dataset(
            config,
            include_holdouts=False,
        )
        return windows, audit
    shard_path = resolve_path(
        config,
        config["data"].get("training_window_shards"),
    )
    audit_path = resolve_path(
        config,
        config["data"].get("training_cohort_audit"),
    )
    if shard_path is None or audit_path is None:
        raise ValueError(
            "real training requires data.training_window_shards and "
            "data.training_cohort_audit"
        )
    return load_validated_window_shards(
        shard_path,
        audit_path,
        config=config,
    )


def validate_training_output_isolation(
    config: dict[str, Any],
    run_dir: Path,
) -> None:
    if bool(config["project"].get("synthetic", False)):
        return
    audit_path = resolve_path(
        config,
        config["data"].get("training_cohort_audit"),
    )
    if audit_path is None:
        raise ValueError("real training requires data.training_cohort_audit")
    if run_dir.resolve() == audit_path.parent.resolve():
        raise ValueError(
            "real training output must not reuse the preprocessing artifact "
            "directory"
        )


def _evaluation_source(config: dict[str, Any], partition: str) -> str:
    if partition == "external_loop":
        return str(config["data"]["external_source"])
    for source in map(str, config["data"].get("stress_sources", [])):
        source_partition = (
            f"stress_{source.lower().replace('-', '_').replace(' ', '_')}"
        )
        if source_partition == partition:
            return source
    raise ValueError(f"unsupported frozen evaluation partition: {partition}")


def prepare_evaluation_dataset(
    config: dict[str, Any],
    *,
    partition: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    expected_source = _evaluation_source(config, partition)
    if bool(config["project"].get("synthetic", False)):
        combined_frame, combined_windows, _ = prepare_dataset(
            config,
            include_holdouts=True,
        )
        frame = combined_frame.loc[
            combined_frame["source_id"].astype(str) == expected_source
        ].copy()
        windows = combined_windows.loc[
            combined_windows["partition"] == partition
        ].copy()
    else:
        frame = load_evaluation_frame(config, partition)
        windows = build_windows(frame, config)
        windows["partition"] = partition
    if frame.empty or windows.empty:
        raise ValueError(f"no eligible data for {partition}")
    participant_keys = set(
        zip(  # noqa: B905
            frame["source_id"].astype(str),
            frame["participant_id"].astype(str),
        )
    )
    audit = build_audit(
        frame,
        windows,
        {partition: participant_keys},
    )
    return frame, windows, audit


def _risk_score(
    frame: pd.DataFrame,
    *,
    baseline: str,
    kind: str,
    horizon: int,
    sample_minutes: int,
) -> np.ndarray:
    if baseline == "persistence":
        forecast = frame["cgm_last"].to_numpy(dtype=float)
    elif baseline == "linear":
        forecast = (
            frame["cgm_last"].to_numpy(dtype=float)
            + frame["cgm_slope_30"].to_numpy(dtype=float)
            * (horizon // sample_minutes)
        )
    else:
        raise ValueError(f"unsupported naïve baseline: {baseline}")
    return -forecast if kind == "hypo" else forecast


def _continuous_baseline_forecast(
    frame: pd.DataFrame,
    *,
    baseline: str,
    horizon: int,
    sample_minutes: int,
) -> np.ndarray:
    if baseline == "persistence":
        return frame["cgm_last"].to_numpy(dtype=float)
    if baseline == "linear":
        return (
            frame["cgm_last"].to_numpy(dtype=float)
            + frame["cgm_slope_30"].to_numpy(dtype=float)
            * (horizon // sample_minutes)
        )
    raise ValueError(f"unsupported continuous baseline: {baseline}")


def run_training(
    config: dict[str, Any],
    *,
    baseline_name: str | None = None,
) -> Path:
    include_holdouts = bool(config["project"].get("synthetic", False))
    baseline_name = baseline_name or str(config["model"]["baseline"])
    training_seed = int(
        config.get("training", {}).get("seed", config["split"]["seed"])
    )
    run_dir = resolve_path(config, config["output"]["run_dir"])
    if run_dir is None:
        raise ValueError("output.run_dir is required")
    protected_artifacts = (
        run_dir / "model_summary.json",
        run_dir / "predictions.parquet",
        run_dir / "baseline_models.joblib",
    )
    if (
        not bool(config["project"].get("synthetic", False))
        and any(path.exists() for path in protected_artifacts)
    ):
        raise FileExistsError(
            f"refusing to overwrite an existing real-data run: {run_dir}"
        )
    if include_holdouts:
        _, windows, audit = prepare_dataset(
            config,
            include_holdouts=True,
        )
    else:
        windows, audit = load_audited_development_windows(config)
    validate_training_output_isolation(config, run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, run_dir / "config_resolved.yaml")
    write_json(audit, run_dir / "cohort_audit.json")

    horizons = [int(value) for value in config["protocol"]["horizons_minutes"]]
    source_order = tuple(
        sorted(map(str, config["data"]["development_sources"]))
    )
    constant_controls: tuple[
        dict[str, float],
        dict[str, dict[str, float]],
        dict[str, Any],
    ] | None = None
    if baseline_name in {
        "constant_global_train_prevalence",
        "constant_source_conditional_train_prevalence",
    }:
        from .pre_freeze_sanity import train_prevalence_controls

        constant_controls = train_prevalence_controls(
            windows.loc[windows["partition"] == "train"],
            source_order=source_order,
        )
    label_shuffle_ledger: dict[str, Any] | None = None
    if baseline_name == "label_shuffle_logistic":
        label_shuffle_ledger = {
            "schema_version": "1.0",
            "protocol_id": "glycoshift-sota-method-v1.3",
            "control": "source_stratified_label_shuffle_logistic",
            "root_seed": 20260806,
            "model_random_state": 42,
            "source_order": list(source_order),
            "tasks": {},
        }
    predictions: list[pd.DataFrame] = []
    model_summaries: dict[str, Any] = {}
    fitted_models: dict[str, dict[str, Any]] = {}
    for kind in ("hypo", "hyper"):
        eligible = windows.loc[~windows[f"{kind}_currently_active"]].copy()
        for horizon in horizons:
            task = f"{kind}_{horizon}"
            label_column = f"{task}_label"
            event_id_column = f"{task}_event_id"
            train = eligible.loc[eligible["partition"] == "train"]
            if train[label_column].nunique() < 2:
                raise ValueError(f"training task has one class only: {task}")
            if constant_controls is not None:
                global_prevalence, source_prevalence, _ = constant_controls
                feature_columns = []
                pipeline = {
                    "kind": baseline_name,
                    "global_probability": global_prevalence[task],
                    "source_probabilities": {
                        source: source_prevalence[source][task]
                        for source in source_order
                    },
                    "unknown_source_policy": "global_train_prevalence",
                }
            elif baseline_name in {"persistence", "linear"}:
                train_features = pd.DataFrame(
                    {
                        "risk_score": _risk_score(
                            train,
                            baseline=baseline_name,
                            kind=kind,
                            horizon=horizon,
                            sample_minutes=int(
                                config["protocol"]["sample_minutes"]
                            ),
                        )
                    }
                )
                feature_columns = ["risk_score"]
                pipeline = Pipeline(
                    [
                        ("scale", StandardScaler()),
                        (
                            "model",
                            LogisticRegression(
                                class_weight=config["model"]["class_weight"],
                                max_iter=int(
                                    config["model"]["maximum_iterations"]
                                ),
                                random_state=training_seed,
                                solver="liblinear",
                            ),
                        ),
                    ]
                )
                pipeline.fit(train_features, train[label_column])
            elif baseline_name == "xgboost":
                try:
                    from xgboost import XGBClassifier
                except ImportError as error:
                    raise RuntimeError(
                        "xgboost baseline requested but xgboost is not installed"
                    ) from error
                feature_columns = list(FEATURE_COLUMNS)
                pipeline = XGBClassifier(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=training_seed,
                    n_jobs=int(config["training"].get("num_workers", 1)),
                )
                pipeline.fit(train[feature_columns], train[label_column])
            elif baseline_name in {"logistic_regression", "label_shuffle_logistic"}:
                feature_columns = list(FEATURE_COLUMNS)
                pipeline = Pipeline(
                    [
                        ("scale", StandardScaler()),
                        (
                            "model",
                            LogisticRegression(
                                class_weight=config["model"]["class_weight"],
                                max_iter=int(
                                    config["model"]["maximum_iterations"]
                                ),
                                random_state=(
                                    42
                                    if baseline_name == "label_shuffle_logistic"
                                    else training_seed
                                ),
                                solver="liblinear",
                            ),
                        ),
                    ]
                )
                fit_labels = train[label_column]
                if baseline_name == "label_shuffle_logistic":
                    from .pre_freeze_sanity import (
                        _shuffled_labels,
                        derive_label_shuffle_seed,
                    )

                    shuffle_seed = derive_label_shuffle_seed(
                        task,
                        root_seed=20260806,
                    )
                    fit_labels, shuffle_audit = _shuffled_labels(
                        train,
                        task=task,
                        source_order=source_order,
                        seed=shuffle_seed,
                    )
                    if fit_labels.nunique() != 2:
                        raise ValueError(
                            f"shuffled training task has one class only: {task}"
                        )
                    assert label_shuffle_ledger is not None
                    label_shuffle_ledger["tasks"][task] = {
                        "seed": shuffle_seed,
                        "fit_eligible_windows": int(len(train)),
                        "fit_positive_windows_before": int(
                            train[label_column].astype(int).sum()
                        ),
                        "fit_positive_windows_after": int(fit_labels.sum()),
                        "by_source": shuffle_audit,
                    }
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    pipeline.fit(train[feature_columns], fit_labels)
                convergence = [
                    warning
                    for warning in caught
                    if issubclass(warning.category, ConvergenceWarning)
                ]
                if convergence:
                    raise ValueError(
                        f"summary logistic did not converge for {task}"
                    )
                if baseline_name == "label_shuffle_logistic":
                    model = pipeline.named_steps["model"]
                    iterations = int(np.max(model.n_iter_))
                    if iterations >= int(config["model"]["maximum_iterations"]):
                        raise ValueError(
                            f"label-shuffle logistic reached max_iter for {task}"
                        )
                    assert label_shuffle_ledger is not None
                    label_shuffle_ledger["tasks"][task][
                        "liblinear_iterations"
                    ] = iterations
            else:
                raise ValueError(f"unknown summary baseline: {baseline_name}")

            task_frames: list[pd.DataFrame] = []
            active_column = f"{kind}_currently_active"
            prediction_partitions = [
                partition
                for partition in sorted(windows["partition"].unique())
                if partition != "train"
            ]
            for partition in prediction_partitions:
                # Score every decision row. Ordinary-task evaluation and
                # calibration exclude its active state below, while the severe
                # endpoint excludes only severe-active rows and therefore needs
                # scores for the 54--70 mg/dL band as well.
                part = windows.loc[windows["partition"] == partition].copy()
                if baseline_name in {"persistence", "linear"}:
                    prediction_features = pd.DataFrame(
                        {
                            "risk_score": _risk_score(
                                part,
                                baseline=baseline_name,
                                kind=kind,
                                horizon=horizon,
                                sample_minutes=int(
                                    config["protocol"]["sample_minutes"]
                                ),
                            )
                        }
                    )
                else:
                    prediction_features = part[feature_columns]
                if constant_controls is not None:
                    if baseline_name == "constant_global_train_prevalence":
                        values = np.full(
                            len(part),
                            pipeline["global_probability"],
                            dtype=float,
                        )
                    else:
                        values = (
                            part["source_id"]
                            .astype(str)
                            .map(pipeline["source_probabilities"])
                            .fillna(pipeline["global_probability"])
                            .to_numpy(dtype=float)
                        )
                    if not np.isfinite(values).all():
                        raise ValueError(
                            "constant control encountered an unknown source"
                        )
                    part[f"{task}_probability"] = values
                elif isinstance(pipeline, Pipeline):
                    part[f"{task}_probability"] = _predict_probability(
                        pipeline,
                        prediction_features,
                    )
                else:
                    part[f"{task}_probability"] = pipeline.predict_proba(
                        prediction_features
                    )[:, 1]
                keep_columns = [
                    "source_id",
                    "participant_id",
                    "context_start",
                    "context_end",
                    "patient_day",
                    "partition",
                    active_column,
                    label_column,
                    event_id_column,
                    f"{task}_event_onset",
                    f"{task}_probability",
                ]
                task_frames.append(part[keep_columns])
            task_predictions = pd.concat(task_frames, ignore_index=True)

            threshold_partition = (
                "calibration"
                if "calibration" in set(task_predictions["partition"])
                else "selection"
            )
            calibration = task_predictions.loc[
                (task_predictions["partition"] == threshold_partition)
                & ~task_predictions[active_column].astype(bool)
            ]
            raw_probability = np.clip(
                task_predictions[f"{task}_probability"].to_numpy(dtype=float),
                1e-6,
                1 - 1e-6,
            )
            raw_logit = np.log(raw_probability / (1 - raw_probability))
            task_predictions[f"{task}_raw_probability"] = raw_probability
            task_predictions[f"{task}_raw_logit"] = raw_logit
            calibration_probability = np.clip(
                calibration[f"{task}_probability"].to_numpy(dtype=float),
                1e-6,
                1 - 1e-6,
            )
            calibration_logit = np.log(
                calibration_probability / (1 - calibration_probability)
            )
            temperature = fit_temperature(
                calibration[label_column].to_numpy(dtype=int),
                calibration_logit,
            )
            task_predictions[f"{task}_probability"] = apply_temperature(
                raw_logit,
                temperature,
            )
            calibration = task_predictions.loc[
                (task_predictions["partition"] == threshold_partition)
                & ~task_predictions[active_column].astype(bool)
            ]
            conformal_threshold = fit_conformal_threshold(
                calibration[label_column].to_numpy(dtype=int),
                calibration[f"{task}_probability"].to_numpy(dtype=float),
                alpha=0.1,
            )
            thresholds: dict[str, float] = {}
            for target in config["evaluation"]["false_alarms_per_patient_day"]:
                thresholds[str(target)] = select_alarm_threshold(
                    calibration,
                    probability_column=f"{task}_probability",
                    label_column=label_column,
                    event_id_column=event_id_column,
                    event_onset_column=f"{task}_event_onset",
                    horizon_minutes=horizon,
                    target_false_alarms_per_day=float(target),
                    candidate_count=int(
                        config["evaluation"].get(
                            "threshold_candidate_count",
                            2001,
                        )
                    ),
                    workers=int(
                        config["evaluation"].get("alarm_sweep_workers", 1)
                    ),
                )
            model_summaries[task] = {
                "baseline": baseline_name,
                "features": feature_columns,
                "threshold_partition": threshold_partition,
                "temperature": temperature,
                "conformal_nonconformity_threshold": conformal_threshold,
                "thresholds_from_calibration": thresholds,
                "thresholds_from_validation": thresholds,
                "coefficient_l2_norm": (
                    float(np.linalg.norm(pipeline.named_steps["model"].coef_))
                    if isinstance(pipeline, Pipeline)
                    else None
                ),
            }
            fitted_models[task] = {
                "model": pipeline,
                "features": feature_columns,
                "baseline": baseline_name,
            }
            predictions.append(task_predictions.drop(columns=[active_column]))

    merged = predictions[0]
    key_columns = [
        "source_id",
        "participant_id",
        "context_start",
        "context_end",
        "patient_day",
        "partition",
    ]
    for task_frame in predictions[1:]:
        value_columns = [
            column for column in task_frame if column not in key_columns
        ]
        merged = merged.merge(
            task_frame[key_columns + value_columns],
            on=key_columns,
            how="outer",
            validate="one_to_one",
        )
    secondary_columns = [
        *key_columns,
        "hypo_currently_active",
        "hyper_currently_active",
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
    secondary_lookup = windows[
        secondary_columns
    ].drop_duplicates(key_columns)
    merged = merged.merge(
        secondary_lookup,
        on=key_columns,
        how="left",
        validate="one_to_one",
    )
    if baseline_name in {"persistence", "linear"}:
        regression_lookup_columns = [
            *key_columns,
            "cgm_last",
            "cgm_slope_30",
            *(
                column
                for horizon in horizons
                for column in (
                    f"cgm_{int(horizon)}_target",
                    f"cgm_{int(horizon)}_target_available",
                )
            ),
        ]
        regression_lookup = windows[
            regression_lookup_columns
        ].drop_duplicates(key_columns)
        merged = merged.merge(
            regression_lookup,
            on=key_columns,
            how="left",
            validate="one_to_one",
        )
        for horizon in horizons:
            merged[f"cgm_{horizon}_prediction"] = (
                _continuous_baseline_forecast(
                    merged,
                    baseline=baseline_name,
                    horizon=horizon,
                    sample_minutes=int(config["protocol"]["sample_minutes"]),
                )
            )
    for horizon in horizons:
        raw_logit = pd.to_numeric(
            merged[f"hypo_{horizon}_raw_logit"], errors="coerce"
        ).to_numpy(dtype=float)
        development_mask = merged["partition"].isin(
            {"selection", "calibration"}
        ).to_numpy()
        if not np.isfinite(raw_logit[development_mask]).all():
            raise ValueError("summary baseline severe-event scores are non-finite")
        merged[f"severe_hypo_{horizon}_raw_logit"] = raw_logit
        merged[f"severe_hypo_{horizon}_raw_probability"] = 1.0 / (
            1.0 + np.exp(-np.clip(raw_logit, -50.0, 50.0))
        )
    selection_predictions = merged.loc[
        merged["partition"] == "selection"
    ].copy()
    calibration_predictions = merged.loc[
        merged["partition"] == "calibration"
    ].copy()
    if selection_predictions.empty or calibration_predictions.empty:
        raise ValueError(
            "summary baseline operational evaluation requires selection and calibration"
        )
    # Local import avoids a module cycle because forecast_training uses the
    # shared summary-baseline evaluator for its legacy secondary metrics.
    from .forecast_training import _calibrate_operational_endpoint

    operational = _calibrate_operational_endpoint(
        selection_predictions,
        calibration_predictions,
        config=config,
        source_order=source_order,
    )
    for current_predictions in (
        selection_predictions,
        calibration_predictions,
    ):
        for horizon in horizons:
            column = f"severe_hypo_{horizon}_probability"
            merged.loc[current_predictions.index, column] = current_predictions[
                column
            ]
    merged.to_parquet(run_dir / "predictions.parquet", index=False)
    joblib.dump(fitted_models, run_dir / "baseline_models.joblib")
    model_summary = {
        "status": (
            "synthetic_plumbing_only"
            if bool(config["project"].get("synthetic", False))
            else "development_baseline"
        ),
        "baseline": baseline_name,
        "training_seed": training_seed,
        "split_seed": int(config["split"]["seed"]),
        "tasks": model_summaries,
        "operational_checkpoint_contract": "not_applicable_no_epochs",
    }
    write_json(model_summary, run_dir / "model_summary.json")
    if constant_controls is not None:
        write_json(constant_controls[2], run_dir / "constant_denominators.json")
    if label_shuffle_ledger is not None:
        write_json(label_shuffle_ledger, run_dir / "label_shuffle_ledger.json")
    write_json(operational, run_dir / "operational_metrics.json")
    metrics = evaluate_predictions(
        merged,
        config=config,
        model_summary=model_summary,
    )
    metrics["operational_v1_3"] = operational
    metrics["legacy_pooled_evaluator_reportable"] = False
    write_json(metrics, run_dir / "metrics.json")
    return run_dir


def evaluate_predictions(
    predictions: pd.DataFrame,
    *,
    config: dict[str, Any],
    model_summary: dict[str, Any],
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "status": "synthetic_plumbing_only"
        if bool(config["project"].get("synthetic", False))
        else "observed",
        "partitions": {},
    }
    horizons = [int(value) for value in config["protocol"]["horizons_minutes"]]
    for partition in sorted(predictions["partition"].dropna().unique()):
        part = predictions.loc[predictions["partition"] == partition]
        partition_metrics: dict[str, Any] = {}
        for kind in ("hypo", "hyper"):
            for horizon in horizons:
                task = f"{kind}_{horizon}"
                label_column = f"{task}_label"
                event_id_column = f"{task}_event_id"
                event_onset_column = f"{task}_event_onset"
                probability_column = f"{task}_probability"
                if f"{kind}_currently_active" in part:
                    part_for_task = part.loc[
                        ~part[f"{kind}_currently_active"].astype(bool)
                    ]
                else:
                    part_for_task = part
                task_part = part_for_task.dropna(
                    subset=[label_column, probability_column]
                ).copy()
                metrics = probability_metrics(
                    task_part[label_column],
                    task_part[probability_column],
                    ece_bins=int(config["evaluation"]["ece_bins"]),
                )
                metrics["event_auprc"] = event_average_precision(
                    task_part,
                    probability_column=probability_column,
                    label_column=label_column,
                    event_id_column=event_id_column,
                    event_onset_column=event_onset_column,
                    horizon_minutes=horizon,
                    candidate_count=int(
                        config["evaluation"].get(
                            "event_ap_candidate_count",
                            201,
                        )
                    ),
                    workers=int(
                        config["evaluation"].get("alarm_sweep_workers", 1)
                    ),
                )
                metrics["alarm_operating_points"] = {}
                threshold_config = model_summary["tasks"][task]
                metrics["risk_coverage"] = selective_risk_curve(
                    task_part[label_column],
                    task_part[probability_column],
                )
                conformal_threshold = threshold_config.get(
                    "conformal_nonconformity_threshold"
                )
                if conformal_threshold is not None:
                    metrics["conformal"] = conformal_classification_metrics(
                        task_part[label_column],
                        task_part[probability_column],
                        threshold=float(conformal_threshold),
                    )
                thresholds = threshold_config.get(
                    "thresholds_from_calibration",
                    threshold_config["thresholds_from_validation"],
                )
                for target, threshold in thresholds.items():
                    metrics["alarm_operating_points"][target] = {
                        "threshold": float(threshold),
                        **alarm_metrics(
                            task_part,
                            probability_column=probability_column,
                            label_column=label_column,
                            event_id_column=event_id_column,
                            event_onset_column=event_onset_column,
                            horizon_minutes=horizon,
                            threshold=float(threshold),
                        ),
                    }
                partition_metrics[task] = metrics
        for horizon in horizons:
            task = f"severe_hypo_{horizon}"
            label_column = f"{task}_label"
            event_id_column = f"{task}_event_id"
            event_onset_column = f"{task}_event_onset"
            probability_column = f"hypo_{horizon}_probability"
            required = {
                label_column,
                event_id_column,
                event_onset_column,
                probability_column,
                "severe_hypo_currently_active",
            }
            if not required.issubset(part.columns):
                continue
            task_part = part.loc[
                ~part["severe_hypo_currently_active"].fillna(False).astype(bool)
            ].dropna(
                subset=[label_column, probability_column]
            )
            metrics = probability_metrics(
                task_part[label_column],
                task_part[probability_column],
                ece_bins=int(config["evaluation"]["ece_bins"]),
            )
            metrics["probability_head"] = f"hypo_{horizon}"
            metrics["event_auprc"] = event_average_precision(
                task_part,
                probability_column=probability_column,
                label_column=label_column,
                event_id_column=event_id_column,
                event_onset_column=event_onset_column,
                horizon_minutes=horizon,
                candidate_count=int(
                    config["evaluation"].get(
                        "event_ap_candidate_count",
                        201,
                    )
                ),
                workers=int(config["evaluation"].get("alarm_sweep_workers", 1)),
            )
            primary_config = model_summary["tasks"][f"hypo_{horizon}"]
            metrics["alarm_operating_points"] = {}
            thresholds = primary_config.get(
                "thresholds_from_calibration",
                primary_config["thresholds_from_validation"],
            )
            for target, threshold in thresholds.items():
                metrics["alarm_operating_points"][target] = {
                    "threshold": float(threshold),
                    **alarm_metrics(
                        task_part,
                        probability_column=probability_column,
                        label_column=label_column,
                        event_id_column=event_id_column,
                        event_onset_column=event_onset_column,
                        horizon_minutes=horizon,
                        threshold=float(threshold),
                    ),
                }
            partition_metrics[task] = metrics
        regression_output: dict[str, dict[str, float]] = {}
        for horizon in horizons:
            target_column = f"cgm_{horizon}_target"
            prediction_column = f"cgm_{horizon}_prediction"
            if target_column in part and prediction_column in part:
                target_available_column = (
                    f"cgm_{horizon}_target_available"
                )
                available = (
                    part[target_available_column].fillna(False).astype(bool)
                    if target_available_column in part
                    else part[target_column].notna()
                )
                regression_part = part.loc[available].dropna(
                    subset=[target_column, prediction_column]
                )
                if not regression_part.empty:
                    regression_output[str(horizon)] = regression_metrics(
                        regression_part[target_column],
                        regression_part[prediction_column],
                    )
        if regression_output:
            partition_metrics["continuous_forecast"] = regression_output
        output["partitions"][partition] = partition_metrics
    return output
