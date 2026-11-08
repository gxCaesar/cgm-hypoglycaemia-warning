"""Explicit fixed-frame callbacks for the dated, target-aligned training path."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from .calibration import apply_temperature, fit_temperature
from .fixed_frame_evaluation import (
    FixedFrameAlarmPolicy,
    FixedFrameEvaluation,
    FixedFrameSpec,
    _validated_inputs,
    evaluate_fixed_frame,
    selection_fixed_frame_curve,
)
from .io import write_json
from .metrics import probability_metrics, regression_metrics
from .revision_inputs import ReplayInputs
from .sequences import REGRESSION_HORIZONS, REVISION_CONTRACT, TASKS
from .target_alignment import severe_source_logit_column

_PARTITIONS = ("selection", "calibration")
_KEYS = ["source_id", "participant_id", "context_end"]
_FLAGS = ("anchor_cgm_observed", "anchor_cgm_interpolated", "anchor_cgm_censored")
_BUDGETS = (0.5, 1.0)


@dataclass(frozen=True)
class _FrozenTable:
    columns: tuple[str, ...]
    records: tuple[tuple, ...]

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> _FrozenTable:
        return cls(
            tuple(frame.columns), tuple(frame.itertuples(index=False, name=None))
        )

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.records, columns=self.columns)


def _boolean(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame:
        raise ValueError(f"revision predictions are missing {column}")
    values = frame[column]
    if values.isna().any() or not pd.api.types.is_bool_dtype(values.dtype):
        raise ValueError(f"{column} must contain complete boolean values")
    return values.to_numpy(dtype=bool)


def _finite(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame:
        raise ValueError(f"revision predictions are missing {column}")
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"revision {column} must be finite")
    return values


def _decision_index(frame: pd.DataFrame) -> pd.MultiIndex:
    if not set(_KEYS).issubset(frame):
        raise ValueError("revision decision identity columns are missing")
    for column in _KEYS[:2]:
        if (
            frame[column].isna().any()
            or frame[column].astype(str).str.strip().eq("").any()
        ):
            raise ValueError("revision decision identifiers must be complete")
    times = pd.to_datetime(frame["context_end"], utc=True, errors="coerce")
    if times.isna().any():
        raise ValueError("revision decision times must be valid")
    index = pd.MultiIndex.from_arrays(
        [frame["source_id"].astype(str), frame["participant_id"].astype(str), times],
        names=_KEYS,
    )
    if not index.is_unique:
        raise ValueError("revision decision keys must be unique")
    return index


def _target_mask(frame: pd.DataFrame, task: str) -> tuple[np.ndarray, np.ndarray]:
    kind, horizon_text = task.split("_")
    base = _boolean(frame, f"base_evaluable_{horizon_text}")
    glucose = _finite(frame, "cgm_last")
    expected = (
        base & (glucose >= 54)
        if kind == "hypo"
        else base & ~_boolean(frame, "hyper_currently_active")
    )
    mask = _boolean(frame, f"{task}_training_mask")
    if not np.array_equal(mask, expected):
        raise ValueError(f"{task} training mask differs from the revision contract")
    label_column = (
        f"severe_hypo_{horizon_text}_label" if kind == "hypo" else f"{task}_label"
    )
    for column in (label_column, f"{task}_training_target"):
        if column not in frame:
            raise ValueError(f"revision predictions are missing {column}")
    labels = pd.to_numeric(frame[label_column], errors="coerce").to_numpy(dtype=float)
    targets = pd.to_numeric(frame[f"{task}_training_target"], errors="coerce").to_numpy(
        dtype=float
    )
    if not np.isin(labels[mask], (0, 1)).all() or not np.array_equal(
        labels[mask], targets[mask]
    ):
        raise ValueError(f"{task} retained targets differ from its actual outcome")
    return targets[mask].astype(int), mask


def _causal(frame: pd.DataFrame) -> None:
    observed, interpolated, censored = (_boolean(frame, flag) for flag in _FLAGS)
    _finite(frame, "cgm_last")
    if not np.all(observed & ~interpolated & ~censored):
        raise ValueError(
            "revision timeline requires observed noninterpolated uncensored anchors"
        )
    for task in TASKS:
        _target_mask(frame, task)


class RevisionTrainingContext:
    """Bind callbacks to complete replay histories and immutable base truth.

    The event/exposure snapshots are tuples of immutable scalar records. Calls
    receive fresh tables, so neither caller mutation nor a returned metric table
    can change the context's base population. The input timeline is defensively
    copied only for the metadata needed to detect missing or changed decisions.
    """

    def __init__(
        self,
        replays: Mapping[str, ReplayInputs],
        *,
        source_order: tuple[str, ...],
        output_dir: str | Path | None = None,
    ) -> None:
        if not source_order or len(set(source_order)) != len(source_order):
            raise ValueError("source_order must contain unique sources")
        if not set(_PARTITIONS).issubset(replays):
            raise ValueError(
                "revision context requires selection and calibration replays"
            )
        if set(replays).difference((*_PARTITIONS, "train")):
            raise ValueError(
                "revision context cannot contain external or locked partitions"
            )
        self._source_order = tuple(source_order)
        self.output_dir = None if output_dir is None else Path(output_dir)
        truth = {}
        self._bindings: dict[str, pd.DataFrame] = {}
        self._indices: dict[str, pd.MultiIndex] = {}
        self._cache: dict[str, Any] | None = None
        for partition in _PARTITIONS:
            replay = replays[partition]
            timeline = replay.timeline
            if (
                timeline.empty
                or "partition" not in timeline
                or not timeline["partition"].eq(partition).all()
            ):
                raise ValueError(
                    f"{partition} replay has missing or mixed partition labels"
                )
            if set(timeline["source_id"].astype(str)) != set(source_order):
                raise ValueError(f"{partition} replay source set mismatch")
            _causal(timeline)
            index = _decision_index(timeline)
            binding_columns = [
                "patient_day",
                "cgm_last",
                *_FLAGS,
                "hyper_currently_active",
            ]
            for horizon in REGRESSION_HORIZONS:
                binding_columns.extend(
                    [
                        f"base_evaluable_{horizon}",
                        f"severe_hypo_{horizon}_label",
                        f"hyper_{horizon}_label",
                        f"cgm_{horizon}_target",
                        f"cgm_{horizon}_target_available",
                        *(
                            f"{kind}_{horizon}_training_{suffix}"
                            for kind in ("hypo", "hyper")
                            for suffix in ("target", "mask")
                        ),
                    ]
                )
            self._bindings[partition] = timeline[binding_columns].copy(deep=True)
            self._indices[partition] = index
            if set(replay.events) != set(REGRESSION_HORIZONS) or set(
                replay.exposure
            ) != set(REGRESSION_HORIZONS):
                raise ValueError(
                    "replay base tables must cover exactly horizons 30 and 60"
                )
            for horizon in REGRESSION_HORIZONS:
                validation = self._timeline(timeline, horizon)
                validation["__revision_validation_score"] = 0.0
                _, events, exposure = _validated_inputs(
                    validation,
                    replay.events[horizon],
                    replay.exposure[horizon],
                    FixedFrameSpec(
                        "__revision_validation_score", horizon, source_order
                    ),
                )
                catalogues = {
                    key: part["event_onset"]
                    .to_numpy(dtype="datetime64[ns]")
                    .astype(np.int64)
                    for key, part in events.groupby(
                        ["source_id", "participant_id"], observed=True
                    )
                }
                for key, part in timeline.groupby(
                    ["source_id", "participant_id"], observed=True
                ):
                    base = _boolean(part, f"base_evaluable_{horizon}")
                    queries = (
                        pd.to_datetime(part.loc[base, "context_end"], utc=True)
                        .to_numpy(dtype="datetime64[ns]")
                        .astype(np.int64)
                    )
                    onsets = catalogues.get(
                        tuple(map(str, key)), np.array([], dtype=int)
                    )
                    positions = np.searchsorted(onsets, queries, side="right")
                    has_event = positions < len(onsets)
                    expected_labels = np.zeros(len(queries), dtype=int)
                    expected_labels[has_event] = (
                        onsets[positions[has_event]] - queries[has_event]
                        <= horizon * 60_000_000_000
                    )
                    if not np.array_equal(
                        part.loc[base, f"severe_hypo_{horizon}_label"], expected_labels
                    ):
                        raise ValueError(
                            "severe outcome labels differ from the fixed base catalogue"
                        )
                truth[partition, horizon] = (
                    _FrozenTable.from_frame(events),
                    _FrozenTable.from_frame(exposure),
                )
        participants = [
            set(index.droplevel("context_end")) for index in self._indices.values()
        ]
        if participants[0].intersection(participants[1]):
            raise ValueError(
                "source-scoped participants overlap selection and calibration"
            )
        self._truth = MappingProxyType(truth)

    @property
    def source_order(self) -> tuple[str, ...]:
        return self._source_order

    def _sources(self, source_order: tuple[str, ...]) -> None:
        if tuple(source_order) != self.source_order:
            raise ValueError("callback source order differs from the bound context")

    def _bind(self, predictions: pd.DataFrame, partition: str) -> pd.MultiIndex:
        if (
            partition not in _PARTITIONS
            or "partition" not in predictions
            or not (predictions["partition"].eq(partition).all())
        ):
            raise ValueError(f"callback requires only the {partition} partition")
        index = _decision_index(predictions)
        expected = self._indices[partition]
        if len(index) != len(expected):
            raise ValueError(
                "revision callback requires the complete decision timeline"
            )
        positions = expected.get_indexer(index)
        if (positions < 0).any():
            raise ValueError(
                "revision callback decision membership differs from replay"
            )
        _causal(predictions)
        truth = self._bindings[partition]
        for column in truth:
            if column not in predictions:
                raise ValueError(f"revision predictions are missing {column}")
            actual = predictions[column].reset_index(drop=True)
            reference = truth[column].iloc[positions].reset_index(drop=True)
            equal = actual.eq(reference) | (actual.isna() & reference.isna())
            if not equal.fillna(False).all():
                raise ValueError(
                    f"revision prediction truth differs from replay: {column}"
                )
        for horizon in REGRESSION_HORIZONS:
            logit = _finite(predictions, severe_source_logit_column(horizon))
            severe = _finite(predictions, f"severe_hypo_{horizon}_raw_logit")
            if not np.array_equal(logit, severe):
                raise ValueError(
                    "severe raw logits differ from the target-aligned hypo head"
                )
        return index

    @staticmethod
    def _timeline(predictions: pd.DataFrame, horizon: int) -> pd.DataFrame:
        timeline = predictions.copy(deep=False)
        timeline["base_evaluable"] = predictions[f"base_evaluable_{horizon}"]
        return timeline

    def _tables(
        self, partition: str, horizon: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        events, exposure = self._truth[partition, horizon]
        return events.frame(), exposure.frame()

    def checkpoint_proxy(
        self,
        predictions: pd.DataFrame,
        *,
        source_order: tuple[str, ...],
        candidate_count: int | None,
        selection_rule: str = "area",
        budget: float = 0.5,
    ) -> tuple[float, float, dict[str, Any]]:
        self._sources(source_order)
        self._bind(predictions, "selection")
        if selection_rule not in {"area", "budget"}:
            raise ValueError("checkpoint selection_rule must be area or budget")
        if not np.isfinite(budget) or budget < 0:
            raise ValueError("checkpoint budget must be finite and non-negative")
        curves, scores, ties = {}, [], []
        for horizon in REGRESSION_HORIZONS:
            events, exposure = self._tables("selection", horizon)
            curve = selection_fixed_frame_curve(
                self._timeline(predictions, horizon),
                events,
                exposure,
                spec=FixedFrameSpec(
                    f"severe_hypo_{horizon}_raw_logit", horizon, self.source_order
                ),
                glucose_cut=54,
                candidate_count=candidate_count,
            )
            curves[str(horizon)] = curve
            scores.append(
                curve["raw_auc"]
                if selection_rule == "area"
                else max(
                    point["best_feasible_fixed_sensitivity"]
                    for point in curve["frontier"]
                    if point["achieved_fapd"] <= budget
                )
            )
            ties.append(curve["fapd_at_best_sensitivity"])
        return (
            float(np.mean(scores)),
            float(np.mean(ties)),
            {
                "horizons": curves,
                "selection_rule": selection_rule,
                "budget": budget,
                "revision_contract": REVISION_CONTRACT,
                "glucose_cut": 54,
                "evaluator": "fixed_frame",
            },
        )

    @staticmethod
    def _config(config: dict[str, Any]) -> tuple[int, tuple[float, ...]]:
        if config.get("training", {}).get("revision_contract") != REVISION_CONTRACT:
            raise ValueError(
                "revision callbacks require the explicit revision contract"
            )
        if (
            tuple(map(int, config["protocol"]["horizons_minutes"]))
            != REGRESSION_HORIZONS
        ):
            raise ValueError("revision callbacks require horizons ordered 30 then 60")
        candidate_count = int(
            config["evaluation"].get("threshold_candidate_count", 2001)
        )
        if candidate_count < 2:
            raise ValueError("threshold candidate count must be at least two")
        if candidate_count != 2001 and not config["project"].get("synthetic", False):
            raise ValueError(
                "real revision calibration requires the original 2001 candidates"
            )
        budgets = tuple(
            map(float, config["evaluation"]["false_alarms_per_patient_day"])
        )
        if len(budgets) != 2 or set(budgets) != set(_BUDGETS):
            raise ValueError(
                "revision calibration requires the original 0.5 and 1.0 budgets"
            )
        return candidate_count, budgets

    def calibrate(
        self,
        selection_predictions: pd.DataFrame,
        calibration_predictions: pd.DataFrame,
        *,
        config: dict[str, Any],
        source_order: tuple[str, ...],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Fit temperatures and policies on calibration only; relink both names."""
        self._sources(source_order)
        candidate_count, budgets = self._config(config)
        predictions_by_partition = {
            "selection": selection_predictions,
            "calibration": calibration_predictions,
        }
        indices = {
            partition: self._bind(frame, partition)
            for partition, frame in (predictions_by_partition.items())
        }
        task_summary: dict[str, Any] = {}
        for task in TASKS:
            labels, mask = _target_mask(calibration_predictions, task)
            logits = _finite(calibration_predictions, f"{task}_logit")
            temperature = fit_temperature(labels, logits[mask])
            for frame in predictions_by_partition.values():
                frame[f"{task}_probability"] = apply_temperature(
                    _finite(frame, f"{task}_logit"), temperature
                )
                if task.startswith("hypo_"):
                    horizon_text = task.split("_")[1]
                    frame[f"severe_hypo_{horizon_text}_probability"] = frame[
                        f"{task}_probability"
                    ].to_numpy(copy=True)
            task_summary[task] = {
                "temperature": float(temperature),
                "temperature_fit_partition": "calibration",
                "temperature_fit_status": (
                    "fitted"
                    if len(np.unique(labels)) == 2
                    else "identity_single_class"
                    if labels.size
                    else "not_run_no_retained_rows"
                ),
                "calibration_retained_count": int(labels.size),
                "calibration_positive_count": int(labels.sum()),
                "target_column": (
                    f"severe_hypo_{task.split('_')[1]}_label"
                    if task.startswith("hypo_")
                    else f"{task}_label"
                ),
                "training_target_column": f"{task}_training_target",
                "mask_column": f"{task}_training_mask",
            }
        evaluations: dict[tuple[str, int, str], FixedFrameEvaluation] = {}
        for horizon in REGRESSION_HORIZONS:
            score_column = f"severe_hypo_{horizon}_probability"
            spec = FixedFrameSpec(score_column, horizon, self.source_order)
            events, exposure = self._tables("calibration", horizon)
            timeline = self._timeline(calibration_predictions, horizon)
            curve = selection_fixed_frame_curve(
                timeline,
                events,
                exposure,
                spec=spec,
                glucose_cut=54,
                candidate_count=candidate_count,
                include_raw_points=True,
            )
            policies, thresholds = {}, {}
            by_threshold = {}
            for budget in budgets:
                feasible = [
                    point
                    for point in curve["raw_points"]
                    if point["achieved_fapd"] <= budget
                ]
                best = max(
                    feasible,
                    key=lambda point: (point["fixed_sensitivity"], point["threshold"]),
                )
                threshold = best["threshold"]
                if threshold not in by_threshold:
                    by_threshold[threshold] = evaluate_fixed_frame(
                        timeline,
                        events,
                        exposure,
                        spec=spec,
                        threshold=threshold,
                        glucose_cut=54,
                    )
                evaluated = by_threshold[threshold]
                summary = evaluated.summary
                sensitivity = summary[
                    "equal_source_participant_macro_fixed_sensitivity"
                ]
                fapd = summary[
                    "equal_source_participant_macro_unmatched_alarms_per_patient_day"
                ]
                if not np.allclose(
                    (sensitivity, fapd),
                    (best["fixed_sensitivity"], best["achieved_fapd"]),
                    rtol=0,
                    atol=1e-12,
                ):
                    raise AssertionError(
                        "calibration sweep and fixed-frame ledger disagree"
                    )
                policy = FixedFrameAlarmPolicy(
                    partition="calibration",
                    endpoint=asdict(spec),
                    glucose_cut=54,
                    nominal_fapd=budget,
                    threshold=threshold,
                    threshold_count=curve["threshold_count"],
                    candidate_rule=curve["candidate_rule"],
                    tie_rule="maximum_fixed_sensitivity_then_highest_threshold",
                    achieved_equal_source_participant_macro_fapd=fapd,
                    achieved_equal_source_participant_macro_fixed_sensitivity=sensitivity,
                    participant_count=summary["participant_count"],
                    base_event_count=summary["base_event_count"],
                    patient_days=summary["patient_days"],
                    observation_minutes=summary["observation_minutes"],
                )
                budget_key = str(budget)
                policies[budget_key], thresholds[budget_key] = (
                    policy.as_dict(),
                    threshold,
                )
                evaluations["calibration", horizon, budget_key] = evaluated
                selected_events, selected_exposure = self._tables("selection", horizon)
                evaluations["selection", horizon, budget_key] = evaluate_fixed_frame(
                    self._timeline(selection_predictions, horizon),
                    selected_events,
                    selected_exposure,
                    spec=spec,
                    threshold=threshold,
                    glucose_cut=54,
                )
            task_summary[f"hypo_{horizon}"].update(
                {
                    "threshold_partition": "calibration",
                    "fixed_frame_policies": policies,
                    "thresholds_from_calibration": thresholds,
                }
            )
        operational = self._operational(task_summary, evaluations)
        self._cache = {
            "indices": indices,
            "evaluations": evaluations,
            "logits": {
                partition: {
                    horizon: frame[severe_source_logit_column(horizon)].to_numpy(
                        copy=True
                    )
                    for horizon in REGRESSION_HORIZONS
                }
                for partition, frame in predictions_by_partition.items()
            },
            "signature": self._policy_signature(task_summary),
        }
        return task_summary, operational

    @staticmethod
    def _policy_signature(tasks: dict[str, Any]) -> tuple:
        return tuple(
            (
                horizon,
                float(tasks[f"hypo_{horizon}"]["temperature"]),
                budget,
                float(
                    tasks[f"hypo_{horizon}"]["fixed_frame_policies"][budget][
                        "threshold"
                    ]
                ),
            )
            for horizon in REGRESSION_HORIZONS
            for budget in map(str, _BUDGETS)
        )

    @staticmethod
    def _operational(
        tasks: dict[str, Any],
        evaluations: Mapping[tuple[str, int, str], FixedFrameEvaluation],
    ) -> dict[str, Any]:
        horizons = {}
        for horizon in REGRESSION_HORIZONS:
            metrics = {
                partition: {
                    str(budget): evaluations[partition, horizon, str(budget)].summary
                    for budget in _BUDGETS
                }
                for partition in _PARTITIONS
            }
            horizons[str(horizon)] = {
                "temperature": tasks[f"hypo_{horizon}"]["temperature"],
                "policies": tasks[f"hypo_{horizon}"]["fixed_frame_policies"],
                "partitions": metrics,
            }
        return {
            "evaluator": "fixed_frame",
            "revision_contract": REVISION_CONTRACT,
            "horizons": horizons,
            "nominal_fapd": 0.5,
            "glucose_cut": 54,
            "threshold_partition": "calibration",
            "evaluation_partition": "selection",
            "primary_equal_horizon_mean_fixed_sensitivity": float(
                np.mean(
                    [
                        horizons[str(h)]["partitions"]["selection"]["0.5"][
                            "equal_source_participant_macro_fixed_sensitivity"
                        ]
                        for h in REGRESSION_HORIZONS
                    ]
                )
            ),
            "equal_horizon_mean_achieved_fapd": float(
                np.mean(
                    [
                        horizons[str(h)]["partitions"]["selection"]["0.5"][
                            "equal_source_participant_macro_unmatched_alarms_per_patient_day"
                        ]
                        for h in REGRESSION_HORIZONS
                    ]
                )
            ),
        }

    def report(
        self,
        predictions: pd.DataFrame,
        *,
        config: dict[str, Any],
        model_summary: dict[str, Any],
    ) -> dict[str, Any]:
        self._config(config)
        if model_summary.get("revision_contract") != REVISION_CONTRACT:
            raise ValueError("reporter requires a revision model summary")
        if "partition" not in predictions or set(predictions["partition"]) != set(
            _PARTITIONS
        ):
            raise ValueError(
                "revision report requires exactly selection and calibration"
            )
        tasks = model_summary["tasks"]
        for task in TASKS:
            target = (
                f"severe_hypo_{task.split('_')[1]}_label"
                if task.startswith("hypo_")
                else f"{task}_label"
            )
            if (
                tasks[task].get("temperature_fit_partition") != "calibration"
                or tasks[task].get("target_column") != target
                or tasks[task].get("mask_column") != f"{task}_training_mask"
            ):
                raise ValueError(
                    "report task target/mask/temperature partition mismatch"
                )
        evaluations: dict[tuple[str, int, str], FixedFrameEvaluation] = {}
        output: dict[str, Any] = {
            "status": "synthetic_plumbing_only"
            if config["project"].get("synthetic", False)
            else "observed_development_revision",
            "revision_contract": REVISION_CONTRACT,
            "evaluator": "fixed_frame",
            "legacy_pooled_evaluator_reportable": False,
            "partitions": {},
        }
        for partition in _PARTITIONS:
            frame = predictions.loc[predictions["partition"].eq(partition)]
            index = self._bind(frame, partition)
            metrics: dict[str, Any] = {"probability_diagnostics": {}, "fixed_frame": {}}
            for task in TASKS:
                labels, mask = _target_mask(frame, task)
                probability = _finite(frame, f"{task}_probability")
                expected = apply_temperature(
                    _finite(frame, f"{task}_logit"), float(tasks[task]["temperature"])
                )
                if not np.allclose(probability, expected, rtol=0, atol=1e-12):
                    raise ValueError(f"{task} probability differs from calibrated head")
                if task.startswith("hypo_"):
                    severe = _finite(
                        frame, f"severe_hypo_{task.split('_')[1]}_probability"
                    )
                    if not np.array_equal(severe, probability):
                        raise ValueError(
                            "severe probability differs from calibrated hypo head"
                        )
                metrics["probability_diagnostics"][task] = {
                    "target_column": tasks[task]["target_column"],
                    "mask_column": f"{task}_training_mask",
                    "n": int(mask.sum()),
                    "metrics": probability_metrics(
                        labels,
                        probability[mask],
                        ece_bins=int(config["evaluation"].get("ece_bins", 10)),
                    )
                    if mask.any()
                    else None,
                }
            cached = self._cache
            cache_valid = (
                cached is not None
                and self._policy_signature(tasks) == cached["signature"]
                and index.equals(cached["indices"][partition])
                and all(
                    np.array_equal(
                        frame[severe_source_logit_column(horizon)].to_numpy(),
                        cached["logits"][partition][horizon],
                    )
                    for horizon in REGRESSION_HORIZONS
                )
            )
            for horizon in REGRESSION_HORIZONS:
                metrics["fixed_frame"][str(horizon)] = {}
                for budget in _BUDGETS:
                    budget_key = str(budget)
                    policy = tasks[f"hypo_{horizon}"]["fixed_frame_policies"][
                        budget_key
                    ]
                    endpoint = policy["endpoint"]
                    if (
                        policy["partition"] != "calibration"
                        or policy["glucose_cut"] != 54
                        or policy["nominal_fapd"] != budget
                        or endpoint["horizon_minutes"] != horizon
                        or tuple(endpoint["source_order"]) != self.source_order
                        or endpoint["score_column"]
                        != f"severe_hypo_{horizon}_probability"
                    ):
                        raise ValueError(
                            "report policy differs from the fixed calibration contract"
                        )
                    if cache_valid:
                        evaluated = cached["evaluations"][
                            partition, horizon, budget_key
                        ]
                    else:
                        events, exposure = self._tables(partition, horizon)
                        evaluated = evaluate_fixed_frame(
                            self._timeline(frame, horizon),
                            events,
                            exposure,
                            spec=FixedFrameSpec(
                                endpoint["score_column"], horizon, self.source_order
                            ),
                            threshold=float(policy["threshold"]),
                            glucose_cut=54,
                        )
                    evaluations[partition, horizon, budget_key] = evaluated
                    metrics["fixed_frame"][str(horizon)][budget_key] = evaluated.summary
                    if self.output_dir is not None:
                        destination = (
                            self.output_dir / partition / str(horizon) / budget_key
                        )
                        destination.mkdir(parents=True, exist_ok=True)
                        for name in ("participants", "sources", "alarms", "events"):
                            getattr(evaluated, name).to_parquet(
                                destination / f"{name}.parquet", index=False
                            )
                target = f"cgm_{horizon}_target"
                predicted = f"cgm_{horizon}_prediction"
                if predicted not in frame:
                    if model_summary.get("continuous_prediction_available") is False:
                        continue
                    raise ValueError("declared forecasting model lacks predictions")
                available = _boolean(frame, f"{target}_available")
                if available.any():
                    values = pd.to_numeric(
                        frame.loc[available, target], errors="coerce"
                    )
                    forecast = pd.to_numeric(
                        frame.loc[available, predicted], errors="coerce"
                    )
                    if not np.isfinite(values).all() or not np.isfinite(forecast).all():
                        raise ValueError(
                            "available continuous targets/predictions must be finite"
                        )
                    metrics.setdefault("continuous_forecast", {})[str(horizon)] = (
                        regression_metrics(values, forecast)
                    )
            output["partitions"][partition] = metrics
        output["operational_fixed_frame"] = self._operational(tasks, evaluations)
        if self.output_dir is not None:
            write_json(output, self.output_dir / "fixed_frame_metrics.json")
        return output
