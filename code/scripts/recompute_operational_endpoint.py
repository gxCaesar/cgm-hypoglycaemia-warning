#!/usr/bin/env python3
"""Recompute the v1.3 operational endpoint from retained prediction artifacts.

The v1.3 operational evaluator (``operational_metrics``) was added after every
retained arm had already been scored by the superseded ``metrics`` path, so no
real-data number exists for the endpoint the v1.3 claim is actually written
against. This driver fits the calibration-only alarm policy and evaluates the
frozen source-then-participant macro endpoint on selection, for each supplied
arm.

It trains nothing, refits nothing, and opens no external, stress or locked
partition. Output is diagnostic: the retained checkpoints were selected under
the superseded macro-AUPRC proxy rather than the v1.3 budget-frontier proxy, so
these values describe the retained artifacts and are not comparator results.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd

from glycoshift.config import load_config
from glycoshift.io import write_json
from glycoshift.operational_metrics import (
    OperationalEndpointSpec,
    fit_operational_alarm_policy,
    operational_endpoint_metrics,
    selection_operational_curve,
)

HORIZONS = (30, 60)
SCORE_TEMPLATE = "hypo_{horizon}_probability"
REQUIRED_PARTITIONS = ("calibration", "selection")


def _spec(horizon: int, source_order: tuple[str, ...]) -> OperationalEndpointSpec:
    return OperationalEndpointSpec(
        score_column=SCORE_TEMPLATE.format(horizon=horizon),
        label_column=f"severe_hypo_{horizon}_label",
        event_id_column=f"severe_hypo_{horizon}_event_id",
        event_onset_column=f"severe_hypo_{horizon}_event_onset",
        active_column="severe_hypo_currently_active",
        horizon_minutes=horizon,
        source_order=source_order,
    )


def _projected_columns(source_order: tuple[str, ...]) -> list[str]:
    columns = {"source_id", "participant_id", "context_end", "patient_day", "partition"}
    for horizon in HORIZONS:
        spec = _spec(horizon, source_order)
        columns.update(
            {
                spec.score_column,
                spec.label_column,
                spec.event_id_column,
                spec.event_onset_column,
                spec.active_column,
            }
        )
    return sorted(columns)


def _load_partitions(
    predictions_path: Path,
    source_order: tuple[str, ...],
) -> dict[str, pd.DataFrame]:
    frame = pd.read_parquet(
        predictions_path,
        columns=_projected_columns(source_order),
    )
    if "partition" not in frame.columns:
        raise ValueError(f"{predictions_path} has no partition column")
    partitions = {
        name: frame.loc[frame["partition"] == name].copy()
        for name in REQUIRED_PARTITIONS
    }
    for name, part in partitions.items():
        if part.empty:
            raise ValueError(f"{predictions_path} has no {name} rows")
    return partitions


def evaluate_arm(
    arm_id: str,
    predictions_path: Path,
    *,
    source_order: tuple[str, ...],
    nominal_fapd: float,
    curve_candidate_count: int | None = None,
) -> dict[str, Any]:
    """Evaluate one arm.

    ``curve_candidate_count`` of ``None`` reproduces the frozen budget-frontier
    proxy, which sweeps every unique selection score. That is millions of
    thresholds per arm and horizon, so a bounded count is available for timing
    probes; a bounded run is not the frozen proxy and must be labelled as such.
    """

    partitions = _load_partitions(predictions_path, source_order)
    horizons: dict[str, Any] = {}
    for horizon in HORIZONS:
        spec = _spec(horizon, source_order)
        policy = fit_operational_alarm_policy(
            partitions["calibration"],
            spec=spec,
            nominal_fapd=nominal_fapd,
        )
        selection = operational_endpoint_metrics(
            partitions["selection"],
            spec=spec,
            threshold=policy.threshold,
        )
        curve = selection_operational_curve(
            partitions["selection"],
            spec=spec,
            candidate_count=curve_candidate_count,
        )
        horizons[str(horizon)] = {
            "calibration_policy": policy.as_dict(),
            "selection": selection,
            "selection_budget_frontier": curve,
        }
    sensitivities = [
        horizons[str(h)]["selection"][
            "equal_source_participant_macro_event_sensitivity"
        ]
        for h in HORIZONS
    ]
    achieved = [
        horizons[str(h)]["selection"][
            "equal_source_participant_macro_false_alarms_per_patient_day"
        ]
        for h in HORIZONS
    ]
    return {
        "arm_id": arm_id,
        "predictions_path": str(predictions_path),
        "nominal_fapd": nominal_fapd,
        "source_order": list(source_order),
        "horizons": horizons,
        "mean_severe_event_sensitivity": sum(sensitivities) / len(sensitivities),
        "mean_achieved_fapd": sum(achieved) / len(achieved),
        "evaluator": "operational_metrics.v1.3",
        "reportability": "diagnostic_only_retained_checkpoint_proxy_mismatch",
    }


def _parse_arm(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "--arm expects arm_id=/path/predictions.parquet"
        )
    arm_id, _, path = value.partition("=")
    if not arm_id or not path:
        raise argparse.ArgumentTypeError("--arm expects a non-empty id and path")
    return arm_id, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--arm", required=True, action="append", type=_parse_arm)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--nominal-fapd", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--curve-candidate-count",
        type=int,
        default=None,
        help=(
            "bound the budget-frontier sweep for timing probes; omit to use "
            "the frozen all-unique-score proxy"
        ),
    )
    arguments = parser.parse_args()

    config = load_config(arguments.config)
    source_order = tuple(config["data"]["development_sources"])
    if len(source_order) != len(set(source_order)):
        raise ValueError("development_sources contains duplicates")

    output = arguments.output
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)

    for _, path in arguments.arm:
        if not path.is_file():
            raise FileNotFoundError(f"missing prediction artifact: {path}")

    if arguments.workers > 1:
        with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
            futures = [
                pool.submit(
                    evaluate_arm,
                    arm_id,
                    path,
                    source_order=source_order,
                    nominal_fapd=arguments.nominal_fapd,
                    curve_candidate_count=arguments.curve_candidate_count,
                )
                for arm_id, path in arguments.arm
            ]
            results = [future.result() for future in futures]
    else:
        results = [
            evaluate_arm(
                arm_id,
                path,
                source_order=source_order,
                nominal_fapd=arguments.nominal_fapd,
                curve_candidate_count=arguments.curve_candidate_count,
            )
            for arm_id, path in arguments.arm
        ]

    for result in results:
        write_json(result, output / f"{result['arm_id']}.json")
    summary = sorted(
        (
            {
                "arm_id": result["arm_id"],
                "mean_severe_event_sensitivity": result[
                    "mean_severe_event_sensitivity"
                ],
                "mean_achieved_fapd": result["mean_achieved_fapd"],
                "budget_frontier_auc": {
                    horizon: result["horizons"][horizon][
                        "selection_budget_frontier"
                    ]["normalized_auc"]
                    for horizon in result["horizons"]
                },
            }
            for result in results
        ),
        key=lambda item: item["mean_severe_event_sensitivity"],
        reverse=True,
    )
    write_json(
        {
            "evaluator": "operational_metrics.v1.3",
            "nominal_fapd": arguments.nominal_fapd,
            "curve_candidate_count": arguments.curve_candidate_count,
            "reportability": "diagnostic_only_retained_checkpoint_proxy_mismatch",
            "ranking": summary,
        },
        output / "summary.json",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
