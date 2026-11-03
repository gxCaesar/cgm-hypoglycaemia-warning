#!/usr/bin/env python3
"""Reading (b) of the M6 pre-registration: the mismatch-free composition.

M6 is trained against `w*sigmoid(f) + (1-w)*A`, a convex combination in probability
space, while the deployed Order-4 result combines two members through their
calibration RANK transforms. The pre-registration declares that mismatch in advance
and names it as a live failure mode. This script removes it from the comparison by
scoring the arm through the SAME composition training used, with the anchor refitted
on calibration so the transform is partition-honest at deployment.

It is read-only over frozen prediction artifacts: no training, no GPU, no new data.

The control is formed the same way from the un-anchored arm's own predictions, so the
contrast is one variable -- whether the member knew about the anchor while it was
trained -- rather than one variable plus a composition change.

Gate G1: `--weight 1.0` drops the anchor term entirely, so every arm must reproduce
its own frozen primary through this new code path. A composition that cannot
reproduce the numbers it is built from is not measuring what it claims to.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from glycoshift.operational_metrics import (
    OperationalEndpointSpec,
    fit_operational_alarm_policy,
    operational_endpoint_metrics,
)

HORIZONS = (30, 60)
KEY = ["source_id", "participant_id", "context_end"]


def _spec(horizon: int, source_order: tuple[str, ...]) -> OperationalEndpointSpec:
    return OperationalEndpointSpec(
        score_column=f"severe_hypo_{horizon}_probability",
        label_column=f"severe_hypo_{horizon}_label",
        event_id_column=f"severe_hypo_{horizon}_event_id",
        event_onset_column=f"severe_hypo_{horizon}_event_onset",
        active_column="severe_hypo_currently_active",
        horizon_minutes=horizon,
        source_order=source_order,
    )


def _read(path: Path, *, base: bool) -> pd.DataFrame:
    columns = list(KEY) + [
        f"severe_hypo_{h}_probability" for h in HORIZONS
    ] + ["partition", "severe_hypo_currently_active"]
    if base:
        columns += ["patient_day", "cgm_last"]
        for h in HORIZONS:
            columns += [
                f"severe_hypo_{h}_label",
                f"severe_hypo_{h}_event_id",
                f"severe_hypo_{h}_event_onset",
            ]
    frame = pq.read_table(path, columns=sorted(set(columns))).to_pandas()
    for column in KEY:
        frame[column] = frame[column].astype(str)
    return frame


def calibration_anchor(
    glucose: np.ndarray, calibration_mask: np.ndarray
) -> np.ndarray:
    """The deployment anchor: the CALIBRATION ECDF of `-cgm_last`.

    The same construction `fit_glucose_anchor` uses at training time, refitted on
    the partition deployment is allowed to fit on. Strictly inside (0, 1) by the
    same half-rank shift, and monotone non-increasing in glucose, so with
    `w = 0` it reproduces the persistence ordering exactly.
    """

    reference = np.sort(glucose[calibration_mask])
    if reference.size == 0:
        raise SystemExit("no calibration rows to fit the anchor on")
    # Fraction of calibration rows at or above this glucose, i.e. the ECDF of -g.
    below = np.searchsorted(reference, glucose, side="left").astype(float)
    return (reference.size - below + 0.5) / (reference.size + 1.0)


def _evaluate(
    *,
    frame: pd.DataFrame,
    scores: dict[int, np.ndarray],
    calibration: np.ndarray,
    selection: np.ndarray,
    source_order: tuple[str, ...],
    nominal_fapd: float,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    sensitivities: list[float] = []
    for horizon in HORIZONS:
        work = frame.copy()
        work[f"severe_hypo_{horizon}_probability"] = scores[horizon]
        policy = fit_operational_alarm_policy(
            work.loc[calibration],
            spec=_spec(horizon, source_order),
            nominal_fapd=nominal_fapd,
        )
        metrics = operational_endpoint_metrics(
            work.loc[selection],
            spec=_spec(horizon, source_order),
            threshold=policy.threshold,
        )
        sensitivity = metrics["equal_source_participant_macro_event_sensitivity"]
        sensitivities.append(sensitivity)
        output[str(horizon)] = {
            "sensitivity": sensitivity,
            "fapd": metrics[
                "equal_source_participant_macro_false_alarms_per_patient_day"
            ],
            "alarm_count": metrics["alarm_count"],
            "detected_event_count": metrics["detected_event_count"],
            "threshold": float(policy.threshold),
        }
    output["primary"] = float(np.mean(sensitivities))
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True, help="name=path")
    parser.add_argument(
        "--weight",
        action="append",
        type=float,
        required=True,
        help="anchor weight w on the NETWORK term; 1.0 is the G1 gate",
    )
    parser.add_argument("--nominal-fapd", type=float, default=0.5)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    specs: dict[str, Path] = {}
    for item in args.arm:
        name, _, path = item.partition("=")
        if name in specs:
            raise SystemExit(f"duplicate arm name: {name}")
        specs[name] = Path(path)

    frame = None
    for index, (name, path) in enumerate(specs.items()):
        part = _read(path, base=(index == 0))
        part = part.rename(
            columns={
                f"severe_hypo_{h}_probability": f"{name}__{h}" for h in HORIZONS
            }
        )
        frame = (
            part
            if frame is None
            else frame.merge(part, on=KEY, how="inner", validate="one_to_one")
        )
    assert frame is not None
    print(f"joined one-to-one over {len(frame):,} rows for {len(specs)} arms")

    calibration = (frame["partition"] == "calibration").to_numpy()
    selection = (frame["partition"] == "selection").to_numpy()
    source_order = tuple(sorted(frame["source_id"].astype(str).unique()))

    glucose = pd.to_numeric(frame["cgm_last"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(glucose).all():
        raise SystemExit("cgm_last carries non-finite values; the anchor needs it")
    anchor = calibration_anchor(glucose, calibration)
    if not (anchor.min() > 0.0 and anchor.max() < 1.0):
        raise SystemExit("the calibration anchor left the open unit interval")
    order = np.argsort(glucose, kind="stable")
    if np.any(np.diff(anchor[order]) > 1e-12):
        raise SystemExit("the calibration anchor is not monotone in glucose")
    print(
        f"calibration anchor over {int(calibration.sum()):,} rows: "
        f"A(54)={np.interp(54.0, np.sort(glucose), anchor[np.argsort(glucose)]):.4f} "
        f"range [{anchor.min():.6f}, {anchor.max():.6f}]"
    )

    report: dict[str, Any] = {
        "rows": int(len(frame)),
        "nominal_fapd": float(args.nominal_fapd),
        "arms": sorted(specs),
        "weights": [float(w) for w in args.weight],
        "cells": {},
    }

    for name in specs:
        for weight in args.weight:
            if not 0.0 <= weight <= 1.0:
                raise SystemExit(f"weight {weight} is outside [0, 1]")
            scores: dict[int, np.ndarray] = {}
            for horizon in HORIZONS:
                network = pd.to_numeric(
                    frame[f"{name}__{horizon}"], errors="coerce"
                ).to_numpy(dtype=float)
                if not np.isfinite(network).all():
                    raise SystemExit(f"{name}__{horizon} carries non-finite values")
                scores[horizon] = weight * network + (1.0 - weight) * anchor
            cell = _evaluate(
                frame=frame,
                scores=scores,
                calibration=calibration,
                selection=selection,
                source_order=source_order,
                nominal_fapd=float(args.nominal_fapd),
            )
            cell["arm"] = name
            cell["weight"] = float(weight)
            key = f"{name}@w{weight:g}"
            report["cells"][key] = cell
            marker = "  <- G1 gate, must equal the arm's frozen primary" if (
                weight == 1.0
            ) else ""
            print(f"  {key:28s} primary {cell['primary']:.9f}{marker}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
