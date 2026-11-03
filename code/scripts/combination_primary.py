#!/usr/bin/env python3
"""Seed sigma on the frozen primary, and whether the combination survives its seed.

Pre-registered in
`experiments/20260827-glycoshift-m6-anchor-aware-v1/PREREG-seed-stability-of-the-combination.md`.

Read-only over frozen prediction artifacts. No training, no GPU, no new data.

Two questions on one pass. First, the seed scale every margin in this project divides
by was measured on a quantity that is not the frozen primary, so measure it there
directly. Second, the result class that clears the promotion floor has only ever been
read at one seed, so read it at four.

The seedless member is built from `cgm_last` directly rather than from the persistence
artifact. That is not a shortcut: `anchor_probability_composition.py` at `w = 0` already
returned `0.806116373`, persistence's own frozen primary, so the calibration ECDF of
`-cgm_last` IS the persistence ordering, verified against a number it could not
influence. Building it here avoids depending on `baseline-cpu-run/persistence`, whose
family carries 265,892 non-finite scores and is superseded, and the v1.5 artifact was
not retained. The gate below re-checks it in this code path anyway.
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
PROMOTION_FLOOR = 0.836116373
TRANSFERRED_SIGMA = 0.009456055


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


def _read(path: Path, *, legacy: bool, base: bool) -> pd.DataFrame:
    score = [
        (f"hypo_{h}_probability" if legacy else f"severe_hypo_{h}_probability")
        for h in HORIZONS
    ]
    columns = list(KEY) + score
    if base:
        # These are properties of the frame, not of an arm. Reading them for
        # every arm duplicates them on merge, which pandas refuses -- correctly,
        # and it refused here before any number was produced.
        columns += [
            "partition",
            "severe_hypo_currently_active",
            "patient_day",
            "cgm_last",
        ]
        for h in HORIZONS:
            columns += [
                f"severe_hypo_{h}_label",
                f"severe_hypo_{h}_event_id",
                f"severe_hypo_{h}_event_onset",
            ]
    frame = pq.read_table(path, columns=sorted(set(columns))).to_pandas()
    if legacy:
        for h in HORIZONS:
            frame[f"severe_hypo_{h}_probability"] = frame[f"hypo_{h}_probability"]
            frame = frame.drop(columns=[f"hypo_{h}_probability"])
    for column in KEY:
        frame[column] = frame[column].astype(str)
    return frame


def calibration_rank(values: np.ndarray, calibration_mask: np.ndarray) -> np.ndarray:
    """Rank against the CALIBRATION distribution alone, applied to every row.

    The identical transform behind the Order-4 result: a function of calibration
    scores only, so scoring one selection window never consults another, and
    strictly increasing between distinct calibration scores.
    """

    reference = np.sort(values[calibration_mask])
    if reference.size == 0:
        raise SystemExit("no calibration rows to fit the rank transform on")
    grid = np.linspace(0.0, 1.0, reference.size)
    return np.interp(values, reference, grid)


def _evaluate(
    *,
    frame: pd.DataFrame,
    members: list[np.ndarray],
    horizon: int,
    calibration: np.ndarray,
    selection: np.ndarray,
    source_order: tuple[str, ...],
    nominal_fapd: float,
) -> dict[str, Any]:
    work = frame.copy()
    work[f"severe_hypo_{horizon}_probability"] = np.mean(members, axis=0)
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
    return {
        "sensitivity": metrics["equal_source_participant_macro_event_sensitivity"],
        "fapd": metrics[
            "equal_source_participant_macro_false_alarms_per_patient_day"
        ],
        "alarm_count": metrics["alarm_count"],
        "detected_event_count": metrics["detected_event_count"],
        # The endpoint is an equal-source macro, so a reader is entitled to ask which
        # cohorts carry it. The per-source block is already computed to form the macro
        # and was simply not being copied out.
        "per_source": metrics.get("per_source", {}),
        "event_count": metrics.get("event_count"),
        "participant_count": metrics.get("participant_count"),
        "source_count": metrics.get("source_count"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True,
                        help="name=path[:legacy]")
    parser.add_argument("--combination", action="append", default=[],
                        help="comma-separated member names")
    parser.add_argument("--frozen", action="append", default=[],
                        help="name=value, that arm's frozen primary, for the gate")
    parser.add_argument("--nominal-fapd", type=float, default=0.5)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    def parse(item: str) -> tuple[str, Path, bool]:
        name, _, rest = item.partition("=")
        legacy = rest.endswith(":legacy")
        return name, Path(rest[: -len(":legacy")] if legacy else rest), legacy

    arms = [parse(item) for item in args.arm]
    frozen = {}
    for item in args.frozen:
        key, _, value = item.partition("=")
        frozen[key] = float(value)

    frame = None
    for index, (name, path, legacy) in enumerate(arms):
        part = _read(path, legacy=legacy, base=(index == 0))
        part = part.rename(
            columns={f"severe_hypo_{h}_probability": f"arm__{name}__{h}"
                     for h in HORIZONS}
        )
        frame = (part if frame is None
                 else frame.merge(part, on=KEY, how="inner", validate="one_to_one"))
        print(f"  joined {name}")
    assert frame is not None
    print(f"joined one-to-one over {len(frame):,} rows for {len(arms)} arms")

    calibration = (frame["partition"] == "calibration").to_numpy()
    selection = (frame["partition"] == "selection").to_numpy()
    source_order = tuple(sorted(frame["source_id"].astype(str).unique()))

    ranks: dict[str, dict[int, np.ndarray]] = {}
    for name, _path, _legacy in arms:
        ranks[name] = {}
        for horizon in HORIZONS:
            values = pd.to_numeric(frame[f"arm__{name}__{horizon}"],
                                   errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise SystemExit(f"{name} horizon {horizon} carries non-finite values")
            ranks[name][horizon] = calibration_rank(values, calibration)

    def primary(members: list[str]) -> dict[str, Any]:
        cells: dict[str, Any] = {"members": list(members)}
        values = []
        for horizon in HORIZONS:
            cell = _evaluate(
                frame=frame, members=[ranks[m][horizon] for m in members],
                horizon=horizon, calibration=calibration, selection=selection,
                source_order=source_order, nominal_fapd=float(args.nominal_fapd),
            )
            cells[str(horizon)] = cell
            values.append(cell["sensitivity"])
        cells["primary"] = float(np.mean(values))
        return cells

    report: dict[str, Any] = {
        "rows": int(len(frame)),
        "nominal_fapd": float(args.nominal_fapd),
        "promotion_floor": PROMOTION_FLOOR,
        "transferred_sigma": TRANSFERRED_SIGMA,
        "cells": {},
    }
    requested = [[n] for n, _p, _l in arms] + [
        c.split(",") for c in args.combination
    ]
    seen: set[str] = set()
    failures = []
    for members in requested:
        label = ",".join(members)
        if label in seen:
            continue
        seen.add(label)
        for member in members:
            if member not in ranks:
                raise SystemExit(f"unknown member {member!r} in {label!r}")
        cell = primary(members)
        report["cells"][label] = cell
        note = ""
        if len(members) == 1 and members[0] in frozen:
            drift = abs(cell["primary"] - frozen[members[0]])
            note = f"   frozen {frozen[members[0]]:.9f}  drift {drift:.3e}"
            if drift > 1e-9:
                failures.append(label)
                note += "  <-- GATE FAILED"
        margin = cell["primary"] - PROMOTION_FLOOR
        print(f"  {label:26s} primary {cell['primary']:.9f}  "
              f"floor {margin:+.9f}{note}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if failures:
        report["gate_failures"] = failures
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {args.output}")
    if failures:
        raise SystemExit(f"single-member gate failed for {failures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
