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
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed-arm",
        action="append",
        required=True,
        help="seed=path[:legacy] for one seed of the trained member",
    )
    parser.add_argument("--frozen", action="append", default=[],
                        help="seed=value, the arm's frozen primary, for gate G1")
    parser.add_argument("--nominal-fapd", type=float, default=0.5)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    def parse(item: str) -> tuple[str, Path, bool]:
        name, _, rest = item.partition("=")
        legacy = rest.endswith(":legacy")
        path = Path(rest[: -len(":legacy")] if legacy else rest)
        return name, path, legacy

    anchor_name = "glucose_anchor"
    seeds = [parse(item) for item in args.seed_arm]
    frozen = {}
    for item in args.frozen:
        key, _, value = item.partition("=")
        frozen[key] = float(value)

    frame = None
    loaded: list[tuple[str, str]] = []
    for index, (name, path, legacy) in enumerate(seeds):
        column = f"arm__{name}"
        part = _read(path, legacy=legacy, base=(index == 0))
        part = part.rename(
            columns={
                f"severe_hypo_{h}_probability": f"{column}__{h}" for h in HORIZONS
            }
        )
        frame = (
            part
            if frame is None
            else frame.merge(part, on=KEY, how="inner", validate="one_to_one")
        )
        loaded.append((name, str(path)))
        print(f"  joined {name}")
    assert frame is not None
    print(f"joined one-to-one over {len(frame):,} rows for {len(loaded)} arms")

    calibration = (frame["partition"] == "calibration").to_numpy()
    selection = (frame["partition"] == "selection").to_numpy()
    source_order = tuple(sorted(frame["source_id"].astype(str).unique()))

    # The seedless member: the calibration ECDF of -cgm_last, which
    # `anchor_probability_composition.py --weight 0` showed reproduces
    # persistence's frozen 0.806116373 exactly. It has no seed, so it is the
    # identical member in every cell and the trained arm is the only thing varying.
    glucose = pd.to_numeric(frame["cgm_last"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(glucose).all():
        raise SystemExit("cgm_last carries non-finite values")
    reference = np.sort(glucose[calibration])
    below = np.searchsorted(reference, glucose, side="left").astype(float)
    anchor_values = (reference.size - below + 0.5) / (reference.size + 1.0)

    ranks: dict[str, dict[int, np.ndarray]] = {}
    ranks[anchor_name] = {h: anchor_values for h in HORIZONS}
    loaded = [(anchor_name, "cgm_last calibration ECDF")] + loaded
    for name, _path in loaded:
        if name == anchor_name:
            continue
        ranks[name] = {}
        for horizon in HORIZONS:
            values = pd.to_numeric(
                frame[f"arm__{name}__{horizon}"], errors="coerce"
            ).to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise SystemExit(f"{name} horizon {horizon} carries non-finite values")
            ranks[name][horizon] = calibration_rank(values, calibration)

    def primary(members: list[str]) -> dict[str, Any]:
        cells = {}
        values = []
        for horizon in HORIZONS:
            cell = _evaluate(
                frame=frame,
                members=[ranks[m][horizon] for m in members],
                horizon=horizon,
                calibration=calibration,
                selection=selection,
                source_order=source_order,
                nominal_fapd=float(args.nominal_fapd),
            )
            cells[str(horizon)] = cell
            values.append(cell["sensitivity"])
        cells["primary"] = float(np.mean(values))
        cells["members"] = list(members)
        return cells

    report: dict[str, Any] = {
        "rows": int(len(frame)),
        "nominal_fapd": float(args.nominal_fapd),
        "promotion_floor": PROMOTION_FLOOR,
        "transferred_sigma": TRANSFERRED_SIGMA,
        "anchor": anchor_name,
        "single": {},
        "combined": {},
    }

    # --- G1: every single-member cell reproduces its own frozen primary --------
    print("\nG1 single-member gates")
    gate_failures = []
    for name, _path in loaded:
        cell = primary([name])
        report["single"][name] = cell
        note = ""
        if name in frozen:
            drift = abs(cell["primary"] - frozen[name])
            note = f"   frozen {frozen[name]:.9f}  drift {drift:.3e}"
            if drift > 1e-9:
                gate_failures.append((name, cell["primary"], frozen[name]))
                note += "  <-- GATE FAILED"
        print(f"  {name:22s} primary {cell['primary']:.9f}{note}")
    if gate_failures:
        report["gate_failures"] = gate_failures
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True))
        raise SystemExit(f"G1 failed for {[f[0] for f in gate_failures]}")

    # --- the combination at every seed ---------------------------------------
    print(f"\ncombination with {anchor_name}")
    for name, _path, _legacy in seeds:
        cell = primary([anchor_name, name])
        report["combined"][name] = cell
        margin = cell["primary"] - PROMOTION_FLOOR
        clears = "CLEARS" if margin > 0 else "below "
        print(
            f"  {anchor_name},{name:16s} primary {cell['primary']:.9f}  "
            f"{clears} floor by {margin:+.9f}"
        )

    single = [report["single"][n]["primary"] for n, _p, _l in seeds]
    combined = [report["combined"][n]["primary"] for n, _p, _l in seeds]
    report["summary"] = {
        "seed_count": len(seeds),
        "single_mean": float(np.mean(single)),
        "single_sd": float(np.std(single, ddof=1)),
        "single_range": [float(min(single)), float(max(single))],
        "combined_mean": float(np.mean(combined)),
        "combined_sd": float(np.std(combined, ddof=1)),
        "combined_range": [float(min(combined)), float(max(combined))],
        "gain_mean": float(np.mean(combined) - np.mean(single)),
        "gain_paired": [float(c - s) for c, s in zip(combined, single)],  # noqa: B905
        "seeds_clearing_floor": int(sum(1 for c in combined if c > PROMOTION_FLOOR)),
    }
    s = report["summary"]
    print(
        f"\nsingle    mean {s['single_mean']:.9f}  sd {s['single_sd']:.9f}  "
        f"range {s['single_range'][0]:.6f}..{s['single_range'][1]:.6f}"
    )
    print(
        f"combined  mean {s['combined_mean']:.9f}  sd {s['combined_sd']:.9f}  "
        f"range {s['combined_range'][0]:.6f}..{s['combined_range'][1]:.6f}"
    )
    print(f"transferred sigma {TRANSFERRED_SIGMA:.9f}")
    print(
        f"seeds clearing the floor: {s['seeds_clearing_floor']} of {s['seed_count']}"
    )
    gains = ", ".join(f"{g:+.6f}" for g in s["gain_paired"])
    print(f"paired gains: {gains}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
