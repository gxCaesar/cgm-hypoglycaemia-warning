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


GRID = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
EQUAL_WEIGHT_REFERENCE = 0.924472288
COMBINATION_SIGMA = 0.011357957


def _score_on(
    frame, ranks, members, weights, horizon, fit_mask, read_mask, source_order, nominal
):
    """Fit the policy on `fit_mask` and score on `read_mask`."""
    blended = sum(w * ranks[m][horizon] for m, w in zip(members, weights))  # noqa: B905
    work = frame.copy()
    work[f"severe_hypo_{horizon}_probability"] = blended
    policy = fit_operational_alarm_policy(
        work.loc[fit_mask], spec=_spec(horizon, source_order), nominal_fapd=nominal
    )
    metrics = operational_endpoint_metrics(
        work.loc[read_mask], spec=_spec(horizon, source_order),
        threshold=policy.threshold,
    )
    return metrics["equal_source_participant_macro_event_sensitivity"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm", action="append", required=True, help="name=path[:legacy]"
    )
    parser.add_argument("--members", required=True,
                        help="two comma-separated names; weight applies to the SECOND")
    parser.add_argument("--nominal-fapd", type=float, default=0.5)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    def parse(item):
        name, _, rest = item.partition("=")
        legacy = rest.endswith(":legacy")
        return name, Path(rest[: -len(":legacy")] if legacy else rest), legacy

    arms = [parse(i) for i in args.arm]
    members = args.members.split(",")
    if len(members) != 2:
        raise SystemExit("this pre-registration covers exactly two members")

    frame = None
    for index, (name, path, legacy) in enumerate(arms):
        part = _read(path, legacy=legacy, base=(index == 0))
        part = part.rename(columns={
            f"severe_hypo_{h}_probability": f"arm__{name}__{h}" for h in HORIZONS})
        frame = (part if frame is None
                 else frame.merge(part, on=KEY, how="inner", validate="one_to_one"))
    assert frame is not None
    print(f"joined one-to-one over {len(frame):,} rows")

    calibration = (frame["partition"] == "calibration").to_numpy()
    selection = (frame["partition"] == "selection").to_numpy()
    source_order = tuple(sorted(frame["source_id"].astype(str).unique()))

    ranks = {}
    for name, _p, _l in arms:
        ranks[name] = {}
        for h in HORIZONS:
            col = pd.to_numeric(frame[f"arm__{name}__{h}"], errors="coerce")
            v = col.to_numpy(float)
            if not np.isfinite(v).all():
                raise SystemExit(f"{name} horizon {h} carries non-finite values")
            ranks[name][h] = calibration_rank(v, calibration)

    # --- STEP 1: the whole grid, scored on CALIBRATION only -------------------
    print()
    print("STEP 1  weight selected on CALIBRATION (selection not read yet)")
    calibration_curve = {}
    for w in GRID:
        vals = [
            _score_on(frame, ranks, members, (1 - w, w), h,
                      calibration, calibration, source_order, args.nominal_fapd)
            for h in HORIZONS
        ]
        calibration_curve[w] = float(np.mean(vals))
        print(f"  w={w:.1f}  calibration {calibration_curve[w]:.9f}")

    chosen = max(calibration_curve, key=lambda k: calibration_curve[k])
    print(f"\n  CHOSEN w = {chosen:.1f} on calibration alone")

    # --- STEP 2: selection, read once, at the chosen weight -------------------
    print()
    print("STEP 2  selection read ONCE, at the chosen weight")
    sel_vals = [
        _score_on(frame, ranks, members, (1 - chosen, chosen), h,
                  calibration, selection, source_order, args.nominal_fapd)
        for h in HORIZONS
    ]
    chosen_selection = float(np.mean(sel_vals))
    gain = chosen_selection - EQUAL_WEIGHT_REFERENCE
    inflation = COMBINATION_SIGMA * float(np.sqrt(2 * np.log(len(GRID))))

    print(f"  w={chosen:.1f}  SELECTION {chosen_selection:.9f}")
    print(f"  equal-weight reference   {EQUAL_WEIGHT_REFERENCE:.9f}")
    print(f"  gain {gain:+.9f}   inflation term {inflation:.9f}")

    if chosen == 0.5:
        outcome = "W-C"
        note = "calibration chooses equal weighting, so a held-out fit agrees"
    elif gain > inflation:
        outcome = "W-A"
        note = "gain exceeds what searching nine weights produces with no effect"
    elif gain > 0:
        outcome = "W-B"
        note = "positive but within selection noise; the headline stays at equal weight"
    else:
        outcome = "W-D"
        note = "calibration and selection disagree about the weight"
    print(f"\n  OUTCOME {outcome}: {note}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "grid": list(GRID), "calibration_curve": calibration_curve,
        "chosen_weight": chosen, "selection_at_chosen": chosen_selection,
        "equal_weight_reference": EQUAL_WEIGHT_REFERENCE, "gain": gain,
        "inflation_term": inflation, "outcome": outcome, "note": note,
        "members": members,
    }, indent=2, sort_keys=True))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
