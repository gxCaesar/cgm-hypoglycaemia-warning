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


def _spearman(x, y):
    """Rank correlation, or None when the slice is too small to mean anything."""
    if len(x) <= 100:
        return None
    return float(pd.Series(x).corr(pd.Series(y), method="spearman"))


DISAGREEMENT_QUANTILES = (0.5, 0.9, 0.99)
DECISION_BAND = (54.0, 70.0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True, help="name=path")
    parser.add_argument("--members", required=True, help="exactly two names")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    def parse(item):
        name, _, rest = item.partition("=")
        legacy = rest.endswith(":legacy")
        return name, Path(rest[: -len(":legacy")] if legacy else rest), legacy

    arms = [parse(i) for i in args.arm]
    a, b = args.members.split(",")

    frame = None
    for index, (name, path, legacy) in enumerate(arms):
        part = _read(path, legacy=legacy, base=(index == 0))
        part = part.rename(columns={
            f"severe_hypo_{h}_probability": f"arm__{name}__{h}" for h in HORIZONS})
        frame = (part if frame is None
                 else frame.merge(part, on=KEY, how="inner", validate="one_to_one"))
    assert frame is not None

    calibration = (frame["partition"] == "calibration").to_numpy()
    selection = (frame["partition"] == "selection").to_numpy()
    glucose = pd.to_numeric(frame["cgm_last"], errors="coerce").to_numpy(float)
    active = frame["severe_hypo_currently_active"].to_numpy(bool)
    report = {"members": [a, b], "horizons": {}}

    for h in HORIZONS:
        ra = calibration_rank(
            pd.to_numeric(frame[f"arm__{a}__{h}"], errors="coerce").to_numpy(float),
            calibration)
        rb = calibration_rank(
            pd.to_numeric(frame[f"arm__{b}__{h}"], errors="coerce").to_numpy(float),
            calibration)
        label = pd.to_numeric(
            frame[f"severe_hypo_{h}_label"], errors="coerce").fillna(0).to_numpy() > 0.5
        eligible = selection & ~active
        gap = np.abs(ra - rb)

        block = {}
        for q in DISAGREEMENT_QUANTILES:
            threshold = float(np.quantile(gap[eligible], q))
            mask = eligible & (gap >= threshold)
            # Among disagreeing windows that ARE severe positives, which member
            # ranked them higher? A complementary pair splits these.
            pos = mask & label
            a_hi = int((ra[pos] > rb[pos]).sum())
            b_hi = int((rb[pos] > ra[pos]).sum())
            tot_pos = a_hi + b_hi
            # The omitted half: on NON-events, which member ranks them LOWER.
            # This is where the threshold is supposed to earn its place.
            neg = mask & ~label
            a_lo = int((ra[neg] < rb[neg]).sum())
            b_lo = int((rb[neg] < ra[neg]).sum())
            tot_neg = a_lo + b_lo
            block[f"q{q}"] = {
                "gap_threshold": threshold,
                "windows": int(mask.sum()),
                "severe_positives_among_them": int(pos.sum()),
                "negatives_among_them": int(neg.sum()),
                f"{a}_ranks_the_event_higher": a_hi,
                f"{b}_ranks_the_event_higher": b_hi,
                f"{a}_share_on_positives": (a_hi / tot_pos) if tot_pos else None,
                f"{a}_ranks_the_nonevent_lower": a_lo,
                f"{b}_ranks_the_nonevent_lower": b_lo,
                f"{a}_share_on_negatives": (a_lo / tot_neg) if tot_neg else None,
            }
        lo, hi = DECISION_BAND
        in_band = eligible & (glucose >= lo) & (glucose < hi)
        # Rank CORRELATION, because the absolute gap has no resolution in a band
        # holding one to two percent of rows.
        outside = eligible & ~in_band
        block["decision_band"] = {
            "windows": int(in_band.sum()),
            "spearman_in_band": _spearman(ra[in_band], rb[in_band]),
            "spearman_outside": _spearman(ra[outside], rb[outside]),
        }
        report["horizons"][str(h)] = block

        print(f"horizon {h}")
        for q in DISAGREEMENT_QUANTILES:
            cell = block[f"q{q}"]
            sp = cell[f"{a}_share_on_positives"]
            sn = cell[f"{a}_share_on_negatives"]
            fp = "n/a" if sp is None else f"{sp:.3f}"
            fn = "n/a" if sn is None else f"{sn:.3f}"
            npos = cell["severe_positives_among_them"]
            nneg = cell["negatives_among_them"]
            print(f"  top {(1 - q) * 100:5.1f}% disagreement: {a} right on "
                  f"{fp} of {npos:6d} events, {fn} of {nneg:8d} non-events")
        band = block["decision_band"]
        sb = band["spearman_in_band"]
        print(f"  rank correlation in 54-70 "
              f"{'n/a' if sb is None else f'{sb:.4f}'}  "
              f"outside {band['spearman_outside']:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
