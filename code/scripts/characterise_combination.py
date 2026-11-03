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


# Fixed before scoring. 54 is the frozen default; the rest walk toward the
# clinical level-1 line at 70, where the threshold is already known to collapse.
ELIGIBILITY_CUTS = (54.0, 60.0, 65.0, 70.0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True, help="name=path")
    parser.add_argument("--combination", required=True, help="comma-separated members")
    parser.add_argument("--nominal-fapd", type=float, default=0.5)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    def parse(item):
        name, _, rest = item.partition("=")
        legacy = rest.endswith(":legacy")
        return name, Path(rest[: -len(":legacy")] if legacy else rest), legacy

    arms = [parse(i) for i in args.arm]
    members = args.combination.split(",")

    frame = None
    for index, (name, path, legacy) in enumerate(arms):
        part = _read(path, legacy=legacy, base=(index == 0))
        part = part.rename(columns={
            f"severe_hypo_{h}_probability": f"arm__{name}__{h}" for h in HORIZONS})
        frame = (part if frame is None
                 else frame.merge(part, on=KEY, how="inner", validate="one_to_one"))
    assert frame is not None
    print(f"joined one-to-one over {len(frame):,} rows")

    glucose = pd.to_numeric(frame["cgm_last"], errors="coerce").to_numpy(float)
    source_order = tuple(sorted(frame["source_id"].astype(str).unique()))
    report = {"combination": members, "cuts": {}, "per_source": {}}

    for cut in ELIGIBILITY_CUTS:
        keep = glucose >= cut if cut > 54.0 else np.ones(len(frame), dtype=bool)
        sub = frame.loc[keep].copy()
        cal = (sub["partition"] == "calibration").to_numpy()
        sel = (sub["partition"] == "selection").to_numpy()
        if cal.sum() == 0 or sel.sum() == 0:
            print(f"  cut {cut}: a partition is empty, skipped", flush=True)
            continue
        ranks = {}
        for name, _p, _l in arms:
            ranks[name] = {}
            for h in HORIZONS:
                col = pd.to_numeric(sub[f"arm__{name}__{h}"], errors="coerce")
                ranks[name][h] = calibration_rank(col.to_numpy(float), cal)

        row = {}
        try:
            cells = [(m, [m]) for m in members]
            cells.append((",".join(members), members))
            for label, mem in cells:
                vals, per_source = [], {}
                for h in HORIZONS:
                    work = sub.copy()
                    work[f"severe_hypo_{h}_probability"] = np.mean(
                        [ranks[m][h] for m in mem], axis=0)
                    policy = fit_operational_alarm_policy(
                        work.loc[cal], spec=_spec(h, source_order),
                        nominal_fapd=args.nominal_fapd)
                    metrics = operational_endpoint_metrics(
                        work.loc[sel], spec=_spec(h, source_order),
                        threshold=policy.threshold)
                    k = "equal_source_participant_macro_event_sensitivity"
                    vals.append(metrics[k])
                    if cut == 54.0:
                        for s, block in metrics.get("per_source", {}).items():
                            per_source.setdefault(s, {})[str(h)] = {
                                "sensitivity": block[
                                    "participant_macro_event_sensitivity"],
                                "participants": block["participant_count"],
                                "events": block["event_count"],
                            }
                row[label] = float(np.mean(vals))
                if cut == 54.0 and per_source:
                    report["per_source"][label] = per_source
        except Exception as exc:  # the evaluator refuses some cuts by design
            print(f"  cut {cut}: evaluator refused, {exc}")
            report["cuts"][str(cut)] = {"refused": str(exc)}
            continue

        report["cuts"][str(cut)] = row
        combo = row[",".join(members)]
        best_member = max(row[m] for m in members)
        verdict = "combination leads" if combo > best_member else "A MEMBER LEADS"
        pieces = "  ".join(f"{m} {row[m]:.6f}" for m in members)
        print(f"  cut {cut:5.1f}   {pieces}   combo {combo:.6f}   {verdict}",
              flush=True)

        # Write after every cut, not only at the end. This run takes hours and the
        # previous version produced nothing until it finished, so nobody could tell a
        # job that was most of the way through from one that was stuck, and a job that
        # died lost every cut it had already computed.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
