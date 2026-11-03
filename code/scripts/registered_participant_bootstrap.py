#!/usr/bin/env python3
'Registered participant bootstrap.'










































from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# A shared 176-core machine. Nothing here fans out, and pinning costs nothing, but a
# BLAS that decides for itself has put this host at load 2263 before.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from glycoshift.operational_metrics import (  # noqa: E402
    OperationalEndpointSpec,
    participant_alarm_table,
    source_stratified_paired_operational_bootstrap_difference,
)

HORIZONS = (30, 60)
TOL = 1e-12
# Declared here rather than taken from the clock, so a rerun is a rerun.
BOOTSTRAP_SEED = 20260906


def spec_for(horizon: int, source_order: tuple[str, ...]) -> OperationalEndpointSpec:
    return OperationalEndpointSpec(
        score_column=f"severe_hypo_{horizon}_probability",
        label_column=f"severe_hypo_{horizon}_label",
        event_id_column=f"severe_hypo_{horizon}_event_id",
        event_onset_column=f"severe_hypo_{horizon}_event_onset",
        active_column="severe_hypo_currently_active",
        horizon_minutes=horizon,
        source_order=source_order,
    )


def read_selection(path: Path) -> pd.DataFrame:
    columns = ["source_id", "participant_id", "context_end", "partition",
               "severe_hypo_currently_active", "patient_day"]
    for h in HORIZONS:
        columns += [f"severe_hypo_{h}_probability", f"severe_hypo_{h}_label",
                    f"severe_hypo_{h}_event_id", f"severe_hypo_{h}_event_onset"]
    frame = pq.read_table(path, columns=sorted(set(columns))).to_pandas()
    return frame.loc[frame["partition"] == "selection"].reset_index(drop=True)


def equal_source_macro(table: pd.DataFrame, source_order: tuple[str, ...]) -> float:
    """The published aggregation: participant macro inside source, then equal by source.

    Sensitivity is defined only among event-positive participants, which is the same
    population the bootstrap resamples.
    """
    per_source = []
    for source in source_order:
        part = table.loc[table["source_id"].astype(str) == source]
        positive = part.loc[part["event_count"] > 0]
        if positive.empty:
            raise SystemExit(f"REFUSE  no event-positive participant for {source}")
        per_source.append(float(positive["event_sensitivity"].mean()))
    return float(np.mean(per_source))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--campaign-runs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--replicates", type=int, default=1000)
    ap.add_argument("--candidate", default="persistence_seed17")
    ap.add_argument("--reference", default="tcn_erm_seed17")
    ap.add_argument("--also", default="balanced_lstm_erm_seed17",
                    help="a second reference scored against the same candidate")
    args = ap.parse_args()

    arms = [args.candidate, args.reference] + (
        [args.also] if args.also and args.also != args.reference else []
    )
    frozen, thresholds, frames = {}, {}, {}
    for arm in arms:
        node = json.loads((args.campaign_runs / arm / "metrics.json").read_text())
        node = node["operational_v1_3"]
        frozen[arm] = {
            "primary_equal_horizon_mean": node[
                "primary_equal_horizon_mean_event_sensitivity"],
            **{h: node["horizons"][h]["selection_metrics"][
                "equal_source_participant_macro_event_sensitivity"]
               for h in ("30", "60")},
        }
        thresholds[arm] = {h: node["horizons"][h]["policy"]["threshold"]
                           for h in ("30", "60")}
        frames[arm] = read_selection(args.campaign_runs / arm / "predictions.parquet")
        print(f"read {arm}: {len(frames[arm]):,} selection rows")

    source_order = tuple(sorted(frames[arms[0]]["source_id"].astype(str).unique()))
    print(f"sources ({len(source_order)}): {', '.join(source_order)}\n")

    tables: dict[str, dict[int, pd.DataFrame]] = {arm: {} for arm in arms}
    for arm in arms:
        for horizon in HORIZONS:
            table = participant_alarm_table(
                frames[arm],
                spec=spec_for(horizon, source_order),
                threshold=thresholds[arm][str(horizon)],
            )
            rebuilt = equal_source_macro(table, source_order)
            published = frozen[arm][str(horizon)]
            flag = "ok " if abs(rebuilt - published) <= TOL else "FAIL"
            print(f"  gate {flag} {arm:26s} h{horizon}  rebuilt {rebuilt:.16f}  "
                  f"published {published:.16f}")
            if flag == "FAIL":
                print("REFUSE  the participant table is not the one behind the "
                      "published number", file=sys.stderr)
                return 1
            tables[arm][horizon] = table
        mean_rebuilt = np.mean([equal_source_macro(tables[arm][h], source_order)
                                for h in HORIZONS])
        if abs(mean_rebuilt - frozen[arm]["primary_equal_horizon_mean"]) > TOL:
            print(f"REFUSE  {arm} horizon mean {mean_rebuilt!r} against published "
                  f"{frozen[arm]['primary_equal_horizon_mean']!r}", file=sys.stderr)
            return 1
        print(f"  gate ok  {arm:26s} horizon mean {mean_rebuilt:.16f}\n")

    results = {}
    for reference in arms[1:]:
        for horizon in HORIZONS:
            out = source_stratified_paired_operational_bootstrap_difference(
                tables[args.candidate][horizon],
                tables[reference][horizon],
                source_order=source_order,
                replicates=args.replicates,
                seed=BOOTSTRAP_SEED,
            )
            results[f"{args.candidate}_vs_{reference}_h{horizon}"] = out
            print(f"{args.candidate} - {reference}  h{horizon}  "
                  f"{out['estimate']:+.10f}  95% [{out['ci_low']:+.6f}, "
                  f"{out['ci_high']:+.6f}]  p={out['p_value_two_sided']:.4f}")

    payload = {
        "procedure": (
            "the REGISTERED source-stratified paired bootstrap: event-positive "
            "source-scoped participants resampled within source, identical "
            "multiplicities for candidate and reference"
        ),
        "estimator": (
            "glycoshift.operational_metrics."
            "source_stratified_paired_operational_bootstrap_difference"
        ),
        "reported_per_horizon_because": (
            "the implemented estimator is single-horizon; the multiseed variant that "
            "averages contrasts within a draw requires one shared truth table per key "
            "and the two horizons carry different event counts"
        ),
        "supersedes_the_claim": (
            "Methods states the registered procedure cannot be computed from what was "
            "retained. The prediction artifacts were never copied back to the checkout "
            "but were retained on the run host, and the frozen thresholds are recorded "
            "per arm, so no retraining and no threshold sweep was required."
        ),
        "candidate": args.candidate,
        "references": arms[1:],
        "sources": list(source_order),
        "replicates": args.replicates,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "gate": {"tolerance": TOL,
                 "rebuilt_published_frozen_primaries": True,
                 "frozen": frozen},
        "thresholds": thresholds,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
