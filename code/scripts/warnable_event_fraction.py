#!/usr/bin/env python3
"""Do severe hypoglycaemia events differ in whether they can be warned about at all?

An event is **warnable at horizon h** if at least one of its eligible lead-in
windows is taken from a non-hypoglycaemic state -- that is, if a warning could
have been issued before the patient was already low. An event whose every
eligible lead-in row already had the patient below 70 mg/dL has no euglycaemic
warning window, and NO system can warn about it early.

The aggregate is already recorded in this bundle: 80.0% of thirty-minute severe
events are warnable and 93.2% at sixty. **This asks whether that fraction is a
property of the disease or a property of the cohort**, by computing it per source
and per participant.

Read-only. Opens completed arms' own prediction artifacts, trains nothing, and
touches no external partition.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

HORIZONS = (30, 60)
EUGLYCAEMIC_FLOOR = 70.0


def _columns() -> list[str]:
    cols = ["source_id", "participant_id", "partition", "cgm_last",
            "severe_hypo_currently_active"]
    for h in HORIZONS:
        cols += [f"severe_hypo_{h}_event_id", f"severe_hypo_{h}_label"]
    return cols


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--floor", type=float, default=EUGLYCAEMIC_FLOOR)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    frame = pq.read_table(args.predictions, columns=_columns()).to_pandas()
    frame["source_id"] = frame["source_id"].astype(str)
    frame["participant_id"] = frame["participant_id"].astype(str)
    eligible = ~frame["severe_hypo_currently_active"].to_numpy(dtype=bool)
    euglycaemic = (
        pd.to_numeric(frame["cgm_last"], errors="coerce").to_numpy(dtype=float)
        >= args.floor
    )
    print(
        f"rows {len(frame):,}   eligible {int(eligible.sum()):,}   "
        f"eligible and euglycaemic {int((eligible & euglycaemic).sum()):,}",
        flush=True,
    )

    report: dict[str, Any] = {
        "rows": int(len(frame)),
        "euglycaemic_floor": args.floor,
        "argv": list(argv) if argv is not None else sys.argv[1:],
        "horizons": {},
    }

    for h in HORIZONS:
        ids = frame[f"severe_hypo_{h}_event_id"]
        present = ids.notna().to_numpy()
        work = pd.DataFrame({
            "source_id": frame["source_id"],
            "participant_id": frame["participant_id"],
            "event_id": ids.astype(str),
            # A lead-in row counts only if the endpoint would have scored it AND
            # the patient was not already hypoglycaemic there.
            "warnable_row": eligible & euglycaemic,
            "eligible_row": eligible,
        }).loc[present]

        per_event = work.groupby(
            ["source_id", "participant_id", "event_id"], observed=True, sort=True
        ).agg(
            warnable=("warnable_row", "any"),
            eligible=("eligible_row", "any"),
            # THE CONFOUND. An event with fewer eligible lead-in rows has fewer
            # chances to contain a euglycaemic one, so a cohort with sparser
            # lead-ins would show lower warnability for a reason that has nothing
            # to do with how its hypoglycaemia arrives. Carried per event so the
            # alternative explanation can be tested rather than assumed away.
            eligible_rows=("eligible_row", "sum"),
        )
        # An event with no eligible lead-in row at all is outside the endpoint's
        # denominator and is excluded rather than counted as unwarnable.
        per_event = per_event.loc[per_event["eligible"]]

        overall = float(per_event["warnable"].mean())
        by_source = per_event.groupby("source_id", observed=True)["warnable"].agg(
            ["size", "sum", "mean"]
        )
        by_participant = per_event.groupby(
            ["source_id", "participant_id"], observed=True
        )["warnable"].agg(["size", "mean"])

        sources = {
            str(s): {
                "events": int(r["size"]),
                "warnable": int(r["sum"]),
                "fraction": float(r["mean"]),
            }
            for s, r in by_source.iterrows()
        }
        # Spread across sources, and across participants holding >= 5 events so a
        # single-event participant cannot manufacture a 0 or a 1.
        dense = by_participant.loc[by_participant["size"] >= 5, "mean"]
        # Confound 1: does warnability track lead-in density across sources?
        density = (
            per_event.groupby("source_id", observed=True)["eligible_rows"].median()
        )
        density_corr = float(by_source["mean"].corr(density, method="spearman"))

        # Confound 2: is the variation between COHORTS or between PATIENTS? If
        # patients inside one cohort vary as much as cohorts do, this is not a
        # cohort property and the source-level table is the wrong summary.
        dense_idx = by_participant["size"] >= 5
        within = (
            by_participant.loc[dense_idx]
            .groupby("source_id", observed=True)["mean"]
            .std(ddof=1)
            .dropna()
        )

        report["horizons"][str(h)] = {
            "confound_lead_in_density": {
                "median_eligible_rows_per_event_by_source": {
                    str(k): float(v) for k, v in density.items()
                },
                "spearman_warnability_vs_density": density_corr,
            },
            "within_source_participant_sd": {
                str(k): float(v) for k, v in within.items()
            },
            "median_within_source_participant_sd": (
                float(within.median()) if len(within) else None
            ),
            "events_with_an_eligible_lead_in": int(len(per_event)),
            "warnable_events": int(per_event["warnable"].sum()),
            "overall_fraction": overall,
            "per_source": sources,
            "source_fraction_min": float(by_source["mean"].min()),
            "source_fraction_max": float(by_source["mean"].max()),
            "source_fraction_sd": float(by_source["mean"].std(ddof=1)),
            "participants_with_5plus_events": int(len(dense)),
            "participant_fraction_sd": (
                float(dense.std(ddof=1)) if len(dense) > 1 else None
            ),
            "participant_fraction_iqr": (
                [float(dense.quantile(0.25)), float(dense.quantile(0.75))]
                if len(dense) > 1 else None
            ),
        }
        print(
            f"h={h}: {int(per_event['warnable'].sum()):,} of {len(per_event):,} "
            f"events warnable ({overall:.4f}); per-source "
            f"{by_source['mean'].min():.4f} to {by_source['mean'].max():.4f}, "
            f"sd {by_source['mean'].std(ddof=1):.4f}",
            flush=True,
        )
        for s, r in by_source.iterrows():
            print(
                f"    {str(s):16} {int(r['sum']):5,} / {int(r['size']):5,}  "
                f"{r['mean']:.4f}   median eligible lead-in rows "
                f"{density.get(s, float('nan')):.0f}",
                flush=True,
            )
        print(
            f"  CONFOUND spearman(warnability, lead-in density) = {density_corr:+.4f}",
            flush=True,
        )
        print(
            f"  between-source sd {by_source['mean'].std(ddof=1):.4f}   "
            f"median WITHIN-source participant sd "
            f"{(within.median() if len(within) else float('nan')):.4f}",
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
