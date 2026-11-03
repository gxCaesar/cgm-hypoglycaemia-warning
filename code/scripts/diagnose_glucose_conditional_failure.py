#!/usr/bin/env python3
"""WHERE do the networks fail to use current glucose? Calibration or ranking?

`RESULT-the-endpoint-has-headroom-and-it-is-in-current-glucose.md` established that
current glucose carries signal the networks do not extract. It did not say which kind
of failure that is, and the two need opposite fixes:

  CALIBRATION failure  the network ranks correctly within a glucose band but its
                       score level is wrong across bands, so one global threshold
                       lands in the wrong place per band. Fixable POST HOC.

  RANKING failure      within a glucose band the network orders windows worse than
                       glucose itself does. Not fixable post hoc; needs retraining.

Both are measured per band: observed event rate against mean predicted risk gives
calibration, and within-band AUPRC of the network against within-band AUPRC of a
glucose-only score gives ranking.

Read-only, no threshold sweep, so this is fast.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score

HORIZONS = (30, 60)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--min-cgm-last", type=float, default=70.0)
    parser.add_argument("--bands", type=int, default=6)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    columns = ["partition", "cgm_last", "severe_hypo_currently_active"]
    for h in HORIZONS:
        columns += [f"severe_hypo_{h}_probability", f"severe_hypo_{h}_label"]
    frame = pq.read_table(args.predictions, columns=columns).to_pandas()
    frame = frame.loc[frame["partition"] == "selection"]
    frame = frame.loc[~frame["severe_hypo_currently_active"].astype(bool)]
    glucose = pd.to_numeric(frame["cgm_last"], errors="coerce").to_numpy(dtype=float)
    frame = frame.loc[glucose >= args.min_cgm_last].copy()
    glucose = glucose[glucose >= args.min_cgm_last]

    edges = np.quantile(glucose, np.linspace(0.0, 1.0, args.bands + 1))
    edges[-1] = np.inf
    band = np.searchsorted(edges[1:-1], glucose, side="right")

    report: dict[str, object] = {
        "arm": args.arm,
        "min_cgm_last": args.min_cgm_last,
        "rows": int(len(frame)),
        "band_edges": [float(e) for e in edges[:-1]] + ["inf"],
        "horizons": {},
    }
    for h in HORIZONS:
        score = pd.to_numeric(
            frame[f"severe_hypo_{h}_probability"], errors="coerce"
        ).to_numpy(dtype=float)
        label = frame[f"severe_hypo_{h}_label"].to_numpy(dtype=int)
        rows = []
        for index in range(args.bands):
            member = band == index
            if member.sum() < 100 or label[member].sum() < 5:
                rows.append({"band": index, "status": "too_few"})
                continue
            # Glucose is protective here: lower glucose means higher risk, so the
            # glucose-only score is its negation.
            rows.append(
                {
                    "band": index,
                    "glucose_low": float(glucose[member].min()),
                    "glucose_high": float(glucose[member].max()),
                    "n": int(member.sum()),
                    "positives": int(label[member].sum()),
                    "observed_rate": float(label[member].mean()),
                    "mean_predicted": float(score[member].mean()),
                    "auprc_network": float(
                        average_precision_score(label[member], score[member])
                    ),
                    "auprc_glucose_only": float(
                        average_precision_score(label[member], -glucose[member])
                    ),
                }
            )
        report["horizons"][str(h)] = rows
        print(f"\nhorizon {h} min, arm {args.arm}")
        print(
            f"  {'band':>4} {'glucose':>13} {'n':>9} {'obs rate':>9} "
            f"{'mean pred':>10} {'AUPRC net':>10} {'AUPRC gluc':>11}"
        )
        for row in rows:
            if row.get("status") == "too_few":
                print(f"  {row['band']:>4}  too few")
                continue
            span = f"{row['glucose_low']:.0f}-{row['glucose_high']:.0f}"
            print(
                f"  {row['band']:>4} {span:>13}"
                f" {row['n']:>9,} {row['observed_rate']:>9.5f}"
                f" {row['mean_predicted']:>10.5f} {row['auprc_network']:>10.5f}"
                f" {row['auprc_glucose_only']:>11.5f}"
            )
    args.output.write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
