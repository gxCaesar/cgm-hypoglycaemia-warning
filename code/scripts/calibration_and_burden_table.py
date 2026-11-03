#!/usr/bin/env python3
"""Calibration and alarm-burden metrics at each arm's own frozen threshold.

Read-only. Computes what Results 4.6 needs from completed arms' prediction
artifacts and their frozen alarm policies: Brier score, calibration slope and
intercept, expected calibration error, achieved alarm burden, and median warning
lead time. It computes nothing at a threshold other than the one the frozen policy
selected, so no value here comes from a swept or re-tuned operating point.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from glycoshift.metrics import (
    alarm_metrics,
    calibration_slope_intercept,
    expected_calibration_error,
)

HORIZONS = (30, 60)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    operational = json.loads(
        (args.run_dir / "operational_metrics.json").read_text(encoding="utf-8")
    )
    columns = ["partition", "source_id", "participant_id", "patient_day",
               "context_end", "severe_hypo_currently_active"]
    for h in HORIZONS:
        columns += [f"severe_hypo_{h}_probability", f"severe_hypo_{h}_label",
                    f"severe_hypo_{h}_event_id", f"severe_hypo_{h}_event_onset"]
    frame = pq.read_table(args.run_dir / "predictions.parquet",
                          columns=columns).to_pandas()
    frame = frame.loc[frame["partition"] == "selection"]
    frame = frame.loc[~frame["severe_hypo_currently_active"].astype(bool)].copy()

    report: dict[str, object] = {"arm": args.arm, "partition": "selection",
                                 "eligible_rows": int(len(frame)), "horizons": {}}
    for h in HORIZONS:
        policy = operational["horizons"][str(h)]["policy"]
        threshold = float(policy["threshold"])
        probability = pd.to_numeric(
            frame[f"severe_hypo_{h}_probability"], errors="coerce"
        ).to_numpy(dtype=float)
        label = frame[f"severe_hypo_{h}_label"].to_numpy(dtype=float)
        slope, intercept = calibration_slope_intercept(label.astype(int), probability)
        burden = alarm_metrics(
            frame,
            probability_column=f"severe_hypo_{h}_probability",
            label_column=f"severe_hypo_{h}_label",
            event_id_column=f"severe_hypo_{h}_event_id",
            threshold=threshold,
            horizon_minutes=h,
            event_onset_column=f"severe_hypo_{h}_event_onset",
        )
        # Reliability curve: bin by predicted probability, not by glucose. Bins are
        # equal-frequency because the score distribution is extremely skewed and
        # equal-width bins would put almost every row in the first one.
        edges = np.unique(np.quantile(probability, np.linspace(0.0, 1.0, 11)))
        assignment = np.digitize(probability, edges[1:-1], right=True)
        reliability = []
        for index in range(len(edges) - 1):
            member = assignment == index
            if member.sum() < 100:
                continue
            reliability.append({
                "bin": index,
                "n": int(member.sum()),
                "mean_predicted": float(probability[member].mean()),
                "observed_rate": float(label[member].mean()),
                "positives": int(label[member].sum()),
            })

        # Per-participant alarm burden at the frozen threshold, so the reader can
        # see the distribution behind a cohort-average false-alarm rate.
        fires = probability >= threshold
        per_participant = (
            frame.assign(_fire=fires)
            .groupby(["source_id", "participant_id"], observed=True)["_fire"]
            .sum()
            .to_numpy(dtype=float)
        )

        report["horizons"][str(h)] = {
            "reliability_bins": reliability,
            "participant_firing_counts": {
                "n_participants": int(len(per_participant)),
                "median": float(np.median(per_participant)),
                "p90": float(np.quantile(per_participant, 0.9)),
                "max": float(per_participant.max()),
                "share_of_firings_in_top_decile": float(
                    np.sort(per_participant)[-max(1, len(per_participant) // 10):].sum()
                    / max(per_participant.sum(), 1.0)
                ),
            },
            "frozen_threshold": threshold,
            "score_transform": policy.get("score_transform"),
            "brier": float(np.mean((probability - label) ** 2)),
            "observed_event_rate": float(label.mean()),
            "mean_predicted": float(probability.mean()),
            "calibration_slope": slope,
            "calibration_intercept": intercept,
            "expected_calibration_error": expected_calibration_error(
                label, probability
            ),
            "median_lead_minutes": burden.get("median_lead_minutes"),
            "false_alarms_per_patient_day": burden.get(
                "false_alarms_per_patient_day"
            ),
            "alarms": burden.get("alarms"),
            "matched_events": burden.get("matched_events"),
        }
        row = report["horizons"][str(h)]
        print(f"  h{h}: brier {row['brier']:.6g}  slope {slope:.4f}  "
              f"intercept {intercept:.4f}  ECE "
              f"{row['expected_calibration_error']:.6g}  "
              f"lead {row['median_lead_minutes']}", flush=True)
    args.output.write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
