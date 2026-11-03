#!/usr/bin/env python3
"""Recalibrate with glucose as a covariate, borrowing strength across all events.

`RESULT-band-conditional-recalibration-fails-structurally.md` measured why the
band-conditional version failed, and the failure was structural rather than a tuning
problem: the band whose calibration is worst holds **25** positives in 973,604 rows,
and isotonic fits each band on that band's own events alone. Its constructive
implication was explicit:

  any correction that needs per-band event counts cannot work here; the fix must
  borrow strength across bands, a smooth parametric function of glucose fitted on all
  events at once, with the band structure entering as a covariate rather than as a
  partition.

This is that. One logistic regression on the calibration partition:

    logit(p_recalibrated) = a + b * logit(p_network) + c1 * u + c2 * u^2,  u = log(g)

Four parameters fit on ~5.6M calibration rows, against isotonic's nonparametric map
fit per band on roughly a dozen events. Glucose is a covariate, not a partition.

Unlike the band version this **reorders across glucose**, which is the intervention
the diagnosis actually calls for: within a glucose band the network already ranks
well, by 3.86x to 406x over glucose alone at 30 minutes, while its level is wrong by
64x to 918x and monotonically so.

Read-only. Trains no network.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression

from glycoshift.operational_metrics import (
    OperationalEndpointSpec,
    fit_operational_alarm_policy,
    operational_endpoint_metrics,
)

HORIZONS = (30, 60)
EPS = 1e-6


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


def _design(score: np.ndarray, glucose: np.ndarray, use_glucose: bool) -> np.ndarray:
    clipped = np.clip(score, EPS, 1.0 - EPS)
    logit = np.log(clipped / (1.0 - clipped))
    if not use_glucose:
        return logit.reshape(-1, 1)
    u = np.log(glucose)
    u = (u - u.mean()) / u.std()
    return np.column_stack([logit, u, u * u])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--min-cgm-last", type=float, default=70.0)
    parser.add_argument("--nominal-fapd", type=float, default=0.5)
    parser.add_argument(
        "--no-glucose",
        action="store_true",
        help=(
            "Drop the glucose covariate, leaving plain Platt scaling. Platt is "
            "STRICTLY monotone in the score, so it cannot change the alarm set and "
            "must reproduce the arm's endpoint exactly. That is the gate, and unlike "
            "the isotonic gate its premise is true."
        ),
    )
    parser.add_argument("--expect-primary", type=float, default=None)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    columns = [
        "partition", "source_id", "participant_id", "patient_day", "context_end",
        "cgm_last", "severe_hypo_currently_active",
    ]
    for h in HORIZONS:
        columns += [
            f"severe_hypo_{h}_probability", f"severe_hypo_{h}_label",
            f"severe_hypo_{h}_event_id", f"severe_hypo_{h}_event_onset",
        ]
    frame = pq.read_table(args.predictions, columns=columns).to_pandas()
    frame["source_id"] = frame["source_id"].astype(str)
    glucose_all = pd.to_numeric(frame["cgm_last"], errors="coerce").to_numpy(float)
    keep = glucose_all >= args.min_cgm_last
    frame = frame.loc[keep].copy()
    glucose = glucose_all[keep]
    is_calibration = (frame["partition"] == "calibration").to_numpy()
    print(f"rows {len(frame):,}  calibration {int(is_calibration.sum()):,}", flush=True)

    report: dict[str, object] = {
        "arm": args.arm, "uses_glucose": not args.no_glucose,
        "min_cgm_last": args.min_cgm_last, "rows": int(len(frame)),
        "coefficients": {}, "horizons": {},
    }
    for h in HORIZONS:
        score = pd.to_numeric(
            frame[f"severe_hypo_{h}_probability"], errors="coerce"
        ).to_numpy(float)
        label = frame[f"severe_hypo_{h}_label"].to_numpy(int)
        design = _design(score, glucose, not args.no_glucose)
        model = LogisticRegression(max_iter=1000, C=1e6, solver="lbfgs")
        model.fit(design[is_calibration], label[is_calibration])
        frame[f"severe_hypo_{h}_probability"] = model.predict_proba(design)[:, 1]
        report["coefficients"][str(h)] = {
            "intercept": float(model.intercept_[0]),
            "beta": [float(v) for v in model.coef_[0]],
            "calibration_positives": int(label[is_calibration].sum()),
        }
        beta = report["coefficients"][str(h)]["beta"]
        print(f"  h{h} coefficients {beta}", flush=True)

    source_order = tuple(sorted(set(frame["source_id"])))
    calibration = frame.loc[frame["partition"] == "calibration"]
    selection = frame.loc[frame["partition"] == "selection"]
    values = []
    for h in HORIZONS:
        spec = _spec(h, source_order)
        policy = fit_operational_alarm_policy(
            calibration, spec=spec, nominal_fapd=args.nominal_fapd
        )
        metrics = operational_endpoint_metrics(
            selection, spec=spec, threshold=policy.threshold
        )
        sens = metrics["equal_source_participant_macro_event_sensitivity"]
        values.append(sens)
        report["horizons"][str(h)] = {
            "sensitivity": sens,
            "fapd": metrics[
                "equal_source_participant_macro_false_alarms_per_patient_day"
            ],
            "alarm_count": metrics["alarm_count"],
            "detected_event_count": metrics["detected_event_count"],
        }
        print(f"  h{h}: sensitivity {sens:.9f}", flush=True)

    primary = sum(values) / len(values)
    report["primary"] = primary
    print(f"  primary {primary:.9f}")
    if args.expect_primary is not None:
        drift = abs(primary - args.expect_primary)
        report["gate_drift"] = drift
        print(f"  gate drift {drift:.12f}")
        if drift > 1e-9:
            raise SystemExit(
                "Platt gate failed: a strictly monotone map moved the endpoint"
            )
    args.output.write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
