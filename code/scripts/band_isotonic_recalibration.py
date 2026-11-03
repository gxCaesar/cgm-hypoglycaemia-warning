#!/usr/bin/env python3
"""Band-conditional isotonic recalibration.

Pre-declared in `PREREG-band-conditional-recalibration.md`.

The diagnosis: inside every glucose band the network RANKS well, beating a
glucose-only score by 3.9x to 460x on AUPRC, but its LEVEL is over-predicted by 64x
at low glucose rising monotonically to 786x at high. One global threshold therefore
fires disproportionately where the score is most inflated.

This maps each band's score to its band-conditional probability, fitted on the
calibration partition alone. It preserves the within-band ordering, which the
diagnosis says is already good, and corrects only the level.

It is the OPPOSITE of the refuted `--state-bins` method, which replaced scores with
within-band ranks and thereby forced an equal alarm rate per band. True event rates
differ across bands by 290x, so equalising was catastrophic.

Read-only. Trains nothing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.isotonic import IsotonicRegression

from glycoshift.operational_metrics import (
    OperationalEndpointSpec,
    fit_operational_alarm_policy,
    operational_endpoint_metrics,
)

HORIZONS = (30, 60)


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--bands", type=int, default=6)
    parser.add_argument("--min-cgm-last", type=float, default=70.0)
    parser.add_argument("--nominal-fapd", type=float, default=0.5)
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
    glucose = pd.to_numeric(frame["cgm_last"], errors="coerce").to_numpy(dtype=float)
    frame = frame.loc[glucose >= args.min_cgm_last].copy()
    glucose = glucose[glucose >= args.min_cgm_last]
    is_calibration = (frame["partition"] == "calibration").to_numpy()
    print(f"rows {len(frame):,}  calibration {int(is_calibration.sum()):,}", flush=True)

    if args.bands == 1:
        band = np.zeros(len(frame), dtype=int)
        edges: list[float] = []
    else:
        quantiles = np.linspace(0.0, 1.0, args.bands + 1)[1:-1]
        edges = list(np.unique(np.quantile(glucose[is_calibration], quantiles)))
        band = np.searchsorted(np.asarray(edges), glucose, side="right")
    print(f"bands {len(set(band.tolist()))}  edges {[round(e,1) for e in edges]}",
          flush=True)

    report: dict[str, object] = {
        "arm": args.arm, "bands": args.bands, "min_cgm_last": args.min_cgm_last,
        "band_edges": [float(e) for e in edges], "rows": int(len(frame)),
        "monotonicity_ok": True, "horizons": {},
    }

    for h in HORIZONS:
        score = pd.to_numeric(
            frame[f"severe_hypo_{h}_probability"], errors="coerce"
        ).to_numpy(dtype=float)
        label = frame[f"severe_hypo_{h}_label"].to_numpy(dtype=float)
        recalibrated = np.empty_like(score)
        for index in sorted(set(band.tolist())):
            member = band == index
            fit_rows = member & is_calibration
            if fit_rows.sum() < 100 or label[fit_rows].sum() < 1:
                # No calibration signal in this band: leave the score untouched
                # rather than invent one. Reported, not silently handled.
                recalibrated[member] = score[member]
                report["monotonicity_ok"] = report["monotonicity_ok"] and True
                continue
            model = IsotonicRegression(out_of_bounds="clip", increasing=True)
            model.fit(score[fit_rows], label[fit_rows])
            recalibrated[member] = model.predict(score[member])
            probe = np.linspace(score[fit_rows].min(), score[fit_rows].max(), 512)
            mapped = model.predict(probe)
            if not bool(np.all(np.diff(mapped) >= -1e-12)):
                report["monotonicity_ok"] = False
        frame[f"severe_hypo_{h}_probability"] = recalibrated

    source_order = tuple(sorted(set(frame["source_id"])))
    calibration = frame.loc[frame["partition"] == "calibration"]
    selection = frame.loc[frame["partition"] == "selection"]
    sensitivities = []
    for h in HORIZONS:
        spec = _spec(h, source_order)
        policy = fit_operational_alarm_policy(
            calibration, spec=spec, nominal_fapd=args.nominal_fapd
        )
        metrics = operational_endpoint_metrics(
            selection, spec=spec, threshold=policy.threshold
        )
        sens = metrics["equal_source_participant_macro_event_sensitivity"]
        sensitivities.append(sens)
        report["horizons"][str(h)] = {
            "sensitivity": sens,
            "fapd": metrics[
                "equal_source_participant_macro_false_alarms_per_patient_day"
            ],
            "alarm_count": metrics["alarm_count"],
            "detected_event_count": metrics["detected_event_count"],
        }
        print(f"  h{h}: sensitivity {sens:.9f}", flush=True)

    primary = sum(sensitivities) / len(sensitivities)
    report["primary"] = primary
    print(f"  primary {primary:.9f}   monotonicity_ok {report['monotonicity_ok']}")
    if args.expect_primary is not None:
        drift = abs(primary - args.expect_primary)
        report["gate_drift"] = drift
        print(f"  gate drift {drift:.12f}")
        if drift > 1e-9:
            raise SystemExit("one-band gate failed: recalibration changed the endpoint")
    args.output.write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
