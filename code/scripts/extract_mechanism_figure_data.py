#!/usr/bin/env python3
"""Emit figure-ready data for the mechanism figure: lead-time gap and three traces.

Read-only. Re-derives the alarm sets under the frozen policy and gates on
reproducing each arm's frozen 60-minute sensitivity before emitting anything,
for the same reason the case selection does: a trace drawn from an alarm set
that does not reproduce the endpoint is a trace from a different analysis.

Emits the FULL difference vector rather than summary statistics, so the figure
draws the distribution it claims to draw.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from glycoshift.metrics import _collapsed_alarm_rows, _event_table, _match_alarms
from glycoshift.operational_metrics import (
    OperationalEndpointSpec,
    fit_operational_alarm_policy,
    operational_endpoint_metrics,
)

H = 60
SCORE = f"hypo_{H}_probability"
WINDOW_BEFORE_MIN = 240
WINDOW_AFTER_MIN = 60


def spec(order):
    return OperationalEndpointSpec(
        score_column=SCORE, label_column=f"severe_hypo_{H}_label",
        event_id_column=f"severe_hypo_{H}_event_id",
        event_onset_column=f"severe_hypo_{H}_event_onset",
        active_column="severe_hypo_currently_active",
        horizon_minutes=H, source_order=tuple(order))


def prepare(path, order, expect, label):
    cols = ["source_id", "participant_id", "context_end", "patient_day", "partition",
            "cgm_last", "severe_hypo_currently_active", SCORE,
            f"severe_hypo_{H}_label", f"severe_hypo_{H}_event_id",
            f"severe_hypo_{H}_event_onset"]
    frame = pq.read_table(path, columns=cols).to_pandas()
    sp = spec(order)
    policy = fit_operational_alarm_policy(
        frame[frame["partition"] == "calibration"], spec=sp, nominal_fapd=0.5)
    sel = frame[frame["partition"] == "selection"]
    got = operational_endpoint_metrics(sel, spec=sp, threshold=policy.threshold)
    rep = got["equal_source_participant_macro_event_sensitivity"]
    if abs(rep - expect) > 5e-9:
        raise SystemExit(f"{label}: reproduced {rep!r} against a frozen {expect!r}")
    print(f"  {label:<12} threshold {policy.threshold:.9f}  reproduces {rep:.9f}")
    eligible = sel[~sel[sp.active_column].astype(bool)]
    alarms = _collapsed_alarm_rows(
        eligible, probability_column=sp.score_column,
        event_id_column=sp.event_id_column, threshold=policy.threshold,
        collapse_minutes=sp.collapse_minutes)
    events = _event_table(eligible, event_id_column=sp.event_id_column,
                          event_onset_column=sp.event_onset_column)
    ids, _, leads = _match_alarms(alarms, events, horizon_minutes=H)
    return eligible, alarms, events, dict(zip(ids, leads, strict=True))


ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--threshold-arm", required=True, type=Path)
ap.add_argument("--network-arm", required=True, type=Path)
ap.add_argument("--sources", required=True)
ap.add_argument("--output", required=True, type=Path)
a = ap.parse_args()
order = tuple(s.strip() for s in a.sources.split(","))

print("gating on the frozen 60-minute sensitivities")
t_el, t_al, events, t_leads = prepare(a.threshold_arm, order, 0.8238759207890576,
                                      "threshold")
n_el, n_al, _, n_leads = prepare(a.network_arm, order, 0.7056468700448905, "network")

both = sorted(set(t_leads) & set(n_leads))
diff = np.array([n_leads[e] - t_leads[e] for e in both], dtype=float)
print(f"\nqualifying events {len(both)}  median diff {np.median(diff):+.1f} min  "
      f"threshold-earlier fraction {(diff < 0).mean():.4f}")

# The registered rule breaks ties by earlier event onset, then source, then
# participant. An earlier version ranked on argsort(diff) alone, which is stable on
# event-id order, so it selected DIFFERENT events from the ones the selection script
# produces while the caption claimed the plan had chosen them. Ordered here exactly
# as select_case_studies.py orders.
ev = events.copy()
key = "event_id" if "event_id" in ev.columns else ev.columns[0]
order = pd.DataFrame({"event_id": both, "diff": diff}).merge(
    ev, left_on="event_id", right_on=key, how="left")
sort_cols = ["diff"] + [c for c in ("event_onset", "source_id", "participant_id")
                        if c in order.columns]
order = order.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
both = list(order["event_id"])
diff = order["diff"].to_numpy(dtype=float)
ev = ev.set_index(key)
rank = np.arange(len(both))
picks = {"p10": int(round(0.10 * (len(both) - 1))),
         "median": int(round(0.50 * (len(both) - 1))),
         "p90": int(round(0.90 * (len(both) - 1)))}

out = {"horizon_minutes": H, "nominal_fapd": 0.5, "n_events": len(both),
       "diff_minutes": [float(x) for x in diff],
       "median_diff": float(np.median(diff)),
       "quartiles": [float(np.percentile(diff, q)) for q in (25, 50, 75)],
       "fraction_threshold_earlier": float((diff < 0).mean()), "cases": {}}

onset_col = "event_onset" if "event_onset" in events.columns else None
for name, r in picks.items():
    eid = both[rank[r]]
    row = ev.loc[eid]
    pid, src = str(row["participant_id"]), str(row["source_id"])
    onset = pd.Timestamp(row[onset_col])
    lo = onset - pd.Timedelta(minutes=WINDOW_BEFORE_MIN)
    hi = onset + pd.Timedelta(minutes=WINDOW_AFTER_MIN)

    def slice_of(frame, cgm=False, *, pid=pid, src=src, lo=lo, hi=hi):
        # bound as defaults: the closure is called inside this iteration so late
        # binding does not bite today, but it would the moment it is deferred
        m = ((frame["participant_id"].astype(str) == pid)
             & (frame["source_id"].astype(str) == src)
             & (frame["context_end"] >= lo) & (frame["context_end"] <= hi))
        s = frame.loc[m].sort_values("context_end")
        if cgm:
            return ([t.isoformat() for t in s["context_end"]],
                    [float(v) for v in s["cgm_last"]])
        return [t.isoformat() for t in s["context_end"]]

    times, cgm = slice_of(t_el, cgm=True)
    out["cases"][name] = {
        "event_id": eid, "source_id": src, "participant_id": pid,
        "event_onset": onset.isoformat(),
        "lead_threshold_min": float(t_leads[eid]),
        "lead_network_min": float(n_leads[eid]),
        "diff_min": float(n_leads[eid] - t_leads[eid]),
        "trace_time": times, "trace_cgm": cgm,
        "alarms_threshold": slice_of(t_al), "alarms_network": slice_of(n_al),
    }
    print(f"  {name:<7} {src} / {pid}  diff {out['cases'][name]['diff_min']:+.1f} min  "
          f"trace points {len(cgm)}  alarms t/n "
          f"{len(out['cases'][name]['alarms_threshold'])}/"
          f"{len(out['cases'][name]['alarms_network'])}")

a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(out, indent=1))
print(f"\nwrote {a.output}")
