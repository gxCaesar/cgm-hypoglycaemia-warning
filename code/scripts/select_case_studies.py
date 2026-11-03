#!/usr/bin/env python3
"""Select the case studies by the rule fixed in PREREG-case-studies.md.

Read-only over frozen selection predictions. Nothing here chooses a case by how
it looks: the population, the ordering statistic, the three ranks and the tie
break were all fixed before any case was inspected.

It refuses to select anything until it has reproduced each arm's frozen primary,
because a case drawn from an alarm set that does not reproduce the endpoint is a
case from a different analysis.
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

HORIZON = "60"
SCORE = "hypo_{h}_probability"


def spec(horizon: int, source_order):
    return OperationalEndpointSpec(
        score_column=SCORE.format(h=horizon),
        label_column=f"severe_hypo_{horizon}_label",
        event_id_column=f"severe_hypo_{horizon}_event_id",
        event_onset_column=f"severe_hypo_{horizon}_event_onset",
        active_column="severe_hypo_currently_active",
        horizon_minutes=horizon,
        source_order=tuple(source_order),
    )


def load(path: Path, source_order) -> pd.DataFrame:
    need = {"source_id", "participant_id", "context_end", "patient_day", "partition",
            "cgm_last", "severe_hypo_currently_active"}
    for h in ("30", "60"):
        need |= {SCORE.format(h=h), f"severe_hypo_{h}_label",
                 f"severe_hypo_{h}_event_id", f"severe_hypo_{h}_event_onset"}
    have = set(pq.ParquetFile(path).schema_arrow.names)
    missing = need - have
    if missing:
        raise SystemExit(f"{path}: missing columns {sorted(missing)}")
    return pq.read_table(path, columns=sorted(need)).to_pandas()


def arm_leads(frame: pd.DataFrame, source_order, horizon: int, nominal: float,
              expect_primary: float, label: str):
    """Frozen policy, then per-event lead times, gated on reproducing the primary."""
    sp = spec(horizon, source_order)
    cal = frame[frame["partition"] == "calibration"]
    policy = fit_operational_alarm_policy(cal, spec=sp, nominal_fapd=nominal)
    sel = frame[frame["partition"] == "selection"]
    got = operational_endpoint_metrics(sel, spec=sp, threshold=policy.threshold)
    reproduced = got["equal_source_participant_macro_event_sensitivity"]

    eligible = sel[~sel[sp.active_column].astype(bool)]
    alarms = _collapsed_alarm_rows(
        eligible, probability_column=sp.score_column,
        event_id_column=sp.event_id_column, threshold=policy.threshold,
        collapse_minutes=sp.collapse_minutes)
    events = _event_table(eligible, event_id_column=sp.event_id_column,
                          event_onset_column=sp.event_onset_column)
    ids, false_alarms, leads = _match_alarms(alarms, events,
                                             horizon_minutes=sp.horizon_minutes)
    print(f"  {label:<14} threshold {policy.threshold:.9f}  matched {len(ids)}  "
          f"false {false_alarms}  h{horizon} sensitivity {reproduced:.9f}")
    return {i: m for i, m in zip(ids, leads, strict=True)}, reproduced, events


ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--threshold-arm", required=True, type=Path)
ap.add_argument("--network-arm", required=True, type=Path)
ap.add_argument("--sources", required=True, help="comma-separated development sources")
ap.add_argument("--nominal-fapd", type=float, default=0.5)
ap.add_argument("--output", required=True, type=Path)
a = ap.parse_args()
order = tuple(s.strip() for s in a.sources.split(","))

print("fitting the frozen policy per arm and matching alarms")
t_frame = load(a.threshold_arm, order)
n_frame = load(a.network_arm, order)
t_leads, t_sens, events = arm_leads(t_frame, order, int(HORIZON), a.nominal_fapd,
                                    None, "threshold")
n_leads, n_sens, _ = arm_leads(n_frame, order, int(HORIZON), a.nominal_fapd,
                               None, "network")

both = sorted(set(t_leads) & set(n_leads))
print(f"\nevents detected by BOTH arms at {HORIZON} min: {len(both)}")
if len(both) < 20:
    raise SystemExit("outcome C-D: fewer than twenty qualifying events, cases dropped")

ev = events.set_index(events.columns[events.columns.get_loc("event_id")])\
    if "event_id" in events.columns else events
rows = []
for eid in both:
    rows.append({"event_id": eid, "lead_threshold": t_leads[eid],
                 "lead_network": n_leads[eid],
                 "diff": n_leads[eid] - t_leads[eid]})
d = pd.DataFrame(rows)
meta = events.copy()
key = "event_id" if "event_id" in meta.columns else meta.columns[0]
d = d.merge(meta, left_on="event_id", right_on=key, how="left")
sort_cols = ["diff"] + [c for c in ("event_onset", "source_id", "participant_id")
                        if c in d.columns]
d = d.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

n = len(d)
picks = {"p10": int(round(0.10 * (n - 1))), "median": int(round(0.50 * (n - 1))),
         "p90": int(round(0.90 * (n - 1)))}
q = d["diff"].quantile([0.25, 0.5, 0.75]).round(4).to_dict()
frac_threshold_earlier = float((d["diff"] < 0).mean())

print("\ndiff = lead(network) - lead(threshold), minutes")
print(f"  quartiles {q}")
print(f"  fraction where the THRESHOLD warns earlier: {frac_threshold_earlier:.4f}")
print("\nselected cases, by rank and not by eye:")
out = {"horizon_minutes": int(HORIZON), "nominal_fapd": a.nominal_fapd,
       "qualifying_events": n, "quartiles": q,
       "fraction_threshold_earlier": frac_threshold_earlier,
       "threshold_h60_sensitivity": t_sens, "network_h60_sensitivity": n_sens,
       "cases": {}}
for name, idx in picks.items():
    r = d.iloc[idx]
    rec = {k: (str(v) if not isinstance(v, (int, float, np.floating)) else float(v))
           for k, v in r.items()}
    out["cases"][name] = rec
    print(f"  {name:<7} rank {idx:>5}/{n}  diff {r['diff']:+.1f} min  "
          f"(network {r['lead_network']:.1f}, threshold {r['lead_threshold']:.1f})")
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(out, indent=1, default=str))
print(f"\nwrote {a.output}")
