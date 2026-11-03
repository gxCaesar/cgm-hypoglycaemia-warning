#!/usr/bin/env python3
"""Per-source cohort composition for Table 1, from the frozen window shards.

Read-only. Counts participants, person-days and distinct event identifiers per
source across all three development partitions. The 60-minute event identifier is
the event inventory because an event is a property of the glucose trace and not of
the horizon; the 30-minute column is counted too so a disagreement shows up rather
than being assumed away.
"""
import argparse
import collections
import glob
import json

import pyarrow.parquet as pq

_ap = argparse.ArgumentParser(
    description="Cohort composition table from window shards.")
_ap.add_argument("--window-shards", required=True,
                 help="path to experiments/<run>/window_shards")
_ap.add_argument("--output", required=True, help="path to write the JSON table")
_args = _ap.parse_args()
BASE = _args.window_shards
COLS = ["source_id", "participant_id", "patient_day", "hypo_60_event_id",
        "hypo_30_event_id", "hyper_60_event_id", "severe_hypo_60_event_id"]

participants = collections.defaultdict(set)
days = collections.defaultdict(set)
stores = {c: collections.defaultdict(set) for c in COLS[3:]}
files = sorted(glob.glob(BASE + "/partition=*/*.parquet"))
print(f"{len(files)} shard files", flush=True)
for i, f in enumerate(files):
    t = pq.read_table(f, columns=COLS).to_pandas()
    for src, g in t.groupby("source_id", observed=True):
        s = str(src)
        pid = g["participant_id"].astype(str)
        participants[s].update(pid.unique())
        days[s].update(zip(pid, g["patient_day"].astype(str), strict=True))
        for col in COLS[3:]:
            v = g[col].dropna().astype(str)
            stores[col][s].update(v[v != ""].unique())
    if (i + 1) % 300 == 0:
        print(f"  {i + 1}/{len(files)}", flush=True)

out = {
    s: {
        "participants": len(participants[s]),
        "person_days": len(days[s]),
        "hypo_events": len(stores["hypo_60_event_id"][s]),
        "hypo_events_30h": len(stores["hypo_30_event_id"][s]),
        "hyper_events": len(stores["hyper_60_event_id"][s]),
        "severe_hypo_events": len(stores["severe_hypo_60_event_id"][s]),
    }
    for s in sorted(participants)
}
with open(_args.output, "w") as _handle:
    json.dump(out, _handle, indent=1)
head = (f"{'source':<16}{'ppts':>7}{'person-days':>13}"
        f"{'hypo':>9}{'hyper':>9}{'severe':>9}")
print(head)
for s, v in out.items():
    print(f"{s:<16}{v['participants']:>7}{v['person_days']:>13,}"
          f"{v['hypo_events']:>9,}{v['hyper_events']:>9,}{v['severe_hypo_events']:>9,}")
tot = {k: sum(v[k] for v in out.values()) for k in
       ("participants", "person_days", "hypo_events",
        "hyper_events", "severe_hypo_events")}
print(f"{'TOTAL':<16}{tot['participants']:>7}{tot['person_days']:>13,}"
      f"{tot['hypo_events']:>9,}{tot['hyper_events']:>9,}{tot['severe_hypo_events']:>9,}")
