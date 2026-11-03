#!/usr/bin/env python3
"""Per-source cohort composition for the EXTERNAL rows of Table 1.

Read-only.  The development rows of that table are counted by
`cohort_composition_table.py` from the frozen window shards, over the complete
development cohort.  The external partitions have no shards -- the frozen evaluator
constructs and scores their windows in one pass -- so the analogue over the complete
external cohort is the frozen prediction artifact, and the column definitions here
are copied from that script rather than restated:

    participants        distinct participant_id
    person_days         distinct (participant_id, patient_day)
    hypo_events         distinct hypo_60_event_id
    hyper_events        distinct hyper_60_event_id
    severe_hypo_events  distinct severe_hypo_60_event_id

The 60-minute identifier is the event inventory because an event is a property of the
glucose trace and not of the horizon; the 30-minute count is carried too so that a
disagreement surfaces instead of being assumed away.

Two known answers are asserted rather than inspected: the Loop partition must return
843 participants, the figure fixed by the duplicate-participant amendment before any
endpoint existed, and HUPA-UCM must return 25, the registered stress-cohort size.
"""
import argparse
import collections
import json

import pyarrow.parquet as pq

KNOWN_PARTICIPANTS = {
    "Loop Observational Study": 843,
    "HUPA-UCM": 25,
}
COLS = ["source_id", "participant_id", "patient_day", "hypo_60_event_id",
        "hypo_30_event_id", "hyper_60_event_id", "severe_hypo_60_event_id"]

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--predictions", required=True, action="append",
                help="a frozen predictions_<partition>.parquet; repeatable")
ap.add_argument("--output", required=True)
args = ap.parse_args()

participants = collections.defaultdict(set)
days = collections.defaultdict(set)
stores = {c: collections.defaultdict(set) for c in COLS[3:]}
rows_read = 0

for path in args.predictions:
    handle = pq.ParquetFile(path)
    print(f"{path}\n  {handle.metadata.num_rows:,} rows, "
          f"{handle.metadata.num_row_groups} row groups", flush=True)
    for batch in handle.iter_batches(batch_size=2_000_000, columns=COLS):
        frame = batch.to_pandas()
        rows_read += len(frame)
        for source, group in frame.groupby("source_id", observed=True):
            key = str(source)
            pid = group["participant_id"].astype(str)
            participants[key].update(pid.unique())
            days[key].update(
                zip(pid, group["patient_day"].astype(str), strict=True))
            for column in COLS[3:]:
                value = group[column].dropna().astype(str)
                stores[column][key].update(value[value != ""].unique())
        print(f"  {rows_read:,} rows", flush=True)

out = {
    source: {
        "participants": len(participants[source]),
        "person_days": len(days[source]),
        "hypo_events": len(stores["hypo_60_event_id"][source]),
        "hypo_events_30h": len(stores["hypo_30_event_id"][source]),
        "hyper_events": len(stores["hyper_60_event_id"][source]),
        "severe_hypo_events": len(stores["severe_hypo_60_event_id"][source]),
    }
    for source in sorted(participants)
}

failures = [
    f"{source}: counted {out[source]['participants']} participants, "
    f"the frozen record says {expected}"
    for source, expected in KNOWN_PARTICIPANTS.items()
    if source in out and out[source]["participants"] != expected
]
if failures:
    raise SystemExit("known-answer check FAILED\n  " + "\n  ".join(failures))

out["_provenance"] = {
    "predictions": list(args.predictions),
    "rows_read": rows_read,
    "known_answer_participants": {
        s: KNOWN_PARTICIPANTS[s] for s in out if s in KNOWN_PARTICIPANTS},
}
with open(args.output, "w") as handle:
    json.dump(out, handle, indent=1)

print(f"\n{'source':<28}{'ppts':>6}{'person-days':>13}"
      f"{'hypo':>9}{'hyper':>9}{'severe':>9}")
for source, value in out.items():
    if source.startswith("_"):
        continue
    print(f"{source:<28}{value['participants']:>6}{value['person_days']:>13,}"
          f"{value['hypo_events']:>9,}{value['hyper_events']:>9,}"
          f"{value['severe_hypo_events']:>9,}")
print(f"\nknown-answer participant check PASSED for "
      f"{sorted(s for s in out if s in KNOWN_PARTICIPANTS)}")
