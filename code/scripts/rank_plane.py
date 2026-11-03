"""Where the two members place each decision, and where the severe events actually are.

The interpretability evidence in this package is aggregate: rank correlation inside the
 decision band, and a share-of-correct split by disagreement level. Neither shows a
 reader
what the members disagree ABOUT. This bins the (threshold rank, network rank) plane and
reports the severe-event rate in each cell, so the disagreement is visible as a region
rather than as a correlation.

Read-only over frozen prediction artefacts. Emits a small summary, not the plane itself.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pyarrow.parquet as pq

persist_path, lstm_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
HORIZON = "60"
PROB = f"severe_hypo_{HORIZON}_probability"
LABEL = f"severe_hypo_{HORIZON}_label"
ACTIVE = "severe_hypo_currently_active"
COLS = [PROB, LABEL, ACTIVE, "partition", "cgm_last"]

def load(path):
    t = pq.read_table(path, columns=COLS).to_pandas()
    return t

a = load(persist_path)
b = load(lstm_path)
assert len(a) == len(b), f"frames differ: {len(a)} vs {len(b)}"

sel = (a["partition"] == "selection").to_numpy()
cal = (a["partition"] == "calibration").to_numpy()
ok = np.isfinite(a[PROB].to_numpy()) & np.isfinite(b[PROB].to_numpy())
# The endpoint excludes decisions taken while a severe event is already running, so the
# plane must too, or it is describing a different set of decisions than the paper
# scores.
eligible = ~a[ACTIVE].to_numpy().astype(bool)
use = sel & ok & eligible

def calib_rank(v, calmask):
    ref = np.sort(v[calmask & np.isfinite(v)])
    return np.searchsorted(ref, v, side="right") / max(len(ref), 1)

ra = calib_rank(a[PROB].to_numpy(), cal & ok)[use]
rb = calib_rank(b[PROB].to_numpy(), cal & ok)[use]
y = a[LABEL].to_numpy()[use].astype(bool)
g = a["cgm_last"].to_numpy()[use]

N = 20
ia = np.clip((ra * N).astype(int), 0, N - 1)
ib = np.clip((rb * N).astype(int), 0, N - 1)
count = np.zeros((N, N), dtype=np.int64)
events = np.zeros((N, N), dtype=np.int64)
np.add.at(count, (ia, ib), 1)
np.add.at(events, (ia[y], ib[y]), 1)

report = {
    "horizon": HORIZON,
    "bins": N,
    "rows_used": int(use.sum()),
    "rows_excluded_active": int((sel & ok & ~eligible).sum()),
    "events_total": int(y.sum()),
    "count": count.tolist(),
    "events": events.tolist(),
    "note": ("rank axes are calibration ECDF ranks; index i is the threshold member, "
             "index j is the network member"),
    "median_glucose_by_threshold_bin": [
        float(np.median(g[ia == i])) if (ia == i).any() else None for i in range(N)
    ],
}
with open(out_path, "w") as fh:
    json.dump(report, fh)

diag = np.trace(events) / max(events.sum(), 1)
off = 1 - diag
top_a_only = events[N - 1, : N // 2].sum()
top_b_only = events[: N // 2, N - 1].sum()
print(f"rows {report['rows_used']:,}  events {report['events_total']:,}")
print(f"  events on the rank diagonal: {diag:.4f}, off it: {off:.4f}")
print(
    f"  events the threshold ranks top-5% and the network ranks below median: "
        f"{top_a_only:,}")
print(
    f"  events the network ranks top-5% and the threshold ranks below median: "
        f"{top_b_only:,}")
