#!/usr/bin/env python3
"""Why does a threshold plus a deep arm win while two deep arms lose?

`RESULT-order-4-clears-the-promotion-floor.md` measured that a partition-honest
rank average of `persistence` and one LSTM reaches 0.924472288 on the frozen
primary against the leader's 0.806116373, while averaging the two DEEP arms
reaches 0.770153695 and loses. The number is established. This asks what produces
it, so the claim can be a mechanism rather than a coincidence.

Three read-only measurements, on the same joined frame the ensemble used:

1. **Pairwise rank agreement.** If the two deep arms agree with each other far
   more than either agrees with the threshold, that alone explains why one pair
   is complementary and the other is redundant.
2. **The same, per current-glucose band.** Agreement is expected to be high where
   the threshold is silent and to break down where it is not.
3. **Where the detected events come from.** For each arm and for the winning
   combination, at the policy fitted on calibration at nominal 0.5 FAPD, attribute
   every detected event to the current-glucose band of the window that raised the
   alarm.

Ranks are fitted on CALIBRATION alone and applied by interpolation, the same
partition-honest transform the ensemble result used, so nothing here consults a
selection score to normalise another.

Trains nothing. Opens completed arms' own prediction artifacts and no other data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from glycoshift.operational_metrics import (
    OperationalEndpointSpec,
    fit_operational_alarm_policy,
)

HORIZONS = (30, 60)
KEY = ["source_id", "participant_id", "context_end"]
BASE = ["partition", "patient_day", "cgm_last", "severe_hypo_currently_active"]

# Clinically meaningful edges. 54 is the severe line the endpoint scores, 61 is
# where the leading threshold alarms, 70 is clinical level 1. The rest are wide
# bands so the counts stay readable.
BAND_EDGES = (-np.inf, 54.0, 61.0, 70.0, 100.0, 140.0, 180.0, np.inf)
BAND_NAMES = ("<54", "54-61", "61-70", "70-100", "100-140", "140-180", ">180")


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


def _read(path: Path, *, with_labels: bool) -> pd.DataFrame:
    columns = list(KEY) + [f"severe_hypo_{h}_probability" for h in HORIZONS]
    if with_labels:
        columns += BASE
        for h in HORIZONS:
            columns += [
                f"severe_hypo_{h}_label",
                f"severe_hypo_{h}_event_id",
                f"severe_hypo_{h}_event_onset",
            ]
    frame = pq.read_table(path, columns=sorted(set(columns))).to_pandas()
    for column in KEY:
        frame[column] = frame[column].astype(str)
    return frame


def _calibration_rank(values: np.ndarray, calibration_mask: np.ndarray) -> np.ndarray:
    """Rank against the CALIBRATION distribution alone, applied to every row.

    Identical to the transform behind the ensemble result: the map is a function
    of calibration scores only, so scoring one selection window never consults
    another, and it is strictly increasing between distinct calibration scores.
    """

    reference = np.sort(values[calibration_mask])
    if reference.size == 0:
        raise SystemExit("no calibration rows to fit the rank transform on")
    grid = np.linspace(0.0, 1.0, reference.size)
    return np.interp(values, reference, grid)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True, help="name=path")
    parser.add_argument("--combination", action="append", default=[])
    parser.add_argument("--nominal-fapd", type=float, default=0.5)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    specs = {}
    for item in args.arm:
        name, _, path = item.partition("=")
        specs[name] = Path(path)

    frame = None
    for i, (name, path) in enumerate(specs.items()):
        part = _read(path, with_labels=(i == 0))
        part = part.rename(
            columns={
                f"severe_hypo_{h}_probability": f"{name}__{h}" for h in HORIZONS
            }
        )
        frame = part if frame is None else frame.merge(part, on=KEY, how="inner",
                                                       validate="one_to_one")
    assert frame is not None
    print(f"joined one-to-one over {len(frame):,} rows")

    calibration_mask = (frame["partition"] == "calibration").to_numpy()
    selection_mask = (frame["partition"] == "selection").to_numpy()
    eligible = ~frame["severe_hypo_currently_active"].to_numpy(dtype=bool)
    band = pd.cut(
        pd.to_numeric(frame["cgm_last"], errors="coerce"),
        bins=list(BAND_EDGES), labels=list(BAND_NAMES), right=False,
    )

    ranks: dict[str, dict[int, np.ndarray]] = {}
    for name in specs:
        ranks[name] = {}
        for h in HORIZONS:
            values = pd.to_numeric(frame[f"{name}__{h}"], errors="coerce")
            arr = values.to_numpy(dtype=float)
            if not np.isfinite(arr).all():
                raise SystemExit(f"{name}__{h} carries non-finite values")
            ranks[name][h] = _calibration_rank(arr, calibration_mask)

    report: dict[str, Any] = {
        "rows": int(len(frame)),
        "nominal_fapd": args.nominal_fapd,
        "band_edges": [float(x) for x in BAND_EDGES],
    }

    # --- 1 and 2: rank agreement, globally and per band, on SELECTION ------
    names = list(specs)
    agreement: dict[str, Any] = {}
    sel = selection_mask & eligible
    for h in HORIZONS:
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                ra, rb = ranks[a][h][sel], ranks[b][h][sel]
                key = f"{a}|{b}|{h}"
                block = {"spearman_global": float(
                    pd.Series(ra).corr(pd.Series(rb), method="spearman")
                )}
                for label in BAND_NAMES:
                    m = (band[sel].to_numpy() == label)
                    if m.sum() > 100:
                        block[f"spearman_{label}"] = float(
                            pd.Series(ra[m]).corr(pd.Series(rb[m]), method="spearman")
                        )
                agreement[key] = block
    report["rank_agreement"] = agreement

    # --- 3: per-band coverage of severe-positive windows at the fitted threshold
    def scored(members: list[str], h: int) -> np.ndarray:
        return np.mean([ranks[m][h] for m in members], axis=0)

    arms = [[n] for n in names] + [c.split(",") for c in args.combination]
    coverage: dict[str, Any] = {}
    for members in arms:
        label = ",".join(members)
        coverage[label] = {}
        for h in HORIZONS:
            work = frame.copy()
            work[f"severe_hypo_{h}_probability"] = scored(members, h)
            cal = work.loc[work["partition"] == "calibration"]
            order = tuple(sorted(set(work.loc[selection_mask, "source_id"])))
            policy = fit_operational_alarm_policy(
                cal, spec=_spec(h, order), nominal_fapd=args.nominal_fapd
            )
            fires = scored(members, h) >= policy.threshold
            pos = (
                pd.to_numeric(frame[f"severe_hypo_{h}_label"], errors="coerce")
                .fillna(0).to_numpy(dtype=float) > 0.5
            )
            block = {"threshold": float(policy.threshold), "bands": {}}
            for lab in BAND_NAMES:
                m = sel & (band.to_numpy() == lab)
                n_pos = int((pos & m).sum())
                block["bands"][lab] = {
                    "rows": int(m.sum()),
                    "severe_positive_windows": n_pos,
                    "positive_windows_fired": int((pos & m & fires).sum()),
                    "coverage": (
                        float((pos & m & fires).sum() / n_pos) if n_pos else None
                    ),
                    "alarming_rows": int((m & fires).sum()),
                }
            coverage[label][str(h)] = block
            print(
                f"  {label:20} h={h}  threshold={policy.threshold:.6f}  "
                f"fires on {int((sel & fires).sum()):,} eligible selection rows"
            )
    report["per_band_coverage"] = coverage

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
