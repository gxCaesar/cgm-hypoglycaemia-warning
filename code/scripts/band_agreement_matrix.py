#!/usr/bin/env python3
"""Band-restricted rank agreement between `persist` and every other arm, one join.

Pre-registered in
the m6 bundle's `PREREG-is-band-agreement-a-law-or-two-anecdotes.md`.

Read-only over frozen prediction artifacts. No training, no GPU, no new data.

The rule under test says composition gain falls as the members' rank agreement
INSIDE the 61-70 mg/dL band rises, and that the global correlation points the
other way. It has two points. This produces the rest of them.

Every helper that decides what a row is comes from `member_disagreement.py` rather than
being written again here. Two implementations of "eligible" is how a table ends up
describing a frame nobody built.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from member_disagreement import (  # noqa: E402
    HORIZONS,
    KEY,
    _read,
    _spearman,
    calibration_rank,
)

BAND = (61.0, 70.0)
# Published in RESULT-a-better-member-makes-the-combination-worse.md, computed by a
# different tool on the same artifacts. If this code path cannot reproduce them the
# matrix is measuring something else and the run stops.
KNOWN = {("persist", "lstm", 30): 0.119, ("persist", "m0", 30): 0.263,
         ("persist", "lstm", 60): 0.086, ("persist", "m0", 60): 0.223}
KNOWN_TOL = 0.001


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--arm", action="append", required=True, help="name=path[:legacy]")
    p.add_argument("--reference", default="persist")
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args(argv)

    def parse(item: str) -> tuple[str, Path, bool]:
        name, _, rest = item.partition("=")
        legacy = rest.endswith(":legacy")
        return name, Path(rest[: -len(":legacy")] if legacy else rest), legacy

    arms = [parse(i) for i in args.arm]
    names = [n for n, _, _ in arms]
    if args.reference not in names:
        raise SystemExit(f"reference {args.reference} is not among {names}")

    frame = None
    for index, (name, path, legacy) in enumerate(arms):
        part = _read(path, legacy=legacy, base=(index == 0))
        part = part.rename(columns={
            f"severe_hypo_{h}_probability": f"arm__{name}__{h}" for h in HORIZONS})
        frame = (part if frame is None
                 else frame.merge(part, on=KEY, how="inner", validate="one_to_one"))
        print(f"  joined {name}", flush=True)
    assert frame is not None
    print(f"  one-to-one over {len(frame):,} rows for {len(arms)} arms", flush=True)

    calibration = (frame["partition"] == "calibration").to_numpy()
    selection = (frame["partition"] == "selection").to_numpy()
    glucose = pd.to_numeric(frame["cgm_last"], errors="coerce").to_numpy(float)
    active = frame["severe_hypo_currently_active"].to_numpy(bool)
    eligible = selection & ~active
    lo, hi = BAND
    in_band = eligible & (glucose >= lo) & (glucose < hi)

    report: dict[str, object] = {
        "band": list(BAND), "reference": args.reference,
        "rows": int(len(frame)), "eligible": int(eligible.sum()),
        "in_band": int(in_band.sum()), "pairs": {},
    }
    ranks = {h: {n: calibration_rank(
        pd.to_numeric(frame[f"arm__{n}__{h}"], errors="coerce").to_numpy(float),
        calibration) for n in names} for h in HORIZONS}

    failures = []
    for name in names:
        if name == args.reference:
            continue
        cell: dict[str, object] = {}
        for h in HORIZONS:
            ra, rb = ranks[h][args.reference], ranks[h][name]
            band_rho = _spearman(ra[in_band], rb[in_band])
            global_rho = _spearman(ra[eligible], rb[eligible])
            cell[str(h)] = {"band": band_rho, "global": global_rho}
            key = (args.reference, name, h)
            if key in KNOWN and abs(band_rho - KNOWN[key]) > KNOWN_TOL:
                failures.append(f"{key} band {band_rho:.6f} != published {KNOWN[key]}")
        report["pairs"][name] = cell
        print(f"  {name:24s} band30 {cell['30']['band']:+.4f}  "
              f"global30 {cell['30']['global']:+.4f}", flush=True)

    if failures:
        raise SystemExit("FAIL known-answer check: " + "; ".join(failures))
    checked = sum(1 for n in names for h in HORIZONS
                  if (args.reference, n, h) in KNOWN)
    report["known_answer_cells_checked"] = checked
    print(f"  known-answer cells reproduced: {checked}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"  wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
