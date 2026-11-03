#!/usr/bin/env python3
"""Measure the seed scale on the family the headline comes from, and restate.

Every margin in this project divides by a sigma measured on a DIFFERENT arm
family, whose own record carries a wide interval on three degrees of freedom and
an open caveat that thread scheduling was unpinned and therefore confounded with
seed variance. No arm in the v1.5 family has ever had a second seed.

This computes the within-family scale from paired seeds of the same arm and
restates every margin supplied to it under both scales, in both directions.
Written before the second seed finished, so the restatement cannot be shaped by
seeing which way it moves.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

TRANSFERRED_SIGMA = 0.009456055
HORIZONS = ("30", "60")
KEY = "equal_source_participant_macro_event_sensitivity"


def primary(run: Path) -> float | None:
    try:
        metrics = json.loads((run / "operational_metrics.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return sum(
        metrics["horizons"][h]["selection_metrics"][KEY] for h in HORIZONS
    ) / len(HORIZONS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", action="append", required=True, help="label=path")
    parser.add_argument(
        "--restate",
        action="append",
        default=[],
        help="label=margin, a difference to restate under both scales",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    values: dict[str, float] = {}
    pending: list[str] = []
    for item in args.seed:
        label, _, path = item.partition("=")
        value = primary(Path(path))
        (pending.append(label) if value is None else values.update({label: value}))

    for label, value in sorted(values.items()):
        print(f"  {label:20s} {value:.9f}")
    for label in pending:
        print(f"  {label:20s} NOT FINISHED")

    if len(values) < 2:
        print()
        print(
            "WITHHELD. A standard deviation needs at least two seeds and "
            f"{len(values)} finished. Reporting a scale from one run would be "
            "reporting a number that does not exist."
        )
        return 2

    ordered = [values[k] for k in sorted(values)]
    measured = statistics.stdev(ordered)
    spread = max(ordered) - min(ordered)
    print()
    print(f"seeds {len(ordered)}   mean {statistics.mean(ordered):.9f}")
    print(f"  WITHIN-FAMILY sd  {measured:.9f}   spread {spread:.9f}")
    ratio = measured / TRANSFERRED_SIGMA
    print(f"  transferred sigma {TRANSFERRED_SIGMA:.9f}   ratio {ratio:.3f}x")
    if len(ordered) == 2:
        print(
            "  NOTE two seeds give one degree of freedom. This sd is a point "
            "estimate with an extremely wide interval and must be reported as such."
        )

    report: dict[str, object] = {
        "seeds": values,
        "within_family_sd": measured,
        # A float's repr contains its own rounding only when the next digit does
        # not round up, so a document quoting six decimals of this value was
        # untraceable by text search. Carry the quoted forms explicitly.
        "within_family_sd_6dp": f"{measured:.6f}",
        "within_family_sd_9dp": f"{measured:.9f}",
        "seed_spread": spread,
        "seed_spread_6dp": f"{spread:.6f}",
        "seed_spread_9dp": f"{spread:.9f}",
        "transferred_sigma": TRANSFERRED_SIGMA,
        "degrees_of_freedom": len(ordered) - 1,
        "restated": {},
    }
    if args.restate:
        print()
        print(f"{'margin':38s} {'value':>12s} {'old sigma':>11s} {'measured':>11s}")
        for item in args.restate:
            label, _, raw = item.partition("=")
            margin = float(raw)
            old = margin / TRANSFERRED_SIGMA
            new = margin / measured
            direction = "SHRINKS" if abs(new) < abs(old) else "GROWS"
            print(f"{label:38s} {margin:+12.6f} {old:+11.2f} {new:+11.2f}  {direction}")
            report["restated"][label] = {
                "margin": margin,
                "sigma_transferred": old,
                "sigma_measured": new,
            }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True))
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
