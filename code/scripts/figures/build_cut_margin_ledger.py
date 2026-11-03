"""Margins of the composition over its best member at every eligibility cut.

The cut curve stores each arm's primary and not the difference between them, so a margin
quoted in prose had no artefact behind it. This derives them once, with the quoted form
beside the float, and records which member each margin is measured against, because the
better member changes hands between cut 54 and cut 60.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

COMPOSITION = "persist,lstm"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuts", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    cuts = json.loads(args.cuts.read_text())["cuts"]

    rows = []
    for key in sorted(cuts, key=float):
        row = cuts[key]
        members = {k: v for k, v in row.items()
                   if k != COMPOSITION and isinstance(v, float)}
        best = max(members, key=members.get)
        margin = row[COMPOSITION] - members[best]
        rows.append({
            "cut": float(key),
            "composition": row[COMPOSITION],
            "best_member": best,
            "best_member_value": members[best],
            "margin": margin,
            "margin_6dp": f"{margin:.6f}",
            "derivation": f"{row[COMPOSITION]:.6f} - {members[best]:.6f}",
        })

    decay = rows[1]["margin"] / rows[-1]["margin"] if rows[-1]["margin"] else None
    report = {
        "composition": COMPOSITION,
        "rows": rows,
        "margin_decay_cut60_to_cut70": decay,
        "margin_decay_6dp": f"{decay:.6f}" if decay else None,
        "note": ("the best member changes between cuts, so each margin names "
                 "the arm it is measured against"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1, sort_keys=True))
    for r in rows:
        print(f"  cut {r['cut']:5.1f}  composition {r['composition']:.6f}  "
              f"best member {r['best_member']:8s} "
              f"{r['best_member_value']:.6f}  margin {r['margin']:+.6f}")
    if decay:
        print(f"  margin at cut 60 is {decay:.2f}x the margin at cut 70")


if __name__ == "__main__":
    main()
