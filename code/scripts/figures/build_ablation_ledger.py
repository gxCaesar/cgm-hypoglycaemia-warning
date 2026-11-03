"""Emit the removal-ablation ledger the cards and the supplement quote.

The four drop values were quoted in the model card and existed nowhere on disk, because
 a difference of two stored primaries is not itself stored. A number a document quotes
 and
 no artefact holds cannot be checked against anything, so this writes each removal with
 its
base, the member removed, the arm that remains, the computed drop, and the drop in units
of the measured seed scale for a combination.

Read-only. Reads `figure-data.json` and writes one JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Measured on four seeds of one arm family, recomputed from seed-combination.json:
# combinations 0.011357957, single members 0.009456055, ratio 1.2011.
COMBINATION_SIGMA = 0.011357957


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    cells = json.loads(args.data.read_text())["panels"]["composition"]["replication"][
        "cells"]
    compositions = {k: v for k, v in cells.items() if "," in k}

    # Emit the rounded forms documents actually quote. A float's repr contains its own
    # rounding only when the next digit does not round up, and this is the fourth value
    # in this package to be untraceable for that reason alone rather than for want of a
    # measurement.
    ledger = {
        "seed_sigma_combination": COMBINATION_SIGMA,
        "seed_sigma_combination_6dp": f"{COMBINATION_SIGMA:.6f}",
        "seed_sigma_combination_9dp": f"{COMBINATION_SIGMA:.9f}",
        "removals": [],
    }
    for name, payload in sorted(compositions.items()):
        members = [m.strip() for m in name.split(",")]
        for removed in members:
            remaining = [m for m in members if m != removed]
            key = ",".join(remaining)
            if key not in cells:
                continue
            drop = payload["primary"] - cells[key]["primary"]
            ledger["removals"].append({
                "composition": name,
                "composition_primary": payload["primary"],
                "member_removed": removed,
                "arm_that_remains": key,
                "remaining_primary": cells[key]["primary"],
                "drop": drop,
                # The document quotes nine decimals; the float's repr does not
                # contain that string once the tenth digit rounds up, so two of
                # these four were untraceable by a text search while the other
                # two matched by luck. Carry the quoted form explicitly.
                "drop_9dp": f"{drop:.9f}",
                "drop_in_seed_sigma": drop / COMBINATION_SIGMA,
                "drop_in_seed_sigma_1dp": f"{drop / COMBINATION_SIGMA:.1f}",
                "derivation": f"{payload['primary']:.9f} - {cells[key]['primary']:.9f}",
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(ledger, indent=1, sort_keys=True))

    for row in ledger["removals"]:
        print(f"  {row['composition']:16s} remove {row['member_removed']:8s} -> "
              f"{row['arm_that_remains']:8s}  drop {row['drop']:+.9f}  "
              f"({row['drop_in_seed_sigma']:+.1f} sigma)")
    print(f"wrote {len(ledger['removals'])} removals to {args.output}")


if __name__ == "__main__":
    main()
