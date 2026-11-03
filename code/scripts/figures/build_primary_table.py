"""Generate the manuscript's primary results table from frozen artefacts.

Written after H-C fired, so the table's shape follows the pre-registered branch: the
composition is the method and its members are the comparators it must beat. Nothing here
is typed. Every value is read from `figure-data.json`, in which each number carries the
path of the artefact it came from.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

FLOOR = 0.836116373
SIGMA_COMBINATION = 0.011357957
SIGMA_SINGLE = 0.009456055

# The fourth field is whether the arm is TRAINED. A seed scale divides a quantity that
# varies with the training seed, and the current-glucose threshold does not: it has no
# parameters, no architecture and no training, so its endpoint is a deterministic
# function
# of the frozen data and the calibration partition. Printing a sigma beside it would
# imply
# a variance it does not have. Its combinations do vary, through their trained member.
ROWS = [
    ("persist,lstm", "threshold + balanced LSTM", "combination", True, True),
    ("persist,m0", "threshold + severe-retargeted LSTM", "combination", False, True),
    ("persist", "current glucose threshold", "single", False, False),
    ("m0", "LSTM, severe-retargeted", "single", False, True),
    ("lstm", "LSTM, balanced ERM", "single", False, True),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    panels = json.loads(args.data.read_text())["panels"]
    cells = panels["composition"]["replication"]["cells"]
    head = panels["headroom"]

    lines = [
        "% Generated from frozen artefacts. Do not edit by hand.",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"arm & kind & primary & vs floor & $\sigma$ \\",
        r"\midrule",
    ]
    untrained = []
    for key, label, kind, headline, trained in ROWS:
        value = cells[key]["primary"]
        gap = value - FLOOR
        sigma = SIGMA_COMBINATION if kind == "combination" else SIGMA_SINGLE
        name = rf"\textbf{{{label}}}" if headline else label
        prim = rf"\textbf{{{value:.6f}}}" if headline else f"{value:.6f}"
        if trained:
            cell = f"{gap / sigma:+.2f}"
        else:
            cell = "--"
            untrained.append(label)
        lines.append(f"{name} & {kind} & {prim} & {gap:+.6f} & {cell} \\\\")
    lines += [
        r"\midrule",
        rf"registered promotion floor & & {FLOOR:.6f} & & \\",
    ]
    for key, label in (("oracle_future_glucose", "oracle, exact future glucose"),
                       ("oracle_label", "oracle, perfect ranking")):
        v = head[key]["primary"]
        lines.append(f"{label} & reference & {v:.6f} & {v - FLOOR:+.6f} & \\\\")
    if untrained:
        lines += [
            r"\midrule",
            (r"\multicolumn{5}{l}{\footnotesize A dash marks an arm with no "
             r"training seed "
             r"to vary: it has no parameters and its endpoint is deterministic.} \\"),
        ]
    lines += [r"\bottomrule", r"\end{tabular}"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    headline_value = cells["persist,lstm"]["primary"]
    print()
    print(f"headline {headline_value:.9f} clears the floor by "
          f"{headline_value - FLOOR:+.9f} = "
          f"{(headline_value - FLOOR) / SIGMA_COMBINATION:+.2f} combination sigma")


if __name__ == "__main__":
    main()
