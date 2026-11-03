"""Figure 5. Every cohort, for the composition and its two members.

The paper's claim is about generalising across sources, and the evidence for it is a
 profile rather than a mean: the composition is not only higher but flatter, and the
 number
a cross-source claim rests on is the lowest cohort. A table carries that; a figure shows
whether the flatness is uniform or whether one cohort is doing the work.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import glycofig as G

KEY = "participant_macro_event_sensitivity"
SERIES = [
    ("persist,lstm", "threshold + balanced LSTM", G.NOMINAL["olive"], "o"),
    ("persist", "current glucose threshold", G.NOMINAL["rose"], "s"),
    ("lstm", "LSTM, balanced ERM", G.NOMINAL["blue"], "^"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--cohort-counts", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    cells = json.loads(args.data.read_text())["cells"]

    G.set_style()
    fig = plt.figure(figsize=(G.COL2_MM * G.MM, 86.0 * G.MM))

    # One order for both panels, taken from the first horizon. Sorting each panel by its
    # own values leaves the second panel unlabelled AND in a different order, so a
    # reader
    # comparing rows across the two would be comparing different cohorts.
    order = sorted(
        cells["persist,lstm"]["30"]["per_source"],
        key=lambda s: cells["persist,lstm"]["30"]["per_source"][s][KEY],
    )
    for panel, (horizon, left) in enumerate(zip(("30", "60"), (
        0.215, 0.640), strict=True)):
        ax = fig.add_axes([left, 0.135, 0.325, 0.760])
        ypos = range(len(order))
        for key, label, colour, marker in SERIES:
            block = cells[key][horizon]["per_source"]
            ax.plot([block[s][KEY] for s in order], list(ypos), marker=marker, ms=3.0,
                    lw=0.0, color=colour, label=label if panel == 0 else None)
        for i, source in enumerate(order):
            lo = min(cells[k][horizon]["per_source"][source][KEY] for k, _, _,
                _ in SERIES)
            hi = max(cells[k][horizon]["per_source"][source][KEY] for k, _, _,
                _ in SERIES)
            ax.plot([lo, hi], [i, i], color=G.NEUTRAL, lw=0.4, zorder=0)
        ax.set_yticks(list(ypos))
        ax.set_yticklabels([s.replace("_", " ") for s in order] if panel == 0
                           else ["" for _ in order])
        ax.set_ylim(-0.7, len(order) - 0.3)
        ax.set_xlim(0, 1.04)
        ax.set_xlabel(f"severe-event sensitivity, {horizon} minutes")
        worst = min(cells["persist,lstm"][horizon]["per_source"][s][KEY] for s in order)
        ax.axvline(worst, color=G.NOMINAL["olive"], lw=0.5, ls=(0, (2.0, 1.6)),
            zorder=0)
        if panel == 0:
            ax.legend(loc="lower left", fontsize=5.0, handlelength=1.2,
                      borderaxespad=0.3, labelspacing=0.35)

    fig.text(0.012, 0.972, "a", fontsize=7.0, fontweight="bold", color=G.INK, va="top")
    fig.text(0.585, 0.972, "b", fontsize=7.0, fontweight="bold", color=G.INK, va="top")
    G.save(fig, str(args.output), caption_words=150)
    print(f"wrote {args.output}")
    for horizon in ("30", "60"):
        vals = {k: [cells[k][horizon]["per_source"][s][KEY]
                    for s in cells[k][horizon]["per_source"]] for k, _, _, _ in SERIES}
        for k in vals:
            print(f"  {horizon} min {k:14s} worst {min(vals[k]):.3f}  "
                  f"spread {max(vals[k]) - min(vals[k]):.3f}")


if __name__ == "__main__":
    main()
