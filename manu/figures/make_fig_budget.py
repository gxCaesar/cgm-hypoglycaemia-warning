'Make fig budget.'






















from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "code" / "scripts" / "figures")
)
import caption_words as _caption_words  # noqa: E402
import glycofig as G  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

FROZEN = 0.5


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--height-mm", type=float, default=74.0)
    parser.add_argument("--caption-words", type=int,
                        default=_caption_words.count("fig_budget"))
    arguments = parser.parse_args(argv)

    derived = json.loads(arguments.summary.read_text())["derived"]
    budgets = sorted(float(k) for k in derived)
    rows = [derived[k] for k in sorted(derived, key=float)]
    if len(budgets) < 4:
        raise SystemExit(f"a sweep of {len(budgets)} points is not a curve")
    mean = np.array([r["mean"] for r in rows])
    sd = np.array([r["sd"] for r in rows])
    control = np.array([r["control"] for r in rows])
    sigma = np.array([r["sigma"] for r in rows])

    G.set_style()
    figure, axes = plt.subplots(
        1, 3, figsize=(G.COL2_MM * G.MM, arguments.height_mm * G.MM))
    figure.subplots_adjust(left=0.070, right=0.988, top=0.905, bottom=0.185,
                           wspace=0.42)
    ax_a, ax_b, ax_c = axes

    for axis in axes:
        axis.axvline(FROZEN, color=G.INK, lw=0.5, ls=(0, (2.4, 1.6)), zorder=1)
        axis.set_xscale("log")
        axis.set_xticks(budgets)
        axis.set_xticklabels([f"{b:g}" for b in budgets], fontsize=5.2)
        axis.minorticks_off()
        axis.set_xlabel("Nominal false alarms per participant-day")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    ax_a.fill_between(budgets, mean - sd, mean + sd, color=G.NOMINAL["blue"],
                      alpha=0.16, lw=0, zorder=2)
    ax_a.plot(budgets, mean, color=G.NOMINAL["blue"], marker="o", markersize=2.8,
              linestyle="--", label="Deep arm, five seeds", zorder=3)
    ax_a.plot(budgets, control, color=G.NOMINAL["rose"], marker="s", markersize=2.8,
              linestyle="-", label="Current glucose (control)", zorder=4)
    ax_a.set_ylim(0.0, 1.0)
    ax_a.set_ylabel("Level 2 event sensitivity")
    ax_a.legend(loc="upper left", handlelength=2.0, borderaxespad=0.3,
                labelspacing=0.28)

    # A lollipop rather than a bar: the quantity is signed and a bar would need a
    # zero baseline it already has, but the sign change is what the panel is about.
    ax_b.axhline(0.0, color=G.INK, lw=0.6, zorder=2)
    for budget, value in zip(budgets, sigma, strict=True):
        colour = G.NOMINAL["rose"] if value > 0 else G.NEUTRAL
        ax_b.plot([budget, budget], [0.0, value], color=colour, lw=0.8, zorder=3)
        ax_b.plot([budget], [value], marker="s" if value > 0 else "o",
                  markersize=3.0, color=colour, zorder=4)
    ax_b.set_ylabel("Control's margin (seed sd at that budget)")

    ax_c.plot(budgets, sd, color=G.NOMINAL["olive"], marker="^", markersize=2.8,
              linestyle="-.", zorder=3)
    ax_c.set_ylim(0.0, max(sd) * 1.12)
    ax_c.set_ylabel("Seed standard deviation")

    for axis, letter in zip(axes, "abc", strict=True):
        G.panel_label(axis, letter, dx=-0.175, dy=1.115)

    G.save(figure, str(arguments.output), caption_words=arguments.caption_words)
    flips = [b for b, s in zip(budgets, sigma, strict=True) if s < 0]
    if flips != [min(budgets)]:
        raise SystemExit(
            f"the caption says the sign inverts only at the tightest budget; {flips}"
        )
    print(f"wrote {arguments.output}  {len(budgets)} budgets, seed sd "
          f"{sd.min():.4f} to {sd.max():.4f} ({sd.max() / sd.min():.1f}x), "
          f"margin inverts at {flips[0]:g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
