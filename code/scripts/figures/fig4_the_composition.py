"""Figure 4. The composition, and the division of labour that makes it work."""
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

PROMOTION_FLOOR = 0.836116373      # the leading single arm plus the registered 0.03
HORIZON = "30"

MEMBERS = [
    ("lstm", "LSTM, balanced ERM", G.NOMINAL["blue"]),
    ("m0", "LSTM, severe-retargeted", G.NOMINAL["teal"]),
    ("persist", "current glucose threshold", G.NOMINAL["rose"]),
    ("persist,m0", "threshold + retargeted", G.NOMINAL["olive"]),
    ("persist,lstm", "threshold + LSTM", G.NOMINAL["olive"]),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    data = json.load(open(args.data))
    cells = data["panels"]["composition"]["replication"]["cells"]
    weight = data["panels"]["composition"]["weight"]
    dis = data["panels"]["disagreement"]["horizons"][HORIZON]

    G.set_style()
    fig = plt.figure(figsize=(G.COL2_MM * G.MM, 76.0 * G.MM))

    # ---- panel a: members and the compositions they form -----------------------------
    rows = [(key, name, colour, cells[key]["primary"]) for key, name, colour in MEMBERS]
    rows.sort(key=lambda r: r[3])
    ax = fig.add_axes([0.205, 0.185, 0.245, 0.680])
    for i, (_key, _name, colour, value) in enumerate(rows):
        ax.plot([0, value], [i, i], color=colour, lw=0.7, solid_capstyle="butt",
            zorder=2)
        ax.plot([value], [i], marker="o", ms=3.0, color=colour, zorder=3)
        ax.text(value + 0.012, i, f"{value:.3f}", fontsize=5.0, va="center",
            color=G.INK)
    ax.axvline(PROMOTION_FLOOR, color=G.INK, lw=0.5, ls=(0, (1.6, 1.6)), zorder=1)
    ax.text(PROMOTION_FLOOR + 0.015, -0.70, "registered\npromotion floor", fontsize=5.0,
            color=G.INK, ha="left", va="bottom")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[1] for r in rows])
    ax.set_ylim(-0.75, len(rows) - 0.35)
    ax.set_xlim(0, 1.06)
    ax.set_xlabel("severe-event sensitivity")

    # ---- panel b: where the members disagree, each is right about a different class
    # ---
    levels = [("q0.5", "median"), ("q0.9", "top 10%"), ("q0.99", "top 1%")]
    bx = fig.add_axes([0.560, 0.185, 0.170, 0.680])
    xs = range(len(levels))
    bx.plot(list(xs), [dis[k]["persist_share_on_negatives"] for k, _ in levels],
            color=G.NOMINAL["rose"], marker="o", ms=3.0, lw=0.9,
)
    bx.plot(list(xs), [dis[k]["persist_share_on_positives"] for k, _ in levels],
            color=G.NOMINAL["blue"], marker="s", ms=3.0, lw=0.9,
)
    bx.set_xticks(list(xs))
    bx.set_xticklabels([n for _, n in levels])
    bx.set_xlabel("disagreement between the\ntwo members")
    bx.set_ylabel("share the threshold gets right")
    bx.set_ylim(0, 1.06)
    bx.set_xlim(-0.3, len(levels) - 0.42)
    # Label the lines where they end, not in a legend box: at this panel width a legend
    # sits on top of the rose series wherever it is placed.
    bx.annotate("non-events,\ncorrectly ranked lower",
                xy=(len(levels) - 1, dis["q0.99"]["persist_share_on_negatives"]),
                xytext=(-4, -14), textcoords="offset points",
                fontsize=5.0, color=G.NOMINAL["rose"], ha="right", va="top")
    bx.annotate("events,\ncorrectly ranked higher",
                xy=(len(levels) - 1, dis["q0.99"]["persist_share_on_positives"]),
                xytext=(-4, 14), textcoords="offset points",
                fontsize=5.0, color=G.NOMINAL["blue"], ha="right", va="bottom")

    # ---- panel c: the weight was swept, and the pre-registration held the headline
    # ----
    cx = fig.add_axes([0.815, 0.185, 0.170, 0.680])
    grid = [float(g) for g in weight["grid"]]
    curve = [weight["calibration_curve"][str(g)] for g in weight["grid"]]
    cx.plot(grid, curve, color=G.NOMINAL["amber"], marker="D", ms=2.6, lw=0.9)
    best = max(range(len(grid)), key=lambda i: curve[i])
    cx.plot([grid[best]], [curve[best]], marker="D", ms=4.2,
            markerfacecolor="none", markeredgecolor=G.INK, markeredgewidth=0.7)
    cx.axvline(0.5, color=G.MUTED, lw=0.5, ls=(0, (1.6, 1.6)), zorder=0)
    cx.set_xlabel("weight on the threshold,\nchosen on calibration")
    cx.set_ylabel("calibration-partition score")
    cx.set_xlim(0.05, 0.95)
    cx.set_ylim(0.68, 0.96)
    cx.text(0.5, 0.700, "equal weight,\nthe headline", fontsize=5.0, color=G.MUTED,
            ha="center", va="bottom")
    cx.annotate(f"best here buys\n+{weight['gain']:.4f} on selection,\n"
                f"against a selection\ninflation term "
                    f"of\n{weight['inflation_term']:.4f}",
                xy=(grid[best], curve[best]), xytext=(0.40, 0.925),
                fontsize=5.0, color=G.INK, ha="left", va="top",
                arrowprops=dict(arrowstyle="-", lw=0.4, color=G.INK))

    for x, letter in ((0.010, "a"), (0.487, "b"), (0.742, "c")):
        fig.text(x, 0.975, letter, fontsize=7.0, fontweight="bold", color=G.INK,
            va="top")

    G.save(fig, args.output, caption_words=150)
    print(f"wrote {args.output}")
    combo = cells["persist,lstm"]["primary"]
    print(f"  combination {combo:.9f}  floor {PROMOTION_FLOOR}  "
          f"clears by {combo - PROMOTION_FLOOR:+.6f}")
    for key, name, _c in MEMBERS:
        print(f"    {name:28s} {cells[key]['primary']:.9f}")
    print(f"  weight sweep: best {grid[best]} at {curve[best]:.9f}, gain "
          f"{weight['gain']:+.6f} against inflation {weight['inflation_term']:.6f} "
          f"-> outcome {weight['outcome']}")


if __name__ == "__main__":
    main()
