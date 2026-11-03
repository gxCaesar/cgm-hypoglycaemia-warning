"""Figure 3. The ordering, and the size of the lead, depend on the alarm budget."""
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

FROZEN_BUDGET = 0.5
SERIES = {
    "persistence_seed17": ("current glucose threshold", G.NOMINAL["rose"], "o"),
    "tcn_erm_seed17": ("TCN, ERM", G.NOMINAL["blue"], "s"),
    "balanced_lstm_erm_seed17": ("LSTM, balanced ERM", G.NOMINAL["teal"], "^"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    data = json.load(open(args.data))
    sweep = data["panels"]["budget_sweep"]
    sigma = {float(k): v["sigma"] for k,
        v in data["panels"]["seed_sigma_by_budget"].items()}
    budgets = sorted(float(b) for b in sweep["persistence_seed17"])

    G.set_style()
    fig = plt.figure(figsize=(G.COL2_MM * G.MM, 78.0 * G.MM))

    # ---- panel a: every arm across the usable range ---------------------------------
    ax = fig.add_axes([0.070, 0.170, 0.255, 0.720])
    for key, (name, colour, marker) in SERIES.items():
        ys = [sweep[key][str(b)]["primary"] for b in budgets]
        ax.plot(budgets, ys, color=colour, marker=marker, ms=3.0, lw=0.9, label=name)
    ax.axvline(FROZEN_BUDGET, color=G.MUTED, lw=0.5, ls=(0, (1.6, 1.6)), zorder=0)
    ax.text(FROZEN_BUDGET + 0.025, 0.98, "frozen\nbudget", fontsize=5.0,
            color=G.MUTED, va="top")
    ax.set_xlabel("nominal alarms per participant-day")
    ax.set_ylabel("severe-event sensitivity")
    ax.set_xlim(0, 1.08)
    ax.set_ylim(0, 1.0)
    ax.legend(loc="lower right", fontsize=5.0, handlelength=1.8, borderaxespad=0.3)

    # ---- panel b: the lead over the strongest OTHER arm, in seed units ---------------
    leads, sigmas, best_other = [], [], []
    for b in budgets:
        lead_arm = sweep["persistence_seed17"][str(b)]["primary"]
        others = {k: sweep[k][str(b)][
            "primary"] for k in SERIES if k != "persistence_seed17"}
        top = max(others, key=others.get)
        best_other.append(top)
        leads.append(lead_arm - others[top])
        sigmas.append(sigma[b])

    bx = fig.add_axes([0.400, 0.170, 0.255, 0.720])
    bx.axhline(0, color=G.INK, lw=0.5, zorder=1)
    for b, lead, colour in zip(budgets, leads,
                               [G.NOMINAL["rose"] if v > 0 else G.NOMINAL[
                                   "blue"] for v in leads], strict=True):
        bx.plot([b, b], [0, lead], color=colour, lw=0.7, solid_capstyle="butt",
            zorder=2)
        bx.plot([b], [lead], marker="o", ms=3.0, color=colour, zorder=3)
    bx.axvline(FROZEN_BUDGET, color=G.MUTED, lw=0.5, ls=(0, (1.6, 1.6)), zorder=0)
    bx.set_xlabel("nominal alarms per participant-day")
    bx.set_ylabel("lead of the threshold over the\nstrongest trained arm")
    bx.set_xlim(0, 1.08)
    bx.set_ylim(-0.105, 0.185)
    bx.annotate("the ordering\ninverts here", xy=(0.125, leads[0]),
                xytext=(10, -2), textcoords="offset points", fontsize=5.0,
                color=G.NOMINAL["blue"], ha="left", va="center")

    # ---- panel c: and the yardstick itself moves ------------------------------------
    cx = fig.add_axes([0.730, 0.170, 0.255, 0.720])
    cx.plot(budgets, sigmas, color=G.NOMINAL["amber"], marker="D", ms=3.0, lw=0.9)
    cx.axvline(FROZEN_BUDGET, color=G.MUTED, lw=0.5, ls=(0, (1.6, 1.6)), zorder=0)
    cx.set_xlabel("nominal alarms per participant-day")
    cx.set_ylabel("seed standard deviation")
    cx.set_xlim(0, 1.08)
    cx.set_ylim(0, 0.055)
    # Alternate the offset: consecutive sigmas are close enough that a fixed
    # upward offset put two labels on the line between their own markers.
    offsets = [(0, 7), (0, -12), (0, 7), (0, -12), (0, 7)]
    for b, value, off in zip(budgets, sigmas, offsets, strict=True):
        cx.annotate(f"{value:.4f}", xy=(b, value), xytext=off,
                    textcoords="offset points", fontsize=5.0, color=G.MUTED,
                    ha="center")

    for x, letter in ((0.008, "a"), (0.338, "b"), (0.668, "c")):
        fig.text(x, 0.975, letter, fontsize=7.0, fontweight="bold", color=G.INK,
            va="top")

    G.save(fig, args.output, caption_words=150)
    print(f"wrote {args.output}")
    print("  budget   lead     sigma      lead/sigma   strongest other")
    for b, lead, s, top in zip(budgets, leads, sigmas, best_other, strict=True):
        print(f"   {b:5.3f}  {lead:+.6f}  {s:.6f}   {lead / s:+7.2f}    {top}")
    print(f"  the lead varies {max(leads) / min(v for v in leads if v > 0):.1f}-fold "
          "across the budgets where it is positive")


if __name__ == "__main__":
    main()
