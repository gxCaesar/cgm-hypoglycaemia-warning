"""Figure 1. Eighteen arms on one eligible frame, and the headroom above all of them."""
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

SEED_SIGMA = 0.009456055     # transferred single-arm scale, measured

# The two campaigns each contain a TCN-ERM arm. Colour separates them and the legend
# names the families, but a reader scanning the axis sees one label twice, so the
# earlier family carries its own suffix.
LABEL = {
    "persistence_seed17": "current glucose threshold",
    "tcn_erm_seed17": "TCN, ERM",
    "balanced_lstm_erm_seed17": "LSTM, balanced ERM",
    "logistic_regression_seed17": "logistic regression",
    "xgboost_seed17": "gradient boosting",
    "linear_seed17": "linear",
    "label_shuffle_logistic_seed17": "label-shuffle control",
    "constant_global_seed17": "constant, global",
    "constant_source_conditional_seed17": "constant, per source",
    "old_tcn_erm": "TCN, ERM (earlier)",
    "old_lstm_erm": "LSTM, ERM (earlier)",
    "old_tls_lstm": "TLS-qexp",
    "old_dro_tau_0p5": r"softmax-DRO, $\tau$ 0.5",
    "old_dro_tau_1p0": r"softmax-DRO, $\tau$ 1.0",
    "old_dro_tau_2p0": r"softmax-DRO, $\tau$ 2.0",
    "old_glycoshift_nomemory_s017": "GlycoShift, seed 17",
    "old_glycoshift_nomemory_s023": "GlycoShift, seed 23",
    "old_patchtst_erm": "PatchTST, ERM",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    data = json.load(open(args.data))
    arms = data["panels"]["eliminations"]
    head = data["panels"]["headroom"]
    comp = data["panels"]["composition"]

    rows = sorted(arms.items(), key=lambda kv: kv[1]["primary"])
    lead = max(r[1]["primary"] for r in rows)
    G.set_style()
    fig = plt.figure(figsize=(G.COL2_MM * G.MM, 112.0 * G.MM))

    # ---- panel a: every arm this project has produced, on one frame -----------------
    ax = fig.add_axes([0.235, 0.085, 0.415, 0.855])
    for i, (_key, payload) in enumerate(rows):
        value = payload["primary"]
        colour = (G.NOMINAL["rose"] if abs(value - lead) < 1e-12
                  else G.NEUTRAL if payload["family"] == "earlier campaigns"
                  else G.NOMINAL["blue"])
        ax.plot([0, value], [i, i], color=colour, lw=0.7, solid_capstyle="butt",
            zorder=1)
        ax.plot([value], [i], marker="o", ms=2.8, color=colour, zorder=2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([LABEL.get(k, k) for k, _ in rows])
    ax.set_ylim(-0.8, len(rows) - 0.2)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("severe-event sensitivity at nominal 0.5 alarms per participant-day")
    for value in (head["oracle_future_glucose"]["primary"],
        head["oracle_label"]["primary"]):
        ax.axvline(value, color=G.NOMINAL["olive"], lw=0.6, ls=(0, (2.2, 1.6)),
            zorder=0)
    ax.annotate("oracles", xy=(0.985, len(rows) - 1.7), xytext=(0.855, len(rows) - 3.6),
                fontsize=5.0, color=G.NOMINAL["olive"], ha="center",
                arrowprops=dict(arrowstyle="-", lw=0.4, color=G.NOMINAL["olive"]))
    ax.legend(handles=[
        plt.Line2D([], [], marker="o", ms=2.8, lw=0.7, color=G.NOMINAL["rose"],
                   label="leading arm"),
        plt.Line2D([], [], marker="o", ms=2.8, lw=0.7, color=G.NOMINAL["blue"],
                   label="this campaign"),
        plt.Line2D([], [], marker="o", ms=2.8, lw=0.7, color=G.NEUTRAL,
                   label="earlier campaigns, rescored"),
    ], loc="lower right", fontsize=5.0, handlelength=1.6, borderaxespad=0.4,
        labelspacing=0.35)

    # ---- panel b: what the headroom above the leader is made of ---------------------
    lead_v = head["persistence_seed17"]["primary"]
    glu_v = head["oracle_future_glucose"]["primary"]
    lab_v = head["oracle_label"]["primary"]
    gap = lab_v - lead_v
    glucose_share = (glu_v - lead_v) / gap

    bx = fig.add_axes([0.760, 0.560, 0.135, 0.330])
    bx.bar([0], [glu_v - lead_v], width=0.62, color=G.NOMINAL["olive"], zorder=2)
    bx.bar([0], [lab_v - glu_v], width=0.62, bottom=[glu_v - lead_v],
           color=G.NOMINAL["amber"], zorder=2)
    bx.set_xlim(-0.55, 0.55)
    bx.set_xticks([])
    bx.set_ylim(0, gap * 1.02)
    bx.set_yticks([0, 0.05, 0.10, 0.15])
    bx.set_ylabel("headroom above the leader")
    bx.spines["left"].set_bounds(0, gap)
    fig.text(0.905, 0.700, f"better glucose\nforecasting\n{glucose_share:.1%}",
             fontsize=5.0, color=G.INK, va="center", ha="left")
    fig.text(0.905, 0.878, f"everything else\n{1 - glucose_share:.1%}",
             fontsize=5.0, color=G.INK, va="center", ha="left")

    # ---- panel c: and it buys them on far fewer alarms -------------------------------
    spend = [
        ("label oracle", head["oracle_label"]["alarms_30"], G.NOMINAL["olive"]),
        ("glucose oracle", head["oracle_future_glucose"]["alarms_30"],
            G.NOMINAL["olive"]),
        ("threshold", comp["seed_family"]["cells"]["persist"]["30"]["alarm_count"],
         G.NOMINAL["rose"]),
        ("best trained arm", comp["replication"]["cells"]["lstm"]["30"]["alarm_count"],
         G.NOMINAL["blue"]),
    ]
    cx = fig.add_axes([0.760, 0.085, 0.225, 0.300])
    for i, (_name, count, colour) in enumerate(spend):
        cx.barh([i], [count], height=0.62, color=colour, zorder=2)
        cx.text(count + 500, i, f"{count:,}", fontsize=5.0, va="center", color=G.INK)
    cx.set_yticks(range(len(spend)))
    cx.set_yticklabels([n for n, _, _ in spend])
    cx.set_ylim(-0.6, len(spend) - 0.4)
    cx.set_xlim(0, 22000)
    cx.set_xticks([0, 10000, 20000])
    cx.set_xticklabels(["0", "10k", "20k"])
    cx.set_xlabel("alarms spent, 30-minute horizon")

    fig.text(0.010, 0.978, "a", fontsize=7.0, fontweight="bold", color=G.INK, va="top")
    fig.text(0.690, 0.978, "b", fontsize=7.0, fontweight="bold", color=G.INK, va="top")
    fig.text(0.690, 0.430, "c", fontsize=7.0, fontweight="bold", color=G.INK, va="top")

    G.save(fig, args.output, caption_words=140)
    print(f"wrote {args.output}")
    print(f"  arms drawn: {len(rows)}   leader: {LABEL.get(rows[-1][0])} {lead:.9f}")
    print(
        f"  headroom {gap:+.6f} = {gap / SEED_SIGMA:.2f} sigma; glucose share "
            f"{glucose_share:.4f}")
    ratio = head["oracle_label"]["alarms_30"] / spend[2][1]
    print(f"  label oracle spends {ratio:.3f} of the threshold's alarms")


if __name__ == "__main__":
    main()
