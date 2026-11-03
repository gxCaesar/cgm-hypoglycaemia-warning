"""Figure 2. The leading arm's advantage lives below its own alarm, and reverses above "
    "it."""
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

EUGLYCAEMIC_CUT = 70          # the clinical level-1 line; the euglycaemic restriction
THRESHOLD_ALARM = 61.0        # the leading arm inverted back to glucose, set-verified

LABEL = {
    "balanced_lstm_erm_seed17": "LSTM, balanced ERM",
    "tcn_erm_seed17": "TCN, ERM",
    "old_tcn_erm": "TCN, ERM (earlier)",
    "old_glycoshift_nomem": "GlycoShift, no memory",
    "old_dro_tau_0p5": r"softmax-DRO, $\tau$ 0.5",
    "old_tls_lstm": "TLS-qexp",
    "persistence_seed17": "current glucose threshold",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--combination-cuts", default=None,
                    help="the composition's own curve across the same cuts")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    data = json.load(open(args.data))
    eug = data["panels"]["euglycaemic"]
    cuts = data["panels"]["cut_curve"]
    arms = data["panels"]["eliminations"]

    G.set_style()
    fig = plt.figure(figsize=(G.COL2_MM * G.MM, 84.0 * G.MM))

    # ---- panel a: the ordering under the euglycaemic restriction --------------------
    rows = sorted(eug.items(), key=lambda kv: kv[1]["primary"])
    ax = fig.add_axes([0.185, 0.150, 0.290, 0.760])
    for i, (key, payload) in enumerate(rows):
        colour = G.NOMINAL["rose"] if key == "persistence_seed17" else G.NOMINAL["blue"]
        ax.plot([0, payload["primary"]], [i, i], color=colour, lw=0.7,
                solid_capstyle="butt", zorder=1)
        ax.plot([payload["primary"]], [i], marker="o", ms=2.8, color=colour, zorder=2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([LABEL.get(k, k) for k, _ in rows])
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlim(0, 0.62)
    ax.set_xlabel("severe-event sensitivity")
    ax.set_title(f"warnings must precede hypoglycaemia\n"
                 f"(decisions restricted to glucose $\\geq$ {EUGLYCAEMIC_CUT} mg/dL)",
                 fontsize=6.0, color=G.INK, pad=4)

    # ---- panel b: the whole eligibility axis, and where the estimand refuses ---------
    bx = fig.add_axes([0.585, 0.150, 0.300, 0.760])
    series = {
        "persistence_seed17": ("current glucose threshold", G.NOMINAL["rose"], "o"),
        "balanced_lstm_erm_seed17": ("LSTM, balanced ERM", G.NOMINAL["blue"], "s"),
    }
    combo = None
    if args.combination_cuts:
        raw = json.load(open(args.combination_cuts))["cuts"]
        combo = sorted((float(k), v["persist,lstm"]) for k, v in raw.items())
    for key, (name, colour, marker) in series.items():
        xs, ys = [], []
        for cut, payload in sorted(cuts["measured"][key].items(),
            key=lambda kv: int(kv[0])):
            xs.append(int(cut))
            ys.append(payload["primary"])
        xs.append(EUGLYCAEMIC_CUT)
        ys.append(eug[key]["primary"])
        bx.plot(xs, ys, color=colour, marker=marker, ms=3.0, lw=0.9, label=name,
            zorder=3)
    if combo:
        bx.plot([c for c, _ in combo], [v for _, v in combo],
                color=G.NOMINAL["olive"], marker="D", ms=3.0, lw=0.9,
                label="threshold + LSTM", zorder=4)
    bx.axvspan(74.0, 88.0, color=G.LADDER[6], zorder=0)
    bx.text(81.0, 0.42,
        "the estimand\nrefuses here:\na source has no\nevent-positive\nparticipant "
            "left",
            fontsize=5.0, color=G.MUTED, ha="center", va="center")
    bx.axvline(THRESHOLD_ALARM, color=G.INK, lw=0.5, ls=(0, (1.6, 1.6)), zorder=1)
    bx.text(THRESHOLD_ALARM - 0.7, 0.86,
        f"the threshold's own\nalarm, {THRESHOLD_ALARM:.0f} mg/dL",
            fontsize=5.0, color=G.INK, ha="right", va="top")
    bx.set_xlim(51, 88)
    # The limit follows the data. It was 0.92 when the panel held only the two members,
    # and adding the composition put its cut-54 value of 0.924 outside the axis, drawn
    # but clipped, which a render showed and a compile could not.
    top = max([v for _, v in combo] if combo else [0.0]
              + [cuts["measured"][k][c]["primary"]
                 for k in cuts["measured"] for c in cuts["measured"][k]])
    bx.set_ylim(0, top * 1.06)
    bx.set_xticks([54, 60, 70, 80])
    bx.set_xlabel("decisions restricted to glucose $\\geq$ cut-off (mg/dL)")
    bx.set_ylabel("severe-event sensitivity")
    bx.legend(loc="lower left", fontsize=5.0, handlelength=1.8, borderaxespad=0.3)

    fig.text(0.010, 0.972, "a", fontsize=7.0, fontweight="bold", color=G.INK, va="top")
    fig.text(0.505, 0.972, "b", fontsize=7.0, fontweight="bold", color=G.INK, va="top")

    G.save(fig, args.output, caption_words=150)
    print(f"wrote {args.output}")
    last_key, last = rows[0]
    print(
        f"  under the restriction the last arm is {LABEL.get(last_key)} at "
            f"{last['primary']:.9f}")
    unres = arms["persistence_seed17"]["primary"]
    print(f"  unrestricted it leads at {unres:.9f}; the restriction costs it "
          f"{unres - last['primary']:.6f}")
    lstm = eug["balanced_lstm_erm_seed17"]["primary"]
    print(
        f"  the arm it beat unrestricted now leads it by {lstm - last['primary']:+.6f}")


if __name__ == "__main__":
    main()
