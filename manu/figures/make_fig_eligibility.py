"""The eligibility rule decides the ordering, and it does so in every cohort.

WHY THIS REPLACES TWO FIGURES. `fig_eligibility_reversal` showed the reversal and
`fig_per_source` showed which of the two effects survives cohort by cohort. They are the
same claim at two resolutions, they belong to one Results subsection, and being separate
floats they were cited out of numerical order and then landed on the same page anyway.
One figure fixes the ordering, halves the float pressure, and puts the aggregate result
beside the per-cohort evidence that decides whether to believe it.

WHAT EACH HALF IS FOR. Panels a and b are the effect: move the eligibility cut and the
ordering inverts, with no model retrained and every policy refitted inside the
restriction. Panels c and d are the reason to trust one of the two effects and not the
other: the model effect is carried by cohorts holding two or three scored participants
and changes sign among them, while the eligibility effect is near half a point in every
cohort with more than three.

Geometry, fonts, palette and the zero-baseline rule are enforced by glycofig.save.
"""

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

CONTROL = "persistence_seed17"
LEADING = "balanced_lstm_erm_seed17"
CUTOFFS = (54, 60, 70)
TOLERANCE = 1e-9

CURVE_ARMS = (
    (CONTROL, "rose", "-", "s"),
    (LEADING, "blue", "--", "o"),
)
BAR_ARMS = (
    LEADING, "tcn_erm_seed17", "old_tcn_erm", "old_glycoshift_nomem",
    "old_dro_tau_0p5", "old_tls_lstm", CONTROL,
)
LABEL = {
    CONTROL: "Current glucose (control)",
    LEADING: "LSTM, balanced ERM",
    "tcn_erm_seed17": "Temporal conv., ERM",
    "old_tcn_erm": "Temporal conv. (earlier)",
    "old_glycoshift_nomem": "Memory transformer",
    "old_dro_tau_0p5": r"GroupDRO, $\tau=0.5$",
    "old_tls_lstm": "Tail-loss shaping",
}
# Splitting LABEL on its comma gave the legend a bare "LSTM", which names an
# architecture rather than an arm.
SHORT = {CONTROL: "Current glucose (control)", LEADING: "LSTM, balanced ERM"}
EFFECTS = (
    ("model", "Control minus strongest trained arm,\nfrozen eligibility", "rose"),
    ("eligibility", "Strongest trained arm minus control,\nat the 70 mg/dL cut-off",
     "blue"),
)


def _primary(path: Path) -> float:
    payload = json.loads(path.read_text())
    value = float(payload["primary_equal_horizon_mean_event_sensitivity"]["global"])
    if not 0.0 <= value <= 1.0:
        raise SystemExit(f"{path.name}: primary {value} is not a sensitivity")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutcurve-dir", required=True, type=Path)
    parser.add_argument("--euglycaemic-dir", required=True, type=Path)
    parser.add_argument("--effects", required=True, type=Path)
    parser.add_argument("--mechanism", required=True, type=Path)
    parser.add_argument("--warnable", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--height-mm", type=float, default=185.0)
    parser.add_argument("--caption-words", type=int,
                        default=_caption_words.count("fig_eligibility"))
    arguments = parser.parse_args(argv)

    curve = {
        arm: [
            _primary(arguments.cutcurve_dir / f"{arm}_cut{cut}.json") if cut != 70
            else _primary(arguments.euglycaemic_dir / f"{arm}.json")
            for cut in CUTOFFS
        ]
        for arm, *_ in CURVE_ARMS
    }
    at70 = {arm: _primary(arguments.euglycaemic_dir / f"{arm}.json")
            for arm in BAR_ARMS}

    payload = json.loads(arguments.effects.read_text())
    per = payload["per_source_contrast"]
    if len(per) != 11:
        raise SystemExit(f"expected eleven sources, found {len(per)}")
    for key in ("model", "eligibility"):
        rebuilt = sum(v[key] for v in per.values()) / len(per)
        stored = payload[f"{key}_effect"]["observed"]
        if abs(rebuilt - stored) > TOLERANCE:
            raise SystemExit(
                f"{key}: per-source values average to {rebuilt:.9f} against a reported "
                f"{stored:.9f}; these are not the same quantity"
            )

    G.set_style()
    figure = plt.figure(figsize=(G.COL2_MM * G.MM, arguments.height_mm * G.MM))
    # Three bands. The top states the reversal, the middle asks whether one cohort
    # carries it, and the bottom prices it on the axes a reader would push back on.
    # Rows are laid out from one gap constant rather than three hand-set numbers.
    # The gaps here were 0.09 to 0.13 of the canvas, which on a 185 mm figure is a
    # blank band of 17 to 24 mm between rows, and only about 10 mm of it is doing
    # any work: the tick labels and axis label of the row above, plus clearance for
    # the panel letter of the row below. The rest was dead. Setting one gap and
    # solving for the row height gives that space back to the panels.
    _bottom, _top, _gap = 0.070, 0.958, 0.070
    _tall = (_top - _bottom - 2 * _gap) / 3
    _r1, _r2, _r3 = _bottom + 2 * (_tall + _gap), _bottom + _tall + _gap, _bottom
    ax_a = figure.add_axes([0.080, _r1, 0.215, _tall])
    ax_b = figure.add_axes([0.500, _r1, 0.175, _tall])
    ax_e = figure.add_axes([0.775, _r1, 0.200, _tall])
    ax_c = figure.add_axes([0.155, _r2, 0.205, _tall])
    ax_d = figure.add_axes([0.400, _r2, 0.205, _tall])
    ax_f = figure.add_axes([0.775, _r2, 0.200, _tall])
    ax_g = figure.add_axes([0.080, _r3, 0.155, _tall])
    ax_h = figure.add_axes([0.325, _r3, 0.155, _tall])
    ax_i = figure.add_axes([0.570, _r3, 0.155, _tall])
    ax_j = figure.add_axes([0.815, _r3, 0.155, _tall])

    # ---- a: move the cut, invert the ordering --------------------------------------
    for arm, hue, dash, mark in CURVE_ARMS:
        ax_a.plot(CUTOFFS, curve[arm], color=G.NOMINAL[hue], linestyle=dash,
                  marker=mark, markersize=3.0, linewidth=0.9,
                  label=SHORT[arm], zorder=3)
    ax_a.set_xticks(CUTOFFS)
    ax_a.set_xlabel("Eligibility cut-off on current glucose (mg/dL)")
    ax_a.set_ylabel("Level 2 event sensitivity")
    ax_a.set_ylim(0.0, 1.0)
    ax_a.legend(loc="lower left", handlelength=2.0, borderaxespad=0.3,
                labelspacing=0.3)

    # ---- b: where the control finishes once decisions start at 70 ------------------
    order = sorted(BAR_ARMS, key=lambda a: at70[a])
    for index, arm in enumerate(order):
        hue = "rose" if arm == CONTROL else "blue" if arm == LEADING else None
        colour = G.NOMINAL[hue] if hue else G.NEUTRAL
        # A lollipop, not a bar: the axis has to start at zero for a bar and the
        # interesting range here is the top half.
        ax_b.plot([0.0, at70[arm]], [index, index], color=colour, lw=0.7,
                  solid_capstyle="butt", zorder=1)
        ax_b.plot([at70[arm]], [index], marker="s" if arm == CONTROL else "o",
                  markersize=3.0, color=colour, zorder=2)
    ax_b.set_yticks(range(len(order)))
    ax_b.set_yticklabels([LABEL[a] for a in order], fontsize=5.2)
    ax_b.set_ylim(-0.7, len(order) - 0.3)
    ax_b.set_xlim(0.0, 0.62)
    ax_b.set_xlabel("Level 2 event sensitivity at a 70 mg/dL cut-off")

    # ---- c, d: which of the two effects survives cohort by cohort ------------------
    rows = sorted(per, key=lambda s: (per[s]["participants"], s))
    positions = list(range(len(rows)))[::-1]
    for axis, (key, title, hue) in zip((ax_c, ax_d), EFFECTS, strict=True):
        mean = payload[f"{key}_effect"]["observed"]
        low, high = payload[f"{key}_effect"]["ci95"]
        axis.axvspan(low, high, color=G.MUTED, alpha=0.12, lw=0, zorder=0)
        axis.axvline(mean, color=G.MUTED, linewidth=0.7, zorder=1)
        axis.axvline(0.0, color=G.INK, linewidth=0.6, linestyle=(0, (2.5, 1.5)),
                     zorder=2)
        for pos, source in zip(positions, rows, strict=True):
            small = per[source]["participants"] < 5
            axis.plot([per[source][key]], [pos], marker="o" if small else "s",
                      markersize=3.4 if small else 2.9,
                      markerfacecolor="none" if small else G.NOMINAL[hue],
                      markeredgecolor=G.NOMINAL[hue], markeredgewidth=0.9,
                      linestyle="none", zorder=4)
        axis.set_title(title, loc="left", fontsize=6.0, pad=3)
        axis.set_xlabel("Difference in level 2 event sensitivity")
        axis.set_xlim(-0.62, 0.72)
        axis.set_ylim(-0.7, len(rows) - 0.3)
    ax_c.set_yticks(positions)
    ax_c.set_yticklabels(
        [f"{s}  ({per[s]['participants']})" for s in rows], fontsize=5.2)
    ax_d.set_yticks(positions)
    ax_d.set_yticklabels([])

    flipped, shared_count = _more_panels(
        (ax_e, ax_f, ax_g, ax_h, ax_i, ax_j),
        json.loads(arguments.mechanism.read_text())["rank_agreement"],
        json.loads(arguments.warnable.read_text())["horizons"],
        arguments.cutcurve_dir, arguments.euglycaemic_dir)

    for axis in (ax_a, ax_b, ax_c, ax_d, ax_e, ax_f, ax_g, ax_h, ax_i, ax_j):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.set_axisbelow(True)
    for axis, letter, dx in (
        (ax_a, "a", 0.062), (ax_b, "b", 0.058), (ax_e, "e", 0.062),
        (ax_c, "c", 0.140), (ax_d, "d", 0.030), (ax_f, "f", 0.062),
        (ax_g, "g", 0.058), (ax_h, "h", 0.058), (ax_i, "i", 0.058),
        (ax_j, "j", 0.058),
    ):
        box = axis.get_position()
        figure.text(box.x0 - dx, box.y1 + 0.026, f"{letter})", fontsize=7.0,
                    fontweight="bold", color=G.INK, va="top")

    G.save(figure, str(arguments.output), caption_words=arguments.caption_words)
    last = order[0]
    small = [s for s in rows if per[s]["participants"] < 5]
    if last != CONTROL:
        raise SystemExit(
            f"the caption says the control finishes last at the 70 mg/dL cut-off; "
            f"{LABEL[last]} does"
        )
    # The caption says ALL of them, so the guard says all of them.
    if flipped != shared_count:
        raise SystemExit(
            f"the caption says the control loses ground in every cohort when the cut "
            f"moves from 54 to 70; it does in {flipped} of {shared_count}"
        )
    print(f"wrote {arguments.output}  control loses ground in {flipped} cohorts; "
          f"{len(order)} arms at the 70 cut-off, control "
          f"last at {at70[CONTROL]:.6f}; {len(small)} of {len(rows)} cohorts hold "
          f"fewer than five scored participants")
    return 0




def _more_panels(axes, agreement, warnable, cutcurve_dir, euglycaemic_dir):
    """The six panels that price the reversal on the axes a reader pushes back on.

    Where the two arms agree and where they do not; what fraction of events could be
    warned about at all; what each cut costs in detections and alarms; whether the gap
    survives a matched alarm count; and whether the reversal is one cohort or eleven.
    """
    ax_e, ax_f, ax_g, ax_h, ax_i, ax_j = axes
    bands = ["<54", "54-61", "61-70", "70-100", "100-140", "140-180", ">180"]
    available = [b for b in bands if f"spearman_{b}" in agreement["persist|lstm|30"]]
    if len(available) < 4:
        raise SystemExit(
            f"the band agreement carries {available}, too few for a curve")

    # e: agreement is global, disagreement is local.
    for key, hue, dash in (("persist|lstm|30", "blue", "-"),
                           ("persist|tcn|30", "amber", "--")):
        ax_e.plot(range(len(available)),
                  [float(agreement[key][f"spearman_{b}"]) for b in available],
                  color=G.NOMINAL[hue], linestyle=dash,
                  marker="o" if "lstm" in key else "^", markersize=2.2,
                  linewidth=0.8, zorder=3,
                  label=key.split("|")[1].upper())
    ax_e.axvspan(0.5, 1.5, color=G.MUTED, alpha=0.12, lw=0, zorder=1)
    ax_e.set_xticks(range(len(available)))
    ax_e.set_xticklabels(available, rotation=40, ha="right", fontsize=5.0)
    ax_e.set_ylim(0.0, 1.0)
    ax_e.set_ylabel("Rank agreement\nwith the control")
    ax_e.set_xlabel("Glucose band (mg/dL)")
    ax_e.legend(loc="lower right", fontsize=5.0, handlelength=1.6,
                borderaxespad=0.2, labelspacing=0.2)

    # f: the ceiling nobody controls.
    per = warnable["30"]["per_source"]
    order_f = sorted(per, key=lambda s: float(per[s]["fraction"]))
    for index, source in enumerate(order_f):
        value = float(per[source]["fraction"])
        ax_f.plot([0.0, value], [index, index], color=G.NEUTRAL, lw=0.6,
                  solid_capstyle="butt", zorder=2)
        ax_f.plot([value], [index], marker="o", markersize=2.4,
                  color=G.NOMINAL["olive"], zorder=3)
    ax_f.set_yticks(range(len(order_f)))
    ax_f.set_yticklabels([s.replace("Shanghai T1DM", "Shanghai") for s in order_f],
                         fontsize=5.0)
    ax_f.set_ylim(-0.6, len(order_f) - 0.4)
    ax_f.set_xlim(0.0, 1.0)
    ax_f.set_xlabel("Level 2 events with an\neligible lead-in, 30 min")

    # g, h: what each cut costs, and whether a matched alarm count keeps the gap.
    cuts = (54, 60)
    arms = (("persistence_seed17", "Control", "rose"),
            ("balanced_lstm_erm_seed17", "LSTM", "blue"))
    detected, alarms_at = {}, {}
    for stem, _label, _hue in arms:
        for cut in cuts:
            path = cutcurve_dir / f"{stem}_cut{cut}.json"
            block = json.loads(path.read_text())["30"]["global"]
            detected[(stem, cut)] = int(block["detected_event_count"])
            alarms_at[(stem, cut)] = int(block["alarm_count"])
    width = 0.34
    for index, (stem, label, hue) in enumerate(arms):
        ax_g.bar([i + (index - 0.5) * width for i in range(len(cuts))],
                 [detected[(stem, c)] for c in cuts], width=width,
                 color=G.NOMINAL[hue], hatch="///" if index else "",
                 edgecolor=G.INK, linewidth=0.35, zorder=3, label=label)
        ax_h.bar([i + (index - 0.5) * width for i in range(len(cuts))],
                 [alarms_at[(stem, c)] for c in cuts], width=width,
                 color=G.NOMINAL[hue], hatch="///" if index else "",
                 edgecolor=G.INK, linewidth=0.35, zorder=3)
    for axis, ylabel in ((ax_g, "Events detected, 30 min"),
                         (ax_h, "Alarms charged, 30 min")):
        axis.set_xticks(range(len(cuts)))
        axis.set_xticklabels([f"{c}" for c in cuts], fontsize=5.0)
        axis.set_xlabel("Eligibility cut-off (mg/dL)")
        axis.set_ylabel(ylabel)
    ax_g.legend(loc="upper right", fontsize=5.0, handlelength=1.1,
                borderaxespad=0.2, labelspacing=0.2)

    # i, j: is the reversal one cohort or eleven?
    at = {}
    for stem, _label, _hue in arms:
        at[(stem, 54)] = json.loads(
            (cutcurve_dir / f"{stem}_cut54.json").read_text())["30"]["global"]
        at[(stem, 70)] = json.loads(
            (euglycaemic_dir / f"{stem}.json").read_text())["30"]["global"]
    flipped = 0
    shared_count = 0
    for axis, (stem, label, hue) in zip((ax_i, ax_j), arms, strict=True):
        low, high = at[(stem, 54)]["per_source"], at[(stem, 70)]["per_source"]
        shared = sorted(set(low) & set(high))
        if stem.startswith("persistence"):
            shared_count = len(shared)
        for source in shared:
            a, b = float(low[source]["sensitivity"]), float(high[source]["sensitivity"])
            axis.plot([0, 1], [a, b], color=G.NOMINAL[hue], lw=0.6, marker="o",
                      markersize=1.8, zorder=3)
            if stem.startswith("persistence") and b < a:
                flipped += 1
        axis.set_xlim(-0.2, 1.2)
        axis.set_xticks([0, 1])
        axis.set_xticklabels(["54", "70"], fontsize=5.0)
        axis.set_ylim(0.0, 1.05)
        axis.set_xlabel(f"{label}, cut-off (mg/dL)")
    ax_i.set_ylabel("Sensitivity, 30 min")
    return flipped, shared_count

if __name__ == "__main__":
    raise SystemExit(main())
