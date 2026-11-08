"""Pairing a trained arm with the control, which is the paper's only positive result.

WHY THIS FIGURE EXISTS. The subsection that clears the registered promotion floor is the
longest in Results and had no display item at all, and the subsection that reports the
sealed cohort had only a table. The positive result was the one thing a reader could not
see.

WHY PANEL B IS THE ONE THAT MATTERS. The paper concedes that the pairing it reports was
selected on the partition it is reported on. Panel b is the answer to that concession: it
draws every arm this project has ever produced against the same arm paired with the
control, so the reader sees whether the effect is one lucky pair or a property of the
construction. Twenty-five of twenty-five pairings excluding the three negative-control
members improve on the arm alone.

WHAT IT DOES NOT CLAIM, and the distinction is load-bearing. Improving on the ARM is not
the same as beating the CONTROL, and only sixteen of the twenty-five do that while twelve
clear the promotion floor. Two pairings finish far BELOW the control: composing with a
per-source constant scores 0.2154 and with a label-shuffled arm 0.1377, against the
control's own 0.8061. Composition reliably lifts the arm it is given and does not
unconditionally beat the control, and a caption that said otherwise would be wrong.

THE THREE NEGATIVE-CONTROL MEMBERS ARE DRAWN AND REPORTED SEPARATELY. Rank-averaging a
global constant score with the control returns the control, and the matrix confirms it:
the pairing with the global constant scores 0.8061163731, which is the control's own
value to ten places. Including it in "every pairing improves" would be counting an
identity as a gain.

Geometry, fonts, palette and the zero-baseline rule are enforced by glycofig.save.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "code" / "scripts" / "figures")
)
import caption_words as _caption_words  # noqa: E402
import glycofig as G  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

CONTROL, MEMBER, PAIR = "persist", "lstm", "persist,lstm"
# These three negative controls are reported separately from the other pairings.
# Only the global-constant pairing is an identity with the control.
DEGENERATE = ("const_global", "const_source", "label_shuffle")
HORIZON = "30"
SOURCE_LABEL = {"Shanghai T1DM": "Shanghai"}


def _more_panels(axes, cells, floor, control_key, member_key, pair_key,
                 degenerate, external):
    """Six panels the composition matrix already held and nothing was reading.

    Whether the whole matrix clears the floor or only this pairing; where the gain comes
    from cohort by cohort; and what it costs, in alarms, in detections, in the horizon
    it is averaged over, and in the rate it was fitted to.
    """
    ax_c, ax_e, ax_g, ax_h, ax_i, ax_j = axes
    arms = ((control_key, "Control", "rose", ""),
            (member_key, "LSTM", "blue", "///"),
            (pair_key, "Composition", "olive", "..."))
    markers = {control_key: "s", member_key: "o", pair_key: "D"}
    horizons = ("30", "60")

    # c: the whole matrix against the floor, not just the cell the paper reports.
    honest = sorted(
        (v["primary"], k) for k, v in cells.items()
        if not any(d in v.get("members", []) for d in degenerate)
    )
    above = sum(1 for value, _k in honest if value >= floor)
    for index, (value, key) in enumerate(honest):
        hue = (G.NOMINAL["olive"] if key == pair_key
               else G.NOMINAL["rose"] if key == control_key else G.NEUTRAL)
        ax_c.plot([value], [index], marker=markers.get(key, "o"), markersize=2.0,
                  markerfacecolor=hue if key in (pair_key, control_key) else "none",
                  markeredgecolor=hue, markeredgewidth=0.6, zorder=3)
    ax_c.axvline(floor, color=G.INK, lw=0.6, ls=(0, (2.2, 1.6)), zorder=2)
    ax_c.set_yticks([])
    ax_c.set_xlim(0.0, 1.0)
    ax_c.set_ylabel(f"{len(honest)} cells (controls excluded)")
    ax_c.set_xlabel("Primary sensitivity")
    # A long diagonal leader across the panel is worse than no leader at all.
    # The cells rise left to right, so the empty corner is the lower right.
    ax_c.text(0.97, len(honest) * 0.13, f"{above} of {len(honest)}\nclear the floor",
              fontsize=5.0, color=G.INK, ha="right", va="center")

    # e: where the gain comes from, cohort by cohort.
    per_pair = cells[pair_key]["30"]["per_source"]
    order = sorted(per_pair, key=lambda s:
                   per_pair[s]["participant_macro_event_sensitivity"])
    for index, source in enumerate(order):
        base = per_pair[source]["participant_macro_event_sensitivity"]
        for key, _label, hue, _hatch in arms[:2]:
            other = cells[key]["30"]["per_source"][source][
                "participant_macro_event_sensitivity"]
            ax_e.plot([base - other], [index], marker=markers[key],
                      markersize=2.2, markerfacecolor=G.NOMINAL[hue],
                      markeredgecolor=G.NOMINAL[hue], zorder=3)
    ax_e.axvline(0.0, color=G.INK, lw=0.6, ls=(0, (2.2, 1.6)), zorder=2)
    ax_e.set_yticks(range(len(order)))
    ax_e.set_yticklabels([s.replace("Shanghai T1DM", "Shanghai") for s in order],
                         fontsize=5.0)
    ax_e.set_ylim(-0.6, len(order) - 0.4)
    ax_e.set_xlabel("Composition minus constituent\n(30 min)")

    # g, h, i, j: what it costs and what it buys.
    width = 0.26
    for index, (key, label, hue, hatch) in enumerate(arms):
        offs = [i + (index - 1) * width for i in range(len(horizons))]
        ax_g.bar(offs, [cells[key][h]["alarm_count"] / 1000.0 for h in horizons],
                 width=width, color=G.NOMINAL[hue], hatch=hatch or None,
                 edgecolor=G.INK, linewidth=0.35, zorder=3, label=label)
        ax_h.bar(offs, [cells[key][h]["detected_event_count"] for h in horizons],
                 width=width, color=G.NOMINAL[hue], hatch=hatch or None,
                 edgecolor=G.INK, linewidth=0.35, zorder=3)
        ax_j.bar(offs, [cells[key][h]["fapd"] for h in horizons], width=width,
                 color=G.NOMINAL[hue], hatch=hatch or None, edgecolor=G.INK,
                 linewidth=0.35, zorder=3)
        ax_i.plot([0, 1], [cells[key][h]["sensitivity"] for h in horizons],
                  color=G.NOMINAL[hue], lw=0.9, marker=markers[key],
                  markersize=2.6, zorder=3)
    for axis, ylabel in ((ax_g, "Surviving alarms (thousands)"),
                         (ax_h, "Events detected"),
                         (ax_j, "Achieved false alarms per\nparticipant-day")):
        axis.set_xticks(range(len(horizons)))
        axis.set_xticklabels([f"{h} min" for h in horizons], fontsize=5.0)
        axis.set_ylabel(ylabel)
    ax_j.axhline(0.5, color=G.INK, lw=0.6, ls=(0, (2.2, 1.6)), zorder=4)
    ax_g.legend(loc="lower center", bbox_to_anchor=(0.5, 1.065), ncol=3, fontsize=5.0,
                handlelength=0.9, handleheight=0.9, columnspacing=0.6,
                borderaxespad=0.1, labelspacing=0.15)
    ax_i.set_xlim(-0.2, 1.2)
    ax_i.set_xticks([0, 1])
    ax_i.set_xticklabels(["30 min", "60 min"], fontsize=5.0)
    ax_i.set_ylim(0.0, 1.0)
    ax_i.set_ylabel("Event sensitivity")
    if external and not all(k in external for k, *_ in arms[:1]):
        raise SystemExit("the external block is missing the control")
    return above, len(honest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--seed-dir", required=True, type=Path)
    parser.add_argument("--external-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--height-mm", type=float, default=185.0)
    parser.add_argument("--caption-words", type=int,
                        default=_caption_words.count("fig_composition"))
    # Stated in the caption in words, which the manuscript's numeral audit cannot see.
    parser.add_argument("--expect-improving", type=int, default=25)
    parser.add_argument("--expect-above-control", type=int, default=16)
    parser.add_argument("--expect-above-floor", type=int, default=12)
    arguments = parser.parse_args(argv)

    matrix = json.loads(arguments.matrix.read_text())
    cells, floor = matrix["cells"], float(matrix["promotion_floor"])
    control = cells[CONTROL]["primary"]

    paired = []
    for _name, cell in cells.items():
        if len(cell["members"]) != 2 or CONTROL not in cell["members"]:
            continue
        other = next(m for m in cell["members"] if m != CONTROL)
        if other in cells:
            paired.append((other, cells[other]["primary"], cell["primary"]))
    honest = [row for row in paired if row[0] not in DEGENERATE]
    counts = (
        sum(1 for _, solo, pair in honest if pair > solo),
        sum(1 for _, _, pair in honest if pair > control),
        sum(1 for _, _, pair in honest if pair >= floor),
    )
    expected = (arguments.expect_improving, arguments.expect_above_control,
                arguments.expect_above_floor)
    if counts != expected:
        raise SystemExit(
            f"panel b counts {counts} against the caption's {expected}; the caption is "
            "now wrong about its own figure"
        )

    seeds = {
        path.stem: json.loads(path.read_text())[
            "primary_equal_horizon_mean_event_sensitivity"]
        for path in sorted(arguments.seed_dir.glob("seed*.json"))
    }
    external = {
        name: float(json.loads(
            (arguments.external_dir / f"{name}-loop.json").read_text()
        )["primary_equal_horizon_mean_event_sensitivity"])
        for name in ("persist", "lstm", "both")
    }

    G.set_style()
    figure = plt.figure(figsize=(G.COL2_MM * G.MM, arguments.height_mm * G.MM))
    # Rows are laid out from one gap constant rather than three hand-set numbers.
    # The gaps here were 0.09 to 0.13 of the canvas, which on a 185 mm figure is a
    # blank band of 17 to 24 mm between rows, and only about 10 mm of it is doing
    # any work: the tick labels and axis label of the row above, plus clearance for
    # the panel letter of the row below. The rest was dead. Setting one gap and
    # solving for the row height gives that space back to the panels.
    _bottom, _top, _gap = 0.072, 0.957, 0.070
    _tall = (_top - _bottom - 2 * _gap) / 3
    _r1, _r2, _r3 = _bottom + 2 * (_tall + _gap), _bottom + _tall + _gap, _bottom
    ax_a = figure.add_axes([0.135, _r1, 0.175, _tall])
    ax_b = figure.add_axes([0.430, _r1, 0.200, _tall])
    ax_c = figure.add_axes([0.760, _r1, 0.215, _tall])
    ax_d = figure.add_axes([0.160, _r2, 0.200, _tall])
    ax_e = figure.add_axes([0.500, _r2, 0.200, _tall])
    ax_f = figure.add_axes([0.815, _r2, 0.160, _tall])
    ax_g = figure.add_axes([0.090, _r3, 0.155, _tall])
    ax_h = figure.add_axes([0.335, _r3, 0.155, _tall])
    ax_i = figure.add_axes([0.580, _r3, 0.155, _tall])
    ax_j = figure.add_axes([0.825, _r3, 0.150, _tall])

    # ---- a: the composition, its members, and the registered floor -----------------
    values = sorted(seeds.values())
    ladder = [
        ("Composition,\nfive seeds", None, G.NOMINAL["olive"], "D"),
        ("LSTM", cells[MEMBER]["primary"], G.NOMINAL["blue"], "o"),
        ("Control", control, G.NOMINAL["rose"], "s"),
    ]
    for index, (_name, value, colour, mark) in enumerate(ladder):
        if value is None:
            for one in values:
                ax_a.plot([one], [index], marker=mark, markersize=3.0, color=colour,
                          markerfacecolor=colour if one >= floor else "none",
                          markeredgecolor=colour, markeredgewidth=0.8,
                          linestyle="none", zorder=3)
            ax_a.plot([min(values), max(values)], [index, index], color=colour,
                      lw=0.6, zorder=2)
        else:
            ax_a.plot([value], [index], marker=mark, markersize=3.2, color=colour,
                      zorder=3)
    ax_a.axvline(floor, color=G.INK, lw=0.6, ls=(0, (2.4, 1.6)), zorder=1)
    ax_a.text(floor, len(ladder) - 0.42, " registered floor", fontsize=5.0,
              color=G.INK, ha="left", va="center")
    ax_a.set_yticks(range(len(ladder)))
    ax_a.set_yticklabels([n for n, *_ in ladder], fontsize=5.4)
    ax_a.set_ylim(-0.6, len(ladder) - 0.2)
    ax_a.set_xlim(0.66, 1.0)
    ax_a.set_xlabel("Primary sensitivity")

    # ---- b: is it one lucky pair, or the construction? ------------------------------
    limits = (0.0, 1.0)
    ax_b.plot(limits, limits, color=G.MUTED, lw=0.5, ls=(0, (2.4, 1.6)), zorder=1)
    ax_b.axhline(control, color=G.NOMINAL["rose"], lw=0.6, zorder=1)
    ax_b.axhline(floor, color=G.INK, lw=0.6, ls=(0, (2.4, 1.6)), zorder=1)
    for other, solo, pair in paired:
        degenerate = other in DEGENERATE
        ax_b.plot([solo], [pair], marker="^" if degenerate else "o", markersize=3.0,
                  markerfacecolor="none" if degenerate else G.NOMINAL["olive"],
                  markeredgecolor=G.NEUTRAL if degenerate else G.NOMINAL["olive"],
                  markeredgewidth=0.8, linestyle="none", zorder=3)
    ax_b.set_xlim(*limits)
    ax_b.set_ylim(*limits)
    ax_b.set_xlabel("Arm primary sensitivity")
    ax_b.set_ylabel("Paired primary sensitivity")
    ax_b.legend(handles=[
        plt.Line2D([], [], marker="o", ms=3.0, ls="none", color=G.NOMINAL["olive"],
                   label="a trained arm"),
        plt.Line2D([], [], marker="^", ms=3.0, ls="none", color=G.NEUTRAL,
                   markerfacecolor="none", label="constant or shuffled member"),
    ], loc="lower right", fontsize=5.0, handlelength=1.2, borderaxespad=0.3,
        labelspacing=0.3)

    # ---- c: and it is not carried by a few cohorts ----------------------------------
    per = {
        key: cells[key][HORIZON]["per_source"]
        for key in (PAIR, MEMBER, CONTROL)
    }
    order = sorted(per[PAIR], key=lambda s: per[PAIR][s]["participant_macro_event_sensitivity"])
    positions = range(len(order))
    for key, colour, mark in ((PAIR, G.NOMINAL["olive"], "D"),
                              (CONTROL, G.NOMINAL["rose"], "s"),
                              (MEMBER, G.NOMINAL["blue"], "o")):
        ax_d.plot(
            [per[key][s]["participant_macro_event_sensitivity"] for s in order],
            list(positions), marker=mark, markersize=2.8, linestyle="none",
            color=colour, zorder=3,
            label={PAIR: "Composition", CONTROL: "Control", MEMBER: "LSTM"}[key],
        )
    for index, source in zip(positions, order, strict=True):
        low = min(per[k][source]["participant_macro_event_sensitivity"] for k in per)
        high = max(per[k][source]["participant_macro_event_sensitivity"] for k in per)
        ax_d.plot([low, high], [index, index], color=G.MUTED, lw=0.4, zorder=2)
    ax_d.set_yticks(list(positions))
    ax_d.set_yticklabels([SOURCE_LABEL.get(s, s) for s in order], fontsize=5.2)
    ax_d.set_ylim(-0.7, len(order) - 0.3)
    ax_d.set_xlim(0.0, 1.02)
    ax_d.set_xlabel("Event sensitivity, 30 min")
    ax_d.legend(loc="lower left", fontsize=5.0, handlelength=1.0, borderaxespad=0.3,
                labelspacing=0.3)

    # ---- d: the cohort opened once --------------------------------------------------
    development = {"persist": control, "lstm": cells[MEMBER]["primary"],
                   "both": cells[PAIR]["primary"]}
    style = {"both": (G.NOMINAL["olive"], "D"), "persist": (G.NOMINAL["rose"], "s"),
             "lstm": (G.NOMINAL["blue"], "o")}
    for name, (colour, mark) in style.items():
        ax_f.plot([0, 1], [development[name], external[name]], color=colour, lw=0.8,
                  marker=mark, markersize=3.0, zorder=3)
    ax_f.set_xlim(-0.28, 1.28)
    ax_f.set_xticks([0, 1])
    ax_f.set_xticklabels(["Development", "Sealed cohort"], fontsize=5.4)
    ax_f.set_ylim(0.6, 1.0)
    ax_f.set_ylabel("Level 2 event sensitivity")

    _more_panels((ax_c, ax_e, ax_g, ax_h, ax_i, ax_j), cells, floor,
                 CONTROL, MEMBER, PAIR, DEGENERATE, external)

    order = (ax_a, ax_b, ax_c, ax_d, ax_e, ax_f, ax_g, ax_h, ax_i, ax_j)
    titles = (
        "Seed variability", "Control pairings", "Ordered values", "Cohort sensitivity",
        "Cohort contrasts", "External transfer", "Alarm burden", "Matched events",
        "Horizon sensitivity", "False-alarm rate",
    )
    for axis, title in zip(order, titles, strict=True):
        axis.set_title(title, loc="left", fontsize=6.0, pad=3)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.set_axisbelow(True)
    for axis, letter, dx in (
        (ax_a, "a", 0.062), (ax_b, "b", 0.058), (ax_c, "c", 0.060),
        (ax_d, "d", 0.130), (ax_e, "e", 0.058), (ax_f, "f", 0.058),
        (ax_g, "g", 0.058), (ax_h, "h", 0.058), (ax_i, "i", 0.058),
        (ax_j, "j", 0.058),
    ):
        box = axis.get_position()
        figure.text(box.x0 - dx, box.y1 + 0.026, f"{letter})", fontsize=7.0,
                    fontweight="bold", color=G.INK, va="top")

    falling = [n for n in style if external[n] < development[n]]
    if falling != ["both"]:
        raise SystemExit(
            "the caption says the composition alone falls between development and the "
            f"sealed cohort; {falling} do"
        )
    G.save(figure, str(arguments.output), caption_words=arguments.caption_words)
    print(f"wrote {arguments.output}  {len(paired)} pairings, {len(honest)} "
          f"excluding negative controls, of which {counts[0]} improve, "
          f"{counts[1]} beat the control "
          f"and {counts[2]} clear the floor; seeds "
          f"{statistics.mean(values):.4f} +/- {statistics.stdev(values):.4f}, "
          f"{sum(1 for v in values if v >= floor)} of {len(values)} above it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
