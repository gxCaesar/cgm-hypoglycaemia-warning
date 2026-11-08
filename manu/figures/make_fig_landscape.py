"""Every arm on one frame, the ceiling above them, and the two horizons underneath.

WHY THIS FIGURE EXISTS. Two Results subsections carried substantive claims with no
display item: that a single-threshold rule leads every trained comparator, and that the
primary averages two horizons which do not agree. The first was a paragraph naming three
of eighteen numbers; the second was one sentence. Both are pictures.

WHY IT DOES NOT COLOUR BY METHOD FAMILY. There are seven families and the nominal
palette holds five plus one neutral, and this repository has already shipped the figure
that results from ignoring that arithmetic: "figure 9's class panel drew seven labels
through each other". Colour instead carries the distinction the paper's own abstract
makes, which is that nine of the fifteen trained fits were selected under a superseded
checkpoint proxy. That is two categories plus the control, it fits the palette, and it
draws a caveat that currently lives only in a subordinate clause.

WHAT FEEDS IT. `arm-table.json`, which is the same artifact the supplementary arm table
is generated from, so the figure and the table cannot drift apart; and the two oracle
result files. Not the retired intermediate a previous draft used, which carried neither
the horizon split nor the checkpoint distinction.

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

HEIGHT_MM = 185.0
# Spelled in the caption as an English word, so checked here.
EXPECT_SINGLE_FIT = 14
# Bars carry a category by colour alone unless a hatch is added, and hatch draws in the
# edge colour, so an edgecolor is not optional here.
HATCH = ("///", "...", "xxx", "\\\\\\")


def _labels(rows: list[dict]) -> dict[str, str]:
    """Disambiguate display names that occur more than once.

    Two arms are called "Temporal convolution, ERM" and differ only by checkpoint
    proxy; two are "Memory transformer, no memory" and differ only by seed. A y-axis
    that prints either name twice is telling the reader nothing about which is which.
    """
    seen: dict[str, list[dict]] = {}
    for row in rows:
        seen.setdefault(row["display"], []).append(row)
    out: dict[str, str] = {}
    for display, group in seen.items():
        for row in group:
            if len(group) == 1:
                out[row["arm"]] = display
            elif len({r["seed"] for r in group}) > 1:
                out[row["arm"]] = f"{display}, seed {row['seed']}"
            else:
                mark = "earlier" if row["checkpoint_proxy"] == "superseded" else "this"
                out[row["arm"]] = f"{display} ({mark})"
    return out


def _oracle_values(oracle):
    """Keep the primary and horizon-specific oracle quantities separate."""
    primary = {
        name: float(payload["primary_equal_horizon_mean_event_sensitivity"]["global"])
        for name, payload in oracle.items()
    }
    alarms = {
        name: int(payload["30"]["global"]["alarm_count"])
        for name, payload in oracle.items()
    }
    sensitivity_30 = {
        name: float(payload["30"]["global"]["sensitivity"])
        for name, payload in oracle.items()
    }
    return primary, alarms, sensitivity_30


def _extra_panels(figure, rows, control, label, axes, oracle_sensitivity_30, oracle_alarms):
    """The six panels that say how the ordering in a could be wrong.

    Each reads only the frozen arm table, and each answers a question the ordering
    invites: does every arm sit at the operating point it was fitted to, what does each
    buy for what it spends, is the method family the discriminator, does the checkpoint
    proxy explain the nine rescored arms, does the ordering survive reading one horizon
    at a time, and how many fits is any of it built on.
    """
    ax_e, ax_f, ax_g, ax_h, ax_i, ax_j = axes
    firing = [r for r in rows if r["alarms_30"] > 0]

    # e: the nominal budget is a fitting target, not an achieved rate.
    ax_e.axvline(0.5, color=G.INK, lw=0.6, ls=(0, (2.2, 1.6)), zorder=2)
    for index, row in enumerate(sorted(firing, key=lambda r: r["achieved_fapd"])):
        colour = (G.NOMINAL["rose"] if row["arm"] == control["arm"]
                  else G.NOMINAL["blue"] if row["checkpoint_proxy"] == "current"
                  else G.NEUTRAL)
        leader = row["arm"] == control["arm"]
        current = row["checkpoint_proxy"] == "current"
        ax_e.plot([row["achieved_fapd"]], [index],
                  marker="s" if leader else "o", markersize=2.4,
                  markerfacecolor=colour if current or leader else "none",
                  markeredgecolor=colour, zorder=3)
    ax_e.set_yticks([])
    ax_e.set_xlim(0.40, 0.62)
    ax_e.set_xticks([0.45, 0.5, 0.55, 0.6])
    ax_e.set_xlabel("False alarms per\nparticipant-day", linespacing=1.35,
                    labelpad=2.0)
    ax_e.set_ylabel("Arms that fire")

    # f: what each arm buys for what it spends, with the two oracles for scale.
    for row in rows:
        if row["alarms_30"] == 0:
            continue
        colour = (G.NOMINAL["rose"] if row["arm"] == control["arm"]
                  else G.NOMINAL["blue"] if row["checkpoint_proxy"] == "current"
                  else G.NEUTRAL)
        leader = row["arm"] == control["arm"]
        current = row["checkpoint_proxy"] == "current"
        ax_f.plot([row["alarms_30"] / 1000.0], [row["sensitivity_30"]],
                  marker="s" if leader else "o", markersize=2.4,
                  markerfacecolor=colour if current or leader else "none",
                  markeredgecolor=colour,
                  linestyle="none", zorder=3)
    for name, mark in (("oracle_label", "*"), ("oracle_future_glucose", "D")):
        ax_f.plot([oracle_alarms[name] / 1000.0], [oracle_sensitivity_30[name]], marker=mark,
                  markersize=3.4, color=G.NOMINAL["olive"], linestyle="none", zorder=4)
    ax_f.set_xlabel("Alarms spent (thousands)")
    ax_f.set_ylabel("Sensitivity, 30 min")
    ax_f.set_ylim(0.0, 1.05)

    # g: the method family is not what separates them.
    families = sorted({r["family"] for r in rows})
    for index, family in enumerate(families):
        values = [r["primary"] for r in rows if r["family"] == family]
        ax_g.plot([min(values), max(values)], [index, index], color=G.NEUTRAL, lw=0.6,
                  solid_capstyle="butt", zorder=2)
        for value in values:
            ax_g.plot([value], [index], marker="o", markersize=2.2,
                      markerfacecolor="none", markeredgecolor=G.MUTED,
                      markeredgewidth=0.6, zorder=3)
    ax_g.axvline(control["primary"], color=G.NOMINAL["rose"], lw=0.6,
                 ls=(0, (2.2, 1.6)), zorder=1)
    ax_g.set_yticks(range(len(families)))
    short = {"cheap control": "cheap", "GroupDRO sweep": "GroupDRO",
             "temporal convolution": "TCN", "memory transformer": "memory",
             "tail-loss shaping": "tail-loss", "patch transformer": "PatchTST"}
    ax_g.set_yticklabels([short.get(f, f) for f in families], fontsize=5.0)
    ax_g.set_ylim(-0.6, len(families) - 0.4)
    ax_g.set_xlim(0.0, 1.0)
    ax_g.set_xlabel("Primary endpoint")

    # h: the nine rescored arms, against the nine selected under the current proxy.
    for index, proxy in enumerate(("current", "superseded")):
        values = [r["primary"] for r in rows if r["checkpoint_proxy"] == proxy]
        colour = G.NOMINAL["blue"] if proxy == "current" else G.NEUTRAL
        for value in values:
            ax_h.plot([index], [value], marker="o", markersize=2.4,
                      markerfacecolor=colour if proxy == "current" else "none",
                      markeredgecolor=colour, markeredgewidth=0.7, zorder=3)
        ax_h.plot([index - 0.22, index + 0.22],
                  [sum(values) / len(values)] * 2, color=colour, lw=0.9, zorder=4)
    ax_h.set_xlim(-0.5, 1.5)
    ax_h.set_xticks([0, 1])
    ax_h.set_xticklabels(["current", "superseded"], fontsize=5.0)
    ax_h.set_ylim(0.0, 1.0)
    ax_h.set_ylabel("Primary endpoint")
    ax_h.set_xlabel("Checkpoint proxy")

    # i: does the ordering survive reading one horizon at a time?
    by30 = {r["arm"]: n for n, r in enumerate(
        sorted(rows, key=lambda r: r["sensitivity_30"]))}
    by60 = {r["arm"]: n for n, r in enumerate(
        sorted(rows, key=lambda r: r["sensitivity_60"]))}
    moved = 0
    for row in rows:
        leader = row["arm"] == control["arm"]
        colour = G.NOMINAL["rose"] if leader else G.NEUTRAL
        ax_i.plot([0, 1], [by30[row["arm"]], by60[row["arm"]]], color=colour,
                  lw=0.9 if leader else 0.4, zorder=3 if leader else 2,
                  marker="s" if leader else "o", markersize=2.2 if leader else 1.6)
        moved += abs(by30[row["arm"]] - by60[row["arm"]])
    ax_i.set_xlim(-0.2, 1.2)
    ax_i.set_xticks([0, 1])
    ax_i.set_xticklabels(["30 min", "60 min"], fontsize=5.0)
    ax_i.set_yticks([])
    ax_i.set_ylabel("Rank")
    ax_i.set_xlabel(f"Total rank movement {moved}")

    # j: how many fits any of this is built on.
    counts: dict[int, int] = {}
    for row in rows:
        counts[row["seeds_available"]] = counts.get(row["seeds_available"], 0) + 1
    keys = sorted(counts)
    ax_j.bar(range(len(keys)), [counts[k] for k in keys], width=0.62,
             color=G.NOMINAL["amber"], hatch="///", edgecolor=G.INK, linewidth=0.4,
             zorder=3)
    for index, key in enumerate(keys):
        ax_j.text(index, counts[key] + 0.35, str(counts[key]), ha="center",
                  fontsize=5.0, color=G.INK)
    ax_j.set_xticks(range(len(keys)))
    ax_j.set_xticklabels([str(k) for k in keys], fontsize=5.0)
    ax_j.set_xlabel("Fits available")
    ax_j.set_ylabel("Arms")
    ax_j.set_ylim(0.0, max(counts.values()) * 1.22)
    return moved, counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-table", required=True, type=Path)
    parser.add_argument("--oracle-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--height-mm", type=float, default=HEIGHT_MM)
    parser.add_argument("--caption-words", type=int,
                        default=_caption_words.count("fig_landscape"))
    # The caption states these counts in words, so the manuscript's number audit, which
    # scans for numerals, cannot see them. Asserting them here is what stops the caption
    # from quietly disagreeing with panel d. The first draft of that caption said every
    # arm above chance falls; six of them rise.
    parser.add_argument("--expect-rising", type=int, default=6)
    parser.add_argument("--expect-falling", type=int, default=10)
    arguments = parser.parse_args(argv)

    table = json.loads(arguments.arm_table.read_text())
    rows = sorted(table["rows"], key=lambda r: r["primary"])
    if len(rows) != table["arms_total"]:
        raise SystemExit("arm table rows disagree with its own total")
    label = _labels(rows)
    control = max(rows, key=lambda r: r["primary"])
    if control["trained"]:
        raise SystemExit("the leading arm is trained; this figure's premise has moved")

    oracle = {
        name: json.loads((arguments.oracle_dir / f"{name}.json").read_text())
        for name in ("oracle_label", "oracle_future_glucose")
    }
    primary, alarms, sensitivity_30 = _oracle_values(oracle)

    G.set_style()
    figure = plt.figure(figsize=(G.COL2_MM * G.MM, arguments.height_mm * G.MM))
    # One tall column for the ordering, and a 3x3 block for everything that qualifies
    # it. The ordering is the claim; the nine beside it are the ways it can be wrong.
    ax_a = figure.add_axes([0.238, 0.062, 0.196, 0.902])
    # Rows are laid out from one gap constant rather than three hand-set numbers.
    # The gaps here were 0.09 to 0.13 of the canvas, which on a 185 mm figure is a
    # blank band of 17 to 24 mm between rows, and only about 10 mm of it is doing
    # any work: the tick labels and axis label of the row above, plus clearance for
    # the panel letter of the row below. The rest was dead. Setting one gap and
    # solving for the row height gives that space back to the panels.
    cols = (0.545, 0.705, 0.865)
    wide = 0.112
    _bottom, _top, _gap = 0.080, 0.958, 0.070
    tall = (_top - _bottom - 2 * _gap) / 3
    rows_y = (_bottom + 2 * (tall + _gap), _bottom + tall + _gap, _bottom)
    ax_b = figure.add_axes([cols[0], rows_y[0], 0.052, tall])
    ax_c = figure.add_axes([cols[1] - 0.012, rows_y[0], 0.140, tall])
    ax_d = figure.add_axes([cols[2] + 0.010, rows_y[0], 0.104, tall])
    ax_e = figure.add_axes([cols[0], rows_y[1], wide, tall])
    ax_f = figure.add_axes([cols[1], rows_y[1], wide, tall])
    ax_g = figure.add_axes([cols[2], rows_y[1], wide, tall])
    ax_h = figure.add_axes([cols[0], rows_y[2], wide, tall])
    ax_i = figure.add_axes([cols[1], rows_y[2], wide, tall])
    ax_j = figure.add_axes([cols[2], rows_y[2], wide, tall])

    # ---- a: every arm this project has produced ------------------------------------
    for index, row in enumerate(rows):
        leader = row["arm"] == control["arm"]
        current = row["checkpoint_proxy"] == "current"
        colour = (G.NOMINAL["rose"] if leader
                  else G.NOMINAL["blue"] if current else G.NEUTRAL)
        ax_a.plot([0.0, row["primary"]], [index, index], color=colour, lw=0.7,
                  solid_capstyle="butt", zorder=1)
        ax_a.plot([row["primary"]], [index],
                  marker="s" if leader else "o", markersize=3.0 if leader else 2.8,
                  markerfacecolor=colour if current or leader else "none",
                  markeredgecolor=colour, markeredgewidth=0.8, zorder=2)
    ax_a.set_yticks(range(len(rows)))
    ax_a.set_yticklabels([label[row["arm"]] for row in rows], fontsize=5.2)
    ax_a.set_ylim(-0.85, len(rows) - 0.15)
    ax_a.set_xlim(0.0, 1.0)
    ax_a.set_xlabel("Level 2 event sensitivity\n"
                    "Nominal 0.5 false alarms / participant-day", linespacing=1.35)
    for value in primary.values():
        ax_a.axvline(value, color=G.NOMINAL["olive"], lw=0.6, ls=(0, (2.2, 1.6)),
                     zorder=0)
    ax_a.annotate("oracles", xy=(0.985, len(rows) - 1.6),
                  xytext=(0.845, len(rows) - 3.8), fontsize=5.0,
                  color=G.NOMINAL["olive"], ha="center",
                  arrowprops=dict(arrowstyle="-", lw=0.4, color=G.NOMINAL["olive"]))
    ax_a.legend(handles=[
        plt.Line2D([], [], marker="s", ms=3.0, lw=0.7, color=G.NOMINAL["rose"],
                   label="control"),
        plt.Line2D([], [], marker="o", ms=2.8, lw=0.7, color=G.NOMINAL["blue"],
                   label="current proxy or rule"),
        plt.Line2D([], [], marker="o", ms=2.8, lw=0.7, color=G.NEUTRAL,
                   markerfacecolor="none", label="superseded proxy, rescored"),
    ], loc="lower right", fontsize=5.0, handlelength=1.7, borderaxespad=0.4,
        labelspacing=0.35)

    # ---- b: what the interval above the control is made of --------------------------
    lead = control["primary"]
    glucose, perfect = primary["oracle_future_glucose"], primary["oracle_label"]
    gap = perfect - lead
    share = (glucose - lead) / gap
    ax_b.bar([0], [glucose - lead], width=0.62, color=G.NOMINAL["olive"],
             hatch=HATCH[0], edgecolor="white", linewidth=0.0, zorder=2)
    ax_b.bar([0], [perfect - glucose], width=0.62, bottom=[glucose - lead],
             color=G.NOMINAL["amber"], hatch=HATCH[1], edgecolor="white",
             linewidth=0.0, zorder=2)
    ax_b.set_xlim(-0.55, 0.55)
    ax_b.set_xticks([])
    ax_b.set_ylim(0.0, gap * 1.02)
    ax_b.set_yticks([0.0, 0.05, 0.10, 0.15])
    ax_b.set_ylabel("Above the control")
    ax_b.spines["left"].set_bounds(0.0, gap)
    # Both labels stay inside b's own x-range. White-on-hatch was unreadable and the
    # right-hand one reached into the next panel's tick labels.
    # Both labels sit to the RIGHT of the bar at different heights. White-on-hatch was
    # unreadable, the left side collides with the tick labels, and reaching past the
    # axes put one of them among the next panel's labels.
    ax_b.text(0.38, (glucose - lead) * 0.5, f"forecast\n{share:.0%}", fontsize=5.0,
              linespacing=1.4,
              color=G.INK, ha="left", va="center", zorder=4)
    ax_b.text(0.38, gap * 0.945, f"rest\n{1 - share:.0%}", fontsize=5.0,
              linespacing=1.4,
              color=G.INK, ha="left", va="center", zorder=4)
    ax_b.set_xlim(-0.45, 1.45)

    # ---- c: and it buys them on far fewer alarms ------------------------------------
    best = max((r for r in rows if r["trained"]), key=lambda r: r["primary"])
    spend = [
        ("Perfect ranking", alarms["oracle_label"], G.NOMINAL["olive"], HATCH[0]),
        ("Perfect forecast", alarms["oracle_future_glucose"], G.NOMINAL["olive"],
         HATCH[1]),
        ("Control", int(control["alarms_30"]), G.NOMINAL["rose"], HATCH[2]),
        ("Best trained arm", int(best["alarms_30"]), G.NOMINAL["blue"], HATCH[3]),
    ]
    for index, (_name, count, colour, hatch) in enumerate(spend):
        ax_c.barh([index], [count], height=0.60, color=colour, hatch=hatch,
                  edgecolor="white", linewidth=0.0, zorder=2)
        ax_c.text(count + 600, index, f"{count:,}", fontsize=5.0, va="center",
                  color=G.INK)
    ax_c.set_yticks(range(len(spend)))
    ax_c.set_yticklabels([n for n, *_ in spend], fontsize=5.2)
    ax_c.set_ylim(-0.62, len(spend) - 0.38)
    ax_c.set_xlim(0.0, 24000.0)
    ax_c.set_xticks([0, 10000, 20000])
    ax_c.set_xticklabels(["0", "10k", "20k"])
    ax_c.set_xlabel("Alarms spent, 30 minutes")

    # ---- d: the primary averages two horizons that disagree -------------------------
    for row in rows:
        leader = row["arm"] == control["arm"]
        current = row["checkpoint_proxy"] == "current"
        colour = (G.NOMINAL["rose"] if leader
                  else G.NOMINAL["blue"] if current else G.NEUTRAL)
        ax_d.plot([0, 1], [row["sensitivity_30"], row["sensitivity_60"]],
                  color=colour, lw=1.0 if leader else 0.5,
                  alpha=1.0 if leader else 0.8,
                  marker="s" if leader else "o", markersize=2.6 if leader else 1.8,
                  markerfacecolor=colour if current or leader else "none",
                  markeredgewidth=0.6, zorder=3 if leader else 2)
    ax_d.set_xlim(-0.22, 1.22)
    ax_d.set_xticks([0, 1])
    ax_d.set_xticklabels(["30 min", "60 min"])
    ax_d.set_ylim(0.0, 1.0)
    ax_d.set_ylabel("Event sensitivity")

    moved, seed_counts = _extra_panels(
        figure, rows, control, label, (ax_e, ax_f, ax_g, ax_h, ax_i, ax_j),
        sensitivity_30, alarms)
    # The caption spells this one out, so the builder owns it.
    single = seed_counts.get(1, 0)
    if single != EXPECT_SINGLE_FIT:
        raise SystemExit(
            f"the caption says fourteen of the eighteen arms rest on one fit; "
            f"{single} of {len(rows)} do"
        )

    for axis in (ax_a, ax_b, ax_c, ax_d, ax_e, ax_f, ax_g, ax_h, ax_i, ax_j):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    figure.text(0.012, 0.982, "a)", fontsize=7.0, fontweight="bold", color=G.INK,
                va="top")
    figure.text(0.034, 0.982, "Arm ordering", fontsize=6.0, fontweight="bold",
                 color=G.INK, va="top")
    for axis, letter, title, dx in (
        (ax_b, "b", "Oracle gap", 0.062),
        (ax_c, "c", "Alarm counts", 0.048),
        (ax_d, "d", "Two horizons", 0.052),
        (ax_e, "e", "Achieved burden", 0.052),
        (ax_f, "f", "Alarm trade-off", 0.052),
        (ax_g, "g", "Method family", 0.052),
        (ax_h, "h", "Checkpoint proxy", 0.052),
        (ax_i, "i", "Rank movement", 0.052),
        (ax_j, "j", "Fit counts", 0.052),
    ):
        box = axis.get_position()
        header_y = box.y1 + (0.013 if letter in "hij" else 0.021)
        header_x = box.x0 if letter in "hij" else box.x0 - dx
        figure.text(header_x, header_y, f"{letter})", fontsize=7.0,
                    fontweight="bold", color=G.INK, va="top")
        figure.text(header_x + 0.022, header_y, title, fontsize=6.0,
                    fontweight="bold", color=G.INK, va="top")

    rising = sum(1 for r in rows if r["sensitivity_60"] > r["sensitivity_30"])
    falling = sum(1 for r in rows if r["sensitivity_60"] < r["sensitivity_30"])
    if (rising, falling) != (arguments.expect_rising, arguments.expect_falling):
        raise SystemExit(
            f"panel d has {rising} arms rising and {falling} falling against the "
            f"caption's {arguments.expect_rising} and {arguments.expect_falling}; "
            "the caption is now wrong about its own figure"
        )
    top_rising = [
        r["display"] for r in sorted(rows, key=lambda x: -x["primary"])[:5]
        if r["sensitivity_60"] > r["sensitivity_30"]
    ]
    if top_rising != [control["display"]]:
        raise SystemExit(
            f"the caption says the control alone rises among the top five; {top_rising}"
        )
    G.save(figure, str(arguments.output), caption_words=arguments.caption_words)
    print(f"wrote {arguments.output}  {len(rows)} arms, "
          f"{sum(1 for r in rows if r['checkpoint_proxy'] == 'superseded')} superseded, "
          f"gap above the control {gap:+.6f} of which {share:.1%} is glucose, "
          f"{rising} rise and {falling} fall between the horizons")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
