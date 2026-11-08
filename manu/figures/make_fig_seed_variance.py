"""Run-to-run variability of the balanced LSTM, drawn from the five fits' own artifacts.

This builder did not exist until 2026-09-04. Commit 8f80f8e added the assembler and the
five per-seed JSON files but no plotting script, so the figure in the manuscript could
not be regenerated and nothing could check it against its data. That is the defect this
file closes; the panels below are drawn only from
``experiments/20260903-erm-lstm-seeds-v1/figure-input/``.

One thing changed while rebuilding it. The previous version showed the 30-minute
frontier alone, while the quantity the paper makes claims about is the equally weighted
30/60-minute average. Showing one horizon and reasoning about the average invites the
obvious objection, so both horizons are drawn and the reader can see that the 60-minute
frontiers are the tighter of the two.

Geometry, fonts, palette and the zero-baseline rule are not enforced here. They are
enforced by ``glycofig.save``, which is the shared module the code/scripts/figures set
already used and which these manuscript builders had bypassed.
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

WIDTH_MM = {"single": 88.0, "double": 180.0}

# Seed order is fixed here rather than read from the ranking, so the colour a seed gets
# does not move when the ranking does.
SEEDS = ["017", "023", "042", "071", "101"]
HUES = ["blue", "rose", "amber", "olive", "teal"]
MARKERS = ["o", "s", "^", "D", "v"]
STYLES = ["solid", (0, (3.2, 1.4)), (0, (1.2, 1.0)),
          (0, (5.0, 1.6, 1.2, 1.6)), (0, (2.0, 1.2, 0.8, 1.2))]

FRONTIER_MAX_FAPD = 2.0  # the range the protocol defines the frontier area over
NOMINAL_FAPD = 0.5


def _load(directory: Path) -> dict[str, dict]:
    runs = {}
    for seed in SEEDS:
        path = directory / f"erm_lstm_seed{seed}.json"
        if not path.exists():
            raise SystemExit(f"missing seed artifact: {path}")
        runs[seed] = json.loads(path.read_text())
    return runs


def _frontier(payload: dict, horizon: str) -> tuple[list[float], list[float]]:
    points = payload["horizons"][horizon]["selection_budget_frontier"]["frontier"]
    x = [p["achieved_fapd"] for p in points]
    y = [p["best_feasible_event_sensitivity"] for p in points]
    keep = [i for i, v in enumerate(x) if v <= FRONTIER_MAX_FAPD]
    if len(keep) < 50:
        # A loader that silently plots what it found draws a "frontier" through three
        # points and looks entirely normal.
        raise SystemExit(
            f"{payload['arm_id']} horizon {horizon}: only {len(keep)} frontier points "
            f"at or below {FRONTIER_MAX_FAPD} FA/day"
        )
    return [x[i] for i in keep], [y[i] for i in keep]


def _panel_seeds(axis, runs: dict[str, dict]) -> None:
    values = [runs[s]["mean_severe_event_sensitivity"] for s in SEEDS]
    burdens = [runs[s]["mean_achieved_fapd"] for s in SEEDS]
    positions = list(range(len(SEEDS)))[::-1]

    # strict, because an unequal zip here silently DROPS a seed and the panel still
    # renders: five fits drawn as four, with nothing saying so.
    for pos, _seed, value, burden, hue, marker in zip(
        positions, SEEDS, values, burdens, HUES, MARKERS, strict=True
    ):
        axis.plot([value], [pos], marker=marker, markersize=4.0,
                  color=G.NOMINAL[hue], linestyle="none", zorder=3)
        axis.annotate(f"{value:.3f}   {burden:.2f} FA/day",
                      xy=(value, pos), xytext=(6, 0), textcoords="offset points",
                      va="center", ha="left", fontsize=6.0, color=G.INK)

    # The figure is about the spread, so draw it. A bare dashed line would also collide
    # in meaning with the nominal-budget line in the other two panels, where the same
    # visual carries something else entirely.
    mean = sum(values) / len(values)
    sd = (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5
    axis.axvspan(mean - sd, mean + sd, color=G.MUTED, alpha=0.13, lw=0, zorder=0)
    axis.axvline(mean, color=G.MUTED, linewidth=0.7, zorder=1)
    # va="bottom" put this string ABOVE its anchor, and the anchor was 0.15 of a row
    # below the axis top, so it climbed out of the axes and landed on the title. It
    # hangs below the anchor instead. The collision gate reported it at 41% of the
    # smaller box, and the panel had been read on screen twice without anyone seeing it.
    axis.annotate(f"mean $\\pm$ s.d.  ({sd:.4f})", xy=(mean, len(SEEDS) - 0.45),
                  xytext=(0, 0), textcoords="offset points",
                  ha="center", va="top", fontsize=6.0, color=G.MUTED)

    axis.set_yticks(positions)
    axis.set_yticklabels([f"seed {s}" for s in SEEDS])
    # A dot plot, so the axis is free not to start at zero. A bar here would overstate
    # every difference on it, and glycofig would refuse it.
    span = max(values) - min(values)
    axis.set_xlim(min(values) - 0.35 * span, max(values) + 1.15 * span)
    axis.set_ylim(-0.6, len(SEEDS) - 0.4)
    axis.set_xlabel("Primary sensitivity\nNominal 0.5 FA/day", linespacing=1.35)


def _panel_frontier(axis, runs: dict[str, dict], horizon: str, show_legend: bool) -> None:
    for seed, hue, style, marker in zip(SEEDS, HUES, STYLES, MARKERS, strict=True):
        x, y = _frontier(runs[seed], horizon)
        axis.plot(x, y, color=G.NOMINAL[hue], linewidth=0.9, linestyle=style,
                  marker=marker, markersize=1.8, markevery=max(len(x) // 10, 1),
                  label=f"seed {seed}", zorder=3)
    axis.axvline(NOMINAL_FAPD, color=G.MUTED, linewidth=0.7,
                 linestyle=(0, (2.5, 1.5)), zorder=1)
    axis.set_xlim(0.0, FRONTIER_MAX_FAPD)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("False alarms per\nparticipant-day", linespacing=1.35)
    if show_legend:
        axis.legend(loc="lower right", handlelength=2.4, borderaxespad=0.3)


def _summary(directory):
    """The run summary the number ledger reads, so the panel cannot drift from it."""
    path = Path(directory) / "summary.json"
    if not path.is_file():
        raise SystemExit(f"{path} is missing; panel f would have to re-derive its own "
                         "frontier areas and would then disagree with the manuscript")
    return json.loads(path.read_text())


def _seed_panels(axes, runs, summary):
    """Seven panels the five seed artifacts already held.

    Where the run-to-run variance actually lives, cohort by cohort and against cohort
    size; whether the frontier is quieter than the point estimate on it; and what moves
    across seeds besides the headline number, in alarms, in detections, in achieved rate
    and between the two horizons.
    """
    ax_d, ax_e, ax_f, ax_g, ax_h, ax_i, ax_j = axes
    metrics = {
        seed: {h: runs[seed]["horizons"][h]["selection_metrics"] for h in ("30", "60")}
        for seed in SEEDS
    }
    sources = sorted(metrics[SEEDS[0]]["30"]["per_source"])

    def per(seed, horizon, source, field):
        return float(metrics[seed][horizon]["per_source"][source][field])

    # d: which cohorts carry the variance.
    spreads = {}
    order = sorted(
        sources,
        key=lambda s: statistics.mean(
            per(x, "30", s, "participant_macro_event_sensitivity") for x in SEEDS))
    for index, source in enumerate(order):
        values = [per(seed, "30", source, "participant_macro_event_sensitivity")
                  for seed in SEEDS]
        spreads[source] = statistics.pstdev(values)
        ax_d.plot([min(values), max(values)], [index, index], color=G.NEUTRAL, lw=0.6,
                  solid_capstyle="butt", zorder=2)
        for value, hue, marker in zip(values, HUES, MARKERS, strict=True):
            ax_d.plot([value], [index], marker=marker, markersize=1.9,
                      color=G.NOMINAL[hue], zorder=3)
    ax_d.set_yticks(range(len(order)))
    ax_d.set_yticklabels([s.replace("Shanghai T1DM", "Shanghai") for s in order],
                         fontsize=5.0)
    ax_d.set_ylim(-0.6, len(order) - 0.4)
    ax_d.set_xlim(0.0, 1.05)
    ax_d.set_xlabel("Event sensitivity across seeds,\n30 min")

    # e: and whether that tracks how many people a cohort holds.
    for source in sources:
        size = float(metrics[SEEDS[0]]["30"]["per_source"][source]["participant_count"])
        ax_e.plot([size], [spreads[source]], marker="o", markersize=2.4,
                  markerfacecolor="none", markeredgecolor=G.NOMINAL["rose"],
                  markeredgewidth=0.8, zorder=3)
    ax_e.set_xscale("log")
    ax_e.set_xlabel("Scored participants")
    ax_e.set_ylabel("Seed sd within cohort")

    # f: the curve is quieter than the point sitting on it.
    #
    # TWO THINGS HERE WERE WRONG IN THE FIRST BUILD AND BOTH CONTRADICTED THE TEXT. It
    # used the POPULATION standard deviation, which drew 0.0452 where the manuscript
    # says 0.0505 everywhere. And it re-derived the frontier area as a mean of the
    # retained points, which is a different quantity from the area the manuscript
    # reports and came out at 0.0149 rather than 0.0315. The areas are precomputed per
    # seed in the run summary and are read from there, so the panel and the ledger
    # cannot disagree.
    point = statistics.stdev(
        [float(runs[s]["mean_severe_event_sensitivity"]) for s in SEEDS])
    ranking = {r["arm_id"]: r for r in summary["ranking"]}
    areas = {
        horizon: statistics.stdev(
            float(ranking[f"erm_lstm_seed{seed}"]["budget_frontier_auc"][horizon])
            for seed in SEEDS)
        for horizon in ("30", "60")
    }
    labels = ["Primary", "Frontier\n30 min", "Frontier\n60 min"]
    heights = [point, areas["30"], areas["60"]]
    ax_f.bar(range(len(labels)), heights, width=0.62, color=G.NOMINAL["amber"],
             hatch="///", edgecolor=G.INK, linewidth=0.4, zorder=3)
    ax_f.set_xticks(range(len(labels)))
    ax_f.set_xticklabels(labels, fontsize=5.0)
    ax_f.set_ylabel("SD across seeds")

    # g, h, i: what else moves across seeds.
    for axis, field, ylabel, scale in (
        (ax_g, "equal_source_participant_macro_false_alarms_per_patient_day",
         "Achieved false alarms per\nparticipant-day", 1.0),
        (ax_h, "alarm_count", "Alarms charged (thousands)", 1e-3),
        (ax_i, "detected_event_count", "Events detected", 1.0),
    ):
        for index, horizon in enumerate(("30", "60")):
            for offset, (seed, hue, marker) in enumerate(zip(SEEDS, HUES, MARKERS,
                                                          strict=True)):
                x = index + (offset - 2) * 0.14
                axis.plot([x], [float(metrics[seed][horizon][field]) * scale],
                          marker=marker, markersize=2.2, color=G.NOMINAL[hue], zorder=3)
        axis.set_xticks([0, 1])
        axis.set_xticklabels(["30 min", "60 min"], fontsize=5.0)
        axis.set_xlim(-0.5, 1.5)
        axis.set_ylabel(ylabel)
    ax_g.axhline(0.5, color=G.INK, lw=0.6, ls=(0, (2.2, 1.6)), zorder=2)

    # j: the two horizons move together, which is why averaging them buys little.
    for seed, hue, marker in zip(SEEDS, HUES, MARKERS, strict=True):
        a = float(metrics[seed]["30"]["equal_source_participant_macro_event_sensitivity"])
        b = float(metrics[seed]["60"]["equal_source_participant_macro_event_sensitivity"])
        ax_j.plot([0, 1], [a, b], color=G.NOMINAL[hue], lw=0.8, marker=marker,
                  markersize=2.4, zorder=3)
    ax_j.set_xticks([0, 1])
    ax_j.set_xticklabels(["30 min", "60 min"], fontsize=5.0)
    ax_j.set_xlim(-0.2, 1.2)
    ax_j.set_ylabel("Event sensitivity")
    # The caption says the noisiest cohort is the smallest and names the two figures at
    # the ends of panel e. Both are checked here rather than transcribed.
    worst = max(spreads, key=lambda s: spreads[s])
    sizes = {s: float(metrics[SEEDS[0]]["30"]["per_source"][s]["participant_count"])
             for s in sources}
    smallest = min(sizes, key=lambda s: (sizes[s], s))
    if worst != smallest:
        raise SystemExit(
            f"the caption says the noisiest cohort is the smallest; the noisiest is "
            f"{worst} at {sizes[worst]:.0f} participants and the smallest is "
            f"{smallest} at {sizes[smallest]:.0f}"
        )
    biggest = max(sizes, key=lambda s: (sizes[s], s))
    return worst, spreads[worst], spreads[biggest], point, areas


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path,
                        help="directory holding erm_lstm_seed<NNN>.json")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--width", choices=sorted(WIDTH_MM), default="double")
    parser.add_argument("--height-mm", type=float, default=185.0)
    parser.add_argument("--caption-words", type=int,
                        default=_caption_words.count("fig_seed_variance"))
    arguments = parser.parse_args(argv)

    runs = _load(arguments.input)
    G.set_style()

    figure = plt.figure(
        figsize=(WIDTH_MM[arguments.width] * G.MM, arguments.height_mm * G.MM))
    # One gap constant, three solved rows. The gaps here were 0.098 and 0.134 of the
    # canvas, an unequal pair that on a 185 mm figure left an 18 and a 25 mm band
    # between rows where about 10 mm is doing work. This figure carried the widest
    # blank band of the whole set after the others were tightened.
    # 0.082 rather than the 0.070 the other figures use. Panel d carries a two-line
    # axis label, and at 0.070 its second line sat on the panel letter of the row
    # below. The gap a row needs is a property of the tallest label under it.
    _bottom, _top, _gap = 0.080, 0.958, 0.082
    _tall = (_top - _bottom - 2 * _gap) / 3
    _r1, _r2, _r3 = _bottom + 2 * (_tall + _gap), _bottom + _tall + _gap, _bottom
    axes = [
        figure.add_axes([0.185, _r1, 0.225, _tall]),
        figure.add_axes([0.560, _r1, 0.175, _tall]),
        figure.add_axes([0.805, _r1, 0.170, _tall]),
    ]
    rest = [
        figure.add_axes([0.170, _r2, 0.215, _tall]),
        figure.add_axes([0.545, _r2, 0.180, _tall]),
        figure.add_axes([0.805, _r2, 0.170, _tall]),
        figure.add_axes([0.085, _r3, 0.160, _tall]),
        figure.add_axes([0.330, _r3, 0.160, _tall]),
        figure.add_axes([0.575, _r3, 0.160, _tall]),
        figure.add_axes([0.820, _r3, 0.155, _tall]),
    ]

    _panel_seeds(axes[0], runs)
    _panel_frontier(axes[1], runs, "30", show_legend=False)
    _panel_frontier(axes[2], runs, "60", show_legend=True)
    axes[1].set_ylabel("Event sensitivity")
    noisiest = _seed_panels(rest, runs, _summary(arguments.input))  # (cohort, its sd, the largest cohort's sd)

    titles = ("Primary sensitivity", "30-minute frontiers", "60-minute frontiers",
              "Cohort variability", "Cohort size", "Endpoint variability",
              "False-alarm rate", "Alarm burden", "Matched events", "Horizon coupling")
    for axis, letter, title in zip(axes + rest, "abcdefghij", titles, strict=True):
        axis.set_title(title, loc="left", fontsize=6.0, fontweight="bold", pad=4)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.set_axisbelow(True)
        box = axis.get_position()
        figure.text(box.x0 - 0.058, box.y1 + 0.024, f"{letter})", fontsize=7.0,
                    fontweight="bold", color=G.INK, va="top")

    G.save(figure, str(arguments.output), caption_words=arguments.caption_words)

    values = [runs[s]["mean_severe_event_sensitivity"] for s in SEEDS]
    spread = max(values) - min(values)
    print(f"wrote {arguments.output}  {len(SEEDS)} seeds, range {spread:.4f}, "
          f"min {min(values):.6f} max {max(values):.6f}; seed sd within a cohort runs "
          f"{noisiest[1]:.2f} in {noisiest[0]} to {noisiest[2]:.2f} in the largest; "
          f"point sd {noisiest[3]:.4f} against frontier areas "
          f"{noisiest[4]['30']:.4f} and {noisiest[4]['60']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
