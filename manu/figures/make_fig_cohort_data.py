"""The eleven cohorts as measured data, including as signal rather than as summary.

WHY IT GREW FROM FOUR PANELS TO ELEVEN. The four it had were all summaries: two
distributions, a coverage matrix and a scatter of counts. A paper about warning from a
glucose trace under incompletely recorded actions contained no glucose trace and no
action channel drawn against time, so a reader could see what the cohorts CONTAIN and
never what they LOOK LIKE. The lab figure standard is explicit that a modality is shown
as itself rather than as a schematic, that every modality belongs on one participant
axis, and that the cohort is shown rather than only the aggregate. Panels a to c are
that, and they are real rows from the release rather than illustrations.

THE PARTICIPANTS IN a TO c WERE NOT CHOSEN BY LOOKING. The extractor fixes the rule
before any trace is drawn: within each cohort, the median participant by count of
readings below the level 2 threshold, ties broken on the identifier, windowed on that
participant's FIRST level 2 event rather than their worst. A cohort whose median
participant gives a dull trace is drawn dull, and one of them has no action channel at
all, which is the point of the panel beside it.

COLOUR CARRIES NOTHING IT CANNOT CARRY. Eleven cohorts exceed the nominal palette, so
they are drawn in the neutral with three named, which is the solution this package
already uses. The composition bars are the only categorical fills and they take hatch,
because a bar has neither a marker nor a linestyle.

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
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

WIDTH_MM = {"single": 88.0, "double": 180.0}

CHANNEL_LABEL = {
    "Glucose": "Glucose",
    "Basal insulin": "Basal",
    "Bolus insulin": "Bolus",
    "Insulin, total": "Insulin",
    "Carbohydrate": "Carbohydrate",
    "Meal label": "Meal label",
    "Heart rate": "Heart rate",
    "Steps": "Steps",
    "Energy expenditure": "Energy",
    "Skin temperature": "Skin temp.",
    "Ambient temperature": "Ambient temp.",
    "Electrodermal activity": "Electrodermal",
    "Workout": "Workout",
}
SOURCE_LABEL = {"Shanghai T1DM": "Shanghai"}

# Three named cohorts, chosen to span the axis the panel is about rather than by size:
# the least hypoglycaemic, the most, and the largest contributor of person-days.
NAMED = (
    ("AZT1D", "blue", "-", "o"),
    ("Shanghai T1DM", "rose", "--", "s"),
    ("ReplaceBG", "olive", "-.", "^"),
)
MARKS = (54.0, 61.0, 70.0)
LEVEL2 = 54.0
# The two participant axes in panels b and c. The first is the cohort recording the most
# action channels, the second the only cohort recording wearables, so between them the
# panels show every kind of channel this release carries.
AXIS_ACTIONS = "IOBP2"
AXIS_WEARABLE = "BrisT1D"
# Printed in the caption, so checked against the artifact on every build.
CAPTION_NOCTURNAL = "1.22"
CAPTION_AGE_RANGE = (4, 65)


def _density(counts: np.ndarray, edges: np.ndarray) -> np.ndarray:
    total = counts.sum()
    if total == 0:
        raise SystemExit("a cohort contributed no in-range glucose reading")
    return counts / total / (edges[1] - edges[0])


def _fraction_below(block: dict, edges: np.ndarray, value: float) -> float:
    counts = np.asarray(block["glucose_histogram"], dtype=float)
    return float(counts[edges[1:] <= value].sum() / counts.sum())


def _series(trace: dict) -> tuple[np.ndarray, np.ndarray]:
    """Minutes and glucose with the gaps kept as gaps, never bridged."""
    minutes = np.asarray(trace["minutes"], dtype=float)
    glucose = np.asarray(
        [np.nan if v is None else float(v) for v in trace["glucose"]], dtype=float
    )
    return minutes, glucose


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--traces", required=True, type=Path)
    parser.add_argument("--cohort-table", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--height-mm", type=float, default=185.0)
    parser.add_argument(
        "--caption-words", type=int,
        default=_caption_words.count("fig_cohort_data"),
    )
    arguments = parser.parse_args(argv)

    profile = json.loads(arguments.profile.read_text())
    signal = json.loads(arguments.traces.read_text())
    table = json.loads(arguments.cohort_table.read_text())
    sources = list(profile["sources"])
    channels = list(profile["channels"])
    if len(sources) != 11:
        raise SystemExit(f"expected eleven development sources, found {len(sources)}")
    traces = signal["traces"]
    # A cohort silently missing from the trace panel would read as a cohort with no
    # event. It happened once, from a run-length test applied over raw rows rather than
    # over readings, so the figure refuses rather than drawing ten.
    absent = [s for s in sources if s not in traces]
    if absent:
        raise SystemExit(
            f"no trace for {absent}; panel a would show fewer cohorts "
            "than the study has"
        )
    edges = np.asarray(profile["bin_edges_mg_dl"], dtype=float)
    fold = 3
    usable = (len(edges) - 1) // fold * fold
    coarse_edges = edges[: usable + 1 : fold]
    coarse_centres = 0.5 * (coarse_edges[:-1] + coarse_edges[1:])

    def coarse(counts: np.ndarray) -> np.ndarray:
        return np.asarray(counts[:usable], dtype=float).reshape(-1, fold).sum(axis=1)

    G.set_style()
    figure = plt.figure(
        figsize=(WIDTH_MM["double"] * G.MM, arguments.height_mm * G.MM)
    )
    grid = figure.add_gridspec(
        5, 12, height_ratios=(3.05, 2.0, 2.05, 2.75, 1.95),
        left=0.070, right=0.988, top=0.968, bottom=0.055, wspace=1.55, hspace=0.62,
    )
    ax_a = figure.add_subplot(grid[0, :])
    # The action lanes get their own band under each trace rather than being overlaid on
    # the glucose axis. Drawn on one axis the insulin stems read as glucose values,
    # which is what the first render of this panel did.
    # 1.15 rather than 1.0 for the lane band. Four lane labels at the 5 pt floor need
    # about 10 mm between them, and 1.0 gave the band 8 mm, so Basal, Bolus, Carbs and
    # Meal were drawn touching. The font cannot go lower, so the band goes taller.
    sub_b = grid[1, 0:8].subgridspec(2, 1, height_ratios=(1.85, 1.15), hspace=0.12)
    sub_c = grid[1, 8:12].subgridspec(2, 1, height_ratios=(1.85, 1.15), hspace=0.12)
    ax_b, ax_b_lanes = figure.add_subplot(sub_b[0]), figure.add_subplot(sub_b[1])
    ax_c, ax_c_lanes = figure.add_subplot(sub_c[0]), figure.add_subplot(sub_c[1])
    ax_d = figure.add_subplot(grid[2, 0:4])
    ax_e = figure.add_subplot(grid[2, 4:8])
    ax_f = figure.add_subplot(grid[2, 8:12])
    ax_g = figure.add_subplot(grid[3, :])
    ax_h = figure.add_subplot(grid[4, 0:3])
    ax_i = figure.add_subplot(grid[4, 3:6])
    ax_j = figure.add_subplot(grid[4, 6:9])
    ax_k = figure.add_subplot(grid[4, 9:12])

    named = {name for name, *_ in NAMED}

    # ---- a: eleven real traces, one per cohort, as strips ---------------------------
    # Lines were tried first and failed for a measurable reason: eleven lanes in 44 mm
    # give each trace about 4 mm, so a 200 mg/dL excursion is a wiggle and the thing the
    # panel exists to show, that these traces fall through 54, is invisible. Glucose is
    # an ordered quantity, so it takes the luminance ladder, and a strip spends the
    # whole 4 mm on colour rather than on amplitude. Readings below the level 2
    # threshold carry a second, non-colour channel: a tick under the strip.
    order = sorted(sources)
    # NOT the seven-step ladder. That is for discrete ordered categories. Glucose here
    # is continuous, and interpolating between the ladder's stops wanders in hue, which
    # the first render showed reading as categories. The standing rule sends continuous
    # quantities to a perceptually uniform map, and cividis is the colour-vision-safe
    # one.
    # Direction matters more than the map. LOW glucose is what the panel is about, so
    # low is dark. Reversed, the first render put every hypoglycaemic reading
    # near white.
    ramp = plt.get_cmap("cividis")
    lo_g, hi_g = 40.0, 300.0
    for index, source in enumerate(order):
        minutes, glucose = _series(traces[source])
        hours_t = minutes / 60.0
        finite = np.isfinite(glucose)
        base = float(index)
        ax_a.scatter(hours_t[finite], np.full(finite.sum(), base + 0.55),
                     c=glucose[finite], cmap=ramp, vmin=lo_g, vmax=hi_g,
                     marker="s", s=5.0, linewidths=0.0, zorder=3)
        low = finite & (glucose < LEVEL2)
        if low.any():
            ax_a.vlines(hours_t[low], base + 0.06, base + 0.26, color=G.INK,
                        linewidth=0.5, zorder=4)
        ax_a.text(-6.45, base + 0.5, SOURCE_LABEL.get(source, source), fontsize=5.0,
                  ha="right", va="center", color=G.INK)
    ax_a.axvline(0.0, color=G.INK, linewidth=0.6, linestyle=(0, (2.2, 1.6)), zorder=5)
    ax_a.set_xlim(-6.3, 6.3)
    ax_a.set_ylim(-0.15, float(len(order)))
    ax_a.set_xticks([-6, -4, -2, 0, 2, 4, 6])
    ax_a.set_xlabel("Hours from the first level 2 event of each cohort's median "
                    "participant")
    ax_a.set_yticks([])
    for spine in ("left", "top", "right"):
        ax_a.spines[spine].set_visible(False)
    # Inset rather than appended: a colourbar attached to the axis steals width from a
    # panel that spans the page, and the first render clipped its own tick labels.
    cax = ax_a.inset_axes((0.845, 1.045, 0.150, 0.055))
    figure.colorbar(
        plt.cm.ScalarMappable(norm=mcolors.Normalize(lo_g, hi_g), cmap=ramp),
        cax=cax, orientation="horizontal")
    # 40 and 54 are 5% of the bar apart and collided as "4054" in the render. The bar's
    # left end already says where the scale starts, so only the threshold is labelled.
    cax.set_xticks([LEVEL2, 150.0, hi_g])
    cax.set_xticklabels(["54", "150", "300"], fontsize=5.0)
    cax.tick_params(labelsize=5.0, length=1.0, pad=1.0)
    cax.set_title("Glucose (mg/dL)", fontsize=5.0, pad=1.5)
    for spine in cax.spines.values():
        spine.set_linewidth(0.4)

    # ---- b, c: every modality of one participant on that participant's own axis -----
    def modality_axis(axis, lanes, source: str,
                      rows: tuple[tuple[str, str, str], ...]) -> int:
        if source not in traces:
            raise SystemExit(f"panel needs a trace for {source} and there is none")
        trace = traces[source]
        minutes, glucose = _series(trace)
        hours_t = minutes / 60.0
        axis.plot(hours_t, glucose, color=G.INK, linewidth=0.7, zorder=4)
        axis.axhline(LEVEL2, color=G.NOMINAL["rose"], linewidth=0.5,
                     linestyle=(0, (2.2, 1.6)), zorder=3)
        for target in (axis, lanes):
            target.axvline(0.0, color=G.INK, linewidth=0.5,
                           linestyle=(0, (2.2, 1.6)), zorder=3)
            target.set_xlim(-6.3, 6.3)
        axis.set_ylim(30.0, 330.0)
        axis.set_xticklabels([])
        drawn = 0
        lanes.set_ylim(0.0, float(len(rows)))
        lanes.set_yticks([len(rows) - i - 0.5 for i in range(len(rows))])
        lanes.set_yticklabels([label for _k, label, _h in rows], fontsize=5.0)
        lanes.tick_params(length=0.0)
        for spine in ("left", "top", "right"):
            lanes.spines[spine].set_visible(False)
        for i, (key, _label, hue) in enumerate(rows):
            block = trace.get(key, {})
            points = block.get("observed") or [(m, 1.0) for m in block.get("marks", [])]
            if not points:
                continue
            drawn += 1
            xs = np.asarray([m for m, _v in points], dtype=float) / 60.0
            vs = np.asarray([v for _m, v in points], dtype=float)
            top = float(np.nanmax(np.abs(vs))) or 1.0
            base = len(rows) - i - 1.0
            lanes.vlines(xs, base + 0.10, base + 0.10 + 0.78 * np.abs(vs) / top,
                         color=G.NOMINAL[hue], linewidth=0.55, zorder=2)
        return drawn

    drawn_b = modality_axis(ax_b, ax_b_lanes, AXIS_ACTIONS, (
        ("basal", "Basal", "blue"), ("bolus", "Bolus", "amber"),
        ("carbs", "Carbs", "olive"), ("meal_label", "Meal", "teal")))
    drawn_c = modality_axis(ax_c, ax_c_lanes, AXIS_WEARABLE, (
        ("steps", "Steps", "olive"), ("heartrate", "Heart rate", "rose")))
    ax_b.set_ylabel("Glucose (mg/dL)")
    ax_b_lanes.set_xlabel(f"Hours from onset, {AXIS_ACTIONS} participant "
                          f"{traces[AXIS_ACTIONS]['participant']}")
    ax_c_lanes.set_xlabel(
        f"Hours, {AXIS_WEARABLE} {traces[AXIS_WEARABLE]['participant']}")
    if drawn_b < 3 or drawn_c < 2:
        raise SystemExit(
            f"the modality axes drew {drawn_b} and {drawn_c} action lanes; the caption "
            "says these two participants carry between them every kind of channel the "
            "release records"
        )

    # ---- d: the glucose distribution of every cohort --------------------------------
    for source in sources:
        if source in named:
            continue
        ax_d.plot(coarse_centres, _density(
            coarse(profile["sources"][source]["glucose_histogram"]), coarse_edges),
            color=G.NEUTRAL, linewidth=0.4, alpha=0.85, zorder=2)
    for source, hue, dash, mark in NAMED:
        block = profile["sources"][source]
        share = _fraction_below(block, edges, 70.0)
        ax_d.plot(
            coarse_centres, _density(coarse(block["glucose_histogram"]), coarse_edges),
            color=G.NOMINAL[hue], linewidth=0.9, linestyle=dash, marker=mark,
            markevery=7, markersize=2.0, zorder=4,
            label=f"{SOURCE_LABEL.get(source, source)} ({share:.1%})",
        )
    for value in MARKS:
        ax_d.axvline(value, color=G.INK, linewidth=0.5,
                     linestyle=(0, (2.2, 1.6)), zorder=3)
    ax_d.set_xlim(40.0, 340.0)
    ax_d.set_xlabel("Sensor glucose (mg/dL)")
    ax_d.set_ylabel("Density")
    ax_d.legend(loc="upper right", handlelength=1.8, borderpad=0.15,
                labelspacing=0.2, fontsize=5.0)

    # ---- e: the same distributions where alarms are actually decided ----------------
    ax_e.axvspan(54.0, 61.0, color=G.NOMINAL["rose"], alpha=0.13, lw=0, zorder=1)
    for source in sources:
        block = profile["sources"][source]
        cumulative = np.cumsum(np.asarray(block["glucose_histogram"], dtype=float))
        cumulative /= cumulative[-1]
        style = next(((h, d, m) for s, h, d, m in NAMED if s == source), None)
        if style is None:
            ax_e.plot(edges[1:], 100.0 * cumulative, color=G.NEUTRAL,
                      linewidth=0.4, alpha=0.85, zorder=2)
        else:
            hue, dash, mark = style
            ax_e.plot(edges[1:], 100.0 * cumulative, color=G.NOMINAL[hue],
                      linewidth=0.9, linestyle=dash, marker=mark, markevery=6,
                      markersize=2.0, zorder=4)
    for value in MARKS:
        ax_e.axvline(value, color=G.INK, linewidth=0.5,
                     linestyle=(0, (2.2, 1.6)), zorder=3)
    ax_e.set_xlim(40.0, 100.0)
    ax_e.set_ylim(0.0, 12.0)
    ax_e.set_xlabel("Sensor glucose (mg/dL)")
    ax_e.set_ylabel("Readings below (%)")

    # ---- f: when the level 2 readings happen ----------------------------------------
    bins = int(signal["time_of_day_bins"])
    hours = (np.arange(bins) + 0.5) * (24.0 / bins)
    pooled_all = np.zeros(bins)
    pooled_low = np.zeros(bins)
    for source in sources:
        block = signal["time_of_day"][source]
        pooled_all += np.asarray(block["all"], dtype=float)
        pooled_low += np.asarray(block["below_54"], dtype=float)
        style = next(((h, d) for s, h, d, _m in NAMED if s == source), None)
        rate = 100.0 * np.asarray(block["below_54"], dtype=float) / np.maximum(
            np.asarray(block["all"], dtype=float), 1.0)
        if style is None:
            ax_f.plot(hours, rate, color=G.NEUTRAL, linewidth=0.35, alpha=0.8, zorder=2)
        else:
            ax_f.plot(hours, rate, color=G.NOMINAL[style[0]], linewidth=0.75,
                      linestyle=style[1], zorder=4)
    pooled = 100.0 * pooled_low / np.maximum(pooled_all, 1.0)
    ax_f.plot(hours, pooled, color=G.INK, linewidth=1.1, zorder=5)
    ax_f.axvspan(0.0, 6.0, color=G.INK, alpha=0.07, lw=0, zorder=1)
    ax_f.set_xlim(0.0, 24.0)
    ax_f.set_xticks([0, 6, 12, 18, 24])
    ax_f.set_ylim(bottom=0.0)
    ax_f.set_xlabel("Hour of day")
    ax_f.set_ylabel("Readings below 54 (%)")

    # ---- g: which cohort records which channel --------------------------------------
    coverage = np.array([
        [100.0 * profile["sources"][s]["channel_participants"][c]
         / profile["sources"][s]["participants"] for s in sources]
        for c in channels
    ])
    present = G.NOMINAL["blue"]
    for i in range(len(channels)):
        for j in range(len(sources)):
            value = coverage[i, j]
            if value == 0.0:
                face, hatch, ink, text = "white", None, G.NEUTRAL, "·"
            elif value == 100.0:
                face, hatch, ink, text = present, None, "white", "100"
            else:
                face, hatch, ink, text = "white", "////", G.INK, f"{value:.1f}"
            ax_g.add_patch(plt.Rectangle(
                (j - 0.5, i - 0.5), 1.0, 1.0, facecolor=face, hatch=hatch,
                edgecolor=present if hatch else "#DCE1E8", linewidth=0.5, zorder=1))
            ax_g.text(
                j, i, text, ha="center", va="center", fontsize=5.0, zorder=3,
                color=ink,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.5)
                if hatch else None,
            )
    ax_g.set_xlim(-0.5, len(sources) - 0.5)
    ax_g.set_ylim(len(channels) - 0.5, -0.5)
    ax_g.set_xticks(range(len(sources)))
    ax_g.set_xticklabels([SOURCE_LABEL.get(s, s) for s in sources],
                         rotation=38, ha="right", fontsize=5.0)
    ax_g.set_yticks(range(len(channels)))
    ax_g.set_yticklabels([CHANNEL_LABEL[c] for c in channels], fontsize=5.0)
    for spine in ax_g.spines.values():
        spine.set_visible(False)
    ax_g.tick_params(length=0.0)

    # ---- h, i: composition, the only categorical fills and so the only hatching ------
    def stacked(axis, field: str, order: tuple[str, ...], missing_label: str) -> None:
        hatches = ("", "///", "...", "xxx")
        hues = ("blue", "amber", "olive", None)
        for row, source in enumerate(sources):
            counts = profile["sources"][source][field]
            total = float(sum(counts.values()))
            left = 0.0
            for index, key in enumerate(order):
                value = float(counts.get(key, 0.0))
                if total == 0.0 or value == 0.0:
                    continue
                share = 100.0 * value / total
                hue = hues[index]
                axis.barh(row, share, left=left, height=0.72,
                          color=G.NOMINAL[hue] if hue else "white",
                          hatch=hatches[index] or None, edgecolor=G.INK,
                          linewidth=0.35, zorder=3)
                left += share
            if total == 0.0:
                axis.barh(row, 100.0, height=0.72, color="white", hatch="xxx",
                          edgecolor=G.NEUTRAL, linewidth=0.35, zorder=3)
        axis.set_xlim(0.0, 100.0)
        axis.set_ylim(len(sources) - 0.5, -0.5)
        axis.set_yticks(range(len(sources)))
        axis.set_yticklabels([SOURCE_LABEL.get(s, s) for s in sources], fontsize=5.0)
        axis.set_xticks([0, 50, 100])
        axis.tick_params(length=1.4)
        handles = [
            plt.Rectangle((0, 0), 1, 1,
                          facecolor=G.NOMINAL[h] if h else "white",
                          hatch=hatches[i] or None, edgecolor=G.INK, linewidth=0.35)
            for i, h in enumerate(hues[: len(order)])
        ]
        labels = list(order)
        if any(sum(profile["sources"][s][field].values()) == 0.0 for s in sources):
            handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="white", hatch="xxx",
                                         edgecolor=G.NEUTRAL, linewidth=0.35))
            labels.append(missing_label)
        axis.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0),
                    ncol=2, fontsize=5.0, handlelength=1.0, handleheight=0.9,
                    columnspacing=0.7, borderaxespad=0.1, labelspacing=0.18)

    def sex_counts(source: str) -> dict[str, float]:
        raw = profile["sources"][source]["demographic_categorical"]["gender"]
        out: dict[str, float] = {}
        for key, value in raw.items():
            label = "Female" if key.lower().startswith("f") else (
                "Male" if key.lower().startswith("m") else "Other")
            out[label] = out.get(label, 0.0) + float(value)
        return out

    for source in sources:
        profile["sources"][source]["_sex"] = sex_counts(source)
    # The modality block lives under `categorical`. Flatten it once so `stacked` reads
    # one shape for both panels rather than carrying a special case.
    for source in sources:
        profile["sources"][source]["_modality"] = {
            k: float(v) for k, v in
            profile["sources"][source]["categorical"].get(
                "insulin_delivery_modality", {}).items()
        }
    stacked(ax_h, "_modality", ("AID", "SAP", "MDI"), "not recorded")
    stacked(ax_i, "_sex", ("Female", "Male", "Other"), "not recorded")
    ax_h.set_xlabel("Insulin delivery (% of rows)")
    ax_i.set_xlabel("Sex (% of participants)")
    ax_i.set_yticklabels([])

    # ---- j: the cohorts are not the same people --------------------------------------
    ages = []
    for row, source in enumerate(sources):
        block = profile["sources"][source]["demographic_numeric"]["age"]
        if block["median"] is None:
            continue
        ages.append(float(block["median"]))
        hue = next((G.NOMINAL[h] for s, h, _d, _m in NAMED if s == source), G.NEUTRAL)
        ax_j.plot([block["p25"], block["p75"]], [row, row], color=hue, linewidth=0.7,
                  solid_capstyle="butt", zorder=3)
        ax_j.plot([block["median"]], [row], marker="o", markersize=2.6, color=hue,
                  zorder=4)
    ax_j.set_ylim(len(sources) - 0.5, -0.5)
    ax_j.set_yticks(range(len(sources)))
    ax_j.set_yticklabels([])
    ax_j.set_xlim(0.0, 75.0)
    ax_j.set_xlabel("Age (years)")

    # ---- k: how unequal the cohorts are ----------------------------------------------
    missing = [s for s in sources if s not in table]
    if missing:
        raise SystemExit(f"cohort table is missing {missing}")
    for source in sources:
        row = table[source]
        style = next(((h, m) for s, h, _d, m in NAMED if s == source), None)
        hue, mark = style if style else (None, "o")
        ax_k.plot(
            [row["person_days"]], [row["severe_hypo_events"]],
            marker=mark, markersize=3.0,
            markerfacecolor=G.NOMINAL[hue] if hue else "none",
            markeredgecolor=G.NOMINAL[hue] if hue else G.NEUTRAL,
            markeredgewidth=0.7, linestyle="none", zorder=4,
        )
        if style:
            ax_k.annotate(
                SOURCE_LABEL.get(source, source),
                (row["person_days"], row["severe_hypo_events"]),
                textcoords="offset points",
                xytext=(-3.4, 4.0) if source == "ReplaceBG" else (3.6, -0.8),
                ha="right" if source == "ReplaceBG" else "left",
                fontsize=5.0, color=G.INK, va="center", zorder=5,
            )
    ax_k.set_xscale("log")
    ax_k.set_yscale("log")
    ax_k.set_xlim(80.0, 1.3e5)
    ax_k.set_xlabel("Person-days")
    ax_k.set_ylabel("Level 2 events")

    for axis in (ax_a, ax_b, ax_c, ax_d, ax_e, ax_f, ax_h, ax_i, ax_j, ax_k):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.set_axisbelow(True)
    for axis, letter, dx, dy in (
        # panel_label hangs the glyph DOWNWARD from dy, so 1.055 on a 34 mm axis
        # put the letter's baseline below the axis top and onto the first tick
        # label. It has to clear its own height, not merely be positive.
        (ax_a, "a", -0.038, 1.095), (ax_b, "b", -0.062, 1.145),
        (ax_c, "c", -0.150, 1.145), (ax_d, "d", -0.235, 1.150),
        (ax_e, "e", -0.235, 1.150), (ax_f, "f", -0.235, 1.150),
        (ax_g, "g", -0.038, 1.070), (ax_h, "h", -0.330, 1.470),
        (ax_i, "i", -0.135, 1.470), (ax_j, "j", -0.135, 1.140),
        (ax_k, "k", -0.330, 1.140),
    ):
        G.panel_label(axis, letter, dx=dx, dy=dy)

    G.save(figure, str(arguments.output), caption_words=arguments.caption_words)

    # The caption states three counts that a future release could move. Each is checked.
    absent_cells = int((coverage == 0.0).sum())
    partial = int(((coverage > 0.0) & (coverage < 100.0)).sum())
    nocturnal = float(pooled[hours < 6.0].mean() / pooled[hours >= 6.0].mean())
    # The caption prints these three, so the builder owns them. A caption number with no
    # code behind it is one that drifts the first time the artifact moves.
    if f"{nocturnal:.2f}" != CAPTION_NOCTURNAL:
        raise SystemExit(
            f"the caption says the overnight rate is {CAPTION_NOCTURNAL} times the "
            f"daytime one; it is {nocturnal:.2f}"
        )
    if (round(min(ages)), round(max(ages))) != CAPTION_AGE_RANGE:
        raise SystemExit(
            f"the caption says median age runs {CAPTION_AGE_RANGE[0]} to "
            f"{CAPTION_AGE_RANGE[1]} years; it runs {min(ages):.0f} to {max(ages):.0f}"
        )
    span = max(ages) / min(ages)
    print(f"wrote {arguments.output}  {len(sources)} cohorts, {len(traces)} traces, "
          f"{absent_cells} channel cells absent and {partial} partial, "
          f"overnight level 2 rate {nocturnal:.2f}x the daytime one, "
          f"median age {min(ages):.0f} to {max(ages):.0f} years ({span:.0f}x)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
