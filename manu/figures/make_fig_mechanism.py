#!/usr/bin/env python3
"""The endpoint scores whether an event was matched, not how early.

Panel a is the distribution of warning-time difference over every level 2 event
BOTH arms match under the frozen policy. Panels b to d are the three cases the
pre-registration selects by rank, at the 10th percentile, the median and the
90th, and the 10th is a tie that is kept because the rule admits it.

Journal figure contract, matching the other builders in this directory: width
exactly 88 or 180 mm at 0.2 mm tolerance, Arial everywhere including
mathtext which has its own font stack, pdf.fonttype 42, authored at final size
with no tight bbox and no downstream resizing.

It renders what it is given and invents nothing. Every value comes from the
frozen extraction, which itself refuses to emit unless both arms reproduce their
frozen 60-minute sensitivity.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.transforms
import numpy as np

matplotlib.use("Agg")
# The manuscript figures are gated by the same module the code/scripts/figures set
# uses. Wiring it in retro-fits eight guards these builders never had, and it is what
# would have refused the 134 mm width and the oversize panel labels.
import sys as _sys  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

_sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "code" / "scripts" / "figures")
)
import caption_words as _caption_words  # noqa: E402
import glycofig as _G  # noqa: E402

MM = 1.0 / 25.4
WIDTH_MM = {"single": 88.0, "double": 180.0}

# Nominal palette: chroma fixed, hues across the whole wheel, one pure neutral
# for the control. Separation never rests on colour alone; each arm also carries
# its own marker, the same pairing the eligibility figure uses.
PALETTE = ["#9D5061", "#8A6635", "#69AF87", "#3CA6CF", "#80679F"]
NEUTRAL = "#888888"
INK = "#28303F"
NETWORK = _G.NOMINAL["blue"]
SEVERE_MG_DL = 54.0
HYPO_MG_DL = 70.0
# The extraction window, fixed. Every trace panel is drawn on THIS range, not on
# its own data extent: three panels with different x ranges and tick labels on
# only the last are read against the last one, which is a lie about the other two.
TRACE_XLIM = (-240.0, 60.0)


def _style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial", "mathtext.it": "Arial",
        "mathtext.bf": "Arial", "mathtext.sf": "Arial",
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.size": 7.0, "axes.labelsize": 7.0, "axes.titlesize": 7.0,
        "xtick.labelsize": 7.0, "ytick.labelsize": 7.0, "legend.fontsize": 6.5,
        "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK, "ytick.color": INK,
        "axes.linewidth": 0.6, "lines.linewidth": 1.0,
        "legend.frameon": False, "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _minutes_from(onset: str, stamps: list[str]) -> np.ndarray:
    base = datetime.fromisoformat(onset)
    return np.array([(datetime.fromisoformat(s) - base).total_seconds() / 60.0
                     for s in stamps], dtype=float)


def _panel_distribution(axis, data: dict) -> None:
    diff = np.asarray(data["diff_minutes"], dtype=float)
    if diff.size != data["n_events"]:
        raise ValueError("the difference vector does not match the event count")
    lo, hi = diff.min(), diff.max()
    edges = np.arange(lo - 2.5, hi + 7.5, 5.0)          # leads are 5-minute multiples
    # Three groups, not two. An earlier version split on diff < 0, which put every
    # TIE into the "network warns earlier" bar and let the caption quote 94.8% when
    # the network is strictly earlier on 81.6%. Ties are 13.3% of these events and
    # they are their own answer, not a win for either arm.
    groups = ((diff > 0, NETWORK, "///", "LSTM earlier"),
              (diff == 0, NEUTRAL, "....", "tied"),
              (diff < 0, _G.NOMINAL["rose"], "xxxx", "control earlier"))
    for mask, colour, hatch, name in groups:
        share = float(mask.mean())
        axis.hist(diff[mask], bins=edges, color=colour, edgecolor=INK,
                  linewidth=0.45, hatch=hatch, label=f"{name} ({share:.1%})")
    axis.axvline(0.0, color=INK, linewidth=0.7, linestyle=(0, (3, 2)))
    median = float(np.median(diff))
    axis.axvline(median, color=INK, linewidth=0.9)
    axis.annotate(f"median {median:+.0f} min", xy=(median, axis.get_ylim()[1]),
                  xytext=(3, -2), textcoords="offset points", va="top", fontsize=6.5)
    axis.set_xlabel("LSTM − control lead time (min)", labelpad=2)
    axis.set_ylabel("Events")
    axis.set_title("a)  Paired warning lead time", loc="left", fontweight="bold")
    axis.legend(loc="upper left", handlelength=1.4)


def _panel_case(axis, case: dict, label: str, show_x: bool) -> None:
    onset = case["event_onset"]
    t = _minutes_from(onset, case["trace_time"])
    cgm = np.asarray(case["trace_cgm"], dtype=float)
    floor = float(min(np.nanmin(cgm), SEVERE_MG_DL))
    # The rails are spaced as a FRACTION of the panel's own range, not in mg/dL. At a
    # fixed 7.5 mg/dL separation the two rails sat 0.75 mm apart in the panel whose
    # trace reaches 300, so a ring drawn on one of them could not be attributed to an
    # arm by eye, which defeats the point of drawing it.
    # just above the 70 mg/dL line at minimum, so a trace that never leaves the
    # hypoglycaemic range still fills its panel instead of hugging the floor
    top = float(min(max(np.nanmax(cgm) * 1.08, 90.0), 260.0))
    span = max(top - floor, 1.0)
    band = 0.26 * span
    rail_threshold, rail_network = 0.06 * span, 0.17 * span
    axis.set_ylim(floor - band, top)
    axis.spines["bottom"].set_position(("data", floor - 2.0))
    axis.set_yticks([y for y in axis.get_yticks() if y >= floor - 1.0])
    axis.plot(t, cgm, color=INK, linewidth=0.9, zorder=3)
    axis.axhline(SEVERE_MG_DL, color=INK, linewidth=0.5, linestyle=(0, (4, 2)))
    axis.axhline(HYPO_MG_DL, color=INK, linewidth=0.5, linestyle=(0, (1, 2)))
    axis.axvline(0.0, color=INK, linewidth=0.8)

    # Alarms go on two rails rather than on the trace. When both arms fire at the
    # same minute, which is exactly what the tied case is, a marker drawn on the
    # trace hides the one beneath it and the panel silently loses an alarm.
    # An arm may fire several times in the window while only one firing is matched to
    # the event, and the endpoint is decided by that one. Drawing them alike leaves the
    # reader unable to check the lead time the panel is chosen for, which is the
    # objection this ring answers. The matched firing is the one at minus the arm's
    # recorded lead, so it is derived rather than asserted.
    for stamps, colour, marker, name, offset, lead in (
            (case["alarms_threshold"], _G.NOMINAL["rose"], "s", "Control", rail_threshold,
             case["lead_threshold_min"]),
            (case["alarms_network"], NETWORK, "o", "Balanced LSTM", rail_network,
             case["lead_network_min"])):
        at = _minutes_from(onset, stamps) if stamps else np.empty(0)
        inside = (at >= TRACE_XLIM[0]) & (at <= TRACE_XLIM[1])
        y = np.full(int(inside.sum()), floor - band + offset)
        shown = at[inside]
        axis.scatter(shown, y, s=14, marker=marker, color=colour, edgecolor=INK,
                     linewidth=0.4, zorder=5, label=name)
        if lead is not None and len(shown):
            hit = np.isclose(shown, -float(lead), atol=1e-6)
            if not hit.any():
                raise SystemExit(
                    f"{case['event_id']}: no {name} alarm at the recorded lead of "
                    f"{lead} min; the marked alarm would not be the matched one"
                )
            # Ring in the ink colour rather than the arm's, so "this is the matched
            # firing" reads the same on both rails. A grey ring on a grey marker was
            # invisible in the render.
            axis.scatter(shown[hit], y[hit], s=58, marker="o", facecolor="none",
                         edgecolor=INK, linewidth=0.8, zorder=6)
    axis.set_ylabel("Glucose (mg/dL)")
    if np.nanmax(cgm) > top:
        axis.annotate(f"peak {np.nanmax(cgm):.0f} mg/dL", xy=(TRACE_XLIM[0] + 4, top),
                      va="top", ha="left", fontsize=6.0)
    case_titles = {"b": "10th percentile (tie)", "c": "50th percentile",
                   "d": "90th percentile"}
    if label == "b":
        # The histogram's abscissa uses the narrow gap above this trace. Keep the
        # case descriptor inside its clear upper margin instead of over that label.
        axis.set_title(f"{label})", loc="left", fontweight="bold")
        axis.text(0.025, 0.985, case_titles[label], transform=axis.transAxes,
                  va="top", fontsize=7.0, fontweight="bold")
    else:
        axis.set_title(f"{label})  {case_titles[label]}", loc="left", fontweight="bold")
    axis.set_xlim(*TRACE_XLIM)
    if show_x:
        # Keep time labels below both alarm rails and their matched-firing rings.
        axis.tick_params(axis="x", pad=18)
        axis.set_xlabel("Time from event onset (min)")
    else:
        axis.tick_params(labelbottom=False)


def _extra_panels(axes, calibration_dir, collapse_dir):
    """The six panels that were two separate supplementary figures.

    What the operating point costs a participant, how early the matched warnings arrive,
    what the refractory rule does across the alarm axis, and how many distinct events
    each arm touches for what it spends. All of it is mechanism for the panels on the
    left, and none of it was in the same figure as them.
    """
    ax_e, ax_f, ax_g, ax_h, ax_i, ax_j = axes
    arms = (("persistence_seed17", "Control", "rose", "s"),
            ("tcn_erm_seed17", "TCN", "amber", "^"),
            ("balanced_lstm_erm_seed17", "LSTM", "blue", "o"))
    horizons = ("30", "60")
    calibration = {
        stem: json.loads((calibration_dir / f"{stem}.json").read_text())
        for stem, *_ in arms
    }

    # e: burden is concentrated, and equally so for every arm.
    for index, (stem, _label, hue, mark) in enumerate(arms):
        counts = calibration[stem]["horizons"]["30"]["participant_firing_counts"]
        ax_e.plot([index, index], [counts["median"], counts["max"]],
                  color=_G.NOMINAL[hue], linewidth=0.7, zorder=2)
        for value, size, filled in ((counts["median"], 3.0, True),
                                    (counts["p90"], 2.4, False),
                                    (counts["max"], 2.4, False)):
            ax_e.plot([index], [value], marker=mark, markersize=size,
                      markerfacecolor=_G.NOMINAL[hue] if filled else "white",
                      markeredgecolor=_G.NOMINAL[hue], markeredgewidth=0.7, zorder=3)
    ax_e.set_yscale("log")
    ax_e.set_xticks(range(len(arms)))
    ax_e.set_xticklabels([label for _s, label, *_ in arms], fontsize=5.0)
    ax_e.set_xlim(-0.6, len(arms) - 0.4)
    ax_e.set_ylabel("Threshold-positive\nwindows per participant", linespacing=1.4)
    ax_e.set_xlabel("Filled median; open p90, max", fontsize=6.5)

    # f: how early the matched warnings arrive.
    for index, (stem, _label, hue, mark) in enumerate(arms):
        for offset, horizon in zip((-0.16, 0.16), horizons, strict=True):
            block = calibration[stem]["horizons"][horizon]
            # The first version read a key that does not exist and skipped when it came
            # back None, which drew an EMPTY panel that looked like a finished one.
            if "median_lead_minutes" not in block:
                raise SystemExit(
                    f"{stem} horizon {horizon} carries no median_lead_minutes; the "
                    "panel would be blank and look finished"
                )
            value = float(block["median_lead_minutes"])
            ax_f.plot([index + offset], [value], marker=mark, markersize=3.0,
                      markerfacecolor=_G.NOMINAL[hue] if horizon == "30" else "white",
                      markeredgecolor=_G.NOMINAL[hue], markeredgewidth=0.7, zorder=3)
            ax_f.annotate(f"{value:g}", (index + offset, value), fontsize=5.0,
                          textcoords="offset points", xytext=(0, 4.0), ha="center",
                          color=INK, zorder=4)
    ax_f.set_xticks(range(len(arms)))
    ax_f.set_xticklabels([label for _s, label, *_ in arms], fontsize=5.0)
    ax_f.set_xlim(-0.6, len(arms) - 0.4)
    ax_f.set_ylabel("Median warning time (min)")
    ax_f.set_xlabel("Filled 30 min; open 60 min", fontsize=6.5)

    # g, h: the collapse ratio is a property of the operating point, not of an arm.
    # i, j: and so, at a matched alarm count, is which arm touches more distinct events.
    curves = {
        stem: json.loads((collapse_dir / f"{stem}.json").read_text())
        for stem, *_ in arms if (collapse_dir / f"{stem}.json").is_file()
    }
    if len(curves) < 2:
        raise SystemExit(
            f"the collapse panels need at least two arms; {sorted(curves)} exist")
    for (ratio_axis, cover_axis), horizon in zip(
            ((ax_g, ax_i), (ax_h, ax_j)), horizons, strict=True):
        for stem, label, hue, mark in arms:
            if stem not in curves:
                continue
            points = sorted(curves[stem][horizon], key=lambda r: r["alarms"])
            alarms = [r["alarms"] / 1000.0 for r in points]
            ratio_axis.plot(alarms, [r["collapse_ratio"] for r in points],
                            color=_G.NOMINAL[hue], marker=mark, markersize=2.0,
                            linewidth=0.8, linestyle="--" if stem.startswith("persistence")
                            else "-", zorder=3, label=label)
            cover_axis.plot(alarms, [r["events_touched"] for r in points],
                            color=_G.NOMINAL[hue], marker=mark, markersize=2.0,
                            linewidth=0.8, linestyle="--" if stem.startswith("persistence")
                            else "-", zorder=3)
        ratio_axis.set_ylabel(f"Collapse ratio, {horizon} min")
        cover_axis.set_ylabel(f"Events touched, {horizon} min")
        for axis in (ratio_axis, cover_axis):
            axis.set_xscale("log")
            axis.set_xlabel("Surviving alarms (×10³)", fontsize=6.5)
    ax_g.legend(loc="upper left", fontsize=5.0, handlelength=1.4,
                borderaxespad=0.2, labelspacing=0.2)
    titles = ("Crossing burden", "Warning lead time", "Refractory collapse",
              "Refractory collapse", "Event coverage", "Event coverage")
    for axis, letter, title in zip(axes, "efghij", titles, strict=True):
        axis.set_title(title, loc="left", fontsize=6.5, fontweight="bold", pad=5)
        box = axis.get_position()
        axis.figure.text(box.x0 - 0.058, box.y1 + 0.024, f"{letter})", fontsize=7.0,
                         fontweight="bold", color=INK, va="top")
    return sorted(curves)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--calibration-dir", required=True, type=Path)
    ap.add_argument("--collapse-dir", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--width", choices=sorted(WIDTH_MM), default="double")
    ap.add_argument("--height-mm", type=float, default=185.0)
    args = ap.parse_args()

    data = json.loads(args.input.read_text())
    for name in ("p10", "median", "p90"):
        if name not in data["cases"]:
            raise SystemExit(f"the extraction is missing case {name!r}")
    _style()
    width_in = WIDTH_MM[args.width] * MM
    figure = plt.figure(figsize=(width_in, args.height_mm * MM))
    # The distribution and the three pre-registered cases keep the left half, because a
    # glucose trace squeezed into a third of the page stops being readable. The six on
    # the right are what used to be two separate supplementary figures about the same
    # mechanism, which meant a reader had to leave the argument to follow it.
    # The two columns are solved from one bottom, one top and one gap. They used to
    # be hand-set and did not agree: the left column reached down to 0.108 of the
    # canvas and the right column stopped at 0.178, so the bottom right of a 185 mm
    # figure held a 13 mm strip of nothing, which is the widest blank band any figure
    # in this set carried. Deriving both columns from the same three numbers makes
    # that kind of disagreement impossible rather than merely fixed.
    _bottom, _top = 0.108, 0.958
    _lead = 0.196                       # the trace panel, which needs the extra height
    _gap_l, _gap_r = 0.055, 0.075
    _short = (_top - _bottom - _lead - 0.052 - 2 * _gap_l) / 3
    _tall = (_top - _bottom - 2 * _gap_r) / 3
    _ly = [_bottom + 2 * (_short + _gap_l) + _short + 0.052,
           _bottom + 2 * (_short + _gap_l),
           _bottom + _short + _gap_l,
           _bottom]
    _ry = [_bottom + 2 * (_tall + _gap_r), _bottom + _tall + _gap_r, _bottom]
    axes = [
        figure.add_axes([0.100, _ly[0], 0.335, _lead]),
        figure.add_axes([0.100, _ly[1], 0.335, _short]),
        figure.add_axes([0.100, _ly[2], 0.335, _short]),
        figure.add_axes([0.100, _ly[3], 0.335, _short]),
    ]
    extra = [
        figure.add_axes([0.585, _ry[0], 0.165, _tall]),
        figure.add_axes([0.820, _ry[0], 0.160, _tall]),
        figure.add_axes([0.585, _ry[1], 0.165, _tall]),
        figure.add_axes([0.820, _ry[1], 0.160, _tall]),
        figure.add_axes([0.585, _ry[2], 0.165, _tall]),
        figure.add_axes([0.820, _ry[2], 0.160, _tall]),
    ]
    assert abs(_ly[0] + _lead - _top) < 1e-9, _ly
    assert abs(_ry[0] + _tall - _top) < 1e-9, _ry

    _panel_distribution(axes[0], data)
    order = [("b", "p10"), ("c", "median"), ("d", "p90")]
    for index, (label, key) in enumerate(order):
        _panel_case(axes[1 + index], data["cases"][key], label,
                    show_x=(index == len(order) - 1))
    _extra_panels(extra, args.calibration_dir, args.collapse_dir)
    handles, labels = axes[1].get_legend_handles_labels()
    seen: dict[str, object] = {}
    for handle, name in zip(handles, labels, strict=True):
        seen.setdefault(name, handle)
    figure.legend(list(seen.values()), list(seen),
                  loc="lower center", ncol=2, handlelength=1.2,
                  bbox_to_anchor=(0.55, 0.0), frameon=False)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    _G.assert_compliant(figure, "make_fig_mechanism.py", caption_words=_caption_words.count("fig_mechanism"))
    with matplotlib.rc_context({"savefig.bbox": None, "savefig.pad_inches": 0}):
        figure.savefig(args.output)
    plt.close(figure)

    native_mm = figure.get_size_inches()[0] / MM
    if abs(native_mm - WIDTH_MM[args.width]) > 0.2:
        raise SystemExit(f"width {native_mm:.3f} mm off target {WIDTH_MM[args.width]}")
    print(f"wrote {args.output} at {native_mm:.2f} mm, "
          f"{data['n_events']} events, median {data['median_diff']:+.1f} min")


if __name__ == "__main__":
    main()
