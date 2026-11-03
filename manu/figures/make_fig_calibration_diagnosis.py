#!/usr/bin/env python3
"""Render the calibration-diagnosis figure from diagnose_glucose_conditional_failure output.

It renders what it is given and invents nothing: an arm absent from the input is
absent from the figure, and a band the diagnostic marked `too_few` is dropped rather
than interpolated.

The finding it draws, measured 2026-08-21 on the selection partition at
`cgm_last >= 70`: within every glucose band the trained arms RANK well, beating a
glucose-only score by 3.9x to 1473x on AUPRC, while over-predicting the event rate by
64x at 70-107 mg/dL rising strictly monotonically to 918x at 216+. The threshold
control's ratio is not monotone, peaking at 646x and falling back to 267x.

The 460x above was a RETRACTED figure and stood here until 2026-09-04. Recomputed
from the same artifacts the ranges are 3.9x-406x for one deep arm and 4.2x-1473x
for the other, so the union is 3.9x-1473x. The retraction sweep that cleared this
number searched manu/ and never looked at the figure builders.

Journal figure contract (npj / Nature-family), matching make_fig_operational.py:
  * width exactly 88 or 180 mm, tolerance 0.2 mm
  * Arial everywhere including mathtext, which has its own font stack
  * pdf.fonttype 42 so text stays live and editable. **At AAAI this must be flipped**:
    the kit forbids the CID/Identity-H that fonttype 42 emits.
  * authored at final size: no bbox_inches="tight", no \\resizebox downstream
"""

from __future__ import annotations

import argparse
import json
import re
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
ARM_STYLE = {
    "Balanced LSTM (ERM)": (_G.NOMINAL["blue"], "o"),
    "TCN (ERM)": (_G.NOMINAL["amber"], "^"),
    "Current glucose (control)": (_G.NOMINAL["rose"], "s"),
}
INK = "#28303F"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "mathtext.sf": "Arial",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.1,
            "legend.frameon": False,
        }
    )


def _load(path: Path, horizon: str) -> tuple[str, list[dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [r for r in payload["horizons"][horizon] if r.get("status") != "too_few"]
    if not rows:
        raise SystemExit(f"no usable bands in {path} at horizon {horizon}")
    return str(payload["arm"]), rows


def _primary(payload: dict) -> float | None:
    value = payload.get("primary")
    if value is None:
        value = payload.get("primary_equal_horizon_mean_event_sensitivity")
    if isinstance(value, dict):
        value = value.get("global")
    return None if value is None else float(value)


def _frame_baselines(root: Path) -> dict[str, dict[str, float]]:
    """Read each frozen comparator on the same eligibility frame as its transform."""
    baselines = {"full": {}, "eugly": {}}
    for arm in ("balanced_lstm_erm_seed17", "tcn_erm_seed17", "persistence_seed17"):
        paths = {
            "full": root / "l20-run-results" / "runs" / arm / "operational_metrics.json",
            "eugly": root / "analysis-euglycaemic" / f"{arm}.json",
        }
        for frame, path in paths.items():
            value = _primary(json.loads(path.read_text()))
            if value is None:
                raise ValueError(f"missing frozen primary endpoint in {path}")
            baselines[frame][arm] = value
    return baselines


def _cell_baseline(baselines: dict, arm: str, family: str, stem: str) -> float:
    if family == "analysis-statecal":
        frame = stem.rsplit("_", 1)[-1]
        if frame not in baselines:
            raise ValueError(f"unknown state-conditional eligibility frame: {stem}")
    else:
        frame = "eugly"
    return baselines[frame][arm]


def _correction_panels(axes, root: Path, calibration_dir: Path, series):
    """What the diagnosis implies, and what happened when it was acted on.

    Every post-hoc correction this project fitted is read here rather than summarised.
    The families are separate because they fail for different reasons: the
    band-conditional ones run out of positives inside a band, the parametric one learns
    a glucose term that reorders the scores, and the state-conditional ones destroy the
    arm outright. The single cell that gains uses no band structure at all.
    """
    ax_e, ax_f, ax_g, ax_h, ax_i, ax_j = axes
    baselines = _frame_baselines(root)
    base = baselines["eugly"]
    families = (("analysis-isotonic", "rank within bands", "blue", "o"),
                ("analysis-parametric", "glucose covariate", "rose", "s"),
                ("analysis-statecal", "state-conditional", "amber", "^"))

    # e: every correction cell against the arm it was fitted on.
    cells, gains = [], []
    for directory, label, hue, mark in families:
        for path in sorted((root / directory).glob("*.json")):
            payload = json.loads(path.read_text())
            value = _primary(payload)
            if value is None:
                continue
            arm = next((a for a in base if a in path.stem), None)
            if arm is None or path.stem.startswith("GATE1BIN"):
                continue
            before = _cell_baseline(baselines, arm, directory, path.stem)
            cells.append((label, hue, mark, before, value, path.stem))
    if len(cells) < 10:
        raise SystemExit(f"only {len(cells)} correction cells were found; the panel "
                         "would understate how many were tried")
    for _label, hue, mark, before, after, stem in cells:
        gains.append((after - before, stem))
        ax_e.plot([before], [after], marker=mark, markersize=2.6,
                  markerfacecolor="none", markeredgecolor=_G.NOMINAL[hue],
                  markeredgewidth=0.8, linestyle="none", zorder=3)
    limits = (0.0, 0.86)
    ax_e.plot(limits, limits, color=INK, lw=0.5, ls=(0, (2.4, 1.6)), zorder=2)
    ax_e.set_xlim(*limits)
    ax_e.set_ylim(*limits)
    ax_e.set_xlabel("Before, matched eligibility frame")
    ax_e.set_ylabel("Primary after recalibration")
    ax_e.legend(handles=[
        plt.Line2D([], [], marker=m, ms=2.6, lw=0, markerfacecolor="none",
                   markeredgecolor=_G.NOMINAL[h], label=lab)
        for _d, lab, h, m in families
    ], loc="upper left", fontsize=5.0, handlelength=1.0, borderaxespad=0.2,
        labelspacing=0.2)

    # f: more bands is not the fix. The one-band control is the only gain and it lives
    # in a run log rather than a result file, which is why it was an unbound number.
    one_band = _one_band_gain(root / "analysis-isotonic" / "GATE1BAND_lstm.log")
    counts = [1, 4, 6, 10]
    for stem, hue, mark in (("balanced_lstm_erm_seed17", "blue", "o"),
                            ("tcn_erm_seed17", "amber", "^")):
        values = []
        for count in counts:
            if count == 1:
                values.append(base[stem] + one_band if "lstm" in stem else None)
                continue
            path = root / "analysis-isotonic" / f"{stem}_b{count}.json"
            values.append(_primary(json.loads(path.read_text()))
                          if path.is_file() else None)
        x = [c for c, v in zip(counts, values, strict=True) if v is not None]
        y = [v for v in values if v is not None]
        ax_f.plot(x, y, color=_G.NOMINAL[hue], marker=mark, markersize=2.6,
                  linewidth=0.9, zorder=3)
        ax_f.axhline(base[stem], color=_G.NOMINAL[hue], lw=0.5, ls=(0, (2.2, 1.6)),
                     zorder=2)
    ax_f.set_xticks(counts)
    ax_f.set_xlabel("Glucose bands the map is fitted in")
    ax_f.set_ylabel("Primary, restricted frame")

    # g: the state-conditional family, which does not merely fail but destroys the arm.
    kinds = ("level_full", "level_eugly", "slope_full", "slope_eugly")
    for index, (stem, hue, mark) in enumerate((("persistence_seed17", "rose", "s"),
                                               ("balanced_lstm_erm_seed17", "blue", "o"),
                                               ("tcn_erm_seed17", "amber", "^"))):
        for offset, kind in enumerate(kinds):
            path = root / "analysis-statecal" / f"{stem}_{kind}.json"
            if not path.is_file():
                continue
            ax_g.plot([offset + (index - 1) * 0.18],
                      [_primary(json.loads(path.read_text()))], marker=mark,
                      markersize=2.4, color=_G.NOMINAL[hue], zorder=3)
            before = _cell_baseline(baselines, stem, "analysis-statecal", path.stem)
            x = offset + (index - 1) * 0.18
            ax_g.plot([x - 0.065, x + 0.065], [before, before],
                      color=_G.NOMINAL[hue], lw=1.0, zorder=2)
    ax_g.set_xticks(range(len(kinds)))
    ax_g.set_xticklabels(["level\nall", "level\n≥70", "slope\nall", "slope\n≥70"], fontsize=5.0)
    ax_g.set_xlim(-0.6, len(kinds) - 0.4)
    ax_g.set_ylim(0.0, 0.86)
    ax_g.set_ylabel("Primary, matched frame")

    # h: what the parametric fit learned, which is a glucose term.
    labels, betas = [], []
    for stem, short in (("balanced_lstm_erm_seed17", "LSTM"),
                        ("tcn_erm_seed17", "TCN")):
        path = root / "analysis-parametric" / f"{stem}_glucose.json"
        if not path.is_file():
            continue
        coefficients = json.loads(path.read_text())["coefficients"]
        for horizon in ("30", "60"):
            labels.append(f"{short}\n{horizon}")
            betas.append(float(coefficients[horizon]["beta"][-1]))
    # A lollipop, not bars. One coefficient is negative, and a bar whose axis does not
    # start at zero overstates every difference on it, which the geometry gate refuses.
    for index, value in enumerate(betas):
        ax_h.plot([index, index], [0.0, value], color=_G.NOMINAL["rose"], lw=0.8,
                  solid_capstyle="butt", zorder=3)
        ax_h.plot([index], [value], marker="o", markersize=2.8,
                  color=_G.NOMINAL["rose"], zorder=4)
    ax_h.axhline(0.0, color=INK, lw=0.6, zorder=2)
    ax_h.set_xticks(range(len(labels)))
    ax_h.set_xticklabels(labels, fontsize=5.0)
    ax_h.set_ylabel("Squared log-glucose\ncoefficient")

    # i: the support a band-conditional map is fitted on.
    rows = series[0][1]
    # Markers, not bars. The counts span three orders of magnitude so the axis has to be
    # logarithmic, and a bar on a logarithmic axis has no base to encode from.
    positives = [float(r["positives"]) for r in rows]
    ax_i.plot(range(len(rows)), positives, color=_G.NOMINAL["olive"], marker="o",
              markersize=2.8, linewidth=0.8, zorder=3)
    ax_i.set_yscale("log")
    ax_i.annotate(f"{positives[-1]:.0f}", (len(rows) - 1, positives[-1]), fontsize=5.0,
                  textcoords="offset points", xytext=(-2.0, 4.5), ha="right",
                  color=INK, zorder=4)
    ax_i.set_xticks(range(len(rows)))
    ax_i.set_xticklabels(_band_labels(rows), rotation=40, ha="right", fontsize=5.0)
    ax_i.set_ylabel("Positive windows (60 min)")
    ax_i.set_xlabel("Current glucose band (mg/dL)")

    # j: the shape of the miscalibration, as a slope and an expected error.
    arms = (("persistence_seed17", "Control", "rose", "s"),
            ("tcn_erm_seed17", "TCN", "amber", "^"),
            ("balanced_lstm_erm_seed17", "LSTM", "blue", "o"))
    # A twin axis was tried and dropped. Two quantities on two scales inside 33 mm, told
    # apart only by marker shape, is unreadable, and the slope = 1 reference line landed
    # among the other axis's values. The slope is the quantity the text refers to.
    for index, (stem, _label, hue, mark) in enumerate(arms):
        block = json.loads((calibration_dir / f"{stem}.json").read_text())["horizons"]
        for offset, horizon in zip((-0.16, 0.16), ("30", "60"), strict=True):
            ax_j.plot([index + offset], [float(block[horizon]["calibration_slope"])],
                      marker=mark, markersize=3.0,
                      markerfacecolor=_G.NOMINAL[hue] if horizon == "30" else "white",
                      markeredgecolor=_G.NOMINAL[hue], markeredgewidth=0.7, zorder=3)
            ax_j.annotate(f"{float(block[horizon]['calibration_slope']):.2f}",
                          (index + offset,
                           float(block[horizon]["calibration_slope"])),
                          fontsize=5.0, textcoords="offset points", xytext=(0, 4.0),
                          ha="center", color=INK, zorder=4)
    ax_j.axhline(1.0, color=INK, lw=0.6, ls=(0, (2.2, 1.6)), zorder=2)
    ax_j.set_xticks(range(len(arms)))
    ax_j.set_xticklabels([lab for _s, lab, *_ in arms], fontsize=5.0)
    ax_j.set_xlim(-0.6, len(arms) - 0.4)
    ax_j.set_ylabel("Calibration slope")
    ax_j.set_xlabel("Filled, 30 minutes; open, 60")
    # Keep both the diagnosed arm and cross-arm maximum, always within matched frames.
    own = max(g for g, stem in gains if "balanced_lstm" in stem)
    across, across_stem = max(gains)
    return len(cells), one_band, own, across, across_stem


def _one_band_gain(path: Path) -> float:
    """The single gain among the corrections, which lives in a log and not a result."""
    match = re.search(r"gate drift\s+([0-9.]+)", path.read_text())
    if match is None:
        raise SystemExit(
            f"{path.name} carries no gate drift; the one-band control is the only "
            "correction that gains and the panel cannot state it without this"
        )
    return float(match.group(1))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnosis", action="append", required=True,
                        help="label=path/to/diagnosis_<arm>.json")
    parser.add_argument("--width", choices=sorted(WIDTH_MM), default="double")
    parser.add_argument("--height-mm", type=float, default=185.0)
    parser.add_argument("--corrections-root", required=True, type=Path,
                        help="the v1.5 bundle, which holds the correction families")
    parser.add_argument("--calibration-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args(argv)

    _style()
    width_in = WIDTH_MM[arguments.width] * MM
    figure = plt.figure(figsize=(width_in, arguments.height_mm * MM))
    # The four diagnostic panels keep the top half at full width, because their abscissa
    # is six glucose bands with labels. The six below are what the diagnosis implies and
    # what happened when it was acted on.
    axes = np.array([
        [figure.add_axes([0.095, 0.780, 0.360, 0.178]),
         figure.add_axes([0.610, 0.780, 0.365, 0.178])],
        [figure.add_axes([0.095, 0.560, 0.360, 0.178]),
         figure.add_axes([0.610, 0.560, 0.365, 0.178])],
    ], dtype=object)
    extra = [
        figure.add_axes([0.095, 0.318, 0.235, 0.170]),
        figure.add_axes([0.470, 0.318, 0.185, 0.170]),
        figure.add_axes([0.790, 0.318, 0.185, 0.170]),
        figure.add_axes([0.095, 0.070, 0.185, 0.172]),
        figure.add_axes([0.415, 0.070, 0.185, 0.172]),
        figure.add_axes([0.735, 0.070, 0.240, 0.172]),
    ]

    for row_index, horizon in enumerate(("30", "60")):
        series = []
        for item in arguments.diagnosis:
            label, _, path = item.partition("=")
            _, rows = _load(Path(path), horizon)
            series.append((label, rows))
        left, right = axes[row_index]
        _draw(left, right, series, horizon, row_index)

    # Lettered panels. This figure identified its four cells by two column titles and
    # a horizon printed inside each y-label, which reads fine on the page and is
    # unciteable in prose: a sentence about over-prediction at 60 minutes had no name
    # to use. Letters run across then down, so a and b are the 30-minute row.
    for axis, letter in zip(list(axes.ravel()) + extra, "abcdefghij", strict=True):
        box = axis.get_position()
        figure.text(box.x0 - 0.056, box.y1 + 0.022, f"{letter})", fontsize=7.0,
                    fontweight="bold", color=INK, va="top")

    corrections = _correction_panels(
        extra, arguments.corrections_root, arguments.calibration_dir, series)
    _G.assert_compliant(figure, "make_fig_calibration_diagnosis.py", caption_words=_caption_words.count("fig_calibration_diagnosis"))
    with matplotlib.rc_context({"savefig.bbox": None, "savefig.pad_inches": 0}):
        figure.savefig(arguments.output)
    native = figure.get_size_inches() * 25.4
    tried, one_band, own, across, across_stem = corrections
    if own > one_band + 1e-9:
        raise SystemExit(
            f"the caption says the one-band control is the only correction of this arm "
            f"that gains; a cell gains {own:.6f} against its {one_band:.6f}"
        )
    print(f"wrote {arguments.output}  native {native[0]:.2f} x {native[1]:.2f} mm; "
          f"{tried} correction cells tried; best on the diagnosed arm {own:+.6f} "
          f"against the one-band control's {one_band:+.6f}; largest gain anywhere "
          f"{across:+.6f} ({across_stem})")
    target = WIDTH_MM[arguments.width]
    if abs(native[0] - target) > 0.2:
        raise SystemExit(f"width {native[0]:.3f} mm is off target {target} by >0.2")
    return 0


def _band_labels(rows: list[dict]) -> list[str]:
    """Range labels for the six bands, the last one open-ended and shown as such."""
    out = []
    for index, row in enumerate(rows):
        low = int(round(row["glucose_low"]))
        if index == len(rows) - 1:
            out.append(f"$\\geq${low}")
        else:
            out.append(f"{low}\u2013{int(round(rows[index + 1]['glucose_low'])) - 1}")
    return out


def _draw(left, right, series, horizon: str, row_index: int) -> None:
    for index, (label, rows) in enumerate(series):
        colour, marker = ARM_STYLE[label]
        # Bands are plotted at their INDEX, not at a glucose midpoint. The top band is
        # open-ended, so its midpoint was the mean of 216 and the observed maximum, 358,
        # which put a 162 mg/dL gap into the axis and implied continuous evidence across
        # it. Six categories on equal spacing is what the analysis actually has.
        x = list(range(len(rows)))
        ratio = [r["mean_predicted"] / r["observed_rate"] for r in rows]
        left.plot(x, ratio, marker=marker, color=colour, label=label,
                  markersize=3.2, linestyle="-" if index < len(series) - 1 else "--")
        lift = [r["auprc_network"] / r["auprc_glucose_only"] for r in rows]
        right.plot(x, lift, marker=marker, color=colour, label=label,
                   markersize=3.2, linestyle="-" if index < len(series) - 1 else "--")

    # The ratios on both panels share one denominator per band, and at the right-hand end
    # that denominator is thin: 25 events in a million rows carries the largest number on
    # the page. Printing the count is the difference between a reader seeing a 900-fold
    # effect and a reader seeing a 900-fold effect resting on 25 events.
    counts = series[0][1]
    if any(r["positives"] != q["positives"]
           for _, rows in series for r, q in zip(rows, counts, strict=True)):
        raise SystemExit(
            "the arms disagree on a band's positive count, so one number cannot label both"
        )
    for axis in (left, right):
        blended = matplotlib.transforms.blended_transform_factory(
            axis.transData, axis.transAxes
        )
        for position, row in enumerate(counts):
            axis.text(position, 1.005, f"{row['positives']:,}",
                      transform=blended, ha="center", va="bottom",
                      fontsize=5.0, color=_G.MUTED)

    labels = _band_labels(series[0][1])
    for axis in (left, right):
        axis.set_xticks(range(len(labels)))
        axis.set_xticklabels(labels, fontsize=5.5)
        axis.set_xlim(-0.45, len(labels) - 0.55)
        axis.set_yscale("log")
        axis.axhline(1.0, color=INK, linewidth=0.5, linestyle=":")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    left.set_ylabel(f"{horizon} min\nPredicted / observed")
    right.set_ylabel(f"{horizon} min\nAUPRC vs glucose-only")
    if row_index == 0:
        left.set_title("Calibration: over-prediction", loc="left",
                       fontweight="bold", pad=17)
        right.set_title("Ranking: within-band lift", loc="left",
                        fontweight="bold", pad=17)
        right.legend(loc="upper left")
    else:
        for axis in (left, right):
            axis.set_xlabel("Current glucose band at the decision, mg/dL")


if __name__ == "__main__":
    raise SystemExit(main())
