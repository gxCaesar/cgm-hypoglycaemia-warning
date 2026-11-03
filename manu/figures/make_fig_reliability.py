#!/usr/bin/env python3
"""Reliability of the frozen scores at the 30 and 60-minute horizons.

Consumes calibration_and_burden_table.py output. Renders what it is given and invents
nothing: a bin the computation dropped for holding under 100 rows is absent here too.

Journal figure contract (npj / Nature-family), matching the other builders in this
directory: width exactly 88 or 180 mm at 0.2 mm tolerance, Arial everywhere
including mathtext, pdf.fonttype 42, authored at final size with no tight bbox.
**At AAAI fonttype 42 must be flipped**, since the kit forbids the CID/Identity-H it
emits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

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
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "custom", "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
        "mathtext.sf": "Arial", "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK, "ytick.color": INK, "axes.linewidth": 0.6,
        "lines.linewidth": 1.1, "legend.frameon": False,
    })


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True,
                        help="label=path/to/<arm>.json")
    parser.add_argument("--width", choices=sorted(WIDTH_MM), default="double")
    parser.add_argument("--height-mm", type=float, default=72.0)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args(argv)

    _style()
    series = []
    for item in arguments.arm:
        label, _, path = item.partition("=")
        series.append((label, json.loads(Path(path).read_text(encoding="utf-8"))))

    width_in = WIDTH_MM[arguments.width] * MM
    figure, axes = plt.subplots(1, 2, figsize=(width_in, arguments.height_mm * MM))

    for index, (label, payload) in enumerate(series):
        colour, marker = ARM_STYLE[label]
        dashes = "--" if index == len(series) - 1 else "-"
        for axis, horizon in zip(axes, ("30", "60")):  # noqa: B905
            bins = payload["horizons"][horizon]["reliability_bins"]
            predicted = [b["mean_predicted"] for b in bins]
            observed = [b["observed_rate"] for b in bins]
            axis.plot(predicted, observed, marker=marker, color=colour,
                      linestyle=dashes, markersize=3.0,
                      label=label if horizon == "30" else None)

    for axis, horizon in zip(axes, ("30", "60")):  # noqa: B905
        low = 1e-5
        axis.plot([low, 1.0], [low, 1.0], color=INK, linewidth=0.5, linestyle=":")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(low, 1.0)
        axis.set_ylim(low, 1.0)
        axis.set_xlabel("mean predicted risk")
        axis.set_title(f"{horizon} min", loc="left", fontweight="bold")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].set_ylabel("observed event rate")
    axes[0].legend(loc="upper left")
    # Lettered for the same reason as the calibration figure: a horizon printed as a
    # panel title cannot be cited from prose.
    for axis, letter in zip(axes, "ab", strict=True):
        _G.panel_label(axis, letter, dx=-0.135, dy=1.135)

    figure.subplots_adjust(left=0.105, right=0.99, top=0.875, bottom=0.165, wspace=0.30)
    _G.assert_compliant(figure, "make_fig_reliability.py", caption_words=_caption_words.count("fig_reliability"))
    with matplotlib.rc_context({"savefig.bbox": None, "savefig.pad_inches": 0}):
        figure.savefig(arguments.output)
    native = figure.get_size_inches() * 25.4
    print(f"wrote {arguments.output}  native {native[0]:.2f} x {native[1]:.2f} mm")
    if abs(native[0] - WIDTH_MM[arguments.width]) > 0.2:
        raise SystemExit("width off target by more than 0.2 mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
