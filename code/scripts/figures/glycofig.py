"""Figure style for the GlycoShift manuscript, Nature-family geometry.

Self-contained on purpose. The lab reference implementation lives in another project
(`scp-crossbench/bench/figures/scpxfig/style.py`) and this module encodes the same rules
rather than importing across project roots, so a change there cannot silently move a
figure here.

Three traps this module exists to close, each of which has shipped at least once:

1. ``savefig(bbox_inches="tight")`` re-crops the canvas, so raising a label's point size
   widens the figure and the extra downscaling cancels the gain exactly. Passing
   ``bbox_inches=None`` does NOT disable it: savefig reads None as "not supplied" and
   falls back to rcParams. `save` therefore writes inside an rc_context that pins
   ``savefig.bbox`` to None.
2. A bar whose y-axis does not start at zero overstates every difference on it. This is
   the defect this lab keeps reshipping, four instances in one package, each found only
   by looking at a render. `assert_compliant` refuses it.
3. Lightness is an ordered channel. Spending it on an unordered variable smuggles a
   ranking into a category axis. The nominal set and the ordered ladder are checked for
   disjointness and for their respective luminance budgets at import time.
"""

from __future__ import annotations

import itertools as _itertools
import sys as _sys
from pathlib import Path as _Path

import matplotlib as mpl
import matplotlib.pyplot as plt

MM = 1.0 / 25.4

# --------------------------------------------------------------- geometry
COL1_MM = 88.0
COL2_MM = 180.0
LEGAL_WIDTHS_MM = (COL1_MM, COL2_MM)
# 134 mm sat here as a "column and a half" and is in NO official guide. Fetched
# 2026-09-04 from nature.com/documents/NRJs-guide-to-preparing-final-artwork.pdf,
# which covers "Original research and review content ... Analysis, Article, Brief
# Communication, Consensus Statement, Hypothesis, Letter, Perspective, Progress
# Article, Resource, Review Article, Technical Report" and gives exactly two widths:
# 1-column 88 mm, 2-column 180 mm. The 135 mm middle width belongs to Nature
# Protocols, whose body is single-column, and does not apply to an npj Article.
# Nothing in this package ever used 134: fig1-fig5 are all COL2_MM. It was removed
# because a constant that is never right is a constant waiting to be copied, which is
# exactly what happened to the five manuscript figures.
# 1.0 would pass a real out-of-spec set; 0.015 is backend quantisation noise.
WIDTH_TOL_MM = 0.2
PAGE_H_MM = 225.0

# Height ceiling is a function of the CAPTION's word count, and of the caption actually
# in the manuscript rather than a planned one.
MAX_H_MM = {
    "1col": {50: 220.0, 150: 180.0, 300: 130.0},
    "2col": {50: 225.0, 150: 210.0, 300: 185.0},
}

FONT_MIN_PT = 5.0
FONT_MAX_PT = 7.0


def height_ceiling_mm(caption_words: int, two_column: bool) -> float:
    table = MAX_H_MM["2col" if two_column else "1col"]
    for limit in sorted(table):
        if caption_words < limit:
            return table[limit]
    raise ValueError(
        f"a caption of {caption_words} words has no ceiling in the guide; shorten it"
    )


# --------------------------------------------------------------- colour
INK = "#28303F"
MUTED = "#5F6B7E"

# Nominal: unordered categories. Chroma fixed at C* 35, lightness free, hues spread as
# widely as the count allows, plus exactly one pure neutral for the baseline category.
# Replaced 2026-09-04. The previous set -- blue #6D98D3, rose #BC6A79, amber #C98E66,
# olive #89A467, teal #3CAC9C -- measures a worst pairwise CIEDE2000 of 2.7 under
# simulated deuteranopia and 3.6 under protanopia, the offending pair being amber
# against olive. Two of the three assertions below passed it, because a luminance gap
# and a luminance spread cannot see a dichromat. That is precisely why the third
# assertion now exists. Measured replacement: 13.0 deutan, 17.6 protan.
NOMINAL = {
    "blue": "#3CA6CF",
    "rose": "#9D5061",
    "amber": "#8A6635",
    "olive": "#69AF87",
    "teal": "#80679F",
}
NEUTRAL = "#8C8C8C"

# Ordered: lightness alone makes the ladder, dark to pale. Palest is fills only.
LADDER = ["#667DB8", "#CB7F9F", "#A4A8B6", "#CAAEDF", "#9CD0F3", "#F3CFED", "#DEEFFF"]


def _relative_luminance(hex_colour: str) -> float:
    r, g, b = (int(hex_colour[i : i + 2], 16) / 255.0 for i in (1, 3, 5))

    def lin(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


def _check_palettes() -> None:
    """The two mirror-image assertions. Both have caught real errors elsewhere."""
    lum = [_relative_luminance(c) for c in LADDER]
    gaps = [abs(lum[i + 1] - lum[i]) for i in range(len(lum) - 1)]
    assert min(gaps) >= 0.05, (
        f"ordered ladder min adjacent luminance gap {min(gaps):.4f} < 0.05; "
        "it would not survive greyscale"
    )
    nom = [_relative_luminance(c) for c in list(NOMINAL.values()) + [NEUTRAL]]
    spread = max(nom) - min(nom)
    assert spread <= 0.22, (
        f"nominal luminance spread {spread:.4f} > 0.22; lightness is smuggling an "
        "ordering into an unordered variable"
    )
    shared = set(NOMINAL.values()) & set(LADDER)
    assert not shared, (
        f"nominal and ladder share {sorted(shared)}; keep the sets disjoint"
    )

    # The third assertion, and the one the other two are blind to. A categorical
    # palette must stay separable for a dichromat, or every categorical mark has to
    # carry a second, non-colour channel. Ten is calibrated rather than picked: it is
    # the highest round value that still accepts every published colour-blind-safe
    # palette while rejecting every set measured defective in this lab.
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
    import palette_cvd_gate as _cvd  # noqa: E402

    for _kind in ("deutan", "protan"):
        _worst, _pair = 1e9, None
        _cols = list(NOMINAL.values()) + [NEUTRAL]
        for _a, _b in _itertools.combinations(_cols, 2):
            _d = _cvd.ciede2000(
                _cvd.srgb_to_lab(_cvd.simulate(_cvd.hex_to_rgb(_a), _kind)),
                _cvd.srgb_to_lab(_cvd.simulate(_cvd.hex_to_rgb(_b), _kind)),
            )
            if _d < _worst:
                _worst, _pair = _d, (_a, _b)
        assert _worst >= _cvd.FLOOR, (
            f"nominal palette worst {_kind} CIEDE2000 {_worst:.1f} < {_cvd.FLOOR}: "
            f"{_pair[0]} and {_pair[1]} are one colour to a dichromat"
        )


_check_palettes()


# --------------------------------------------------------------- rcParams
def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            # mathtext has its OWN font stack and ignores font.sans-serif entirely, so a
            # figure that is Arial everywhere a person looks still renders every $...$
            # in
            # DejaVu unless these five are set.
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "mathtext.sf": "Arial",
            "mathtext.default": "regular",
            "font.size": 6.0,
            "axes.labelsize": 6.0,
            "axes.titlesize": 6.5,
            "xtick.labelsize": 5.5,
            "ytick.labelsize": 5.5,
            "legend.fontsize": 5.5,
            "figure.titlesize": 7.0,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.0,
            "ytick.major.size": 2.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 0.9,
            "lines.markersize": 3.0,
            "legend.frameon": False,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            # Journals want live, editable text. This is REQUIRED here and BANNED at
            # AAAI, where the kit forbids the CID/Identity-H that fonttype 42 emits.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_label(ax, letter: str, dx: float = -0.085, dy: float = 1.045) -> None:
    # npj Digital Medicine specifies lowercase bold labels as a), b), c).
    ax.text(
        dx, dy, f"{letter})", transform=ax.transAxes,
        fontsize=7.0, fontweight="bold", color=INK, va="top", ha="left",
    )


# --------------------------------------------------------------- compliance
def assert_compliant(fig, name: str, caption_words: int) -> None:
    w_mm = fig.get_figwidth() / MM
    h_mm = fig.get_figheight() / MM

    nearest = min(LEGAL_WIDTHS_MM, key=lambda t: abs(t - w_mm))
    assert abs(w_mm - nearest) <= WIDTH_TOL_MM, (
        f"{name}: width {w_mm:.2f} mm is not one of "
        f"{LEGAL_WIDTHS_MM} within {WIDTH_TOL_MM} mm"
    )
    ceiling = height_ceiling_mm(caption_words, two_column=(nearest == COL2_MM))
    assert h_mm <= ceiling + 0.2, (
        f"{name}: height {h_mm:.2f} mm exceeds the {ceiling} mm ceiling that a "
        f"{caption_words}-word caption allows"
    )

    for ax in fig.get_axes():
        # A bar encodes its value from zero. A dot or lollipop is the fix, never a
        # truncated axis.
        if ax.containers and any(
            isinstance(c, mpl.container.BarContainer) for c in ax.containers
        ):
            # Ask the container, never infer from width vs height: those are DATA
            # units on two different axes and are not comparable. Inferring it called a
            # stacked vertical bar horizontal and then checked the wrong axis.
            bars = [c for c in ax.containers
                    if isinstance(c, mpl.container.BarContainer)]
            vertical = all(getattr(c, "orientation", "vertical") == "vertical"
                           for c in bars)
            lo = ax.get_ylim()[0] if vertical else ax.get_xlim()[0]
            assert abs(lo) < 1e-12, (
                f"{name}: a bar axis starts at {lo:g}, not zero; use a dot or "
                "lollipop plot instead"
            )

    for txt in fig.findobj(mpl.text.Text):
        if not txt.get_text().strip():
            continue
        pt = txt.get_fontsize()
        assert FONT_MIN_PT - 1e-9 <= pt <= FONT_MAX_PT + 1e-9, (
            f"{name}: text {pt:.2f} pt is outside the {FONT_MIN_PT} to "
            f"{FONT_MAX_PT} pt guide: {txt.get_text()[:40]!r}"
        )


def save(fig, path: str, caption_words: int) -> None:
    """Save at the authored size.

    bbox_inches is never passed. See the module docstring.
    """
    assert_compliant(fig, path, caption_words)
    with mpl.rc_context({"savefig.bbox": None, "savefig.pad_inches": 0.0}):
        fig.savefig(path, dpi=mpl.rcParams["savefig.dpi"])
    plt.close(fig)
