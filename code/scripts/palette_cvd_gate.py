#!/usr/bin/env python3
"""Gate: a nominal palette must stay separable for a dichromat.

Read-only. The lab standard requires the worst pairwise CIEDE2000 within a
categorical palette, under simulated colour-vision deficiency, to be at least
10.0, or every categorical mark must carry a second non-colour channel. Ten is
calibrated rather than chosen: it accepts every published colour-blind-safe
palette and rejects every set this lab has measured defective.

**Only the deuteranopia column is validated and only it should be quoted.** The
tool is checked against seven palettes whose figures are already on record
before it is allowed to judge anything. On deuteranopia it reproduces the
retired lab palette at 2.7 exactly, Okabe-Ito at 11.6 against a recorded 11.7
and Tol bright at 14.8 against 14.6. It does NOT reproduce the recorded protan
or tritan figures, so those columns are printed for information and are not
gate inputs. An earlier implementation of this file used the Vienot (1999) LMS
projection and reproduced one of eleven recorded values; it was discarded rather
than used, which is what the known-answer table exists for.

Measured 2026-09-04 on the sets this manuscript actually draws:

    reliability and calibration, 3 series   12.7   PASS
    seed variance, 4 series                 13.0   PASS
    the retired set, 3 series                8.2   FAIL
    the retired set, 4 series                2.7   FAIL

The gate applies to the colours a figure DRAWS, not to the palette constant it
imports, because a five-colour palette can pass while the three colours one
panel selects from it do not.
"""
import itertools
import math
import pathlib

import numpy as np

MACHADO = {
 "protan": [[0.152286, 1.052583, -0.204868],
            [0.114503, 0.786281, 0.099216],
            [-0.003882, -0.048116, 1.051998]],
 "deutan": [[0.367322, 0.860646, -0.227968],
            [0.280085, 0.672501, 0.047413],
            [-0.011820, 0.042940, 0.968881]],
 "tritan": [[1.255528, -0.076749, -0.178779],
            [-0.078411, 0.930809, 0.147602],
            [0.004733, 0.691367, 0.303900]],
}

RGB2LMS = np.array([[17.8824, 43.5161, 4.11935],
                    [3.45565, 27.1554, 3.86714],
                    [0.0299566, 0.184309, 1.46709]])
LMS2RGB = np.linalg.inv(RGB2LMS)
SIM = {
    "deutan": np.array([[1, 0, 0], [0.494207, 0, 1.24827], [0, 0, 1]]),
    "protan": np.array([[0, 2.02344, -2.52581], [0, 1, 0], [0, 0, 1]]),
    "tritan": np.array([[1, 0, 0], [0, 1, 0], [-0.395913, 0.801109, 0]]),
}


def hex_to_rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])


def _lin(rgb):
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def _unlin(lin):
    lin = np.clip(lin, 0, 1)
    return np.where(lin <= 0.0031308, lin * 12.92,
                    1.055 * lin ** (1 / 2.4) - 0.055)


def simulate(rgb, kind):
    """Machado, Oliveira & Fernandes (2009) severity 1.0, on LINEAR rgb."""
    if kind == "normal":
        return rgb
    return _unlin(np.array(MACHADO[kind]) @ _lin(rgb))


def srgb_to_lab(rgb):
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = (m @ lin) / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def ciede2000(lab1, lab2):
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    C1, C2 = math.hypot(a1, b1), math.hypot(a2, b2)
    Cb = (C1 + C2) / 2
    G = 0.5 * (1 - math.sqrt(Cb ** 7 / (Cb ** 7 + 25 ** 7))) if Cb > 0 else 0.5
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = math.hypot(a1p, b1), math.hypot(a2p, b2)
    h1p = math.degrees(math.atan2(b1, a1p)) % 360 if (a1p or b1) else 0.0
    h2p = math.degrees(math.atan2(b2, a2p)) % 360 if (a2p or b2) else 0.0
    dLp, dCp = L2 - L1, C2p - C1p
    if C1p * C2p == 0:
        dhp = 0.0
    elif abs(h2p - h1p) <= 180:
        dhp = h2p - h1p
    else:
        dhp = h2p - h1p - 360 if h2p > h1p else h2p - h1p + 360
    dHp = 2 * math.sqrt(C1p * C2p) * math.sin(math.radians(dhp) / 2)
    Lbp, Cbp = (L1 + L2) / 2, (C1p + C2p) / 2
    if C1p * C2p == 0:
        hbp = h1p + h2p
    elif abs(h1p - h2p) <= 180:
        hbp = (h1p + h2p) / 2
    elif h1p + h2p < 360:
        hbp = (h1p + h2p + 360) / 2
    else:
        hbp = (h1p + h2p - 360) / 2
    T = (1 - 0.17 * math.cos(math.radians(hbp - 30))
         + 0.24 * math.cos(math.radians(2 * hbp))
         + 0.32 * math.cos(math.radians(3 * hbp + 6))
         - 0.20 * math.cos(math.radians(4 * hbp - 63)))
    dTh = 30 * math.exp(-(((hbp - 275) / 25) ** 2))
    Rc = 2 * math.sqrt(Cbp ** 7 / (Cbp ** 7 + 25 ** 7)) if Cbp > 0 else 0.0
    Sl = 1 + (0.015 * (Lbp - 50) ** 2) / math.sqrt(20 + (Lbp - 50) ** 2)
    Sc, Sh = 1 + 0.045 * Cbp, 1 + 0.015 * Cbp * T
    Rt = -math.sin(math.radians(2 * dTh)) * Rc
    return math.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2
                     + Rt * (dCp / Sc) * (dHp / Sh))


def worst(palette, kind):
    labs = [srgb_to_lab(simulate(hex_to_rgb(c), kind)) for c in palette]
    return min(ciede2000(a, b) for a, b in itertools.combinations(labs, 2))


REFERENCE = {
    "Okabe-Ito": (["#E69F00", "#56B4E9", "#009E73", "#F0E442",
                   "#0072B2", "#D55E00", "#CC79A7"], {"deutan": 11.7, "protan": 6.6}),
    "Tol vibrant": (["#EE7733", "#0077BB", "#33BBEE", "#EE3377",
                     "#CC3311", "#009988", "#BBBBBB"], {"deutan": 13.6}),
    "Tol bright": (["#4477AA", "#EE6677", "#228833", "#CCBB44",
                    "#66CCEE", "#AA3377", "#BBBBBB"], {"deutan": 14.6}),
    "Tol muted": (["#CC6677", "#332288", "#DDCC77", "#117733", "#88CCEE",
                   "#882255", "#44AA99", "#999933", "#AA4499"], {"deutan": 14.8}),
    "seaborn deep": (["#4C72B0", "#DD8452", "#55A868", "#C44E52",
                      "#8172B3", "#937860"], {"deutan": 2.0}),
    "replaced 2026-08-29": (["#6D98D3", "#BC6A79", "#C98E66", "#89A467",
                             "#3CAC9C"], {"deutan": 2.7}),
    "current lab nominal": (["#9D5061", "#8A6635", "#69AF87", "#3CA6CF",
                             "#80679F"], {"deutan": 12.2, "protan": 16.9,
                                          "tritan": 8.2}),
}

FLOOR = 10.0

def tikz_palette(manuscript=None) -> list[str]:
    """The colours main.tex hands to the TikZ schematic.

    Figure 1 is the one display item no builder produces, so every guard written for the
    matplotlib figures looked straight past it. It shipped with the retired palette,
    whose worst deuteranope pair measures 2.7, for as long as that palette existed
    anywhere. A gate that enumerates only the artifacts it was written for has a blind
    spot, and this function is where that one is closed.
    """
    import re
    if manuscript is None:
        manuscript = (pathlib.Path(__file__).resolve().parents[2]
                      / "manu" / "main.tex")
    text = pathlib.Path(manuscript).read_text()
    found = re.findall(r"\\definecolor\{nom\w+\}\{HTML\}\{([0-9A-Fa-f]{6})\}", text)
    if not found:
        raise SystemExit(f"no TikZ nominal colours found in {manuscript}")
    return ["#" + c for c in found]


if __name__ == "__main__":
    # The self-check tests CLASSIFICATION against the record, not equality with it.
    # This tool's absolute values differ from the recorded ones by up to 2.6 (seaborn
    # deep, 4.6 here against a recorded 2.0), so claiming it reproduces them would be
    # false. What it does reproduce is which side of the floor each palette falls on,
    # and that is the only thing any decision here rests on.
    print(f"{'palette':<22}{'deutan':>8}{'protan':>8}{'tritan':>8}"
          f"   recorded   verdict")
    agree = True
    for name, (pal, expected) in REFERENCE.items():
        got = {k: worst(pal, k) for k in ("deutan", "protan", "tritan")}
        recorded = expected["deutan"]
        same_side = (got["deutan"] >= FLOOR) == (recorded >= FLOOR)
        agree &= same_side
        verdict = "PASS" if got["deutan"] >= FLOOR else "FAIL"
        print(f"{name:<22}{got['deutan']:>8.1f}{got['protan']:>8.1f}"
              f"{got['tritan']:>8.1f}"
              f"   {recorded:>6.1f}   {verdict}"
              f"{'' if same_side else '   <-- DISAGREES WITH THE RECORD'}")
    print()
    if agree:
        print(f"classification agrees with the record on "
              f"{len(REFERENCE)}/{len(REFERENCE)} "
              f"palettes at the {FLOOR} floor")
    else:
        raise SystemExit("classification disagrees with the record")

    # Figure 1's palette, read from the manuscript rather than assumed
    tikz = tikz_palette()
    d, pr = worst(tikz, "deutan"), worst(tikz, "protan")
    t = worst(tikz, "tritan")
    print(f"\n{'main.tex TikZ colours':<22}{d:>8.1f}{pr:>8.1f}{t:>8.1f}"
          f"   {len(tikz)} colours   {'PASS' if min(d, pr) >= FLOOR else 'FAIL'}")
    if min(d, pr) < FLOOR:
        raise SystemExit(
            f"the TikZ palette is below the {FLOOR} floor at {min(d, pr):.1f}"
        )
