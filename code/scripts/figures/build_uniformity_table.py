"""Per-cohort spread of the composition and its members, from the frozen decomposition.

The endpoint weights cohorts equally, so a mean can be carried by the easy ones. This
prints the number a cross-source claim actually rests on, which is the worst
cohort, next to the mean that is usually quoted instead. Read-only over one artefact.
to the mean that is usually quoted instead. Read-only over one artefact.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

KEY = "participant_macro_event_sensitivity"
LABEL = {
    "persist": "current glucose threshold",
    "lstm": "LSTM, balanced ERM",
    "m0": "LSTM, severe-retargeted",
    "persist,m0": "threshold + severe-retargeted",
    "persist,lstm": "threshold + balanced LSTM",
}
ORDER = ["persist", "lstm", "m0", "persist,m0", "persist,lstm"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--horizon", action="append", default=None,
                    help="repeatable; both horizons are reported side by side")
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    cells = json.loads(args.data.read_text())["cells"]

    horizons = args.horizon or ["30", "60"]
    lines = ["% Generated from the frozen decomposition. Do not edit by hand.",
             r"\begin{tabular}{lr" + "rr" * len(horizons) + "}", r"\toprule",
             " & & " + " & ".join(
                 rf"\multicolumn{{2}}{{c}}{{{h} min}}" for h in horizons) + r" \\",
             "".join(rf"\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}"
                     for i in range(len(horizons))),
             r"arm & primary & " + " & ".join(
                 "worst & spread" for _ in horizons) + r" \\", r"\midrule"]
    for key in ORDER:
        cells_by_h = []
        for h in horizons:
            block = cells[key][h].get("per_source", {})
            if not block:
                cells_by_h = []
                break
            values = {s: b[KEY] for s, b in block.items()}
            cells_by_h.append(
                f"{min(values.values()):.3f} & "
                f"{max(values.values()) - min(values.values()):.3f}")
        if not cells_by_h:
            continue
        bold = key == "persist,lstm"
        name = rf"\textbf{{{LABEL[key]}}}" if bold else LABEL[key]
        lines.append(f"{name} & {cells[key]['primary']:.6f} & "
                     + " & ".join(cells_by_h) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
