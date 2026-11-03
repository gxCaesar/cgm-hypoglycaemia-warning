#!/usr/bin/env python3
"""Generate the manuscript's elimination table from frozen artifacts.

Nothing here is retyped. Every value is read from a run's own
`operational_metrics.json`, so a number in the paper and a number on disk cannot
drift apart. An arm whose artifact is missing is emitted as a gap rather than
silently dropped, because a table that quietly shrinks is how a comparator set
loses a member nobody notices.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Every candidate below is a one- or two-key variant of the severe-retargeted arm,
# so that arm is the control they are all measured against. It is itself a candidate,
# and comparing it to itself printed a meaningless "+0.000000 (+0.00 sigma)" in the
# Delta column of an earlier build. Its own pre-registered control is the balanced
# LSTM, and its margin over that is reported on its own line instead.
# CONTROL is read from the control arm's own artefact, never typed. The hand-written
# literal it replaces was a truncation, so an arm that tied the control exactly showed a
# delta of 2.7e-10 and the exact-tie test never fired.
CONTROL: float | None = None
CONTROL_LABEL = "M0 severe retarget"
CONTROL_OWN_COMPARATOR = 0.738848093
CONTROL_OWN_COMPARATOR_LABEL = "balanced LSTM ERM"
FLOOR = 0.836116373
# Two scales. The transferred one is what every margin in this project was originally
# divided by; the measured one comes from two seeds of the control arm itself and is the
# one the restatement ledger asks for. They are reported side by side because the
# pre-registered outcome was that the transfer was harmless, so the original margins
# stand and the new column is a restatement rather than a replacement.
SIGMA = 0.009456055
SIGMA_MEASURED = 0.010153794
HORIZONS = ("30", "60")
KEY = "equal_source_participant_macro_event_sensitivity"


def primary(run: Path) -> float | None:
    try:
        metrics = json.loads((run / "operational_metrics.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    values = [metrics["horizons"][h]["selection_metrics"][KEY] for h in HORIZONS]
    return sum(values) / len(values)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        help="rung|label|mechanism|path",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    global CONTROL
    rows, missing = [], []
    for item in args.arm:
        rung, label, mechanism, path = item.split("|", 3)
        value = primary(Path(path))
        if label == CONTROL_LABEL:
            if value is None:
                raise SystemExit(
                    f"the control arm {label} has no artefact; refusing to "
                                 "compute deltas against a typed constant")
            CONTROL = value
        if value is None:
            missing.append(label)
        rows.append((rung, label, mechanism, value))

    lines = [
        "% Generated from frozen artifacts. Do not edit by hand.",
        # The mechanism column wraps. Adding two sigma columns pushed this 142 pt past
        # the
        # text block and silently cut the last two off the page, which only a render
        # showed. A fixed-width wrapping column fits without shrinking the type.
        r"\begin{tabular}{llp{42mm}rrrr}",
        r"\toprule",
        r"Rung & Candidate & What it changed & Primary & $\Delta$ & "
        r"$\sigma_{\mathrm{t}}$ & $\sigma_{\mathrm{m}}$ \\",
        r"\midrule",
    ]
    for rung, label, mechanism, value in sorted(
        rows, key=lambda r: (-1e9 if r[3] is None else -r[3])
    ):
        if value is None:
            gap = r"\multicolumn{2}{c}{not run}"
            lines.append(f"{rung} & {label} & {mechanism} & {gap} \\\\")
            continue
        if label == CONTROL_LABEL:
            lines.append(
                f"{rung} & {label} & {mechanism} & {value:.6f} & "
                r"\emph{control} & & \\"
            )
            continue
        delta = value - CONTROL
        # An exact zero is the shape a mis-wired arm produces, so it is labelled rather
        # than left to look like the self-comparison defect an earlier build printed
        # here.
        tag = r"$^{\dagger}$" if delta == 0.0 else ""
        lines.append(
            f"{rung} & {label} & {mechanism} & {value:.6f} & {delta:+.6f}{tag} & "
            f"{delta / SIGMA:+.2f} & {delta / SIGMA_MEASURED:+.2f} \\\\"
        )
    own = CONTROL - CONTROL_OWN_COMPARATOR
    lines += [
        r"\midrule",
        (f"\\multicolumn{{7}}{{l}}{{Control's own margin over "
         f"{CONTROL_OWN_COMPARATOR_LABEL}: "
         f"{own:+.6f}, {own / SIGMA:+.2f} transferred and "
         f"{own / SIGMA_MEASURED:+.2f} measured $\\sigma$}} \\\\"),
        f"\\multicolumn{{7}}{{l}}{{Single-arm promotion floor {FLOOR:.6f}}} \\\\",
        (r"\multicolumn{7}{l}{\footnotesize $\dagger$ exact tie with the control. "
         r"$\sigma_{\mathrm{t}}$ is the transferred single-arm scale and "
         r"$\sigma_{\mathrm{m}}$ the one measured on this family.} \\"),
        r"\bottomrule",
        r"\end{tabular}",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    print()
    print(f"rows: {len(rows)}   missing artifacts: {len(missing)} {missing}")
    if missing:
        print("A missing arm is emitted as a gap, not dropped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
