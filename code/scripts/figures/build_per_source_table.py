#!/usr/bin/env python3
'Build per source table.'

















from __future__ import annotations

import argparse
import json
from pathlib import Path

HORIZONS = ("30", "60")
SHARED = ("participant_count", "event_positive_participant_count", "event_count")


def thousands(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--control", default="persist")
    parser.add_argument("--member", default="lstm")
    parser.add_argument("--pair", default="persist,lstm")
    arguments = parser.parse_args(argv)

    cells = json.loads(arguments.matrix.read_text())["cells"]
    arms = [("Control", arguments.control), ("LSTM", arguments.member),
            ("Composition", arguments.pair)]
    for _label, key in arms:
        if key not in cells:
            raise SystemExit(f"the matrix has no cell {key!r}")

    per = {key: {h: cells[key][h]["per_source"] for h in HORIZONS}
           for _l, key in arms}
    sources = sorted(per[arguments.pair]["30"])

    # One frame, so the denominators cannot differ between arms or between horizons at
    # the participant level. Checked rather than trusted.
    for source in sources:
        for field in SHARED:
            seen = {
                (key, h): per[key][h][source][field]
                for _l, key in arms for h in HORIZONS
                if not (field == "event_count" and h == "60")
            }
            if len(set(seen.values())) != 1:
                raise SystemExit(
                    f"{source}: {field} differs across arms or horizons ({seen}), so "
                    "the three arms are not scored on one frame and a shared column "
                    "would be wrong"
                )

    rows = []
    for source in sources:
        block = per[arguments.pair]["30"][source]
        cells_out = [
            source.replace("&", r"\&"),
            str(block["participant_count"]),
            str(block["event_positive_participant_count"]),
            thousands(block["event_count"]),
            thousands(per[arguments.pair]["60"][source]["event_count"]),
        ]
        for _label, key in arms:
            for h in HORIZONS:
                v = per[key][h][source]["participant_macro_event_sensitivity"]
                cells_out.append(f"${v:.3f}$")
        rows.append(" & ".join(cells_out) + r" \\")

    header = (r"Source & \multicolumn{1}{c}{$n$} & \multicolumn{1}{c}{$n^{+}$} & "
              r"\multicolumn{2}{c}{Events} & \multicolumn{2}{c}{Control} & "
              r"\multicolumn{2}{c}{LSTM} & \multicolumn{2}{c}{Composition} \\")
    sub = (r" & & & 30 min & 60 min & 30 min & 60 min & 30 min & 60 min & "
           r"30 min & 60 min \\")
    # Reading the rendered table showed something the aggregate hides: the composition
    # is the best of the three in almost every source, and there is exactly one source
    # where it is not. Counting it here rather than in prose means the sentence cannot
    # drift away from the table it describes.
    def best(h: str, source: str) -> str:
        ranked = sorted(arms, key=lambda a:
                        per[a[1]][h][source]["participant_macro_event_sensitivity"])
        top = per[ranked[-1][1]][h][source]["participant_macro_event_sensitivity"]
        runner = per[ranked[-2][1]][h][source]["participant_macro_event_sensitivity"]
        return "tie" if top == runner else ranked[-1][0]

    wins = {h: sum(1 for s in sources if best(h, s) == "Composition") for h in HORIZONS}
    small = sum(1 for s in sources
                if per[arguments.pair]["30"][s]["participant_count"] <= 3)
    losses = {h: sorted(s for s in sources if best(h, s) not in ("Composition", "tie"))
              for h in HORIZONS}
    consistency = (
        f"{small} strata contain three or fewer scored participants. "
        f"The composition strictly exceeds both comparators in {wins['30']} sources "
        f"at 30 minutes and {wins['60']} at 60 minutes. "
    )
    for h in HORIZONS:
        tied_control = [
            s for s in sources if best(h, s) == "tie" and
            per[arguments.pair][h][s]["participant_macro_event_sensitivity"] ==
            per[arguments.control][h][s]["participant_macro_event_sensitivity"]
        ]
        if tied_control:
            tied_sources = ", ".join(tied_control)
            consistency += f"It ties the control in {tied_sources} at {h} minutes. "
    if losses["30"] == losses["60"] and len(losses["30"]) == 1:
        lost = losses["30"][0]
        # The clause below describes that stratum, so it is checked against it. A
        # rescoring that moved the exception elsewhere would otherwise leave a true
        # sentence attached to the wrong source.
        control_key = arguments.control
        control_at = [per[control_key][h][lost]["participant_macro_event_sensitivity"]
                      for h in HORIZONS]
        n_at = per[control_key]["30"][lost]["participant_count"]
        if any(v != 1.0 for v in control_at):
            raise SystemExit(
                f"the caption says the control detects every event in {lost}; "
                f"it scores {control_at}")
        consistency += (
            f"It is lower only in {lost} at both horizons, where the control detects "
            f"every event across {n_at} scored participants. ")
    else:
        consistency += (
            f"It is beaten in {losses['30']} at 30 minutes and {losses['60']} at 60. ")

    macro = ["Equal-source macro", "--", "--", "--", "--"]
    for _label, key in arms:
        for h in HORIZONS:
            macro.append(f"${cells[key][h]['sensitivity']:.3f}$")

    body = "\n".join([
        "% Generated by code/scripts/figures/build_per_source_table.py. "
        "Do not edit by hand.",
        r"\begin{table}[t]",
        r"\centering",
        r"{\footnotesize\setlength{\tabcolsep}{3.0pt}",
        r"\begin{tabular}{lrrrrrrrrrr}",
        r"\toprule",
        header,
        r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}",
        sub,
        r"\midrule",
        *rows,
        r"\midrule",
        " & ".join(macro) + r" \\",
        r"\bottomrule",
        r"\end{tabular}}",
        r"\caption{Level 2 event sensitivity by development source on the unrestricted "
        r"selection partition. Columns $n$ and $n^{+}$ report scored participants and "
        r"participants experiencing at least one level 2 event, respectively, and are "
        r"shared by all three arms at both horizons. Sensitivity is the mean of "
        r"participant-level event sensitivities among the $n^{+}$ event-positive "
        r"participants in each source, rather than a pooled ratio of detected to total "
        r"events. Higher values are better. Event counts differ between horizons "
        r"because "
        r"the 60-minute horizon admits additional onsets. Control denotes the "
        r"current-glucose "
        r"rule, LSTM the source-balanced long short-term memory network, and "
        r"Composition "
        r"their equal-weight calibration-rank composition, using the registered "
        r"trained fit. "
        r"The frozen policy is fitted at a nominal 0.5 false alarms per "
        r"participant-day. "
        r"The bottom row averages the eleven source sensitivities equally. Its count "
        r"cells "
        r"are inapplicable and shown as en-dashes. "
        + consistency.rstrip() + "}",
        r"\label{tab:per_source}",
        r"\end{table}",
        "",
    ])
    arguments.output.write_text(body)
    print(f"wrote {arguments.output}  {len(sources)} sources x {len(arms)} arms x "
          f"{len(HORIZONS)} horizons; shared denominators verified; composition best "
          f"in {wins['30']}/{len(sources)} and {wins['60']}/{len(sources)} sources, "
          f"beaten in {losses['30']} and {losses['60']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
