"""Generate the supplement's per-source tables from frozen run artefacts.

The main text reports an equal-source participant-macro mean. That aggregate hides
which sources carry it, and this project has already been bitten once by exactly that:
four sources holding two or three selection participants carried two thirds of a
headline difference, and one source with two participants and eleven events was worth
0.65 of it on its own. A supplement that shows only the aggregate cannot be audited for
that, so every table here is per source and carries the count that entered.

Read-only. Opens each arm's own `operational_metrics.json` and trains nothing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HORIZONS = ("30", "60")
KEY_SENS = "participant_macro_event_sensitivity"
KEY_FAPD = "participant_macro_false_alarms_per_patient_day"


def load(run: Path) -> dict | None:
    path = run / "operational_metrics.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def escape(name: str) -> str:
    return name.replace("_", r"\_").replace("&", r"\&")


def cohort_table(payload: dict) -> list[str]:
    """S1. What each source contributes, before any arm is scored on it."""
    rows = []
    for horizon in HORIZONS:
        rows.append(payload["horizons"][horizon]["selection_metrics"])
    sources = sorted(rows[0]["per_source"])
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r" & & & \multicolumn{2}{c}{severe events} "
        r"& \multicolumn{2}{c}{event-positive participants} \\",
        r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"source & participants & participant-days & 30 min & 60 min "
        r"& 30 min & 60 min \\",
        r"\midrule",
    ]
    for source in sources:
        a, b = (row["per_source"][source] for row in rows)
        lines.append(
            f"{escape(source)} & {a['participant_count']} & {a['patient_days']:,} & "
            f"{a['event_count']:,} & {b['event_count']:,} & "
            f"{a['event_positive_participant_count']} & "
            f"{b['event_positive_participant_count']} \\\\"
        )
    a, b = rows
    lines += [
        r"\midrule",
        f"all {a['source_count']} sources & {a['participant_count']} & "
        f"{a['patient_days']:,} & {a['event_count']:,} & {b['event_count']:,} & "
        f"{a['event_positive_participant_count']} & "
        f"{b['event_positive_participant_count']} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return lines


def per_source_arm_table(arms: dict[str, dict], horizon: str) -> list[str]:
    """S2/S3. Every arm's sensitivity per source, with the source's own denominator."""
    first = next(iter(arms.values()))
    sources = sorted(first["horizons"][horizon]["selection_metrics"]["per_source"])
    header = " & ".join(escape(name) for name in arms)
    lines = [
        r"\begin{tabular}{lr" + "r" * len(arms) + "}",
        r"\toprule",
        f"source & events & {header} \\\\",
        r"\midrule",
    ]
    for source in sources:
        cells = []
        n_events = None
        for payload in arms.values():
            block = payload["horizons"][horizon]["selection_metrics"]["per_source"][
                source]
            n_events = block["event_count"]
            cells.append(f"{block[KEY_SENS]:.3f}")
        lines.append(f"{escape(source)} & {n_events:,} & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    macro = []
    for payload in arms.values():
        block = payload["horizons"][horizon]["selection_metrics"]
        macro.append(f"{block['equal_source_participant_macro_event_sensitivity']:.3f}")
    lines += [
        r"equal-source macro & & " + " & ".join(macro) + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return lines


def decomposition(arms: dict[str, dict], left: str, right: str, horizon: str) -> dict:
    """Per-source contributions to the macro difference between two arms.

    The supplement quotes these and they existed nowhere: a difference of two stored
    per-source values, divided by the source count, is not itself stored. Each entry
    carries the rounded form the prose uses as well as the full value, because a float's
    repr contains its own rounding only when the next digit does not round up.
    """
    a = arms[left]["horizons"][horizon]["selection_metrics"]
    b = arms[right]["horizons"][horizon]["selection_metrics"]
    n = a["source_count"]
    rows, total = [], 0.0
    for source in sorted(a["per_source"]):
        left_v = a["per_source"][source][KEY_SENS]
        right_v = b["per_source"][source][KEY_SENS]
        contribution = (left_v - right_v) / n
        total += contribution
        rows.append({
            "source": source,
            "events": a["per_source"][source]["event_count"],
            "event_positive_participants":
                a["per_source"][source]["event_positive_participant_count"],
            left: left_v, right: right_v,
            "difference": left_v - right_v,
            "contribution_to_macro": contribution,
            "contribution_6dp": f"{contribution:.6f}",
        })
    gap = (a["equal_source_participant_macro_event_sensitivity"]
           - b["equal_source_participant_macro_event_sensitivity"])
    return {
        "horizon": horizon, "left": left, "right": right, "source_count": n,
        "macro_left": a["equal_source_participant_macro_event_sensitivity"],
        "macro_right": b["equal_source_participant_macro_event_sensitivity"],
        "macro_difference": gap,
        "macro_difference_6dp": f"{gap:.6f}",
        "sum_of_contributions": total,
        "identity_holds": abs(total - gap) < 1e-12,
        "per_source": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", required=True, type=Path)
    ap.add_argument("--arm", action="append", required=True, help="label=dirname")
    ap.add_argument("--outdir", required=True, type=Path)
    args = ap.parse_args()

    arms, missing = {}, []
    for item in args.arm:
        label, _, dirname = item.partition("=")
        payload = load(args.runs_root / dirname)
        if payload is None:
            missing.append(label)
            continue
        arms[label] = payload
    if not arms:
        raise SystemExit("no arm artefacts found")

    args.outdir.mkdir(parents=True, exist_ok=True)
    reference = next(iter(arms.values()))
    (args.outdir / "table-s1-cohort.tex").write_text(
        "\n".join(cohort_table(reference)) + "\n")
    for horizon in HORIZONS:
        (args.outdir / f"table-s2-per-source-{horizon}.tex").write_text(
            "\n".join(per_source_arm_table(arms, horizon)) + "\n")

    if "threshold" in arms and "TCN" in arms:
        decomp = {h: decomposition(arms, "threshold", "TCN", h) for h in HORIZONS}
        (args.outdir / "per-source-decomposition.json").write_text(
            json.dumps(decomp, indent=1, sort_keys=True))
        for h, d in decomp.items():
            print(f"  {h} min macro difference {d['macro_difference_6dp']}, "
                  f"contributions sum to it: {d['identity_holds']}")

    print(f"arms: {len(arms)}   missing: {missing}")
    block = reference["horizons"]["30"]["selection_metrics"]
    print(
        f"sources {block['source_count']}, participants {block['participant_count']}, "
          f"participant-days {block['patient_days']:,}, "
          f"30-min severe events {block['event_count']:,}")
    smallest = min(block["per_source"].items(),
                   key=lambda kv: kv[1]["event_positive_participant_count"])
    print(f"thinnest source at 30 min: {smallest[0]} with "
          f"{smallest[1]['event_positive_participant_count']} event-positive "
          f"participant(s) and {smallest[1]['event_count']} events")
    print(
        f"wrote {len(list(args.outdir.glob('table-s*.tex')))} tables to {args.outdir}")


if __name__ == "__main__":
    main()
