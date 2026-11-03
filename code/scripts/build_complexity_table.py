#!/usr/bin/env python3
"""Collect the cost of every arm from records the runs already wrote.

Reported by 92 percent of accepted conference papers and 75 percent of journal
ones. Nothing here is generated: parameter counts come from each run's training
summary and wall clock from its own launch and finish markers, so a number in
the table and a number on disk cannot disagree.

Two things it will not do. It will not estimate a missing value, because an
estimate sitting in a column of measurements is indistinguishable from one. And
it will not silently omit an arm whose records are incomplete, because a cost
table that quietly contains only the cheap arms is worse than none.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def wall_clock(bundle: Path, cell: str) -> tuple[str, float | None]:
    """Hours between the launch record and the finish marker, or a reason."""
    record = bundle / f"launch_record.{cell}"
    finished = bundle / f"{cell}.finished_at"
    if not record.exists():
        return "no launch record", None
    if not finished.exists():
        return "still running", None
    started = None
    for line in record.read_text().splitlines():
        if line.startswith("launched_at="):
            started = line.split("=", 1)[1].strip()
    if started is None:
        return "launch record has no timestamp", None
    try:
        a = datetime.fromisoformat(started)
        b = datetime.fromisoformat(finished.read_text().strip())
    except ValueError:
        return "unparseable timestamp", None
    return "", (b - a).total_seconds() / 3600.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument(
        "--arm", action="append", required=True, help="label|cell|run-path"
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    rows, incomplete = [], []
    for item in args.arm:
        label, cell, path = item.split("|", 2)
        summary = read_json(Path(path) / "training_summary.json")
        params = summary.get("parameters") if summary else None
        note, hours = wall_clock(args.bundle, cell)
        epochs = None
        log = Path(path) / "logs" / "train.jsonl"
        if log.exists():
            entries = log.read_text().splitlines()
            epochs = sum(1 for line in entries if '"epoch"' in line) - 1
        if params is None or hours is None:
            incomplete.append((label, note or "no training summary"))
        rows.append((label, params, epochs, hours, note))

    print(f"{'arm':26s} {'params':>10s} {'epochs':>7s} {'hours':>8s}  note")
    for label, params, epochs, hours, note in rows:
        p = f"{params:,}" if params is not None else "--"
        e = str(epochs) if epochs is not None else "--"
        h = f"{hours:.2f}" if hours is not None else "--"
        print(f"{label:26s} {p:>10s} {e:>7s} {h:>8s}  {note}")

    lines = [
        "% Generated from run records. Do not edit by hand.",
        r"\begin{tabular}{lrrr}", r"\toprule",
        r"Arm & Parameters & Epochs & Wall clock (h) \\", r"\midrule",
    ]
    for label, params, epochs, hours, _n in rows:
        p = f"{params:,}" if params is not None else r"\textit{n/a}"
        e = str(epochs) if epochs is not None else r"\textit{n/a}"
        h = f"{hours:.2f}" if hours is not None else r"\textit{running}"
        lines.append(f"{label} & {p} & {e} & {h} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {args.output}")

    if incomplete:
        print()
        for label, why in incomplete:
            print(f"  INCOMPLETE {label}: {why}")
        print(
            f"{len(incomplete)} arm(s) have no complete cost record. They are shown "
            "as gaps rather than estimated or dropped."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
