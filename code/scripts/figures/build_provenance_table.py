"""Per-arm provenance, collected from each run's own training summary.

Nothing is typed. Seeds, parameter counts, the selected epoch and the run's own
 self-reported external status all come from `model_summary.json`, and wall clock from
 the
 timestamps of artefacts the run wrote. An arm whose summary is missing is emitted as a
 gap
rather than dropped, because a provenance table that quietly contains only the arms with
tidy records is worse than none.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LABEL = {
    "severe-retarget-run": "M0 severe retarget",
    "m0-seed23-run": "M0, second seed",
    "c1-uniform-weight-run": "C1 uniform weight",
    "c3-baseline-memory-recovery-run": "C3r retrieval memory",
    "c5-endpoint-checkpoint-run": "C5 endpoint checkpoint",
    "anchor-w0p25-run": r"M6 anchor $w=0.25$",
    "anchor-w0p50-run": r"M6 anchor $w=0.50$",
    "anchor-w0p75-run": r"M6 anchor $w=0.75$",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summaries", required=True, type=Path)
    ap.add_argument("--wallclock", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    wall = json.loads(args.wallclock.read_text())

    lines = ["% Generated from run records. Do not edit by hand.",
             r"\begin{tabular}{llrrrl}", r"\toprule",
             r"arm & split / train seed & parameters & epoch & hours & external \\",
             r"\midrule"]
    missing, statuses = [], set()
    for run in sorted(LABEL):
        path = args.summaries / f"{run}.json"
        if not path.is_file():
            missing.append(run)
            lines.append(
                f"{LABEL[run]} & \\multicolumn{{5}}{{c}}{{summary missing}} \\\\")
            continue
        s = json.loads(path.read_text())
        status = str(s.get("status", "?"))
        statuses.add(status)
        short = "untouched" if status.endswith("external_untouched") else status
        lines.append(
            f"{LABEL[run]} & {s.get('split_seed')} / {s.get('training_seed')} & "
            f"{s.get('parameters'):,} & {s.get('best_epoch')} & "
            f"{wall[run]['hours']:.2f} & {short} \\\\")
    lines += [
        r"\midrule",
        (r"\multicolumn{6}{l}{\footnotesize Every arm reports the same "
         r"external status, "
         r"recorded by the run itself and not by us.} \\"),
        r"\bottomrule", r"\end{tabular}"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print()
    print(f"arms {len(LABEL)}  missing summaries {len(missing)}  "
          f"distinct external statuses {sorted(statuses)}")


if __name__ == "__main__":
    main()
