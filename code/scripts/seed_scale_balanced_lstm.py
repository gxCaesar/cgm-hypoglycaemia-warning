#!/usr/bin/env python3
'Seed scale balanced lstm.'















import argparse
import glob
import json
import math
import statistics as st
from pathlib import Path

CHI2_975_025 = {1: (0.000982, 5.023886), 2: (0.050636, 7.377759),
                3: (0.215795, 9.348404), 4: (0.484419, 11.143287)}


def load(run_dir: Path) -> dict:
    op = json.loads((run_dir / "operational_metrics.json").read_text())
    ts = json.loads((run_dir / "training_summary.json").read_text())
    if op["evaluation_partition"] != "selection":
        raise SystemExit(
            f"{run_dir}: scored on {op['evaluation_partition']}, not selection")
    if op["threshold_partition"] != "calibration":
        raise SystemExit(f"{run_dir}: threshold fitted on {op['threshold_partition']}")
    if abs(op["nominal_fapd"] - 0.5) > 1e-12:
        raise SystemExit(f"{run_dir}: nominal budget {op['nominal_fapd']}, not 0.5")
    return {
        "primary": op["primary_equal_horizon_mean_event_sensitivity"],
        "fapd": op["equal_horizon_mean_achieved_fapd"],
        "h30": op["horizons"]["30"]["selection_metrics"][
            "equal_source_participant_macro_event_sensitivity"],
        "h60": op["horizons"]["60"]["selection_metrics"][
            "equal_source_participant_macro_event_sensitivity"],
        "epoch": ts["best_epoch"],
        "window_auprc": ts.get("best_selection_macro_window_auprc"),
    }


ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--run", action="append", required=True, help="seed=path/to/run_dir")
ap.add_argument("--recorded-sigma", type=float, default=0.009456055)
ap.add_argument("--margin", type=float, default=0.088355915,
                help="composition primary minus the registered promotion floor")
ap.add_argument("--output", type=Path)
args = ap.parse_args()

runs = {}
for spec in args.run:
    seed, _, path = spec.partition("=")
    hits = glob.glob(path)
    if len(hits) != 1:
        raise SystemExit(f"seed {seed}: {path} matched {len(hits)} directories")
    runs[int(seed)] = load(Path(hits[0]))

primary = [r["primary"] for r in runs.values()]
n = len(primary)
sd = st.stdev(primary)
lo_q, hi_q = CHI2_975_025[n - 1]
sd_lo, sd_hi = sd * math.sqrt((n - 1) / hi_q), sd * math.sqrt((n - 1) / lo_q)

print(f"balanced_lstm_erm, frozen primary, {n} fits differing only in training.seed\n")
print(f"{'seed':>6}{'primary':>16}{'h30':>13}{'h60':>13}{'FAPD':>9}{'epoch':>7}")
for s in sorted(runs):
    r = runs[s]
    print(f"{s:>6}{r['primary']:>16.9f}{r['h30']:>13.6f}{r['h60']:>13.6f}"
          f"{r['fapd']:>9.4f}{r['epoch']:>7}")
print(f"\n  mean  {st.mean(primary):.9f}")
print(f"  sd    {sd:.9f}   95% CI [{sd_lo:.6f}, {sd_hi:.6f}], {n-1} df")
print(f"  range {max(primary) - min(primary):.9f}")
print(f"  recorded sigma {args.recorded_sigma:.9f}  ->  measured is "
      f"{sd / args.recorded_sigma:.2f}x it")

win = [r["window_auprc"] for r in runs.values() if r["window_auprc"] is not None]
if len(win) > 1:
    wsd = st.stdev(win)
    print(f"\n  window AUPRC over {len(win)} fits: sd {wsd:.9f}, "
          f"range {max(win) - min(win):.9f}")
    print(f"  the reported endpoint is {sd / wsd:.1f}x noisier across seeds than the "
          f"window metric it replaced")

print(f"\nmargin {args.margin:.9f} divided by each candidate scale:")
for label, s in (("recorded, single arm", args.recorded_sigma),
                 ("measured here", sd),
                 ("measured, optimistic CI end", sd_lo),
                 ("measured, pessimistic CI end", sd_hi)):
    print(f"  {label:<30} {args.margin / s:6.2f} sigma")

if args.output:
    args.output.write_text(json.dumps({
        "runs": {str(s): runs[s] for s in sorted(runs)},
        "n": n, "mean": st.mean(primary), "sd": sd,
        "sd_ci95": [sd_lo, sd_hi], "range": max(primary) - min(primary),
        "recorded_sigma": args.recorded_sigma,
        "ratio_to_recorded": sd / args.recorded_sigma,
        "margin": args.margin,
        "margin_sigma_measured": args.margin / sd,
        "margin_sigma_recorded": args.margin / args.recorded_sigma,
    }, indent=1))
    print(f"\nwrote {args.output}")
