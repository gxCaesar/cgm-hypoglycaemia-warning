#!/usr/bin/env python3
'Assemble seed variance figure input.'
















import argparse
import glob
import json
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--run", action="append", required=True, help="arm_id=path/to/run_dir")
ap.add_argument("--output", required=True, type=Path)
args = ap.parse_args()

args.output.mkdir(parents=True, exist_ok=True)
ranking = []
for spec in args.run:
    arm_id, _, pattern = spec.partition("=")
    hits = glob.glob(pattern)
    if len(hits) != 1:
        raise SystemExit(f"{arm_id}: {pattern} matched {len(hits)} directories")
    run = Path(hits[0])
    op = json.loads((run / "operational_metrics.json").read_text())
    proxy = json.loads((run / "selection_operational_proxy.json").read_text())

    if op["evaluation_partition"] != "selection":
        raise SystemExit(f"{arm_id}: scored on {op['evaluation_partition']}")
    if abs(op["nominal_fapd"] - 0.5) > 1e-12:
        raise SystemExit(f"{arm_id}: nominal budget {op['nominal_fapd']}")

    horizons = {}
    for h in ("30", "60"):
        frontier = proxy["horizons"][h].get("frontier")
        if not frontier:
            raise SystemExit(f"{arm_id} horizon {h}: no frontier points in the proxy")
        horizons[h] = {
            "selection_budget_frontier": {
                "frontier": frontier,
                "normalized_auc": proxy["horizons"][h]["normalized_auc"],
            },
            "selection_metrics": op["horizons"][h]["selection_metrics"],
        }

    payload = {
        "arm_id": arm_id,
        "evaluator": "operational_metrics.v1.3",
        "nominal_fapd": op["nominal_fapd"],
        "mean_severe_event_sensitivity":
            op["primary_equal_horizon_mean_event_sensitivity"],
        "mean_achieved_fapd": op["equal_horizon_mean_achieved_fapd"],
        "horizons": horizons,
        "reportability": "frozen_budget_frontier_proxy",
        "run_dir": str(run),
    }
    (args.output / f"{arm_id}.json").write_text(json.dumps(payload, indent=1))
    ranking.append({
        "arm_id": arm_id,
        "mean_severe_event_sensitivity": payload["mean_severe_event_sensitivity"],
        "mean_achieved_fapd": payload["mean_achieved_fapd"],
        "budget_frontier_auc": {
            h: horizons[h]["selection_budget_frontier"]["normalized_auc"]
            for h in ("30", "60")},
    })

ranking.sort(key=lambda r: r["mean_severe_event_sensitivity"], reverse=True)
(args.output / "summary.json").write_text(json.dumps({
    "evaluator": "operational_metrics.v1.3",
    "nominal_fapd": 0.5,
    "reportability": "frozen_budget_frontier_proxy",
    "ranking": ranking,
}, indent=1))
print(f"wrote {len(ranking)} arms to {args.output}")
for r in ranking:
    print(f"  {r['arm_id']:<28}{r['mean_severe_event_sensitivity']:.9f}"
          f"   FAPD {r['mean_achieved_fapd']:.4f}")
