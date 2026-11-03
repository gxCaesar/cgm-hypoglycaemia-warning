"""Collect every number the manuscript figures draw, with the file each came from.

A figure that re-parses the evidence tree itself cannot be audited, because nobody can
see which of several similarly-named artifacts it actually read. This writes one
`figure-data.json` in which every value carries `source`, a repository-relative path.

Read-only. Touches no cohort shard and trains nothing.
"""

from __future__ import annotations

import argparse
import glob
import json
import os


def _primary(payload):
    value = payload.get("primary_equal_horizon_mean_event_sensitivity")
    if isinstance(value, dict):
        return value.get("global")
    return value


def _load(path):
    with open(path) as handle:
        return json.load(handle)


def _rel(path, root):
    return os.path.relpath(path, root)


def build(v15_root: str, m6_root: str, repo_root: str) -> dict:
    out: dict = {"panels": {}}

    # ---- headroom: the two oracles against the deployed leaders -------------------
    headroom = {}
    for key, rel in (
        ("oracle_label", "analysis-oracle/oracle_label.json"),
        ("oracle_future_glucose", "analysis-oracle/oracle_future_glucose.json"),
    ):
        path = os.path.join(v15_root, rel)
        payload = _load(path)
        headroom[key] = {
            "primary": _primary(payload),
            "alarms_30": payload["30"]["global"]["alarm_count"],
            "alarms_60": payload["60"]["global"]["alarm_count"],
            "source": _rel(path, repo_root),
        }

    # ---- the campaign frame: nine v1.5 arms --------------------------------------
    arms = {}
    for path in sorted(glob.glob(os.path.join(v15_root,
        "l20-run-results/runs/*/operational_metrics.json"))):
        arm = path.split(os.sep)[-2]
        payload = _load(path)
        arms[arm] = {"primary": _primary(payload), "source": _rel(path, repo_root),
                     "family": "v1.5 campaign"}

    # ---- the earlier families, rescored on the SAME eligible frame ----------------
    for path in sorted(glob.glob(os.path.join(v15_root,
        "analysis-replication/*.json"))):
        arm = "old_" + os.path.basename(path)[:-5]
        payload = _load(path)
        arms[arm] = {"primary": _primary(payload), "source": _rel(path, repo_root),
                     "family": "earlier campaigns"}
    out["panels"]["eliminations"] = arms
    for name in ("persistence_seed17", "balanced_lstm_erm_seed17"):
        if name in arms:
            headroom[name] = dict(arms[name])
    out["panels"]["headroom"] = headroom

    # ---- the alarm-budget sweep ---------------------------------------------------
    sweep: dict = {}
    for path in sorted(glob.glob(os.path.join(v15_root, "analysis-fapd-sweep/*.json"))):
        base = os.path.basename(path)[:-5]
        if "-fapd" not in base:
            continue
        arm, budget = base.rsplit("-fapd", 1)
        payload = _load(path)
        sweep.setdefault(arm, {})[float(budget)] = {
            "primary": _primary(payload),
            "sens_30": payload["30"]["global"]["sensitivity"],
            "sens_60": payload["60"]["global"]["sensitivity"],
            "alarms_30": payload["30"]["global"]["alarm_count"],
            "source": _rel(path, repo_root),
        }
    out["panels"]["budget_sweep"] = sweep

    # ---- the seed scale, measured separately AT EACH BUDGET ------------------------
    # Seed sigma is budget-dependent, so a margin must never be divided by the 0.5
    # figure without checking which budget it was measured at.
    import statistics
    from collections import defaultdict

    per_budget = defaultdict(dict)
    for path in sorted(glob.glob(os.path.join(v15_root,
        "analysis-sigma-sweep/tcn-seed-*.json"))):
        base = os.path.basename(path)[:-5]
        seed = base.split("-")[2]
        budget = float(base.split("fapd")[1])
        per_budget[budget][seed] = {"primary": _primary(_load(path)),
                                    "source": _rel(path, repo_root)}
    sigma = {}
    for budget, seeds in per_budget.items():
        values = [v["primary"] for v in seeds.values()]
        sigma[budget] = {
            "sigma": statistics.stdev(values) if len(values) > 1 else None,
            "n_seeds": len(values),
            "seeds": sorted(seeds),
            "sources": [v["source"] for v in seeds.values()],
        }
    out["panels"]["seed_sigma_by_budget"] = sigma

    # ---- the composition, and how its members divide the labour --------------------
    comp_path = os.path.join(m6_root, "analysis/combination-primary-replication.json")
    seed_path = os.path.join(m6_root, "analysis/seed-combination.json")
    dis_path = os.path.join(m6_root, "analysis/disagreement2.json")
    weight_path = os.path.join(m6_root, "analysis/global-weight.json")
    out["panels"]["composition"] = {
        "replication": {"cells": _load(comp_path)["cells"], "source": _rel(comp_path,
            repo_root)},
        "seed_family": {"cells": _load(seed_path)["cells"], "source": _rel(seed_path,
            repo_root)},
        "weight": {**_load(weight_path), "source": _rel(weight_path, repo_root)},
    }
    out["panels"]["disagreement"] = {**_load(dis_path), "source": _rel(dis_path,
        repo_root)}

    # ---- the eligibility-cut curve, and where the estimand refuses -----------------
    cuts: dict = {}
    for path in sorted(glob.glob(os.path.join(v15_root, "analysis-cutcurve/*.json"))):
        base = os.path.basename(path)[:-5]
        arm, cut = base.rsplit("_cut", 1)
        cuts.setdefault(arm, {})[int(cut)] = {
            "primary": _primary(_load(path)), "source": _rel(path, repo_root)}
    refusals = {}
    for path in sorted(glob.glob(os.path.join(v15_root, "analysis-cutcurve/*.log"))):
        base = os.path.basename(path)[:-4]
        arm, cut = base.rsplit("_cut", 1)
        if os.path.exists(path[:-4] + ".json"):
            continue
        with open(path) as handle:
            tail = handle.read().strip().splitlines()[-1]
        refusals.setdefault(arm, {})[int(cut)] = {
            "refusal": tail.strip(), "source": _rel(path, repo_root)}
    out["panels"]["cut_curve"] = {"measured": cuts, "refused": refusals}

    # ---- the euglycaemic restriction: the reversal ----------------------------------
    eug = {}
    for path in sorted(glob.glob(os.path.join(v15_root,
        "analysis-euglycaemic/*.json"))):
        base = os.path.basename(path)[:-5]
        if base.startswith(("GATE", "MATCHED", "ROBUSTNESS")):
            continue
        eug[base] = {"primary": _primary(_load(path)), "source": _rel(path, repo_root)}
    out["panels"]["euglycaemic"] = eug
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    exp = os.path.join(args.repo_root, "experiments")
    data = build(
        v15_root=os.path.join(exp, "20260817-glycoshift-v1-5-baseline-closure-v1"),
        m6_root=os.path.join(exp, "20260827-glycoshift-m6-anchor-aware-v1"),
        repo_root=args.repo_root,
    )
    with open(args.output, "w") as handle:
        json.dump(data, handle, indent=1, sort_keys=True)
    for panel, payload in data["panels"].items():
        print(f"  {panel:16s} {len(payload)} entries")
    print(f"written {args.output}")


if __name__ == "__main__":
    main()
