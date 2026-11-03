#!/usr/bin/env python3
"""Single-seed paired participant bootstrap for the frozen contrast.

**This does NOT discharge the promotion criterion.** The active run's frozen
protocol specifies a MULTISEED source-stratified paired event-positive-participant
bootstrap over model_seed_order [17, 23, 42, 71, 101], and the multiseed function
refuses tables that do not exactly cover that order. Only seed 17 exists, so what
follows is the k=1 case of the same resampling and is reported as descriptive.

It is also computed PER HORIZON. The primary endpoint is the mean over thirty and
sixty minutes, and a participant table carries one horizon, so two intervals are
produced where the criterion asks for one. Both facts are written into the output
so a reader cannot mistake this for the registered quantity.

Replicates, bootstrap seed and interval come from the frozen protocol rather than
being chosen here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from composition_frozen_evaluation import (  # noqa: E402
    CALIBRATION,
    HORIZONS,
    JOIN_KEY,
    calibration_rank,
    read_member_parts,
)

from glycoshift.operational_metrics import (  # noqa: E402
    OperationalEndpointSpec,
    fit_operational_alarm_policy,
    participant_alarm_table,
    source_stratified_paired_operational_bootstrap_difference,
)


def _spec(horizon: int, source_order: tuple[str, ...]) -> OperationalEndpointSpec:
    return OperationalEndpointSpec(
        score_column=f"severe_hypo_{horizon}_probability",
        label_column=f"severe_hypo_{horizon}_label",
        event_id_column=f"severe_hypo_{horizon}_event_id",
        event_onset_column=f"severe_hypo_{horizon}_event_onset",
        active_column="severe_hypo_currently_active",
        horizon_minutes=horizon,
        source_order=source_order,
    )


def _scored(frame: pd.DataFrame, members: list[str], calibration_mask) -> pd.DataFrame:
    """Rank-average the named members, exactly as the frozen composition does."""
    out = frame.copy()
    for horizon in HORIZONS:
        ranks = []
        for name in members:
            values = pd.to_numeric(
                frame[f"member__{name}__{horizon}"], errors="coerce"
            ).to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise SystemExit(f"{name} horizon {horizon} carries non-finite values")
            ranks.append(calibration_rank(values, calibration_mask))
        out[f"severe_hypo_{horizon}_probability"] = np.mean(np.vstack(ranks), axis=0)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--member", action="append", required=True, help="name=path[,path]")
    p.add_argument("--score", action="append", default=[],
                   help="name=severe|hypo|neg_cgm_last")
    p.add_argument("--candidate", required=True, help="comma-separated member names")
    p.add_argument("--reference", action="append", required=True,
                   help="comma-separated member names, repeatable")
    p.add_argument("--protocol", required=True, type=Path,
                   help="frozen benchmark-protocol.json with the uncertainty block")
    p.add_argument("--evaluate-partition", default="selection")
    p.add_argument("--nominal-fapd", type=float, default=0.5)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args(argv)

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    unc = protocol["uncertainty"]
    replicates = int(unc["replicates"])
    bootstrap_seed = int(unc["bootstrap_seed"])
    seed_order = [int(s) for s in unc["model_seed_order"]]

    scores = dict(item.split("=", 1) for item in args.score)
    frame = None
    for item in args.member:
        name, _, spec = item.partition("=")
        part = read_member_parts([Path(x) for x in spec.split(",") if x], HORIZONS,
                                 scores.get(name, "severe"))
        part = part.rename(columns={
            f"severe_hypo_{h}_probability": f"member__{name}__{h}" for h in HORIZONS})
        if frame is None:
            frame = part
        else:
            drop = [c for c in part.columns if c in frame.columns and c not in JOIN_KEY]
            frame = frame.merge(part.drop(columns=drop), on=JOIN_KEY, how="inner",
                                validate="one_to_one")
        print(f"  joined {name}, {len(frame):,} rows", flush=True)
    assert frame is not None

    calibration_mask = (frame["partition"] == CALIBRATION).to_numpy()
    evaluate_mask = (frame["partition"] == args.evaluate_partition).to_numpy()
    order = tuple(sorted(
        frame.loc[evaluate_mask, "source_id"].dropna().unique().tolist()))
    cal_order = tuple(sorted(
        frame.loc[calibration_mask, "source_id"].dropna().unique().tolist()))

    def arm_tables(members: list[str]) -> dict[str, pd.DataFrame]:
        scored = _scored(frame, members, calibration_mask)
        tables = {}
        for horizon in HORIZONS:
            policy = fit_operational_alarm_policy(
                scored.loc[calibration_mask], spec=_spec(int(horizon), cal_order),
                nominal_fapd=args.nominal_fapd)
            tables[horizon] = participant_alarm_table(
                scored.loc[evaluate_mask], spec=_spec(int(horizon), order),
                threshold=policy.threshold)
        return tables

    candidate = [m.strip() for m in args.candidate.split(",")]
    cand_tables = arm_tables(candidate)
    report: dict[str, object] = {
        "NOT_THE_REGISTERED_QUANTITY": (
            "the frozen protocol specifies a MULTISEED bootstrap over model seeds "
            f"{seed_order}; only seed 17 exists, so this is the k=1 case and is "
            "descriptive. It is also computed per horizon, while the primary endpoint "
            "is the mean over horizons."),
        "protocol_id": protocol.get("protocol_id"),
        "replicates": replicates, "bootstrap_seed": bootstrap_seed,
        "confidence_interval": float(unc["confidence_interval"]),
        "model_seeds_required": seed_order, "model_seeds_available": [17],
        "candidate": candidate, "evaluate_partition": args.evaluate_partition,
        "source_order": list(order), "contrasts": {},
    }
    for ref_spec in args.reference:
        reference = [m.strip() for m in ref_spec.split(",")]
        ref_tables = arm_tables(reference)
        block = {}
        for horizon in HORIZONS:
            block[horizon] = source_stratified_paired_operational_bootstrap_difference(
                cand_tables[horizon], ref_tables[horizon],
                source_order=order, replicates=replicates, seed=bootstrap_seed)
            b = block[horizon]
            print(f"  {'+'.join(candidate)} vs {'+'.join(reference)} @{horizon}min: "
                  f"{ {k: v for k, v in b.items()} }", flush=True)
        report["contrasts"]["+".join(reference)] = block

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1, sort_keys=True))
    print(f"  wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
