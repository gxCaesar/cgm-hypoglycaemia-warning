#!/usr/bin/env python3
"""Is the clinically meaningful endpoint saturated, or is there room above 0.521?

Pre-declared in `PREREG-headroom-on-the-clinically-meaningful-endpoint.md`.

Rank averaging is a measuring instrument here, not a proposed method. It asks
whether the information exists across arms at all. It fits no parameter, so it
cannot overfit calibration, and it is scale-free, which matters because the arms
come from two run families whose scores share no scale.

Ranks are computed over the pooled restricted frame, calibration and selection
together. That is transductive: it uses selection SCORES, never selection labels, to
normalise. For a headroom question that is the right instrument and the honest
framing is that it gives an **upper bound** on what the information could buy. If the
transductive version cannot beat the single best arm, a partition-honest version
certainly cannot.

Read-only. Opens completed arms' own prediction artifacts and trains nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from glycoshift.operational_metrics import (
    OperationalEndpointSpec,
    fit_operational_alarm_policy,
    operational_endpoint_metrics,
)

HORIZONS = (30, 60)
KEY = ["source_id", "participant_id", "context_end"]
BASE = [
    "partition",
    "patient_day",
    "cgm_last",
    "severe_hypo_currently_active",
]


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


def _read(
    path: Path,
    *,
    legacy: bool,
    with_labels: bool,
    with_frame_check: bool = False,
) -> pd.DataFrame:
    score = [
        (f"hypo_{h}_probability" if legacy else f"severe_hypo_{h}_probability")
        for h in HORIZONS
    ]
    columns = list(KEY) + score
    if with_frame_check:
        columns += ["partition", "severe_hypo_currently_active"]
    if with_labels:
        columns += BASE
        for h in HORIZONS:
            columns += [
                f"severe_hypo_{h}_label",
                f"severe_hypo_{h}_event_id",
                f"severe_hypo_{h}_event_onset",
            ]
    frame = pq.read_table(path, columns=sorted(set(columns))).to_pandas()
    if legacy:
        for h in HORIZONS:
            frame[f"severe_hypo_{h}_probability"] = frame[f"hypo_{h}_probability"]
            frame = frame.drop(columns=[f"hypo_{h}_probability"])
    for column in KEY:
        frame[column] = frame[column].astype(str)
    return frame


def _rank_within(frame: pd.DataFrame, column: str) -> np.ndarray:
    """AVERAGE rank in [0,1]. Scale free, and tied scores share one rank.

    The first version used `argsort` ordinal ranks, which give tied scores
    DISTINCT values. That breaks monotonicity: a rank threshold can split a tie
    group no raw-score threshold could, so a single-arm ensemble stopped
    reproducing that arm's own endpoint, returning 0.508358636 against the frozen
    0.521160563. The single-arm gate caught it, which is what it is for.
    `method="average"` keeps ties together and restores the monotone map.
    """

    values = pd.to_numeric(frame[column], errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise SystemExit(f"{column} carries non-finite values")
    ranks = values.rank(method="average").to_numpy(dtype=float)
    return (ranks - 1.0) / max(len(ranks) - 1, 1)


def _calibration_rank(
    frame: pd.DataFrame, column: str, calibration_mask: np.ndarray
) -> np.ndarray:
    """Rank against the CALIBRATION distribution alone, applied to every row.

    This is the deployable form. The transform is a function of the calibration
    scores only, so scoring one selection window never consults another, and the
    map is strictly increasing between distinct calibration scores rather than a
    step, which keeps the single-arm case an exact monotone reparameterisation.
    """

    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise SystemExit(f"{column} carries non-finite values")
    reference = np.sort(values[calibration_mask])
    if reference.size == 0:
        raise SystemExit("no calibration rows to fit the rank transform on")
    grid = np.linspace(0.0, 1.0, reference.size)
    return np.interp(values, reference, grid)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm", action="append", required=True,
        help="name=path[:legacy] ; legacy reads the untempered hypo head",
    )
    parser.add_argument("--combination", action="append", required=True,
                        help="comma-separated arm names")
    parser.add_argument(
        "--honest-ranks",
        action="store_true",
        help=(
            "Fit each arm's rank transform on CALIBRATION ONLY and map selection "
            "through it by interpolation. Without this the ranks are pooled, which "
            "is transductive and gives an upper bound. With it the combination is "
            "deployable: nothing at inference time uses another window's score."
        ),
    )
    parser.add_argument("--min-cgm-last", type=float, default=70.0)
    parser.add_argument("--nominal-fapd", type=float, default=0.5)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    specs: dict[str, tuple[Path, bool]] = {}
    for item in args.arm:
        name, _, rest = item.partition("=")
        path, _, flag = rest.partition(":")
        specs[name] = (Path(path), flag == "legacy")

    # The first arm supplies labels, events and the eligibility columns; every
    # other arm supplies only its score and must join one-to-one onto it.
    names = list(specs)
    base_name = names[0]
    base_path, base_legacy = specs[base_name]
    frame = _read(base_path, legacy=base_legacy, with_labels=True)
    frame = frame.rename(
        columns={f"severe_hypo_{h}_probability": f"{base_name}__{h}" for h in HORIZONS}
    )
    rows_before = len(frame)
    frame_checks: list[str] = []
    for name in names[1:]:
        path, legacy = specs[name]
        # The base arm supplies partition and eligibility for everyone. An
        # adversarial audit on 2026-08-22 found that only the JOIN KEY was
        # verified, so equal aggregate counts were the only evidence that the
        # arms shared an eligible frame -- and equal aggregates do not establish
        # identical rows. Read those two columns from every arm and compare them
        # rowwise. Premise: two arms scored on the same frame must agree on which
        # partition each row is in and whether its severe event is already active.
        other = _read(
            path, legacy=legacy, with_labels=False, with_frame_check=True
        ).rename(
            columns={f"severe_hypo_{h}_probability": f"{name}__{h}" for h in HORIZONS}
        )
        if len(other) != rows_before:
            raise SystemExit(f"{name} has {len(other)} rows, base has {rows_before}")
        frame = frame.merge(
            other, on=KEY, how="inner", validate="one_to_one",
            suffixes=("", f"__{name}_check"),
        )
        if len(frame) != rows_before:
            raise SystemExit(
                f"join with {name} was not total: {len(frame)} of {rows_before}"
            )
        for column in ("partition", "severe_hypo_currently_active"):
            mine = frame[f"{column}__{name}_check"]
            base = frame[column]
            if column == "severe_hypo_currently_active":
                mismatch = int(
                    (mine.astype(bool).to_numpy() != base.astype(bool).to_numpy()).sum()
                )
            else:
                mismatch = int(
                    (mine.astype(str).to_numpy() != base.astype(str).to_numpy()).sum()
                )
            if mismatch:
                raise SystemExit(
                    f"{name} disagrees with the base arm on {column} in "
                    f"{mismatch:,} rows; the arms are not on one frame"
                )
            frame = frame.drop(columns=[f"{column}__{name}_check"])
        frame_checks.append(name)
    print(f"join verified one-to-one and total over {rows_before:,} rows", flush=True)
    if frame_checks:
        print(
            "rowwise frame check passed for "
            + ", ".join(frame_checks)
            + " on partition and severe_hypo_currently_active",
            flush=True,
        )

    keep = pd.to_numeric(frame["cgm_last"], errors="coerce") >= args.min_cgm_last
    frame = frame.loc[keep].copy()
    print(f"cgm_last >= {args.min_cgm_last}: {len(frame):,} rows", flush=True)

    # The declared code gate was single-member identity, which proves the rank
    # transform is inert for ONE member and says nothing about averaging. The
    # 2026-08-22 audit called that out and it was right. This gate covers the
    # two-member path: averaging a member's rank vector with ITSELF must return
    # that vector bitwise, so any defect in the stacking arithmetic shows here
    # before a single endpoint is computed. Premise: mean(x, x) == x exactly in
    # IEEE754 for every finite x.
    _probe = names[0]
    for h in HORIZONS:
        single = (
            _calibration_rank(frame, f"{_probe}__{h}",
                              (frame["partition"] == "calibration").to_numpy())
            if args.honest_ranks
            else _rank_within(frame, f"{_probe}__{h}")
        )
        doubled = np.mean([single, single], axis=0)
        if not np.array_equal(single, doubled):
            raise SystemExit(
                f"two-member idempotence gate FAILED at horizon {h}: "
                f"mean(x, x) != x for {_probe}"
            )
    print("two-member idempotence gate passed: mean(x, x) == x bitwise", flush=True)

    source_order = tuple(sorted(set(frame["source_id"])))
    report: dict[str, Any] = {
        "min_cgm_last": args.min_cgm_last,
        "nominal_fapd": args.nominal_fapd,
        # An audit on 2026-08-22 could not tell from the artifact whether the
        # honest branch had run, because `--honest-ranks` is opt-in and the
        # DEFAULT is transductive. An output that cannot name its own execution
        # mode cannot be reviewed, so it names it.
        "honest_ranks": bool(args.honest_ranks),
        "rank_transform": (
            "calibration-fitted, interpolated (partition-honest)"
            if args.honest_ranks
            else "pooled over calibration and selection (TRANSDUCTIVE)"
        ),
        "argv": list(argv) if argv is not None else sys.argv[1:],
        "arms": {n: str(specs[n][0]) for n in specs},
        "rowwise_frame_check": frame_checks,
        "rows": int(len(frame)),
        "join_rows_verified": int(rows_before),
        "results": {},
    }

    for combo in args.combination:
        members = [m.strip() for m in combo.split(",") if m.strip()]
        missing = [m for m in members if m not in specs]
        if missing:
            raise SystemExit(f"unknown arms in combination {combo}: {missing}")
        work = frame.copy()
        calibration_mask = (work["partition"] == "calibration").to_numpy()
        for h in HORIZONS:
            if args.honest_ranks:
                stacked = np.mean(
                    [
                        _calibration_rank(work, f"{m}__{h}", calibration_mask)
                        for m in members
                    ],
                    axis=0,
                )
            else:
                stacked = np.mean(
                    [_rank_within(work, f"{m}__{h}") for m in members], axis=0
                )
            work[f"severe_hypo_{h}_probability"] = stacked
        calibration = work.loc[work["partition"] == "calibration"]
        selection = work.loc[work["partition"] == "selection"]
        per_horizon = {}
        for h in HORIZONS:
            spec = _spec(h, source_order)
            policy = fit_operational_alarm_policy(
                calibration, spec=spec, nominal_fapd=args.nominal_fapd
            )
            metrics = operational_endpoint_metrics(
                selection, spec=spec, threshold=policy.threshold
            )
            per_horizon[str(h)] = {
                "sensitivity": metrics[
                    "equal_source_participant_macro_event_sensitivity"
                ],
                "fapd": metrics[
                    "equal_source_participant_macro_false_alarms_per_patient_day"
                ],
                "alarm_count": metrics["alarm_count"],
                "detected_event_count": metrics["detected_event_count"],
            }
        primary = sum(v["sensitivity"] for v in per_horizon.values()) / len(HORIZONS)
        report["results"][combo] = {"members": members, "horizons": per_horizon,
                                    "primary": primary}
        print(f"  {combo:<58} primary {primary:.9f}", flush=True)
        args.output.write_text(json.dumps(report, indent=1), encoding="utf-8")

    args.output.write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
