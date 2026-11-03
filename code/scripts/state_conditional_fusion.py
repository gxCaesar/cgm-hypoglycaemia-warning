#!/usr/bin/env python3
"""SCRF: fuse arms with a weight that is a smooth function of the patient state.

Every measurement in this bundle points at one sentence: this endpoint punishes
noise in the lower tail so severely that exact information about the present
beats noisy information about the future unless the future information is nearly
exact. The consequence is not "ensemble the arms" -- it is that **each member is
reliable in a different region of the state space and a single scalar weight
cannot express that**. `persistence` covers 100% of severe positives below
61 mg/dL and 0% above 70; the LSTM covers 0.26-0.55 everywhere. A uniform average
spends half its weight on a member that is silent wherever the other works.

    f(x) = sum_k w_k(s) * r_k(x)          s = cgm_last
    w(s) = softmax_k( Theta . B(s) )      sum_k w_k(s) = 1

`r_k` is arm k's CALIBRATION-fitted rank, the transform already gated elsewhere.
`B` is M Gaussian RBFs whose centres and width come from calibration quantiles of
`cgm_last`, so neighbouring regions share strength -- which is precisely what the
failed per-band isotonic construction lacked, its worst band holding 25 positives
in 973,604 rows.

**The uniform baseline is NESTED**: `Theta = 0` gives `w(s) = 1/K` exactly, which
is the rank average already measured at 0.924472288. SCRF cannot be credited with
a gain uniform weighting already produced, and that is gate G1.

Pre-registered in `PREREG-state-conditional-reliability-fusion.md`. Read-only over
frozen prediction artifacts: trains no network, opens no external partition.
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
from scipy.optimize import minimize

from glycoshift.operational_metrics import (
    OperationalEndpointSpec,
    fit_operational_alarm_policy,
    operational_endpoint_metrics,
)

HORIZONS = (30, 60)
KEY = ["source_id", "participant_id", "context_end"]
BASE = ["partition", "patient_day", "cgm_last", "severe_hypo_currently_active"]
#: Basis centres are read at these calibration percentiles of `cgm_last`. Fixed
#: before any fit and deliberately NOT the 54/61/70 lines used elsewhere here,
#: which are partly derived from a previously measured operating point.
CENTRE_PERCENTILES = (10.0, 30.0, 50.0, 70.0, 90.0)


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


def _read(path: Path, *, legacy: bool, base: bool) -> pd.DataFrame:
    score = [
        (f"hypo_{h}_probability" if legacy else f"severe_hypo_{h}_probability")
        for h in HORIZONS
    ]
    columns = list(KEY) + score + ["partition", "severe_hypo_currently_active"]
    if base:
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


def calibration_rank(values: np.ndarray, calibration: np.ndarray) -> np.ndarray:
    """Rank against the CALIBRATION distribution alone, applied to every row.

    Identical in form to the transform behind the Order 4 result: the map is a
    function of calibration scores only, so scoring one selection window never
    consults another, and it is strictly increasing between distinct calibration
    values, which is what keeps the single-arm case an exact reparameterisation.
    """

    reference = np.sort(values[calibration])
    if reference.size == 0:
        raise SystemExit("no calibration rows to fit the rank transform on")
    grid = np.linspace(0.0, 1.0, reference.size)
    return np.interp(values, reference, grid)


def rbf_basis(state: np.ndarray, centres: np.ndarray, width: float) -> np.ndarray:
    """M overlapping Gaussians. Overlap is the point: it shares strength."""

    z = (state[:, None] - centres[None, :]) / width
    return np.exp(-0.5 * z * z)


def fuse(theta: np.ndarray, basis: np.ndarray, ranks: np.ndarray) -> np.ndarray:
    """`sum_k softmax_k(Theta . B(s))_k * r_k`.

    `theta` is `(K-1, M)`; the first arm's row is pinned to zero because softmax
    is shift-invariant, so the free parameters are exactly the degrees of freedom.
    """

    k = ranks.shape[1]
    full = np.vstack([np.zeros((1, basis.shape[1])), theta.reshape(k - 1, -1)])
    logits = basis @ full.T
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= weights.sum(axis=1, keepdims=True)
    return np.einsum("nk,nk->n", weights, ranks), weights


def distinct_events_covered(
    score: np.ndarray, budget: int, event_key: np.ndarray, has_event: np.ndarray
) -> int:
    """How many DISTINCT severe events the top-`budget` windows touch.

    Event-level and budget-matched, which is much closer to the endpoint than a
    likelihood. It omits the refractory collapse and the one-to-one matching the
    endpoint applies; those run unchanged at evaluation time on selection, and
    the omission is disclosed in the pre-registration rather than hidden.
    """

    if budget >= score.size:
        top = np.arange(score.size)
    else:
        top = np.argpartition(-score, budget)[:budget]
    hit = top[has_event[top]]
    return int(np.unique(event_key[hit]).size)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True,
                        help="name=path[:legacy]")
    parser.add_argument("--nominal-fapd", type=float, default=0.5)
    parser.add_argument("--restarts", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument(
        "--centres",
        choices=("marginal", "decision"),
        default="marginal",
        help=(
            "Where the RBF centres go. `marginal` reads percentiles of cgm_last "
            "over ALL calibration rows -- the first run's rule, which placed every "
            "centre above the decision region and could not express a weight that "
            "goes from 1 to 0 across 54-70. `decision` reads the same percentiles "
            "among calibration windows that are SEVERE-POSITIVE, putting capacity "
            "where the decisions are. Same M, same width rule, same parameter "
            "count: only the placement moves."
        ),
    )
    parser.add_argument("--uniform-only", action="store_true",
                        help="G1: skip the fit and score Theta = 0 only")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    specs: dict[str, tuple[Path, bool]] = {}
    for item in args.arm:
        name, _, rest = item.partition("=")
        path, _, flag = rest.partition(":")
        specs[name] = (Path(path), flag == "legacy")
    names = list(specs)

    frame = _read(specs[names[0]][0], legacy=specs[names[0]][1], base=True).rename(
        columns={f"severe_hypo_{h}_probability": f"{names[0]}__{h}" for h in HORIZONS}
    )
    rows_before = len(frame)
    for name in names[1:]:
        path, legacy = specs[name]
        other = _read(path, legacy=legacy, base=False).rename(
            columns={f"severe_hypo_{h}_probability": f"{name}__{h}" for h in HORIZONS}
        )
        frame = frame.merge(other, on=KEY, how="inner", validate="one_to_one",
                            suffixes=("", f"__{name}_chk"))
        if len(frame) != rows_before:
            raise SystemExit(f"join with {name} was not total")
        # Rowwise frame identity, not equal aggregates. An audit on 2026-08-22
        # established that equal counts are corroboration and not proof.
        for column in ("partition", "severe_hypo_currently_active"):
            a = frame[f"{column}__{name}_chk"]
            b = frame[column]
            bad = (
                int((a.astype(bool).to_numpy() != b.astype(bool).to_numpy()).sum())
                if column.endswith("active")
                else int((a.astype(str).to_numpy() != b.astype(str).to_numpy()).sum())
            )
            if bad:
                raise SystemExit(f"{name} disagrees on {column} in {bad:,} rows")
            frame = frame.drop(columns=[f"{column}__{name}_chk"])
    print(f"join one-to-one and total over {rows_before:,} rows; "
          f"rowwise frame check passed for {', '.join(names[1:]) or 'n/a'}", flush=True)

    calibration = (frame["partition"] == "calibration").to_numpy()
    selection = (frame["partition"] == "selection").to_numpy()
    state = pd.to_numeric(frame["cgm_last"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(state).all():
        raise SystemExit("cgm_last carries non-finite values")

    # G3: every fitted quantity below is a function of calibration rows only.
    if args.centres == "decision":
        # The decision-bearing subpopulation: calibration windows carrying a
        # severe event at EITHER horizon. Calibration rows and calibration
        # labels only, both of which the alarm policy already uses.
        bearing = calibration & np.logical_or.reduce([
            pd.to_numeric(frame[f"severe_hypo_{h}_label"], errors="coerce")
            .fillna(0).to_numpy(dtype=float) > 0.5
            for h in HORIZONS
        ])
        if int(bearing.sum()) < 100:
            raise SystemExit("too few decision-bearing calibration rows for a basis")
        print(f"decision-aligned basis from {int(bearing.sum()):,} severe-positive "
              f"calibration windows", flush=True)
        centres = np.percentile(state[bearing], CENTRE_PERCENTILES)
    else:
        centres = np.percentile(state[calibration], CENTRE_PERCENTILES)
    width = float(np.mean(np.diff(centres)))
    if not np.isfinite(width) or width <= 0:
        raise SystemExit("degenerate basis width")
    basis = rbf_basis(state, centres, width)
    print(f"basis centres [{args.centres}] (percentiles {CENTRE_PERCENTILES}): "
          f"{np.round(centres, 2).tolist()}  width {width:.3f}", flush=True)

    ranks = {
        h: np.column_stack([
            calibration_rank(
                pd.to_numeric(frame[f"{n}__{h}"], errors="coerce").to_numpy(float),
                calibration,
            )
            for n in names
        ])
        for h in HORIZONS
    }

    source_order = tuple(sorted(set(frame.loc[selection, "source_id"])))
    report: dict[str, Any] = {
        "rows": int(rows_before),
        "arms": {n: str(specs[n][0]) for n in names},
        "nominal_fapd": args.nominal_fapd,
        "basis_centres": centres.tolist(),
        "basis_width": width,
        "centre_percentiles": list(CENTRE_PERCENTILES),
        "centre_rule": args.centres,
        "argv": list(argv) if argv is not None else sys.argv[1:],
        "uniform_only": bool(args.uniform_only),
        "horizons": {},
    }
    primaries: dict[str, list[float]] = {"uniform": [], "scrf": []}

    for h in HORIZONS:
        r = ranks[h]
        k = r.shape[1]
        ids = frame[f"severe_hypo_{h}_event_id"]
        has_event = ids.notna().to_numpy()
        event_key = (
            frame["source_id"].astype(str) + "|"
            + frame["participant_id"].astype(str) + "|" + ids.astype(str)
        ).to_numpy()

        # Bound through defaults rather than closed over: these are defined
        # inside a loop and late binding would silently score the wrong horizon.
        def endpoint(score: np.ndarray, _h: int = h) -> dict[str, Any]:
            work = frame.copy()
            work[f"severe_hypo_{_h}_probability"] = score
            policy = fit_operational_alarm_policy(
                work.loc[calibration], spec=_spec(_h, source_order),
                nominal_fapd=args.nominal_fapd,
            )
            metrics = operational_endpoint_metrics(
                work.loc[selection], spec=_spec(_h, source_order),
                threshold=policy.threshold,
            )
            return {
                "threshold": float(policy.threshold),
                "sensitivity": metrics[
                    "equal_source_participant_macro_event_sensitivity"],
                "fapd": metrics[
                    "equal_source_participant_macro_false_alarms_per_patient_day"],
                "alarm_count": metrics["alarm_count"],
                "detected_event_count": metrics["detected_event_count"],
            }

        theta0 = np.zeros((k - 1) * basis.shape[1])
        uniform_score, uniform_w = fuse(theta0, basis, r)
        # G4: softmax output is on the simplex, so the fusion interpolates
        # deployable arms rather than extrapolating past them.
        if not np.allclose(uniform_w.sum(axis=1), 1.0, atol=1e-12):
            raise SystemExit("G4 FAILED: weights are not on the simplex")
        if k > 1 and not np.allclose(uniform_w, 1.0 / k, atol=1e-12):
            raise SystemExit("G1 FAILED: Theta = 0 did not give uniform weights")
        uniform_metrics = endpoint(uniform_score)
        primaries["uniform"].append(uniform_metrics["sensitivity"])
        print(f"h={h} uniform (Theta=0): sensitivity "
              f"{uniform_metrics['sensitivity']:.9f}  alarms "
              f"{uniform_metrics['alarm_count']:,}", flush=True)

        block: dict[str, Any] = {"uniform": uniform_metrics}

        if not args.uniform_only and k > 1:
            # The budget the UNIFORM combination spends on calibration, so the
            # fit cannot win by firing more.
            cal_uniform = uniform_score[calibration]
            budget = int((cal_uniform >= uniform_metrics["threshold"]).sum())
            cal_key = event_key[calibration]
            cal_has = has_event[calibration]
            cal_basis = basis[calibration]
            cal_r = r[calibration]
            base_cover = distinct_events_covered(
                cal_uniform, budget, cal_key, cal_has)
            print(
                f"     calibration budget {budget:,} windows; uniform covers "
                f"{base_cover:,} distinct events",
                flush=True,
            )

            def negative_cover(
                theta: np.ndarray,
                _b: np.ndarray = cal_basis,
                _r: np.ndarray = cal_r,
                _n: int = budget,
                _k: np.ndarray = cal_key,
                _e: np.ndarray = cal_has,
            ) -> float:
                fused, _ = fuse(theta, _b, _r)
                return -float(distinct_events_covered(fused, _n, _k, _e))

            rng = np.random.default_rng(args.seed + h)
            best = (negative_cover(theta0), theta0)
            starts = [theta0] + [
                rng.normal(0.0, 1.0, theta0.size) for _ in range(args.restarts - 1)
            ]
            for start in starts:
                res = minimize(negative_cover, start, method="Nelder-Mead",
                               options={"maxiter": 400, "xatol": 1e-3,
                                        "fatol": 0.5, "disp": False})
                if res.fun < best[0]:
                    best = (float(res.fun), np.asarray(res.x, dtype=float))
            theta = best[1]
            print(f"     fitted: calibration cover {int(-best[0]):,} "
                  f"(uniform {base_cover:,})", flush=True)

            score, weights = fuse(theta, basis, r)
            if not np.allclose(weights.sum(axis=1), 1.0, atol=1e-10):
                raise SystemExit("G4 FAILED after fitting")
            scrf_metrics = endpoint(score)
            primaries["scrf"].append(scrf_metrics["sensitivity"])
            grid = np.linspace(float(np.percentile(state[calibration], 1)),
                               float(np.percentile(state[calibration], 99)), 9)
            _, grid_w = fuse(theta, rbf_basis(grid, centres, width),
                             np.zeros((grid.size, k)))
            block.update({
                "scrf": scrf_metrics,
                "theta": theta.tolist(),
                "calibration_budget_windows": budget,
                "calibration_cover_uniform": base_cover,
                "calibration_cover_fitted": int(-best[0]),
                "weight_grid_state": grid.tolist(),
                "weight_grid": {n: grid_w[:, i].tolist() for i, n in enumerate(names)},
            })
            print(f"h={h} SCRF: sensitivity {scrf_metrics['sensitivity']:.9f}"
                  f"  alarms {scrf_metrics['alarm_count']:,}", flush=True)

        report["horizons"][str(h)] = block

    for key, values in primaries.items():
        if len(values) == len(HORIZONS):
            report[f"primary_{key}"] = float(np.mean(values))
            print(f"PRIMARY {key}: {np.mean(values):.9f}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
