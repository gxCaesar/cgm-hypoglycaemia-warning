"""Does a per-source alarm threshold close the gap to the cheap control?

Secondary and descriptive. The frozen primary endpoint fits **one** global
threshold on calibration, and nothing here changes that. This asks a separate
question that a reviewer asks first: if the deficiency is distinct-event
coverage at a fixed alarm budget, can it be recovered by stratifying the alarm
policy rather than by a new objective?

Every alarm semantic is imported from `operational_metrics` -- eligibility,
refractory collapse, one-to-one event matching, the source-then-participant
macro -- so the only thing that varies between the two arms below is how many
thresholds are fitted. The global arm is expected to reproduce the arm's own
`operational_metrics.json` exactly, and that agreement is the correctness check
for the stratified arm.

**Per-participant stratification is not evaluable on this protocol and is not
attempted.** The split is participant-disjoint, so a threshold fitted on a
calibration participant has no selection participant to transfer to. Sources
appear in every partition, which is why per-source is the only stratification
this design can measure.

Read-only: opens one completed arm's own prediction artifact, trains nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from glycoshift.operational_metrics import (
    OperationalEndpointSpec,
    fit_operational_alarm_policy,
    operational_endpoint_metrics,
)

HORIZONS = (30, 60)
SCORE_COLUMNS = (
    "partition",
    "source_id",
    "participant_id",
    "context_end",
    "patient_day",
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


def _columns(
    *,
    legacy: bool = False,
    oracle: str = "none",
    score_source: str = "head",
    synthetic: str = "none",
) -> list[str]:
    columns = list(SCORE_COLUMNS) + [
        "severe_hypo_currently_active",
        "cgm_last",
        "cgm_slope_30",
    ]
    if oracle != "none":
        columns += [f"cgm_{horizon}_target" for horizon in HORIZONS]
    if synthetic != "none":
        columns += [
            f"cgm_{horizon}_{suffix}"
            for horizon in HORIZONS
            for suffix in ("prediction", "target")
        ]
    if score_source == "forecast":
        # The truth column is read only for the persistence-collapse diagnostic
        # declared with the pre-registration; it never enters the score.
        columns += [
            f"cgm_{horizon}_{suffix}"
            for horizon in HORIZONS
            for suffix in ("prediction", "target")
        ]
    for horizon in HORIZONS:
        columns += [
            (
                f"hypo_{horizon}_probability"
                if legacy
                else f"severe_hypo_{horizon}_probability"
            ),
            f"severe_hypo_{horizon}_label",
            f"severe_hypo_{horizon}_event_id",
            f"severe_hypo_{horizon}_event_onset",
        ]
    return columns


def _apply_oracle(frame: pd.DataFrame, *, mode: str) -> pd.DataFrame:
    """Replace each severe score by information the deployed system can never have.

    `research-sota-loop` Step 0 asks for an oracle reference arm BEFORE any
    component is designed, because it is what separates "the task is saturated"
    from "the method is not good enough", and those two have opposite correct
    responses. This project had never run one. Its only recorded `oracle` is a
    transductive rank-average bound on the SCORING side, which is a different
    object and answers a different question.

    Two modes, and they bracket different things:

    ``label``           perfect ranking. The ceiling of the frozen endpoint under
                        its own refractory, matching, macro estimand and FAPD
                        budget -- what ANY method could reach.
    ``future_glucose``  the true glucose at the horizon. What a PERFECT glucose
                        forecaster could reach, so the gap to ``label`` is what
                        no amount of forecasting accuracy can buy.

    The oracle gets oracle INPUT, never an oracle threshold: the policy is still
    fitted on calibration alone and evaluated on selection, exactly as every
    frozen arm was.

    **`future_glucose` can UNDERSHOOT and that is stated rather than hidden.**
    `cgm_<h>_target` is the glucose at exactly the horizon, one point, while the
    label is an event onset anywhere in `(0, h]`. An excursion that begins and
    ends inside the window is invisible to it, so it is a conservative reference
    point rather than a bound.
    """

    for horizon in HORIZONS:
        column = f"cgm_{horizon}_target"
        future = pd.to_numeric(frame[column], errors="coerce")
        censored = int(future.isna().sum())
        print(
            f"oracle {mode}: {column} censored on {censored} of {len(future)} rows "
            f"({100.0 * censored / max(1, len(future)):.3f}%)"
        )
        # A NaN target means the sensor read BELOW its floor -- `windows.py` sets
        # `target_available = not lower_censored` -- so it is the most severe
        # reading, not a missing one. `na_option="bottom"` gives it the largest
        # rank, which is the highest risk. Ranking rather than negating keeps the
        # score bounded for the quantile candidate set; a rank is a monotone
        # transform of `-future` up to ties, so the alarm sets are unchanged.
        ranked = (-future).rank(method="average", na_option="bottom", pct=True)
        if mode == "future_glucose":
            score = ranked
        elif mode == "label":
            # Every positive above every negative. `0.5 * ranked` lands in
            # (0, 0.5], so the two classes cannot interleave, and within each
            # class the lower future glucose ranks higher.
            label = frame[f"severe_hypo_{horizon}_label"].astype(float)
            score = label + 0.5 * ranked
        else:
            raise ValueError(f"unknown oracle mode: {mode}")
        frame[f"severe_hypo_{horizon}_probability"] = score.astype(float)
    return frame


def _apply_forecast_score(frame: pd.DataFrame) -> pd.DataFrame:
    """Score the endpoint through the model's own glucose forecast head.

    **This is NOT an oracle and must never be reported beside one.** It uses only
    what the deployed system has at prediction time. `cgm_<h>_prediction` is the
    auxiliary regression head trained at `auxiliary_regression_weight: 0.1`, and
    it is written into every prediction artifact and has never been scored.

    The oracle arm measured that 91.7% of the headroom above the leading threshold
    is reachable from glucose alone. This asks whether that information is already
    inside the model and is being discarded by the scoring path, which reads the
    classification head -- itself the 70 mg/dL hypo head bound to the severe score
    verbatim.
    """

    for horizon in HORIZONS:
        predicted = pd.to_numeric(
            frame[f"cgm_{horizon}_prediction"], errors="coerce"
        )
        if not np.isfinite(predicted.to_numpy(dtype=float)).all():
            raise ValueError(f"cgm_{horizon}_prediction is not finite")
        # Lower predicted glucose is higher severe risk. A rank is a monotone
        # transform of `-predicted` up to ties, so it leaves every alarm set
        # unchanged while keeping the quantile candidate set well behaved.
        frame[f"severe_hypo_{horizon}_probability"] = (
            (-predicted).rank(method="average", pct=True).astype(float)
        )
    return frame


def _apply_synthetic_forecast(
    frame: pd.DataFrame, *, family: str, lam: float, seed: int
) -> pd.DataFrame:
    """A forecaster of CONTROLLED quality, to find what quality the endpoint needs.

    **Neither family is deployable.** Both read the true future glucose, so both are
    reference constructions and must never be reported as methods. They exist to
    answer two questions one rescoring pass can settle before a GPU is spent: at
    what forecast quality does scoring through a forecast overtake the leading
    threshold, and is this project's forecaster failing on the SIZE of its error or
    on its SHAPE.

    ``blend``        ``lam * target + (1 - lam) * prediction``. Keeps this
                     forecaster's bias and shrinks it toward the truth, so its
                     error is ``(1 - lam)(prediction - target)`` and its RMSE is
                     exactly ``(1 - lam)`` of the measured one.
    ``noisy_truth``  ``target + N(0, sigma^2)`` with ``sigma`` set per horizon to
                     ``(1 - lam) * RMSE_h``, so its RMSE MATCHES the blend's at the
                     same ``lam`` while the error is unbiased.

    The gap between the two at equal ``lam`` is therefore the cost of the bias,
    measured rather than argued. See ``PREREG-how-good-must-the-forecast-be.md``.
    """

    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lambda must lie in [0, 1], got {lam}")
    rng = np.random.default_rng(seed)
    for horizon in HORIZONS:
        truth = pd.to_numeric(
            frame[f"cgm_{horizon}_target"], errors="coerce"
        ).to_numpy(dtype=float)
        predicted = pd.to_numeric(
            frame[f"cgm_{horizon}_prediction"], errors="coerce"
        ).to_numpy(dtype=float)
        if not (np.isfinite(truth).all() and np.isfinite(predicted).all()):
            raise ValueError(f"horizon {horizon} carries non-finite glucose")
        rmse = float(np.sqrt(np.mean((predicted - truth) ** 2)))
        if family == "blend":
            values = lam * truth + (1.0 - lam) * predicted
        elif family == "noisy_truth":
            sigma = (1.0 - lam) * rmse
            if sigma > 0:
                values = truth + rng.normal(0.0, sigma, size=truth.shape)
            else:
                values = truth
        else:
            raise ValueError(f"unknown synthetic family: {family}")
        achieved = float(np.sqrt(np.mean((values - truth) ** 2)))
        print(
            f"synthetic {family} lam={lam:.2f} h={horizon}: "
            f"pred_rmse={rmse:.4f}  target_rmse={(1.0 - lam) * rmse:.4f}  "
            f"achieved_rmse={achieved:.4f} mg/dL"
        )
        frame[f"severe_hypo_{horizon}_probability"] = (
            pd.Series(-values).rank(method="average", pct=True).to_numpy(dtype=float)
        )
    return frame


def _apply_within_state_calibration(
    frame: pd.DataFrame, *, variable: str, bin_count: int
) -> pd.DataFrame:
    """Replace each score by its within-bin CALIBRATION CDF value.

    The bin edges and every CDF are fit on the calibration partition alone and
    then applied to both partitions. Fitting either on selection would leak the
    evaluation partition into the policy.

    Per bin the map is monotone, so one global threshold on the transformed score
    is exactly one threshold per bin. With `bin_count == 1` the map is monotone on
    the whole score, the ordering and therefore the alarm set are unchanged, and
    the endpoint must reproduce the global policy. That is the code gate.
    """

    values = pd.to_numeric(frame[variable], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise SystemExit(f"{variable} carries non-finite values; cannot bin on it")
    calibration_mask = (frame["partition"] == "calibration").to_numpy()
    if not calibration_mask.any():
        raise SystemExit("within-state calibration needs calibration rows")
    if bin_count < 1:
        raise SystemExit("state bin count must be at least 1")

    if bin_count == 1:
        edges = np.array([], dtype=float)
    else:
        quantiles = np.linspace(0.0, 1.0, bin_count + 1)[1:-1]
        edges = np.unique(np.quantile(values[calibration_mask], quantiles))
    assignments = np.searchsorted(edges, values, side="right")
    observed = sorted(set(assignments.tolist()))
    print(f"state bins on {variable}: {len(observed)} occupied, edges {edges}")

    out = frame.copy()
    for horizon in HORIZONS:
        column = f"severe_hypo_{horizon}_probability"
        scores = pd.to_numeric(out[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(scores).all():
            raise SystemExit(f"{column} carries non-finite values")
        transformed = np.empty_like(scores)
        for index in observed:
            member = assignments == index
            reference = np.sort(scores[member & calibration_mask])
            if reference.size == 0:
                raise SystemExit(
                    f"state bin {index} has no calibration rows for {column}"
                )
            # Interpolated CDF rather than the step ECDF. The step version maps
            # every selection score falling between two calibration scores to one
            # shared value, which creates ties the original scores did not have
            # and coarsens the achievable alarm sets. Interpolating keeps the map
            # strictly increasing between distinct calibration scores, so the
            # ordering of the selection scores survives and the single-bin case is
            # an exact monotone reparameterisation.
            grid = np.linspace(0.0, 1.0, reference.size)
            transformed[member] = np.interp(scores[member], reference, grid)
        out[column] = transformed
    return out


def _global_arm(
    calibration: pd.DataFrame,
    selection: pd.DataFrame,
    *,
    horizon: int,
    source_order: tuple[str, ...],
    nominal_fapd: float,
) -> dict:
    spec = _spec(horizon, source_order)
    policy = fit_operational_alarm_policy(
        calibration, spec=spec, nominal_fapd=nominal_fapd
    )
    metrics = operational_endpoint_metrics(
        selection, spec=spec, threshold=policy.threshold
    )
    return {
        "thresholds": {"__global__": policy.threshold},
        "sensitivity": metrics["equal_source_participant_macro_event_sensitivity"],
        "fapd": metrics[
            "equal_source_participant_macro_false_alarms_per_patient_day"
        ],
        "detected_event_count": metrics["detected_event_count"],
        "alarm_count": metrics["alarm_count"],
        "per_source": {
            source: {
                "sensitivity": block["participant_macro_event_sensitivity"],
                "fapd": block["participant_macro_false_alarms_per_patient_day"],
            }
            for source, block in metrics["per_source"].items()
        },
    }


def _per_source_arm(
    calibration: pd.DataFrame,
    selection: pd.DataFrame,
    *,
    horizon: int,
    source_order: tuple[str, ...],
    nominal_fapd: float,
) -> dict:
    """One threshold per source, each fitted on that source's calibration rows.

    The per-source budget is the same nominal FAPD as the global policy, so the
    two arms are compared at the same alarm allowance rather than at the same
    total alarm count.
    """

    thresholds: dict[str, float] = {}
    per_source: dict[str, dict[str, float]] = {}
    detected = 0
    alarms = 0
    for source in source_order:
        spec = _spec(horizon, (source,))
        calibration_rows = calibration.loc[
            calibration["source_id"].astype(str) == source
        ]
        selection_rows = selection.loc[selection["source_id"].astype(str) == source]
        policy = fit_operational_alarm_policy(
            calibration_rows, spec=spec, nominal_fapd=nominal_fapd
        )
        metrics = operational_endpoint_metrics(
            selection_rows, spec=spec, threshold=policy.threshold
        )
        thresholds[source] = policy.threshold
        per_source[source] = {
            "sensitivity": metrics["per_source"][source][
                "participant_macro_event_sensitivity"
            ],
            "fapd": metrics["per_source"][source][
                "participant_macro_false_alarms_per_patient_day"
            ],
        }
        detected += metrics["detected_event_count"]
        alarms += metrics["alarm_count"]
    count = len(source_order)
    return {
        "thresholds": thresholds,
        "sensitivity": sum(v["sensitivity"] for v in per_source.values()) / count,
        "fapd": sum(v["fapd"] for v in per_source.values()) / count,
        "detected_event_count": detected,
        "alarm_count": alarms,
        "per_source": per_source,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--nominal-fapd", type=float, default=0.5)
    parser.add_argument("--expect-primary", type=float, default=None)
    parser.add_argument(
        "--state-bins",
        choices=("none", "slope", "level"),
        default="none",
        help=(
            "Within-state alarm calibration. Bin decision rows by calibration "
            "quantiles of `cgm_slope_30` (slope) or `cgm_last` (level), then "
            "replace each score by its within-bin CALIBRATION CDF value on both "
            "partitions. That is a per-bin monotone transform, so one global "
            "threshold on the transformed scores implements one threshold per "
            "bin, while the estimand, refractory rule and matcher stay frozen. "
            "With a single bin the transform is globally monotone and the "
            "endpoint must be unchanged, which is the gate."
        ),
    )
    parser.add_argument(
        "--state-bin-count",
        type=int,
        default=3,
        help="Number of equal-frequency state bins. 1 exercises the code gate.",
    )
    parser.add_argument(
        "--min-cgm-last",
        type=float,
        default=None,
        help=(
            "Restrict calibration and selection to rows whose `cgm_last` is at "
            "least this value, then fit and evaluate entirely inside the "
            "restriction. Used for the pre-declared analysis of warnings issued "
            "from a non-hypoglycaemic state. Unset leaves the frame untouched."
        ),
    )
    parser.add_argument(
        "--legacy-hypo-score",
        action="store_true",
        help=(
            "Read the severe-event score from `hypo_<h>_probability` instead of "
            "`severe_hypo_<h>_probability`. Older run artifacts predate the "
            "severe-event column, which `deep_training.py:585` binds to the same "
            "hypo-head logit and then temperature-scales. Temperature scaling is "
            "strictly monotone, and both the candidate set and the threshold "
            "comparison depend on the score only through its order, so the fitted "
            "alarm set is unchanged. That equivalence is verified on a v1.5 arm "
            "rather than assumed; see the sigma-remeasurement record."
        ),
    )
    parser.add_argument(
        "--global-only",
        action="store_true",
        help=(
            "Skip the per-source policy arm. The per-source arm fits eleven "
            "policies per horizon and dominates the cost; the alarm-budget "
            "sweep only needs the global one. Without this flag the output is "
            "unchanged."
        ),
    )
    parser.add_argument(
        "--oracle",
        choices=("none", "label", "future_glucose"),
        default="none",
        help=(
            "Replace the severe score by information the deployed system can "
            "never have, to measure the endpoint's ceiling. `label` is a perfect "
            "ranking and bounds what ANY method could reach; `future_glucose` is "
            "the true glucose at the horizon and bounds what a perfect glucose "
            "forecaster could reach. Default `none` reads no extra column and "
            "leaves every code path exactly as it was, which is what makes "
            "`--expect-primary` an identity gate on this flag."
        ),
    )
    parser.add_argument(
        "--score-source",
        choices=("head", "forecast", "current_glucose"),
        default="head",
        help=(
            "Which of the model's own outputs supplies the severe score. `head` "
            "is the classification head, which is what every frozen number in "
            "this bundle used. `forecast` is the auxiliary glucose regression "
            "head, `cgm_<h>_prediction`, which is written into every artifact and "
            "has never been scored. `current_glucose` ranks by `-cgm_last` and is "
            "a POSITIVE CONTROL: `persistence_seed17` scores through a one-feature "
            "logistic on the same column, which is monotone, so this must "
            "reproduce its frozen 0.806116373 through a path that touches the "
            "forecast machinery. All three are DEPLOYABLE -- this is not "
            "`--oracle` and they must never be reported as if they were."
        ),
    )
    parser.add_argument(
        "--synthetic-forecast",
        choices=("none", "blend", "noisy_truth"),
        default="none",
        help=(
            "A forecaster of CONTROLLED quality. NOT DEPLOYABLE -- both families "
            "read the true future glucose and exist to measure what forecast "
            "quality the endpoint needs, and whether this project's forecaster "
            "fails on the SIZE of its error or its SHAPE. `blend` shrinks the "
            "model's own prediction toward the truth, keeping its bias; "
            "`noisy_truth` adds unbiased noise of the SAME per-horizon RMSE. "
            "Never report either as a method."
        ),
    )
    parser.add_argument("--blend-lambda", type=float, default=0.0)
    parser.add_argument("--noise-seed", type=int, default=20260821)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.synthetic_forecast != "none" and (
        args.oracle != "none" or args.score_source != "head"
    ):
        raise SystemExit("--synthetic-forecast excludes --oracle and --score-source")
    if args.oracle != "none" and args.score_source != "head":
        raise SystemExit("--oracle and --score-source are mutually exclusive")
    if args.score_source != "head" and args.legacy_hypo_score:
        raise SystemExit("a non-head score source ignores the classification head")

    frame = pq.read_table(
        args.predictions,
        columns=_columns(
            legacy=args.legacy_hypo_score,
            oracle=args.oracle,
            score_source=args.score_source,
            synthetic=args.synthetic_forecast,
        ),
    ).to_pandas()
    if args.legacy_hypo_score:
        for horizon in HORIZONS:
            frame[f"severe_hypo_{horizon}_probability"] = frame[
                f"hypo_{horizon}_probability"
            ]
    if args.oracle != "none":
        frame = _apply_oracle(frame, mode=args.oracle)
    if args.synthetic_forecast != "none":
        frame = _apply_synthetic_forecast(
            frame,
            family=args.synthetic_forecast,
            lam=args.blend_lambda,
            seed=args.noise_seed,
        )
    if args.score_source == "current_glucose":
        # Positive control for the forecast path: same ranking machinery, same
        # direction, a column whose endpoint value is already known.
        current = pd.to_numeric(frame["cgm_last"], errors="coerce")
        if not np.isfinite(current.to_numpy(dtype=float)).all():
            raise SystemExit("cgm_last is not finite")
        for horizon in HORIZONS:
            frame[f"severe_hypo_{horizon}_probability"] = (
                (-current).rank(method="average", pct=True).astype(float)
            )
    if args.score_source == "forecast":
        frame = _apply_forecast_score(frame)
        for horizon in HORIZONS:
            predicted = pd.to_numeric(frame[f"cgm_{horizon}_prediction"])
            truth = pd.to_numeric(frame[f"cgm_{horizon}_target"], errors="coerce")
            current = pd.to_numeric(frame["cgm_last"], errors="coerce")
            # Declared with the pre-registration: a forecaster that collapses onto
            # persistence makes any endpoint result mean something weaker.
            print(
                f"forecast diagnostic h={horizon}: "
                f"corr(prediction, cgm_last)={predicted.corr(current):.6f}  "
                f"corr(target, cgm_last)={truth.corr(current):.6f}  "
                f"corr(prediction, target)={predicted.corr(truth):.6f}  "
                f"RMSE={float(((predicted - truth) ** 2).mean() ** 0.5):.4f} mg/dL"
            )
    frame["source_id"] = frame["source_id"].astype(str)
    if args.min_cgm_last is not None:
        keep = pd.to_numeric(frame["cgm_last"], errors="coerce")
        before = len(frame)
        frame = frame.loc[keep >= args.min_cgm_last].copy()
        print(
            f"cgm_last >= {args.min_cgm_last}: {len(frame)} of {before} rows "
            f"({100.0 * len(frame) / before:.2f}%)"
        )
        for horizon in HORIZONS:
            label = f"severe_hypo_{horizon}_label"
            print(
                f"  severe_hypo_{horizon} positives retained: "
                f"{int(frame[label].sum())}"
            )
        if frame.empty:
            raise SystemExit("the cgm_last restriction retained no rows")
    if args.state_bins != "none":
        variable = "cgm_slope_30" if args.state_bins == "slope" else "cgm_last"
        frame = _apply_within_state_calibration(
            frame, variable=variable, bin_count=args.state_bin_count
        )
    calibration = frame.loc[frame["partition"] == "calibration"].copy()
    selection = frame.loc[frame["partition"] == "selection"].copy()
    del frame
    source_order = tuple(sorted(set(selection["source_id"])))
    if set(calibration["source_id"]) != set(source_order):
        raise SystemExit("calibration and selection do not carry the same sources")

    policy_arms = ("global",) if args.global_only else ("global", "per_source")
    report: dict[str, dict] = {"arm": args.arm, "nominal_fapd": args.nominal_fapd}
    means: dict[str, list[float]] = {arm: [] for arm in policy_arms}
    for horizon in HORIZONS:
        block = {
            "global": _global_arm(
                calibration,
                selection,
                horizon=horizon,
                source_order=source_order,
                nominal_fapd=args.nominal_fapd,
            ),
        }
        if not args.global_only:
            block["per_source"] = _per_source_arm(
                calibration,
                selection,
                horizon=horizon,
                source_order=source_order,
                nominal_fapd=args.nominal_fapd,
            )
        report[str(horizon)] = block
        for arm in means:
            means[arm].append(block[arm]["sensitivity"])
        print(f"\nhorizon {horizon} min")
        for arm in policy_arms:
            print(
                f"  {arm:<11s} sensitivity {block[arm]['sensitivity']:.9f}"
                f"   FAPD {block[arm]['fapd']:.6f}"
                f"   events {block[arm]['detected_event_count']}"
                f"   alarms {block[arm]['alarm_count']}"
            )

    primary = {arm: sum(v) / len(v) for arm, v in means.items()}
    report["primary_equal_horizon_mean_event_sensitivity"] = primary
    print("\nprimary (mean of 30/60)")
    print(f"  global      {primary['global']:.9f}   <- reproduces the frozen policy")
    if not args.global_only:
        print(f"  per_source  {primary['per_source']:.9f}")
        print(f"  delta       {primary['per_source'] - primary['global']:+.9f}")

    if args.expect_primary is not None:
        drift = abs(primary["global"] - args.expect_primary)
        print(f"\nagreement with the arm's own frozen endpoint: {drift:.12f}")
        if drift > 1e-9:
            raise SystemExit(
                "the global arm did not reproduce the frozen endpoint, so the "
                "stratified arm is not trustworthy either"
            )

    if args.output:
        args.output.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
