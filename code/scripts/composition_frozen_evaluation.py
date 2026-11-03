'Composition frozen evaluation.'






















from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HORIZONS = ("30", "60")
CALIBRATION = "calibration"
# The members' prediction files are NOT row-aligned. They mismatch from row zero, so
# averaging by position combines different decisions. The first version of this file did
# exactly that and its known-answer check returned 0.168 against a frozen 0.924. Members
# are joined on this key, one-to-one, as the tool that produced the frozen number does.
JOIN_KEY = ["source_id", "participant_id", "context_end"]


def _columns(horizon: str) -> list[str]:
    return [
        f"severe_hypo_{horizon}_probability",
        f"severe_hypo_{horizon}_label",
        f"severe_hypo_{horizon}_event_id",
        f"severe_hypo_{horizon}_event_onset",
    ]


SCORES = ("severe", "hypo", "neg_cgm_last")


def read_member(path: Path, horizons: tuple[str, ...],
                score: str = "severe") -> pd.DataFrame:
    """One prediction file, with the member's score placed under the severe name.

    `run_frozen_evaluation` writes `hypo_<h>_probability` and no severe score, so an
    external file cannot be read under the development convention. Two substitutions
    are available and NEITHER is assumed: `hypo` is inert only where the severe score
    is a strictly increasing function of it, which holds for the trained member and
    measurably does not hold for the baseline, and `neg_cgm_last` reconstructs the
    threshold's own ordering. Both are proved end to end by the known-answer run on
    the development selection partition before either is used on a held-out set.
    """
    if score not in SCORES:
        raise SystemExit(f"unknown score {score!r}, expected one of {SCORES}")
    base = ["source_id", "participant_id", "context_end", "patient_day", "partition",
            "severe_hypo_currently_active", "cgm_last"]
    keep = [c for h in horizons for c in _columns(h)]
    if score == "severe":
        cols = base + keep
    elif score == "hypo":
        cols = base + [c for c in keep if "probability" not in c] + [
            f"hypo_{h}_probability" for h in horizons]
    else:
        cols = base + [c for c in keep if "probability" not in c]
    frame = pq.read_table(path, columns=sorted(set(cols))).to_pandas()
    if score == "hypo":
        for h in horizons:
            frame[f"severe_hypo_{h}_probability"] = frame.pop(f"hypo_{h}_probability")
    elif score == "neg_cgm_last":
        glucose = pd.to_numeric(frame["cgm_last"], errors="coerce")
        for h in horizons:
            frame[f"severe_hypo_{h}_probability"] = -glucose
    return frame


def read_member_parts(paths: list[Path], horizons: tuple[str, ...],
                      score: str = "severe") -> pd.DataFrame:
    """One member, assembled from one or more prediction files.

    The calibration ECDF is fitted on the DEVELOPMENT calibration partition and the
    held-out rows come from a frozen external evaluation, and `run_frozen_evaluation`
    writes only the external partition. So scoring the composition on an external set
    needs both files for the same member, and the tool refused with "the joined frame
    carries no calibration rows" until it could take them.

    The parts must be partition-disjoint. Overlap would put one decision in the frame
    twice, and a duplicated calibration row silently reweights the ECDF that every
    held-out score is read against.
    """
    parts, seen = [], {}
    for path in paths:
        part = read_member(path, horizons, score)
        here = set(pd.unique(part["partition"]))
        clash = {p: seen[p] for p in here if p in seen}
        if clash:
            raise SystemExit(
                f"{path.name} repeats partition(s) already read from "
                f"{sorted(set(clash.values()))}: {sorted(clash)}")
        seen.update({p: path.name for p in here})
        parts.append(part)
        print(f"    read {path.name}: {len(part):,} rows, partitions {sorted(here)}")
    frame = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
    before = len(frame)
    frame = frame.drop_duplicates(subset=JOIN_KEY)
    if len(frame) != before:
        raise SystemExit(
            f"assembled member repeats {before - len(frame)} decision key(s) across "
            f"its parts, which partition-disjointness should have prevented")
    return frame


def calibration_rank(values: np.ndarray, calibration_mask: np.ndarray) -> np.ndarray:
    """Calibration ECDF rank. A function of calibration scores alone, so scoring one
    held-out row never consults another, and strictly increasing between distinct
    calibration values, so a single member passed through it is a reparameterisation."""
    reference = np.sort(values[calibration_mask & np.isfinite(values)])
    if reference.size == 0:
        raise SystemExit("no finite calibration scores to fit the rank transform on")
    return np.searchsorted(reference, values, side="right") / reference.size


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--member", action="append", required=True,
                    help="name=predictions.parquet[,more.parquet ...]")
    ap.add_argument("--score", action="append", default=[],
                    help=f"name=<{'|'.join(SCORES)}>, default severe")
    ap.add_argument("--evaluate-partition", required=True,
                    help="selection, or a held-out partition")
    ap.add_argument("--nominal-fapd", type=float, default=0.5)
    ap.add_argument("--expect-primary", type=float, default=None,
                    help="the frozen value this must reproduce; refuses on mismatch")
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args(argv)

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from glycoshift.operational_metrics import (  # noqa: E402
        OperationalEndpointSpec,
        fit_operational_alarm_policy,
        operational_endpoint_metrics,
    )

    # Copied from `combination_primary.py`, the tool that produces the frozen
    # composition primary, rather than reconstructed. The first version of this file
    # guessed the signature and failed on an argument that does not exist.
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

    scores = dict(item.split("=", 1) for item in args.score)
    frame = None
    member_names: list[str] = []
    for item in args.member:
        name, _, spec = item.partition("=")
        member_names.append(name)
        part = read_member_parts([Path(p) for p in spec.split(",") if p],
                                 HORIZONS, scores.get(name, "severe"))
        part = part.rename(columns={
            f"severe_hypo_{h}_probability": f"member__{name}__{h}" for h in HORIZONS})
        if frame is None:
            frame = part
        else:
            drop = [c for c in part.columns
                    if c in frame.columns and c not in JOIN_KEY]
            frame = frame.merge(part.drop(columns=drop), on=JOIN_KEY,
                                how="inner", validate="one_to_one")
        print(f"  joined {name}, {len(frame):,} rows")
    if frame is None:
        raise SystemExit("no members supplied")

    calibration_mask = (frame["partition"] == CALIBRATION).to_numpy()
    if not calibration_mask.any():
        raise SystemExit("the joined frame carries no calibration rows")
    evaluate_mask = (frame["partition"] == args.evaluate_partition).to_numpy()
    if not evaluate_mask.any():
        raise SystemExit(f"no rows in partition {args.evaluate_partition!r}")

    combined = frame.copy()
    for horizon in HORIZONS:
        ranks = []
        for name in member_names:
            values = pd.to_numeric(
                frame[f"member__{name}__{horizon}"], errors="coerce"
            ).to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise SystemExit(f"{name} horizon {horizon} carries non-finite values")
            ranks.append(calibration_rank(values, calibration_mask))
        combined[f"severe_hypo_{horizon}_probability"] = np.mean(
            np.vstack(ranks), axis=0)

    report: dict[str, object] = {
        "members": member_names,
        "member_scores": {n: scores.get(n, "severe") for n in member_names},
        "evaluate_partition": args.evaluate_partition,
        "nominal_fapd": args.nominal_fapd,
        "join_key": JOIN_KEY,
        "joined_rows": int(len(frame)),
        "calibration_rows": int(calibration_mask.sum()),
        "evaluated_rows": int(evaluate_mask.sum()),
        "horizons": {},
    }
    sensitivities = []
    # The policy is FITTED on development calibration and APPLIED to the evaluated
    # partition, and on a held-out cohort those two carry different sources. Deriving
    # one order from the combined frame gives a spec that names eleven sources the
    # external rows do not contain, and the sweep refuses. Each side therefore gets the
    # order of the rows it actually sees.
    #
    # This is not a relaxation of the estimand, and the known-answer run is what proves
    # it: on the development selection partition both sides carry the same eleven
    # sources, the two orders coincide, and the frozen primary must still return. On a
    # single-source external partition the equal-source macro degenerates to a
    # participant macro inside that source, which is the only reading available there.
    def _order(mask):
        return tuple(sorted(combined.loc[mask, "source_id"].dropna().unique().tolist()))

    calibration_order = _order(calibration_mask)
    evaluate_order = _order(evaluate_mask)
    report["source_order"] = list(evaluate_order)
    report["calibration_source_order"] = list(calibration_order)
    for horizon in HORIZONS:
        policy = fit_operational_alarm_policy(
            combined.loc[calibration_mask],
            spec=_spec(int(horizon), calibration_order),
            nominal_fapd=args.nominal_fapd,
        )
        metrics = operational_endpoint_metrics(
            combined.loc[evaluate_mask],
            spec=_spec(int(horizon), evaluate_order),
            threshold=policy.threshold,
        )
        value = metrics["equal_source_participant_macro_event_sensitivity"]
        sensitivities.append(value)
        report["horizons"][horizon] = {
            "event_sensitivity": value,
            "achieved_fapd": metrics[
                "equal_source_participant_macro_false_alarms_per_patient_day"],
            "alarm_count": metrics["alarm_count"],
            "detected_event_count": metrics["detected_event_count"],
            "threshold_fitted_on_calibration": policy.threshold,
        }
    primary = float(np.mean(sensitivities))
    report["primary_equal_horizon_mean_event_sensitivity"] = primary
    report["primary_9dp"] = f"{primary:.9f}"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1, sort_keys=True))
    print(f"partition {args.evaluate_partition}  rows {report['evaluated_rows']:,}  "
          f"primary {primary:.9f}")
    for horizon, block in report["horizons"].items():
        print(f"  {horizon} min  sens {block['event_sensitivity']:.9f}  "
              f"fapd {block['achieved_fapd']:.4f}  alarms {block['alarm_count']:,}")

    if args.expect_primary is not None:
        # The frozen literal is recorded to nine decimals, so nine decimals is the most
        # that can be checked against it and exact float equality is the wrong test. The
        # tolerance is set at half a unit in the last recorded place. It is not loose:
        # the alignment bug this check exists to catch produced 0.168 against 0.924, off
        # by 0.76, which is nine orders of magnitude outside it.
        tolerance = 5e-10
        difference = primary - args.expect_primary
        report["known_answer"] = {
            "expected": args.expect_primary,
            "observed": primary,
            "difference": difference,
            "tolerance": tolerance,
            "passed": abs(difference) < tolerance,
        }
        args.output.write_text(json.dumps(report, indent=1, sort_keys=True))
        if abs(difference) >= tolerance:
            raise SystemExit(
                f"KNOWN-ANSWER CHECK FAILED: got {primary!r}, expected "
                f"{args.expect_primary!r}, difference {difference!r}, "
                f"tolerance {tolerance!r}"
            )
        print(f"  known-answer check PASSED: {primary:.12f} against "
              f"{args.expect_primary}, difference {difference:.3e}, "
              f"tolerance {tolerance:.0e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
