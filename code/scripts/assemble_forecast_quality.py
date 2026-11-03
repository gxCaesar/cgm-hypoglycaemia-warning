#!/usr/bin/env python3
'Assemble forecast quality.'

























from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

RMSE = re.compile(
    r"synthetic (?P<kind>blend|noisy_truth) lam=(?P<lam>[\d.]+) h=(?P<h>\d+): "
    r"pred_rmse=(?P<pred>[\d.]+)\s+target_rmse=(?P<target>[\d.]+)\s+"
    r"achieved_rmse=(?P<achieved>[\d.]+)"
)
DIAGNOSTIC = re.compile(
    r"forecast diagnostic h=(?P<h>\d+): corr\(prediction, cgm_last\)=(?P<pc>[\d.]+)\s+"
    r"corr\(target, cgm_last\)=(?P<tc>[\d.]+)\s+"
    r"corr\(prediction, target\)=(?P<pt>[\d.]+)\s+"
    r"RMSE=(?P<rmse>[\d.]+)"
)
HORIZONS = ("30", "60")
# Matched to four significant figures is the claim; 0.01 mg/dL is two orders below the
# smallest non-zero abscissa in the sweep.
MATCH_TOLERANCE_MG_DL = 0.01


def primary(path: Path) -> float:
    payload = json.loads(path.read_text())
    value = payload["primary_equal_horizon_mean_event_sensitivity"]
    value = value["global"] if isinstance(value, dict) else value
    if not 0.0 <= float(value) <= 1.0:
        raise SystemExit(f"{path.name}: primary {value} is not a sensitivity")
    return float(value)


def sweep(directory: Path, kind: str) -> list[dict]:
    rows = []
    prefix = "blend" if kind == "blend" else "noisy"
    for path in sorted(directory.glob(f"{prefix}_l*.json")):
        log = path.with_suffix(".log")
        if not log.is_file():
            raise SystemExit(
                f"{path.name} has no run log; the abscissa lives only there")
        matches = {
            m.group("h"): m
            for m in RMSE.finditer(log.read_text())
            if m.group("kind") == kind
        }
        missing = [h for h in HORIZONS if h not in matches]
        if missing:
            raise SystemExit(f"{log.name}: no {kind} RMSE line for horizon {missing}")
        rows.append({
            "cell": path.stem,
            "lambda": float(matches["30"].group("lam")),
            "primary": primary(path),
            "achieved_rmse_mg_dl": {
                h: float(matches[h].group("achieved")) for h in HORIZONS
            },
            "model_forecast_rmse_mg_dl": {
                h: float(matches[h].group("pred")) for h in HORIZONS
            },
        })
    if len(rows) < 4:
        raise SystemExit(f"a {kind} sweep of {len(rows)} points is not a dose-response")
    return rows


def crossing(rows: list[dict], level: float, horizon: str = "30") -> dict:
    """Where the sweep meets a reference score, on the RMSE axis.

    Interpolated when the reference lies between two measured points, extrapolated from
    the last segment when it does not. The two are reported under different keys because
    a figure that draws them alike is claiming something it did not measure.
    """
    points = sorted((r["achieved_rmse_mg_dl"][horizon], r["primary"]) for r in rows)
    for (x0, y0), (x1, y1) in zip(points[:-1], points[1:], strict=False):
        if (y0 - level) * (y1 - level) <= 0 and y0 != y1:
            return {"mg_dl": x0 + (level - y0) * (x1 - x0) / (y1 - y0),
                    "basis": "interpolated between measured points"}
    (xa, ya), (xb, yb) = points[0], points[1]
    slope = (yb - ya) / (xb - xa)
    return {"mg_dl": xa + (level - ya) / slope,
            "basis": "extrapolated from the two lowest-error measured points"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast-dir", required=True, type=Path)
    parser.add_argument("--control", required=True, type=Path,
                        help="the arm the sweep is read against")
    parser.add_argument("--diagnostic", required=True, type=Path,
                        help="the run log of the arm the sweep is built on")
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args(argv)

    blend, noisy = sweep(arguments.forecast_dir, "blend"), sweep(
        arguments.forecast_dir, "noisy_truth")

    # The unbiased sweep has no zero-error cell of its own, because unbiased error of
    # magnitude zero IS the perfect forecast, and that is measured once as the shrunk
    # arm's lambda = 1. Without it the crossing has to be extrapolated past the end of
    # the data and lands at 3.3 mg/dL; with it the crossing is an interpolation at 5.0,
    # which is also the figure this project's own record carries. Extrapolating past
    # measured points when a measured point exists is the error, not the disagreement.
    perfect = [r for r in blend if r["achieved_rmse_mg_dl"]["30"] == 0.0]
    if len(perfect) != 1:
        raise SystemExit(
            f"expected exactly one zero-error cell to share between the sweeps, "
            f"found {len(perfect)}"
        )
    noisy = noisy + [dict(perfect[0], cell=perfect[0]["cell"] + " (shared endpoint)")]

    # The controlled-comparison claim, checked rather than asserted. The sweeps are not
    # the same length -- the unbiased arm has no zero-noise cell, because zero noise is
    # the perfect forecast the other arm already carries -- so they are matched on the
    # lambdas they share, and the overlap must be non-trivial for the claim to mean
    # anything.
    shared = {r["lambda"]: r for r in blend}.keys() & {r["lambda"] for r in noisy}
    if len(shared) < 3:
        raise SystemExit(
            f"the two sweeps share only {len(shared)} settings, which is not a "
            "controlled comparison"
        )
    by_lambda = ({r["lambda"]: r for r in blend}, {r["lambda"]: r for r in noisy})
    for lam in sorted(shared):
        a, b = by_lambda[0][lam], by_lambda[1][lam]
        for h in HORIZONS:
            gap = abs(a["achieved_rmse_mg_dl"][h] - b["achieved_rmse_mg_dl"][h])
            if gap > MATCH_TOLERANCE_MG_DL:
                raise SystemExit(
                    f"lambda {lam} h={h}: the two arms differ by {gap:.4f} mg/dL, so "
                    "they are not matched on error magnitude and the comparison "
                    "between them is not controlled"
                )

    log = arguments.diagnostic.read_text()
    diagnostic = {
        m.group("h"): {
            "rmse_mg_dl": float(m.group("rmse")),
            "corr_prediction_with_current": float(m.group("pc")),
            "corr_truth_with_current": float(m.group("tc")),
            "corr_prediction_with_truth": float(m.group("pt")),
        }
        for m in DIAGNOSTIC.finditer(log)
    }
    if sorted(diagnostic) != sorted(HORIZONS):
        raise SystemExit("the forecast diagnostic is missing a horizon")
    # The guard against pairing this sweep with another arm's diagnostic.
    for h in HORIZONS:
        sweep_rmse = blend[0]["model_forecast_rmse_mg_dl"][h]
        if abs(diagnostic[h]["rmse_mg_dl"] - sweep_rmse) > MATCH_TOLERANCE_MG_DL:
            raise SystemExit(
                f"h={h}: the diagnostic reports RMSE {diagnostic[h]['rmse_mg_dl']} "
                f"against the sweep's {sweep_rmse}, so it describes a different arm"
            )

    control = primary(arguments.control)
    payload = {
        "control_primary": control,
        "control_arm": arguments.control.stem,
        "own_forecast_primary": primary(
            arguments.forecast_dir / "blend_l000.json"),
        "diagnostic_source": str(arguments.diagnostic.name),
        "second_architecture_forecast_primary": primary(
            arguments.forecast_dir / "tcn_forecast.json"),
        "horizons": list(HORIZONS),
        "matched_on_achieved_rmse": True,
        "match_tolerance_mg_dl": MATCH_TOLERANCE_MG_DL,
        "shrunk_toward_truth": blend,
        "unbiased_noise": noisy,
        "diagnostic": diagnostic,
        "shared_settings": sorted(shared),
        "crossing_of_control": {
            "shrunk_toward_truth": crossing(blend, control),
            "unbiased_noise": crossing(noisy, control),
        },
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=1) + "\n")

    cs = payload["crossing_of_control"]
    print(f"wrote {arguments.output}")
    print(f"  control {control:.6f}; its own forecast "
          f"{payload['own_forecast_primary']:.6f}")
    for name, block in cs.items():
        print(f"  {name} crosses at {block['mg_dl']:.2f} mg/dL ({block['basis']})")
    same = next(r for r in noisy if r["lambda"] == 0.0)
    own = next(r for r in blend if r["lambda"] == 0.0)
    print(f"  at a matched {own['achieved_rmse_mg_dl']['30']:.2f} mg/dL the model "
          f"scores {own['primary']:.4f} against noise's {same['primary']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
