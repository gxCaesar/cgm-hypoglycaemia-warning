#!/usr/bin/env python3
"""Compute the D0 seed standard deviation and apply the pre-registered read-out.

Consumes the four-seed scoring produced by recompute_operational_endpoint.py for
one architecture and reports sigma with a confidence interval, the seed cost
curve, and the axis verdicts. Every threshold here was fixed in
sota-method-gate-2026-08-v1-4.md before any of these numbers existed; this script
substitutes, it does not decide.

It reports an upper bound on seed variance, not seed variance. The D0 lanes ran
with thread counts measured at 165 / 175 / 160 and no *_NUM_THREADS pinned, so
per-lane scheduling noise is folded in and cannot be separated from the seed
effect. A verdict that lands near a threshold must be re-measured with threads
pinned rather than read as final.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

# Pre-registered in v1.4 D2/D6 before D0 reported.
Z_80_POWER = 2.80          # two-sided 95% at 80% power
Z_50_POWER = 2.00          # "interval just excludes zero"
BUDGET_SEEDS_PER_ARM = 10  # what the frozen 400 card-hour ceiling buys
AXIS_SPANS = {"objective": 0.0797, "architecture": 0.1694}
VIABLE_FRACTION = 0.50     # <= this share of an axis span is viable
MARGINAL_FRACTION = 0.75   # above this is not viable at the stated budget
CURRENT_FLOOR = 0.03


def _load_endpoint(directory: Path) -> dict[str, float]:
    summary = json.loads((directory / "summary.json").read_text())
    values = {
        entry["arm_id"]: float(entry["mean_severe_event_sensitivity"])
        for entry in summary["ranking"]
    }
    if len(values) < 3:
        raise ValueError(
            f"seed variance needs at least three arms, found {len(values)}"
        )
    return values


def _sigma_interval(values: list[float]) -> tuple[float, float, float, int]:
    """Return (sigma, lower, upper, df) using the chi-square interval."""

    n = len(values)
    df = n - 1
    sigma = statistics.stdev(values)
    # Chi-square quantiles for 95% two-sided, df 2..5; beyond that the caller
    # should widen this table rather than extrapolate.
    chi2 = {
        2: (0.0506, 7.378),
        3: (0.2158, 9.348),
        4: (0.4844, 11.143),
        5: (0.8312, 12.833),
    }
    if df not in chi2:
        raise ValueError(f"no tabulated chi-square interval for df={df}")
    lower_q, upper_q = chi2[df]
    lower = sigma * math.sqrt(df / upper_q)
    upper = sigma * math.sqrt(df / lower_q)
    return sigma, lower, upper, df


def _seeds_for(margin: float, sigma: float, z: float) -> int:
    return math.ceil(2.0 * (z * sigma / margin) ** 2)


def _budget_margin(sigma: float) -> float:
    return Z_80_POWER * sigma * math.sqrt(2.0 / BUDGET_SEEDS_PER_ARM)


def _verdict(sigma: float, span: float) -> str:
    fraction = _budget_margin(sigma) / span
    if fraction <= VIABLE_FRACTION:
        return "viable"
    if fraction <= MARGINAL_FRACTION:
        return "marginal"
    return "not viable at this budget"


def build_report(directory: Path) -> dict[str, Any]:
    values = _load_endpoint(directory)
    ordered = [values[key] for key in sorted(values)]
    sigma, lower, upper, df = _sigma_interval(ordered)

    margin_at_budget = Z_80_POWER * sigma * math.sqrt(2.0 / BUDGET_SEEDS_PER_ARM)
    report: dict[str, Any] = {
        "arms": values,
        "n": len(ordered),
        "df": df,
        "mean": statistics.fmean(ordered),
        "sigma": sigma,
        "sigma_ci95": [lower, upper],
        "sigma_is_upper_bound": True,
        "sigma_caveat": (
            "thread counts 165/175/160 with no *_NUM_THREADS pinned, so per-lane "
            "scheduling noise is confounded with seed variance"
        ),
        "detectable_margin_at_budget_80power": margin_at_budget,
        "current_floor": CURRENT_FLOOR,
        "floor_is_detectable": bool(CURRENT_FLOOR >= margin_at_budget),
        "seed_cost_80power": {
            f"{margin:.3f}": _seeds_for(margin, sigma, Z_80_POWER)
            for margin in (0.030, 0.050, 0.079, 0.100, 0.169)
        },
        "seed_cost_50power": {
            f"{margin:.3f}": _seeds_for(margin, sigma, Z_50_POWER)
            for margin in (0.030, 0.050, 0.079, 0.100, 0.169)
        },
        "axis_verdicts": {
            axis: {
                "span": span,
                "fraction_of_span_required": margin_at_budget / span,
                "verdict": _verdict(sigma, span),
            }
            for axis, span in AXIS_SPANS.items()
        },
    }
    # The straddling case is the one contamination can flip, and it flips only
    # toward abandonment, so it is named explicitly rather than left to reading.
    report["requires_thread_pinned_remeasurement"] = bool(
        lower < CURRENT_FLOOR < upper
        or any(
            VIABLE_FRACTION
            < margin_at_budget / span
            <= MARGINAL_FRACTION
            for span in AXIS_SPANS.values()
        )
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    report = build_report(arguments.input)
    text = json.dumps(report, indent=2, sort_keys=True)
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
