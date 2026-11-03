"""Paired bootstrap over the eleven development sources, for the frozen primary.

WHAT THIS IS NOT. The protocol registers a source-stratified paired bootstrap that
resamples event-positive PARTICIPANTS. That procedure needs the per-participant matched
events, which live with the prediction artifacts on the execution host and are not in
this checkout. It has not been run and this does not stand in for it.

WHAT THIS IS. The primary endpoint is an equal-source macro: each source contributes one
participant-macro sensitivity and the eleven are averaged with equal weight. That makes
the source the unit of the outer average, and the per-source values ARE archived. This
resamples those eleven units with replacement, paired, so both arms are always scored on
the same drawn multiset of sources.

It therefore quantifies between-source variability while holding each source's internal
participant-macro fixed. It is the weaker of the two procedures and it is directionally
the more relevant one for this particular result: four sources holding two or three
scored participants carry two thirds of the 60-minute gap, so a reader's first question
is what happens when the source composition is perturbed, which is exactly what this
answers and what a participant-level resample inside fixed sources would not.

Every run first reproduces both arms' frozen primaries from the per-source values it is
about to resample. If that fails the numbers are not the endpoint's and the run refuses.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HORIZONS = ("30", "60")
REPRODUCTION_TOLERANCE = 1e-12


def _load_arm(root: Path, arm: str, restricted: bool) -> dict:
    """Read one arm. The two analyses archive the same quantity under two schemas."""
    path = ((root / f"{arm}.json") if restricted
            else (root / arm / "operational_metrics.json"))
    if not path.exists():
        raise SystemExit(f"no archived endpoint for {arm}: {path}")
    payload = json.loads(path.read_text())
    if payload["nominal_fapd"] != 0.5:
        raise SystemExit(f"{arm} is at nominal {payload['nominal_fapd']}, not 0.5")
    if not restricted:
        if payload["evaluation_partition"] != "selection":
            raise SystemExit(f"{arm} is scored on {payload['evaluation_partition']}")
        if payload["threshold_partition"] != "calibration":
            raise SystemExit(
                f"{arm} fits its threshold on {payload['threshold_partition']}"
            )
    return payload


def _stored_primary(payload: dict, restricted: bool) -> float:
    value = payload["primary_equal_horizon_mean_event_sensitivity"]
    return float(value["global"] if restricted else value)


def _per_source(payload: dict, restricted: bool) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for horizon in HORIZONS:
        if restricted:
            block = payload[horizon]["global"]["per_source"]
            field = "sensitivity"
        else:
            block = payload["horizons"][horizon]["selection_metrics"]["per_source"]
            field = "participant_macro_event_sensitivity"
        for source, values in block.items():
            out.setdefault(source, {})[horizon] = float(values[field])
    return out


def _primary(per_source: dict[str, dict[str, float]], order: list[str]) -> float:
    """The endpoint: equal-source macro per horizon, then the equal horizon mean."""
    return float(
        np.mean([np.mean([per_source[s][h] for s in order]) for h in HORIZONS])
    )


def _gate(arm: str, payload: dict, per_source: dict, order: list[str],
          restricted: bool) -> float:
    stored = _stored_primary(payload, restricted)
    rebuilt = _primary(per_source, order)
    if abs(stored - rebuilt) > REPRODUCTION_TOLERANCE:
        raise SystemExit(
            f"{arm}: per-source values rebuild {rebuilt:.12f} against a frozen primary "
            f"of {stored:.12f}, a difference of {abs(stored - rebuilt):.3e}. These are "
            f"not the endpoint's numbers."
        )
    return stored


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--restricted", action="store_true",
                        help="read the euglycaemic-restriction schema instead")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--comparator", required=True)
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)

    payloads = {a: _load_arm(arguments.runs, a, arguments.restricted)
                for a in (arguments.candidate, arguments.comparator)}
    sources = {a: _per_source(p, arguments.restricted) for a, p in payloads.items()}

    order = sorted(sources[arguments.candidate])
    if order != sorted(sources[arguments.comparator]):
        raise SystemExit("the two arms do not share a source set; they are not paired")
    if len(order) != 11:
        raise SystemExit(f"expected eleven development sources, found {len(order)}")

    frozen = {a: _gate(a, payloads[a], sources[a], order, arguments.restricted)
              for a in payloads}
    observed = frozen[arguments.candidate] - frozen[arguments.comparator]

    rng = np.random.default_rng(arguments.seed)
    # One draw of sources per replicate, used for BOTH arms. Drawing separately would
    # measure the sum of two variances and inflate the interval.
    index = rng.integers(0, len(order), size=(arguments.replicates, len(order)))
    matrix = {
        arm: np.array([[sources[arm][s][h] for s in order] for h in HORIZONS])
        for arm in payloads
    }
    draws = {
        arm: matrix[arm][:, index].mean(axis=2).mean(axis=0) for arm in payloads
    }
    differences = draws[arguments.candidate] - draws[arguments.comparator]

    lower, upper = np.percentile(differences, [2.5, 97.5])
    result = {
        "procedure": "paired nonparametric bootstrap over the eleven sources",
        "not_the_registered_procedure": (
            "the protocol registers a participant-level resample within source strata; "
            "that needs per-participant matched events, which are not in this checkout"
        ),
        "candidate": arguments.candidate,
        "comparator": arguments.comparator,
        "analysis": ("euglycaemic restriction" if arguments.restricted
                     else "frozen primary"),
        "nominal_fapd": payloads[arguments.candidate]["nominal_fapd"],
        "sources": order,
        "frozen_primary": frozen,
        "observed_difference": observed,
        "replicates": arguments.replicates,
        "seed": arguments.seed,
        "ci95": [float(lower), float(upper)],
        "bootstrap_mean": float(differences.mean()),
        "bootstrap_sd": float(differences.std(ddof=1)),
        "fraction_at_or_below_zero": float((differences <= 0.0).mean()),
    }
    if arguments.output:
        arguments.output.write_text(json.dumps(result, indent=1) + "\n")

    print(f"{arguments.candidate} minus {arguments.comparator}")
    for arm, value in frozen.items():
        print(f"  frozen primary {arm:28s} {value:.9f}  (rebuilt from per-source)")
    print(f"  observed difference          {observed:+.9f}")
    print(f"  95% interval over sources    [{lower:+.9f}, {upper:+.9f}]")
    print(f"  bootstrap sd                 {differences.std(ddof=1):.9f}")
    print(f"  replicates at or below zero  {(differences <= 0.0).mean():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
