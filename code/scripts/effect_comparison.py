'Effect comparison.'






















from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HORIZONS = ("30", "60")
TOLERANCE = 1e-12


def _scored_participants(runs: Path, arm: str) -> dict[str, dict[str, int]]:
    """Participants and event-positive participants per source, from the frozen run."""
    payload = json.loads((runs / arm / "operational_metrics.json").read_text())
    block = payload["horizons"]["30"]["selection_metrics"]["per_source"]
    return {s: {"participants": int(v["participant_count"]),
                "event_positive": int(v["event_positive_participant_count"])}
            for s, v in block.items()}


def _frozen(runs: Path, arm: str) -> tuple[float, dict[str, float]]:
    payload = json.loads((runs / arm / "operational_metrics.json").read_text())
    if payload["evaluation_partition"] != "selection" or payload["nominal_fapd"] != 0.5:
        raise SystemExit(f"{arm}: unexpected partition or budget")
    per: dict[str, float] = {}
    for horizon in HORIZONS:
        block = payload["horizons"][horizon]["selection_metrics"]["per_source"]
        for source, values in block.items():
            per.setdefault(source, {})[horizon] = float(
                values["participant_macro_event_sensitivity"]
            )
    return float(payload["primary_equal_horizon_mean_event_sensitivity"]), per


def _restricted(directory: Path, arm: str) -> tuple[float, dict[str, float]]:
    payload = json.loads((directory / f"{arm}.json").read_text())
    if payload["nominal_fapd"] != 0.5:
        raise SystemExit(f"{arm}: unexpected budget")
    per: dict[str, float] = {}
    for horizon in HORIZONS:
        block = payload[horizon]["global"]["per_source"]
        for source, values in block.items():
            per.setdefault(source, {})[horizon] = float(values["sensitivity"])
    return float(payload["primary_equal_horizon_mean_event_sensitivity"]["global"]), per


def _macro(per: dict[str, dict[str, float]], order: list[str]) -> float:
    return float(np.mean([np.mean([per[s][h] for s in order]) for h in HORIZONS]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--restricted-dir", required=True, type=Path)
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)

    # model effect: the control's lead over the strongest trained arm, frozen primary.
    # eligibility effect: the trained arm's lead over the control once eligibility
    # moves.
    frozen_control, per_control = _frozen(arguments.runs, "persistence_seed17")
    frozen_tcn, per_tcn = _frozen(arguments.runs, "tcn_erm_seed17")
    rest_control, per_rc = _restricted(arguments.restricted_dir, "persistence_seed17")
    rest_lstm, per_rl = _restricted(
        arguments.restricted_dir, "balanced_lstm_erm_seed17"
    )

    order = sorted(per_control)
    if len(order) != 11:
        raise SystemExit(f"expected eleven sources, found {len(order)}")
    for name, per, stored in (
        ("persistence frozen", per_control, frozen_control),
        ("tcn frozen", per_tcn, frozen_tcn),
        ("persistence restricted", per_rc, rest_control),
        ("lstm restricted", per_rl, rest_lstm),
    ):
        if sorted(per) != order:
            raise SystemExit(f"{name}: source set differs; the effects are not paired")
        if abs(_macro(per, order) - stored) > TOLERANCE:
            raise SystemExit(
                f"{name}: per-source rebuild {_macro(per, order):.12f} against a "
                f"stored {stored:.12f}. These are not the endpoint's numbers."
            )

    def matrix(per):
        return np.array([[per[s][h] for s in order] for h in HORIZONS])

    m_control, m_tcn = matrix(per_control), matrix(per_tcn)
    m_rc, m_rl = matrix(per_rc), matrix(per_rl)

    model_observed = frozen_control - frozen_tcn
    eligibility_observed = rest_lstm - rest_control
    gap_observed = eligibility_observed - model_observed

    rng = np.random.default_rng(arguments.seed)
    index = rng.integers(0, len(order), size=(arguments.replicates, len(order)))

    def draw(m):
        return m[:, index].mean(axis=2).mean(axis=0)

    model = draw(m_control) - draw(m_tcn)
    eligibility = draw(m_rl) - draw(m_rc)
    gap = eligibility - model

    def interval(x):
        low, high = np.percentile(x, [2.5, 97.5])
        return float(low), float(high)

    # leave one source out, which is what an interval over eleven units invites
    loo = {}
    for dropped in order:
        kept = [s for s in order if s != dropped]
        loo[dropped] = {
            "model": _macro(per_control, kept) - _macro(per_tcn, kept),
            "eligibility": _macro(per_rl, kept) - _macro(per_rc, kept),
        }

    # and the raw per-source contrasts, so nothing hides inside the average
    counts = _scored_participants(arguments.runs, "persistence_seed17")
    per_source = {
        s: {
            "model": float(
                np.mean([per_control[s][h] - per_tcn[s][h] for h in HORIZONS])),
            "eligibility": float(
                np.mean([per_rl[s][h] - per_rc[s][h] for h in HORIZONS])),
            **counts[s],
        }
        for s in order
    }

    result = {
        "procedure": "paired source resample; both effects share one drawn multiset",
        "not_the_registered_procedure": (
            "the protocol registers a participant-level resample, which "
            "from retained artifacts, and this holds each source macro fixed"
        ),
        "sources": order,
        "replicates": arguments.replicates,
        "seed": arguments.seed,
        "model_effect": {
            "definition": "persistence_seed17 minus tcn_erm_seed17, frozen primary",
            "observed": model_observed,
            "ci95": interval(model),
        },
        "eligibility_effect": {
            "definition": "balanced_lstm_erm_seed17 minus persistence_seed17, cut 70",
            "observed": eligibility_observed,
            "ci95": interval(eligibility),
        },
        "difference_of_effects": {
            "definition": "eligibility effect minus model effect, paired within draw",
            "observed": gap_observed,
            "ci95": interval(gap),
            "fraction_at_or_below_zero": float((gap <= 0.0).mean()),
        },
        "leave_one_source_out": loo,
        "per_source_contrast": per_source,
    }
    if arguments.output:
        arguments.output.write_text(json.dumps(result, indent=1) + "\n")

    print("effect comparison on the eleven development sources")
    for key in ("model_effect", "eligibility_effect", "difference_of_effects"):
        block = result[key]
        low, high = block["ci95"]
        print(f"  {key:24s} {block['observed']:+.9f}  [{low:+.9f}, {high:+.9f}]")
    print(f"  replicates with eligibility <= model  "
          f"{result['difference_of_effects']['fraction_at_or_below_zero']:.4f}")
    print("\n  leave-one-source-out")
    for source, values in loo.items():
        print(f"    drop {source:16s} model {values['model']:+.4f}   "
              f"eligibility {values['eligibility']:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
