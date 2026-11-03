#!/usr/bin/env python3
'Fixed pair interaction.'































from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
M6 = ROOT / "experiments" / "20260827-glycoshift-m6-anchor-aware-v1" / "analysis"
SB = ROOT / "experiments" / "20260903-erm-lstm-seeds-v1" / "source-bootstrap"
MATRIX = M6 / "composition-matrix.json"
ECOMP = SB / "effect-comparison.json"
UNRES = SB / "persistence_vs_balanced_lstm_erm_seed17.json"
REST = SB / "RESTRICTED_balanced_lstm_erm_seed17_vs_persistence.json"
OUT = SB / "fixed-pair-interaction.json"

CANDIDATE = "balanced_lstm_erm_seed17"
CONTROL = "persistence_seed17"
TOL = 1e-12


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def original_frame_per_source(arm: str) -> dict[str, float]:
    """The primary is the equal mean of the two horizons, per source."""
    cells = load(MATRIX)["cells"][arm]
    return {
        source: (cells["30"]["per_source"][source]
                 ["participant_macro_event_sensitivity"]
                 + cells["60"]["per_source"][source]
                 ["participant_macro_event_sensitivity"]) / 2
        for source in cells["30"]["per_source"]
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true",
                    help=f"write {OUT.relative_to(ROOT)}; otherwise print only")
    args = ap.parse_args()

    ecomp = load(ECOMP)
    sources = sorted(ecomp["sources"])

    control = original_frame_per_source("persist")
    candidate = original_frame_per_source("lstm")
    if sorted(control) != sources or sorted(candidate) != sources:
        print("REFUSE  the composition matrix and the bootstrap disagree about the "
              "source list", file=sys.stderr)
        return 1

    # Fail closed before using the reconstruction for anything.
    frozen = load(UNRES)["frozen_primary"]
    for arm, values, key in ((CONTROL, control, CONTROL),
                             (CANDIDATE, candidate, CANDIDATE)):
        rebuilt = float(np.mean([values[s] for s in sources]))
        if abs(rebuilt - frozen[key]) > TOL:
            print(f"REFUSE  reconstructed macro for {arm} is {rebuilt!r} "
                  f"against the archived {frozen[key]!r}", file=sys.stderr)
            return 1

    original = np.array([candidate[s] - control[s] for s in sources])
    restricted = np.array([ecomp["per_source_contrast"][s]["eligibility"]
                           for s in sources])
    archived_rest = load(REST)["observed_difference"]
    if abs(float(restricted.mean()) - archived_rest) > TOL:
        print(f"REFUSE  restricted per-source values mean {restricted.mean()!r} "
              f"against the archived {archived_rest!r}", file=sys.stderr)
        return 1

    replicates = int(ecomp["replicates"])
    seed = int(ecomp["seed"])
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(sources), size=(replicates, len(sources)))
    boot_original = original[draw].mean(axis=1)
    boot_restricted = restricted[draw].mean(axis=1)
    boot_interaction = boot_restricted - boot_original

    def summarise(observed: float, draws: np.ndarray) -> dict:
        lo, hi = (float(v) for v in np.percentile(draws, [2.5, 97.5]))
        return {
            "observed": float(observed),
            "ci95": [lo, hi],
            "bootstrap_mean": float(draws.mean()),
            "bootstrap_sd": float(draws.std(ddof=1)),
            "fraction_at_or_below_zero": float(np.mean(draws <= 0)),
        }

    payload = {
        "procedure": "paired source resample; one drawn multiset serves both frames",
        "supersedes": (
            "the difference_of_effects cell of effect-comparison.json, which subtracts "
            "contrasts that differ in arm, orientation and frame at once"
        ),
        "not_the_registered_procedure": load(REST)["not_the_registered_procedure"],
        "candidate": CANDIDATE,
        "control": CONTROL,
        "nominal_fapd": load(REST)["nominal_fapd"],
        "sources": sources,
        "replicates": replicates,
        "seed": seed,
        "reconstruction_check": {
            "original_frame_per_source_rebuilt_from": str(MATRIX.relative_to(ROOT)),
            "tolerance": TOL,
            "archived_frozen_primary": frozen,
            "archived_restricted_difference": archived_rest,
        },
        "original_frame": summarise(original.mean(), boot_original),
        "restricted_frame": summarise(restricted.mean(), boot_restricted),
        "interaction": summarise(restricted.mean() - original.mean(), boot_interaction),
        "per_source": {
            s: {"original": float(original[i]), "restricted": float(restricted[i])}
            for i, s in enumerate(sources)
        },
    }

    for name in ("original_frame", "restricted_frame", "interaction"):
        cell = payload[name]
        print(f"{name:18s} {cell['observed']:+.10f}  "
              f"95% [{cell['ci95'][0]:+.6f}, {cell['ci95'][1]:+.6f}]  "
              f"frac<=0 {cell['fraction_at_or_below_zero']:.4f}")

    if args.write:
        OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
