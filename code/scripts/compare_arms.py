#!/usr/bin/env python3
"""Per-task comparison across completed arms, with paired participant CIs.

This produces the manuscript's main comparison table. Three properties matter
and are enforced rather than assumed:

1. **Per task, not just macro.** The summary baselines already showed macro
   hides structure: going from one feature to nineteen moved the hyperglycemia
   tasks by nothing while the entire gain landed on hypoglycemia.

2. **Paired at the participant level**, over an identical participant and row
   set, so an eligibility difference between arms cannot masquerade as skill.

3. **Corrected for multiplicity** with Holm across the reported family.

Selection split only; this script must never read a held-out partition.

## Why the bootstrap is reimplemented here

`metrics.paired_participant_bootstrap_difference` is the reference, and it is
kept as the oracle. It cannot be used directly at this scale: it deep-copies
and concatenates roughly a thousand DataFrames per replicate, so 1000
replicates over 5.8M rows does not finish.

For an AUPRC metric that work is unnecessary. `_bootstrap_draw_copy` only
suffixes `participant_id` and `*_event_id`, which AUPRC never reads, and
average precision is invariant to row order. So the same computation reduces to
concatenating per-participant label/score arrays. The draw sequence is
reproduced exactly (same seed, same `sorted(keys)`, same `rng.choice`), and
`--verify-oracle` checks agreement against the reference implementation on a
subsample.
"""
from __future__ import annotations

import argparse
import json
import os
from functools import partial
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score

from glycoshift.metrics import holm_adjust, paired_participant_bootstrap_difference

TASKS = ("hypo_30", "hypo_60", "hyper_30", "hyper_60")
KEY_COLUMNS = ["source_id", "participant_id"]


def _needed_columns() -> list[str]:
    columns = [*KEY_COLUMNS, "partition", "context_end"]
    for task in TASKS:
        columns.append(f"{task}_label")
        columns.append(f"{task}_probability")
    for kind in ("hypo", "hyper"):
        columns.append(f"{kind}_currently_active")
    return columns


def _load_selection(run_dir: Path) -> pd.DataFrame:
    """Read only the columns the comparison needs.

    predictions.parquet carries event ids, onsets and continuous targets that
    this script never touches; reading them costs minutes and gigabytes.
    """
    path = run_dir / "predictions.parquet"
    if not path.exists():
        raise FileNotFoundError(f"arm has no predictions.parquet: {run_dir}")
    # Arms differ in which optional columns they carry: the summary-baseline
    # path drops the *_currently_active flags because run_training already
    # applied that filter, leaving a NaN probability on rows a task does not
    # apply to. Selecting only columns that exist keeps both shapes readable,
    # and `_eligible` handles the flag being absent.
    available = set(pq.read_schema(path).names)
    wanted = [column for column in _needed_columns() if column in available]
    missing_required = {*KEY_COLUMNS, "partition"}.difference(available)
    if missing_required:
        raise ValueError(
            f"predictions lack required columns {sorted(missing_required)}: {run_dir}"
        )
    frame = pd.read_parquet(path, columns=wanted)
    selection = frame.loc[frame["partition"] == "selection"].copy()
    if selection.empty:
        raise ValueError(f"arm has no selection rows: {run_dir}")
    return selection


def _eligible(frame: pd.DataFrame, task: str) -> pd.DataFrame:
    kind = task.split("_")[0]
    active = f"{kind}_currently_active"
    if active in frame:
        frame = frame.loc[~frame[active].astype(bool)]
    return frame.dropna(subset=[f"{task}_label", f"{task}_probability"])


def _auprc_arrays(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(average_precision_score(labels, scores))


def _auprc(frame: pd.DataFrame, task: str) -> float:
    return _auprc_arrays(
        frame[f"{task}_label"].to_numpy(dtype=int),
        frame[f"{task}_probability"].to_numpy(dtype=float),
    )


def _per_participant_arrays(
    frame: pd.DataFrame, task: str
) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    keyed = frame.assign(
        _s=frame["source_id"].astype(str),
        _p=frame["participant_id"].astype(str),
    )
    out: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for key, group in keyed.groupby(["_s", "_p"], sort=False):
        out[(str(key[0]), str(key[1]))] = (
            group[f"{task}_label"].to_numpy(dtype=int),
            group[f"{task}_probability"].to_numpy(dtype=float),
        )
    return out


def paired_auprc_bootstrap(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    task: str,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    """Fast equivalent of the reference paired bootstrap, for AUPRC only."""
    candidate_groups = _per_participant_arrays(candidate, task)
    reference_groups = _per_participant_arrays(reference, task)
    if set(candidate_groups) != set(reference_groups):
        raise ValueError("paired bootstrap requires identical participants")

    keys = sorted(candidate_groups)
    # Outcome agreement: the arms must be scoring the same events, otherwise a
    # difference in labels would be read as a difference in skill.
    for key in keys:
        left = candidate_groups[key][0]
        right = reference_groups[key][0]
        if left.shape != right.shape or not np.array_equal(
            np.sort(left), np.sort(right)
        ):
            raise ValueError(f"paired bootstrap outcome mismatch at {key}")

    estimate = float(_auprc(candidate, task) - _auprc(reference, task))
    rng = np.random.default_rng(seed)
    differences: list[float] = []
    for _ in range(replicates):
        sampled = rng.choice(len(keys), size=len(keys), replace=True)
        picked = [keys[int(index)] for index in sampled]
        candidate_value = _auprc_arrays(
            np.concatenate([candidate_groups[k][0] for k in picked]),
            np.concatenate([candidate_groups[k][1] for k in picked]),
        )
        reference_value = _auprc_arrays(
            np.concatenate([reference_groups[k][0] for k in picked]),
            np.concatenate([reference_groups[k][1] for k in picked]),
        )
        difference = float(candidate_value - reference_value)
        if np.isfinite(difference):
            differences.append(difference)
    if not differences:
        return {
            "estimate": estimate,
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "p_value_two_sided": float("nan"),
        }
    values = np.asarray(differences, dtype=float)
    lower_tail = (float(np.sum(values <= 0)) + 1) / (len(values) + 1)
    upper_tail = (float(np.sum(values >= 0)) + 1) / (len(values) + 1)
    return {
        "estimate": estimate,
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "p_value_two_sided": float(min(1.0, 2 * min(lower_tail, upper_tail))),
    }


def _bootstrap_cell(
    combo: tuple[str, str],
    *,
    frames: dict[str, pd.DataFrame],
    reference: str,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    """One (arm, task) paired bootstrap. Top level so it can be forked."""
    name, task = combo
    # One BLAS thread per worker; otherwise N workers each spawn N threads and
    # the box thrashes instead of parallelising.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        return paired_auprc_bootstrap(
            _eligible(frames[name], task),
            _eligible(frames[reference], task),
            task=task,
            replicates=replicates,
            seed=seed,
        )
    except ValueError as error:
        # An eligibility mismatch is a finding, not something to paper over by
        # silently dropping to an unpaired test.
        return {"error": str(error)}


def _verify_against_oracle(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    task: str,
    participants: int,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    """Agreement check on a participant subsample.

    The reference implementation cannot run at full scale, which is the whole
    reason for the fast path, so equivalence is established on a subset and the
    fast path is then trusted at scale.
    """
    keys = sorted(
        set(
            zip(  # noqa: B905
                candidate["source_id"].astype(str),
                candidate["participant_id"].astype(str),
            )
        )
    )[:participants]
    wanted = set(keys)

    def restrict(frame: pd.DataFrame) -> pd.DataFrame:
        mask = [
            (str(s), str(p)) in wanted
            for s, p in zip(  # noqa: B905
                frame["source_id"], frame["participant_id"]
            )
        ]
        return frame.loc[np.asarray(mask)].copy()

    small_candidate = restrict(candidate)
    small_reference = restrict(reference)

    fast = paired_auprc_bootstrap(
        small_candidate,
        small_reference,
        task=task,
        replicates=replicates,
        seed=seed,
    )
    oracle = paired_participant_bootstrap_difference(
        small_candidate,
        small_reference,
        metric=lambda df: _auprc(df, task),
        replicates=replicates,
        seed=seed,
    )
    deltas = {
        key: abs(float(fast[key]) - float(oracle[key]))
        for key in ("estimate", "ci_low", "ci_high", "p_value_two_sided")
        if np.isfinite(fast[key]) and np.isfinite(oracle[key])
    }
    return {
        "participants": len(keys),
        "replicates": replicates,
        "fast": fast,
        "oracle": oracle,
        "max_abs_delta": max(deltas.values()) if deltas else float("nan"),
        "identical": all(value == 0.0 for value in deltas.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        metavar="NAME=RUN_DIR",
        help="repeatable, e.g. --arm xgboost=../experiments/baseline-cpu-run/xgboost",
    )
    parser.add_argument(
        "--reference",
        required=True,
        help="arm name every other arm is compared against",
    )
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="parallel (arm, task) bootstrap cells; each holds the frames "
        "copy-on-write, so raise it against free RAM, not core count",
    )
    parser.add_argument(
        "--verify-oracle",
        type=int,
        default=0,
        metavar="N_PARTICIPANTS",
        help="check the fast bootstrap against the reference on N participants",
    )
    parser.add_argument("--output", default="-")
    args = parser.parse_args()

    arms: dict[str, Path] = {}
    for entry in args.arm:
        name, _, path = entry.partition("=")
        if not name or not path:
            raise ValueError(f"--arm must be NAME=RUN_DIR, got {entry!r}")
        arms[name] = Path(path).expanduser().resolve()
    if args.reference not in arms:
        raise ValueError(f"reference {args.reference!r} is not among --arm names")

    frames = {name: _load_selection(path) for name, path in arms.items()}
    print(
        json.dumps({"loaded": {n: int(len(f)) for n, f in frames.items()}}),
        flush=True,
    )

    report: dict[str, object] = {
        "reference": args.reference,
        "replicates": args.replicates,
        "seed": args.seed,
        "arms": {name: str(path) for name, path in arms.items()},
        "point_estimates": {},
        "paired_vs_reference": {},
    }

    for name, frame in frames.items():
        per_task: dict[str, float] = {}
        counts: dict[str, int] = {}
        for task in TASKS:
            eligible = _eligible(frame, task)
            per_task[task] = _auprc(eligible, task)
            counts[task] = int(len(eligible))
        finite = [v for v in per_task.values() if np.isfinite(v)]
        macro = float(np.mean(finite)) if finite else float("nan")
        report["point_estimates"][name] = {
            "per_task": per_task,
            "n": counts,
            "macro": macro,
        }
        print(json.dumps({"arm": name, "macro": macro, **per_task}), flush=True)

    reference_frame = frames[args.reference]

    if args.verify_oracle:
        other = next(n for n in frames if n != args.reference)
        verification = _verify_against_oracle(
            _eligible(frames[other], TASKS[0]),
            _eligible(reference_frame, TASKS[0]),
            task=TASKS[0],
            participants=args.verify_oracle,
            replicates=min(args.replicates, 200),
            seed=args.seed,
        )
        report["oracle_verification"] = verification
        print(
            json.dumps(
                {
                    "oracle_check": other,
                    "identical": verification["identical"],
                    "max_abs_delta": verification["max_abs_delta"],
                }
            ),
            flush=True,
        )

    # Each (arm, task) cell is an independent 1000-replicate bootstrap whose
    # cost is dominated by sorting ~5.8M scores per replicate. That is inherent
    # to average precision, not an implementation defect, so the lever is
    # parallelism across cells. Workers are forked after the frames are loaded,
    # so the read-only arrays are shared copy-on-write rather than duplicated.
    combos = [
        (name, task)
        for name in frames
        if name != args.reference
        for task in TASKS
    ]
    worker = partial(
        _bootstrap_cell,
        frames=frames,
        reference=args.reference,
        replicates=args.replicates,
        seed=args.seed,
    )
    if args.workers > 1 and len(combos) > 1:
        with get_context("fork").Pool(processes=args.workers) as pool:
            outcomes = pool.map(worker, combos)
    else:
        outcomes = [worker(combo) for combo in combos]

    raw_p: dict[str, float] = {}
    for (name, task), result in zip(combos, outcomes):  # noqa: B905
        report["paired_vs_reference"].setdefault(name, {})[task] = result
        if "p_value_two_sided" in result:
            raw_p[f"{name}:{task}"] = float(result["p_value_two_sided"])
        print(json.dumps({"arm": name, "task": task, **result}), flush=True)

    if raw_p:
        report["holm_adjusted_p"] = holm_adjust(raw_p)

    payload = json.dumps(report, indent=2, default=float)
    if args.output == "-":
        print(payload)
    else:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
