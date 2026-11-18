"""Accept five-fold participant ledgers and report prespecified OOF point estimands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
from .revision_crossfit_evaluation import (
    CUTS,
    HORIZONS,
    KEYS,
    aggregate_oof_participants,
)
from .revision_crossfit_prepare import CROSSFIT_CONTRACT, validated_crossfit_metadata
from .revision_external import _bound_file, _json, _recheck_bindings

SEEDS = (17, 23, 42, 71, 101)
METHODS = ("lstm", "tcn", "persistence", "linear", "summary_logistic", "xgboost")
SENSITIVITY = "equal_source_participant_macro_fixed_sensitivity"
BURDEN = "equal_source_participant_macro_unmatched_alarms_per_patient_day"


def summarize_paired_points(points: dict, *, metric: str) -> dict:
    """Keep equal-horizon/equal-model-seed arithmetic after participant aggregation."""
    values = {}
    for arm in METHODS:
        seeds = SEEDS if arm in {"lstm", "tcn"} else (17,)
        for cut in CUTS:
            values[arm, cut] = float(
                np.mean(
                    [
                        points[arm, seed, cut, horizon][metric]
                        for seed in seeds
                        for horizon in HORIZONS
                    ]
                )
            )
    contrasts = {}
    for seed in SEEDS:
        at_cut = {
            cut: float(
                np.mean(
                    [
                        points["lstm", seed, cut, horizon][metric]
                        - points["persistence", 17, cut, horizon][metric]
                        for horizon in HORIZONS
                    ]
                )
            )
            for cut in CUTS
        }
        contrasts[str(seed)] = {
            "lstm_minus_current_glucose_54": at_cut[54],
            "lstm_minus_current_glucose_70": at_cut[70],
            "contrast_change_70_minus_54": at_cut[70] - at_cut[54],
        }
    return {
        "metric": metric,
        "method_points": [
            {"arm": arm, "glucose_cut": cut, "value": value}
            for (arm, cut), value in values.items()
        ],
        "model_seed_contrasts": contrasts,
        "mean_contrast_change_70_minus_54": float(
            np.mean([row["contrast_change_70_minus_54"] for row in contrasts.values()])
        ),
        "generalization_ci": None,
        "interval_status": "not_computed_unvalidated_generalization_interval",
    }


def collect_oof(config: dict) -> Path:
    """Read only completed bound outputs and metadata, never raw glucose or weights."""
    output = Path(config["output_dir"]).resolve()
    prepared = Path(config["prepared_dir"]).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("refusing to overwrite OOF summaries")
    expected_arms = {
        (arm, seed)
        for arm in METHODS
        for seed in (SEEDS if arm in {"lstm", "tcn"} else (17,))
    }
    arms = config["arms"]
    if (
        len(arms) != len(expected_arms)
        or {(row["arm"], row["model_seed"]) for row in arms} != expected_arms
    ):
        raise ValueError("OOF summary requires all six methods and five deep seeds")
    all_tables = {}
    folds = []
    source_order = None
    original_roles = None
    receipts = []
    for arm in arms:
        if len(arm["fold_outputs"]) != 5:
            raise ValueError("OOF method must supply exactly five outer folds")
        seen_folds = set()
        tables = {(cut, horizon): [] for cut in CUTS for horizon in HORIZONS}
        for directory in arm["fold_outputs"]:
            directory = Path(directory).resolve()
            receipt_path = directory / "receipt.json"
            receipt = json.loads(receipt_path.read_text())
            if (
                receipt.get("status") != "complete"
                or receipt.get("crossfit_contract") != CROSSFIT_CONTRACT
                or receipt.get("outer_refitting") is not False
                or receipt.get("prepared_dir") != str(prepared)
            ):
                raise ValueError(
                    "OOF producer is incomplete or belongs to another pool"
                )
            fold = receipt["outer_fold"]
            if isinstance(fold, bool) or fold not in range(5) or fold in seen_folds:
                raise ValueError("OOF producer fold identity is invalid or duplicated")
            seen_folds.add(fold)
            _recheck_bindings(receipt)
            freeze = json.loads(_bound_file(receipt["policy_freeze"]).read_text())
            _recheck_bindings(freeze)
            if freeze["outer_fold"] != fold or freeze["run_dir"] != receipt["run_dir"]:
                raise ValueError("OOF producer/freeze fold or run identity differs")
            model = freeze["summary"]
            model_arm = model.get("architecture", model.get("baseline"))
            model_config = load_config(Path(freeze["run_dir"]) / "config_resolved.yaml")
            if (
                model_arm != arm["arm"]
                or model_config["training"]["seed"] != arm["model_seed"]
            ):
                raise ValueError("OOF model identity or seed differs from declaration")
            audit, _, manifest, roles = validated_crossfit_metadata(
                prepared, model_config, outer_fold=fold
            )
            sources = tuple(audit["source_order"])
            if source_order is not None and sources != source_order:
                raise ValueError("OOF source order differs across producers")
            source_order = sources
            original = roles[[*KEYS, "original_partition"]].set_index(KEYS)
            if original_roles is not None:
                pd.testing.assert_frame_equal(
                    original.sort_index(), original_roles.sort_index()
                )
            original_roles = original
            outer_keys = set(
                roles.loc[roles.role.eq("outer_evaluation"), KEYS].itertuples(
                    index=False, name=None
                )
            )
            inventory = {
                (row["source_id"], row["participant_id"]): row for row in manifest
            }
            bound_paths = {
                Path(binding["path"]).resolve() for binding in receipt["outputs"]
            }
            for cut in CUTS:
                for horizon in HORIZONS:
                    path = (
                        directory
                        / f"private/evaluations/{cut}/{horizon}/participants.parquet"
                    )
                    if path not in bound_paths:
                        raise ValueError(
                            "OOF participant ledger is not bound by its receipt"
                        )
                    table = pd.read_parquet(path)
                    actual = set(table[KEYS].itertuples(index=False, name=None))
                    expected = {
                        key
                        for key in outer_keys
                        if inventory[key]["endpoint_rows"][str(horizon)]["exposure"] > 0
                    }
                    if actual != expected or table.duplicated(KEYS).any():
                        raise ValueError(
                            "OOF population differs from its full outer assignment"
                        )
                    for row in table.itertuples():
                        count = inventory[row.source_id, row.participant_id][
                            "endpoint_rows"
                        ][str(horizon)]["events"]
                        if row.base_event_count != count:
                            raise ValueError(
                                "OOF base event count differs from prepared truth"
                            )
                    table = table.merge(
                        original.reset_index(), on=KEYS, validate="one_to_one"
                    )
                    table["outer_fold"] = fold
                    tables[cut, horizon].append(table)
                    folds.append(
                        {
                            "arm": arm["arm"],
                            "model_seed": arm["model_seed"],
                            "outer_fold": fold,
                            "glucose_cut": cut,
                            "horizon_minutes": horizon,
                            **aggregate_oof_participants(table, sources),
                        }
                    )
            receipts.append(str(receipt_path))
        for (cut, horizon), parts in tables.items():
            table = pd.concat(parts, ignore_index=True)
            if table.duplicated(KEYS).any():
                raise ValueError("participant appears in multiple outer folds")
            all_tables[arm["arm"], arm["model_seed"], cut, horizon] = table
    # Ensure every method/seed/cut has the same horizon-specific exposed universe.
    for horizon in HORIZONS:
        reference = all_tables["lstm", 17, 54, horizon].set_index(KEYS).sort_index()
        for key, table in all_tables.items():
            if key[-1] != horizon:
                continue
            actual = table.set_index(KEYS).sort_index()
            columns = [
                "base_event_count",
                "patient_days",
                "observation_minutes",
                "outer_fold",
            ]
            pd.testing.assert_frame_equal(reference[columns], actual[columns])
    output.mkdir(parents=True, exist_ok=True)
    for population in ("all_development", "original_selection"):
        points, records = {}, []
        for key, table in all_tables.items():
            selected = (
                table
                if population == "all_development"
                else table.loc[table.original_partition.eq("selection")]
            )
            points[key] = aggregate_oof_participants(selected, source_order)
            arm, seed, cut, horizon = key
            records.append(
                {
                    "arm": arm,
                    "model_seed": seed,
                    "glucose_cut": cut,
                    "horizon_minutes": horizon,
                    **points[key],
                }
            )
        _json(
            output / f"{population}.json",
            {
                "population": population,
                "points": records,
                "sensitivity": summarize_paired_points(points, metric=SENSITIVITY),
                "burden": summarize_paired_points(points, metric=BURDEN),
                "aggregation": (
                    "participant contributions pooled across folds before source macro"
                ),
            },
        )
    _json(output / "fold-variation.json", {"rows": folds})
    _json(
        output / "receipt.json",
        {
            "status": "complete",
            "crossfit_contract": CROSSFIT_CONTRACT,
            "producer_count": len(receipts),
            "condition_count": len(all_tables),
            "source_order": source_order,
            "input_receipts": receipts,
            "overall_generalization_ci_computed": False,
        },
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    collect_oof(json.loads(args.config.read_text()))


if __name__ == "__main__":
    main()
