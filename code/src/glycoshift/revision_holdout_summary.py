"""Summarize the fixed fold-0, seed-17 LSTM/current-glucose holdout check.

This entrypoint consumes completed crossfit producers without treating the single
fixed fold as five-fold OOF or estimating a generalization confidence interval.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .config import load_config
from .revision_crossfit_evaluation import (
    CUTS,
    HORIZONS,
    KEYS,
    _binding,
    aggregate_oof_participants,
)
from .revision_crossfit_prepare import (
    CROSSFIT_CONTRACT,
    INNER_ROLES,
    validated_crossfit_metadata,
)
from .revision_crossfit_summary import BURDEN, SENSITIVITY
from .revision_external import _bound_file, _json, _recheck_bindings

ARMS = ("lstm", "persistence")
OUTER_FOLD = 0
MODEL_SEED = 17
REAL_OUTER_PARTICIPANTS = 207
TRUTH_COLUMNS = [
    "base_event_count", "patient_days", "observation_minutes", "base_patient_days",
]


def summarize_holdout_points(points: dict, *, metric: str) -> dict:
    """Pair the same participants, cuts and horizons; propagate undefined means."""
    def contrast(values):
        differences = {
            cut: values["lstm", cut] - values["persistence", cut] for cut in CUTS
        }
        return {
            "lstm_minus_current_glucose_54": differences[54],
            "lstm_minus_current_glucose_70": differences[70],
            "contrast_change_70_minus_54": differences[70] - differences[54],
        }

    horizon_contrasts = [
        {
            "horizon_minutes": horizon,
            **contrast({(arm, cut): points[arm, cut, horizon][metric]
                        for arm in ARMS for cut in CUTS}),
        }
        for horizon in HORIZONS
    ]
    source_metric = (
        "fixed_sensitivity" if metric == SENSITIVITY
        else "participant_macro_unmatched_alarms_per_patient_day"
    )
    source_order = points["lstm", 54, 30]["source_order"]
    source_contrasts = []
    for index, source in enumerate(source_order):
        for horizon in HORIZONS:
            source_contrasts.append({
                "source_id": source,
                "horizon_minutes": horizon,
                **contrast({
                    (arm, cut): points[arm, cut, horizon]["sources"][index][
                        source_metric
                    ]
                    for arm in ARMS for cut in CUTS
                }),
            })
    return {
        "metric": metric,
        "method_points": [
            {"arm": arm, "model_seed": MODEL_SEED, "glucose_cut": cut,
             "value": float(np.mean([
                 points[arm, cut, horizon][metric] for horizon in HORIZONS
             ]))}
            for arm in ARMS for cut in CUTS
        ],
        "horizon_contrasts": horizon_contrasts,
        "source_horizon_contrasts": source_contrasts,
        "equal_horizon_contrast": {
            key: float(np.mean([row[key] for row in horizon_contrasts]))
            for key in (
                "lstm_minus_current_glucose_54", "lstm_minus_current_glucose_70",
                "contrast_change_70_minus_54",
            )
        },
        "generalization_ci": None,
        "interval_status": "not_computed_fixed_holdout_conditional_point_check",
    }


def _check_prediction_roles(path: Path, roles: pd.DataFrame, *, outer: bool) -> None:
    """Scan only identity columns in bounded batches, never glucose or scores."""
    # Preparation rejects zero-exposure original participants; both native
    # producers score every full-online selection/calibration/outer timeline.
    admitted = ("outer_evaluation",) if outer else ("selection", "calibration")
    expected = set(roles.loc[roles.role.isin(admitted), [*KEYS, "role"]].itertuples(
        index=False, name=None
    ))
    columns = [*KEYS, "partition", *(["outer_fold"] if outer else [])]
    observed = set()
    for batch in pq.ParquetFile(path).iter_batches(columns=columns):
        frame = batch.to_pandas()
        if frame[columns].isna().any().any():
            raise ValueError("holdout predictions contain missing role identities")
        if outer and not frame.outer_fold.eq(OUTER_FOLD).all():
            raise ValueError("holdout predictions have a different outer fold")
        identities = set(frame[[*KEYS, "partition"]].itertuples(
            index=False, name=None
        ))
        if not identities.issubset(expected):
            raise ValueError("holdout predictions violate inner/outer role isolation")
        observed.update(identities)
    if observed != expected:
        raise ValueError("holdout predictions omit assigned participants or roles")


def collect_holdout(config: dict) -> Path:
    """Accept exactly the two completed producers from the existing shared pool."""
    output = Path(config["output_dir"]).resolve()
    prepared = Path(config["prepared_dir"]).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("refusing to overwrite fixed holdout summaries")
    arms = config["arms"]
    if (
        len(arms) != 2
        or {(row["arm"], row["model_seed"]) for row in arms}
        != {(arm, MODEL_SEED) for arm in ARMS}
        or any(len(row["fold_outputs"]) != 1 for row in arms)
    ):
        raise ValueError("fixed holdout requires only LSTM/persistence seed17 fold0")
    tables, receipts = {}, []
    source_order, reference_roles = None, None
    outer_count = None
    for arm in arms:
        directory = Path(arm["fold_outputs"][0]).resolve()
        receipt_path = directory / "receipt.json"
        receipt = json.loads(receipt_path.read_text())
        if (
            receipt.get("status") != "complete"
            or receipt.get("crossfit_contract") != CROSSFIT_CONTRACT
            or receipt.get("outer_refitting") is not False
            or type(receipt.get("outer_fold")) is not int
            or receipt["outer_fold"] != OUTER_FOLD
            or receipt.get("prepared_dir") != str(prepared)
        ):
            raise ValueError("holdout producer is incomplete or has wrong fold/pool")
        _recheck_bindings(receipt)
        freeze_path = _bound_file(
            receipt["policy_freeze"], directory / "private/policy-freeze.json"
        )
        freeze = json.loads(freeze_path.read_text())
        _recheck_bindings(freeze)
        run = Path(receipt["run_dir"])
        if (
            freeze.get("crossfit_contract") != CROSSFIT_CONTRACT
            or type(freeze.get("outer_fold")) is not int
            or freeze["outer_fold"] != OUTER_FOLD
            or freeze.get("run_dir") != str(run)
            or freeze.get("prepared_dir") != str(prepared)
            or freeze.get("policy_fit_role") != "calibration"
            or freeze.get("outer_values_opened") is not False
            or freeze.get("base54_policy_parity_checked") is not True
            or set(freeze.get("policies", {}))
            != {f"{cut}/{horizon}" for cut in CUTS for horizon in HORIZONS}
        ):
            raise ValueError("holdout policy freeze differs from run/fold/isolation")
        bindings = freeze["bindings"]
        for name in (
            "crossfit-audit.json", "shared/input-audit.json",
            "shared/participant-manifest.private.json",
            f"folds/fold-{OUTER_FOLD}/participant-roles.private.json",
        ):
            _bound_file(bindings[f"prepared/{name}"], prepared / name)
        model_config = load_config(_bound_file(
            bindings["config_resolved.yaml"], run / "config_resolved.yaml"
        ))
        model = json.loads(_bound_file(
            bindings["model_summary.json"], run / "model_summary.json"
        ).read_text())
        if (
            freeze["summary"] != model
            or model.get("architecture", model.get("baseline")) != arm["arm"]
            or model_config["training"]["seed"] != MODEL_SEED
            or model.get("training_seed") != MODEL_SEED
        ):
            raise ValueError("holdout model identity or seed differs from declaration")
        audit, _, manifest, roles = validated_crossfit_metadata(
            prepared, model_config, outer_fold=OUTER_FOLD
        )
        sources = tuple(audit["source_order"])
        if (
            tuple(freeze.get("source_order", ())) != sources
            or (source_order is not None and source_order != sources)
        ):
            raise ValueError("holdout source order differs across producers")
        source_order = sources
        if reference_roles is not None:
            pd.testing.assert_frame_equal(
                reference_roles.sort_values(KEYS, ignore_index=True),
                roles.sort_values(KEYS, ignore_index=True),
            )
        reference_roles = roles
        outer_keys = set(roles.loc[roles.role.eq("outer_evaluation"), KEYS].itertuples(
            index=False, name=None
        ))
        outer_count = len(outer_keys)
        if (
            not model_config["project"].get("synthetic", False)
            and outer_count != REAL_OUTER_PARTICIPANTS
        ):
            raise ValueError(
                "real fixed holdout must retain all 207 outer participants"
            )
        cohort = json.loads(_bound_file(
            bindings["cohort_audit.json"], run / "cohort_audit.json"
        ).read_text())
        if (
            cohort.get("crossfit_contract") != CROSSFIT_CONTRACT
            or cohort.get("outer_fold") != OUTER_FOLD
            or cohort.get("crossfit_prepared_dir") != str(prepared)
            or cohort.get("bundle_roles") != list(INNER_ROLES)
            or cohort.get("loaded_participants") != len(roles) - outer_count
        ):
            raise ValueError("holdout fit audit violates inner role isolation")
        for cut in CUTS:
            for horizon in HORIZONS:
                policy = freeze["policies"][f"{cut}/{horizon}"]
                if (
                    policy.get("partition") != "calibration"
                    or policy.get("glucose_cut") != cut
                    or policy.get("nominal_fapd") != 0.5
                    or policy.get("endpoint") != {
                        "score_column": f"severe_hypo_{horizon}_probability",
                        "horizon_minutes": horizon, "source_order": list(sources),
                    }
                ):
                    raise ValueError(
                        "holdout policy has wrong calibration/task binding"
                    )
        _check_prediction_roles(_bound_file(
            bindings["predictions.parquet"], run / "predictions.parquet"
        ), roles, outer=False)
        bound_outputs = {Path(row["path"]).resolve() for row in receipt["outputs"]}
        outer_predictions = directory / "private/predictions.parquet"
        if outer_predictions not in bound_outputs:
            raise ValueError("holdout predictions are not bound by the receipt")
        _check_prediction_roles(outer_predictions, roles, outer=True)
        inventory = {(row["source_id"], row["participant_id"]): row for row in manifest}
        for cut in CUTS:
            for horizon in HORIZONS:
                path = directory / (
                    f"private/evaluations/{cut}/{horizon}/participants.parquet"
                )
                if path not in bound_outputs:
                    raise ValueError(
                        "holdout participant ledger is not bound by receipt"
                    )
                table = pd.read_parquet(path)
                actual = set(table[KEYS].itertuples(index=False, name=None))
                expected = {
                    key for key in outer_keys
                    if inventory[key]["endpoint_rows"][str(horizon)]["exposure"] > 0
                }
                if actual != expected or table.duplicated(KEYS).any():
                    raise ValueError(
                        "holdout ledger differs from full outer assignment"
                    )
                for row in table.itertuples():
                    count = inventory[row.source_id, row.participant_id][
                        "endpoint_rows"
                    ][str(horizon)]["events"]
                    if row.base_event_count != count:
                        raise ValueError("holdout base event count differs from truth")
                tables[arm["arm"], cut, horizon] = table
        _recheck_bindings(freeze)
        _recheck_bindings(receipt)
        receipts.append(_binding(receipt_path))
    for horizon in HORIZONS:
        reference = tables["lstm", 54, horizon].set_index(KEYS).sort_index()
        for (_arm, _cut, current_horizon), table in tables.items():
            if current_horizon == horizon:
                try:
                    pd.testing.assert_frame_equal(
                        reference[TRUTH_COLUMNS],
                        table.set_index(KEYS).sort_index()[TRUTH_COLUMNS],
                    )
                except AssertionError as error:
                    raise ValueError(
                        "same-horizon holdout participant truth/exposure differs"
                    ) from error
    points = {
        key: {
            **aggregate_oof_participants(table, source_order),
            "uncertainty": "conditional fixed fold/seed points; no generalization CI",
        }
        for key, table in tables.items()
    }
    _recheck_bindings(receipts)
    output.mkdir(parents=True, exist_ok=True)
    _json(output / "holdout.json", {
        "population": "fixed_outer_holdout", "outer_fold": OUTER_FOLD,
        "model_seed": MODEL_SEED, "outer_participant_count": outer_count,
        "source_order": source_order,
        "points": [
            {"arm": arm, "model_seed": MODEL_SEED, "glucose_cut": cut,
             "horizon_minutes": horizon, **points[arm, cut, horizon]}
            for arm in ARMS for cut in CUTS for horizon in HORIZONS
        ],
        "sensitivity": summarize_holdout_points(points, metric=SENSITIVITY),
        "burden": summarize_holdout_points(points, metric=BURDEN),
        "aggregation": "participant contributions within source, then equal sources",
        "interpretation": (
            "fixed fold0 seed17 conditional point check; not five-fold OOF, "
            "seed-robust generalization, or untouched confirmation"
        ),
    })
    _json(output / "receipt.json", {
        "status": "complete", "crossfit_contract": CROSSFIT_CONTRACT,
        "population": "fixed_outer_holdout", "outer_fold": OUTER_FOLD,
        "model_seed": MODEL_SEED, "outer_participant_count": outer_count,
        "producer_count": len(receipts), "condition_count": len(tables),
        "source_order": source_order, "input_receipts": receipts,
        "overall_generalization_ci_computed": False,
    })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    collect_holdout(json.loads(args.config.read_text()))


if __name__ == "__main__":
    main()
