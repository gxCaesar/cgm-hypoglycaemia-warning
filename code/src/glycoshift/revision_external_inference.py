"""Paired inference from frozen external producers' private evaluation ledgers.

This consumer never reads predictions, glucose records, model parameters or ECDF
arrays, and never fits a policy. Native completed producer receipts bind every
input ledger. Only aggregate counts and percentile intervals are exported.

The CLI takes an explicit JSON ``producers`` list: one persistence fit (seed 17),
five LSTM fits, and their five frozen compositions. Each record names ``arm``,
``model_seed``, ``output_dir`` and either ``run_dir`` or ``recipe``. A shared
persistence fit is reused under every LSTM seed solely for paired arithmetic;
these aliases are never additional fits or independent participants.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .fixed_frame_evaluation import (
    FixedFrameEvaluation,
    FixedFrameKey,
    FixedFrameSpec,
    _aggregate,
    joint_fixed_frame_bootstrap,
)
from .revision_external import (
    _ALLOWED,
    _bound_file,
    _evidence,
    _json,
    _plain,
    _policies,
    _recheck_bindings,
    load_development_freeze,
)
from .sequences import REVISION_CONTRACT

MODEL_SEEDS = (17, 23, 42, 71, 101)
HORIZONS = (30, 60)
ARMS = ("persistence", "lstm", "composition")
_IDS = ["source_id", "participant_id"]
_EVENT_IDS = [*_IDS, "event_id"]
_EVENT_TRUTH = [*_EVENT_IDS, "event_onset"]
_EXPOSURE = [*_IDS, "patient_days", "observation_minutes"]
_COUNTS = (
    "normalized_participants",
    "predicted_participants",
    "zero_base_exposure_participants",
    "decision_rows",
    "tail_rows",
    "raw_rows",
)
_METRICS = (
    "fixed_sensitivity",
    "unmatched_alarms_per_patient_day",
    "delivered_alarms_per_patient_day",
)


@dataclass(frozen=True)
class ExternalProducer:
    arm: str
    model_seed: int
    output_dir: Path
    run_dir: Path | None = None
    recipe: Path | None = None


@dataclass(frozen=True)
class _AcceptedProducer:
    specification: ExternalProducer
    receipt: dict
    receipt_binding: dict
    bindings: dict[str, dict]
    policies: dict[tuple[int, int], dict]
    report: dict


def _binding(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError("inference input must be a regular non-symlink file")
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _cuts(values: Sequence[int]) -> tuple[int, ...]:
    result = tuple(values)
    if result not in ((54,), (54, 70)):
        raise ValueError(
            "main cut 54 is required; only explicitly frozen cut 70 is optional"
        )
    return result


def _contrasts(cuts: tuple[int, ...]) -> dict[str, dict[tuple[str, float], float]]:
    output = {}
    for cut in cuts:
        for arm in ARMS:
            output[f"{arm}_at_{cut}"] = {(arm, cut): 1.0}
        for left, right in (
            ("lstm", "persistence"),
            ("composition", "persistence"),
            ("composition", "lstm"),
        ):
            output[f"{left}_minus_{right}_at_{cut}"] = {
                (left, cut): 1.0,
                (right, cut): -1.0,
            }
    return output


def _sensitivity_points(
    evaluations: Mapping[FixedFrameKey, FixedFrameEvaluation],
    contrasts: Mapping[str, Mapping[tuple[str, float], float]],
    source_order: tuple[str, ...],
    seeds: tuple[int, ...],
) -> dict:
    """Descriptive points retain an undefined source/horizon instead of dropping it."""
    results = {}
    for name, terms in contrasts.items():
        points = np.empty((len(HORIZONS), len(seeds)))
        for h_index, horizon in enumerate(HORIZONS):
            for s_index, seed in enumerate(seeds):
                source_values = []
                for source in source_order:
                    value = 0.0
                    for (arm, cut), coefficient in terms.items():
                        table = evaluations[
                            FixedFrameKey(arm, cut, horizon, seed)
                        ].participants
                        positive = table.loc[
                            table.source_id.eq(source) & table.base_event_count.gt(0)
                        ]
                        value += coefficient * positive.fixed_sensitivity.mean()
                    source_values.append(value)
                points[h_index, s_index] = np.mean(source_values)
        results[name] = {
            "estimate": float(points.mean()),
            "ci_low": None,
            "ci_high": None,
            "coefficients": [
                {"arm": arm, "glucose_cut": cut, "coefficient": coefficient}
                for (arm, cut), coefficient in terms.items()
            ],
            "per_horizon": {
                h: {
                    "estimate": float(points[index].mean()),
                    "ci_low": None,
                    "ci_high": None,
                }
                for index, h in enumerate(HORIZONS)
            },
            "per_model_seed": {
                seed: float(points[:, index].mean()) for index, seed in enumerate(seeds)
            },
        }
    return results


def paired_external_inference(
    evaluations: Mapping[FixedFrameKey, FixedFrameEvaluation],
    *,
    source_order: tuple[str, ...],
    glucose_cuts: tuple[int, ...] = (54,),
    model_seed_order: tuple[int, ...] = MODEL_SEEDS,
    replicates: int = 1000,
    bootstrap_seed: int = 20260810,
    include_replicates: bool = False,
) -> dict:
    """Use the fixed-frame joint oracle; smaller complete seed sets are toy-only.

    Sensitivity has no interval for an undefined source/horizon or fewer than
    two event-positive participants in any source. All-base-participant burden
    is inferred separately, with identical multiplicities within each draw.
    """
    cuts = _cuts(glucose_cuts)
    contrasts = _contrasts(cuts)
    common = dict(
        contrasts=contrasts,
        source_order=source_order,
        horizon_order=HORIZONS,
        model_seed_order=model_seed_order,
        replicates=replicates,
        bootstrap_seed=bootstrap_seed,
        include_replicates=include_replicates,
    )
    # Burden validates the complete same-horizon truth even when sensitivity is
    # undefined. No sparse-case exception is allowed to hide an alignment error.
    metrics = {
        metric: joint_fixed_frame_bootstrap(evaluations, metric=metric, **common)
        for metric in _METRICS[1:]
    }
    for cut in cuts:
        for horizon in HORIZONS:
            control = evaluations[
                FixedFrameKey("persistence", cut, horizon, model_seed_order[0])
            ]
            reference_table = control.participants.sort_values(_IDS).reset_index(
                drop=True
            )
            for seed in model_seed_order[1:]:
                alias = evaluations[FixedFrameKey("persistence", cut, horizon, seed)]
                if (
                    alias.threshold != control.threshold
                    or alias.spec != control.spec
                    or not alias.participants.sort_values(_IDS)
                    .reset_index(drop=True)
                    .equals(reference_table)
                ):
                    raise ValueError("shared persistence aliases differ across seeds")
    reference = {
        h: evaluations[FixedFrameKey("persistence", cuts[0], h, model_seed_order[0])]
        for h in HORIZONS
    }
    counts = []
    for h, evaluation in reference.items():
        table = evaluation.participants
        for source in source_order:
            part = table.loc[table.source_id.eq(source)]
            counts.append(
                {
                    "source_id": source,
                    "horizon_minutes": h,
                    "participant_count": len(part),
                    "event_positive_participant_count": int(
                        part.base_event_count.gt(0).sum()
                    ),
                    "base_event_count": int(part.base_event_count.sum()),
                    "patient_days": int(part.patient_days.sum()),
                    "observation_minutes": float(part.observation_minutes.sum()),
                }
            )
    # This also validates sensitivity/count consistency if no resampling is
    # possible. Keep NaNs for event-free participants, never impute zeros.
    for evaluation in evaluations.values():
        table = evaluation.participants
        detected = table.detected_event_count.to_numpy(dtype=float)
        total = table.base_event_count.to_numpy(dtype=float)
        sensitivity = table.fixed_sensitivity.to_numpy(dtype=float)
        positive = total > 0
        if (
            not np.isfinite(detected).all()
            or (detected < 0).any()
            or (detected > total).any()
            or not np.equal(detected, np.floor(detected)).all()
            or not np.isnan(sensitivity[~positive]).all()
            or not np.allclose(
                sensitivity[positive],
                detected[positive] / total[positive],
                rtol=0,
                atol=1e-12,
            )
        ):
            raise ValueError(
                "participant sensitivity disagrees with base/detected event counts"
            )
    positives = [row["event_positive_participant_count"] for row in counts]
    if all(positives):
        sensitivity_result = joint_fixed_frame_bootstrap(
            evaluations, metric="fixed_sensitivity", **common
        )
        sensitivity_result["interval_status"] = "participant_percentile_bootstrap"
    else:
        sensitivity_result = {
            "metric": "fixed_sensitivity",
            "replicates": 0,
            "bootstrap_seed": bootstrap_seed,
            "model_seeds": list(model_seed_order),
            "horizons": list(HORIZONS),
            "source_order": list(source_order),
            "resampling_unit": "source_scoped_participant",
            "population": "event_positive_in_at_least_one_horizon",
            "stratification": "source_by_horizon_positivity_pattern",
            "aggregation": "equal_source_then_equal_horizon_and_model_seed",
            "interval_status": "undefined_base_sensitivity",
            "contrasts": _sensitivity_points(
                evaluations, contrasts, source_order, model_seed_order
            ),
        }
    if all(positives) and min(positives) < 2:
        sensitivity_result["interval_status"] = (
            "descriptive_sparse_event_positive_participants"
        )
        sparse_horizons = {
            row["horizon_minutes"]
            for row in counts
            if row["event_positive_participant_count"] < 2
        }
        for result in sensitivity_result["contrasts"].values():
            result["ci_low"] = result["ci_high"] = None
            result.pop("replicate_estimates", None)
            result.pop("replicate_horizon_seed_estimates", None)
            for horizon in sparse_horizons:
                result["per_horizon"][horizon]["ci_low"] = None
                result["per_horizon"][horizon]["ci_high"] = None
    metrics["fixed_sensitivity"] = sensitivity_result
    for metric in _METRICS[1:]:
        metrics[metric]["interval_status"] = "participant_percentile_bootstrap"
    return {
        "counts": counts,
        "metrics": metrics,
        "model_seed_aggregation": (
            "mean_of_complete_LSTM_seed_set_with_one_shared_control_fit"
        ),
        "independent_unit": "source_scoped_participant",
        "seed_variability_is_participant_uncertainty": False,
        "horizon_truth": "each_horizon_own_Ebase_and_Dbase; exact_within_horizon",
        "burden_population": "all_base_participants_including_event_free",
    }


def _producer_specifications(
    values: Sequence[ExternalProducer | Mapping],
) -> list[ExternalProducer]:
    result = []
    for value in values:
        if isinstance(value, ExternalProducer):
            producer = value
        else:
            expected_fields = {"arm", "model_seed", "output_dir"} | {
                "recipe" if value.get("arm") == "composition" else "run_dir"
            }
            if set(value) != expected_fields:
                raise ValueError(
                    "producer records require exact arm/seed/output/run/recipe fields"
                )
            producer = ExternalProducer(
                arm=value["arm"],
                model_seed=value["model_seed"],
                output_dir=Path(value["output_dir"]),
                run_dir=Path(value["run_dir"]) if value.get("run_dir") else None,
                recipe=Path(value["recipe"]) if value.get("recipe") else None,
            )
        if (
            isinstance(producer.model_seed, bool)
            or not isinstance(producer.model_seed, int)
            or producer.arm not in ARMS
            or (producer.arm == "composition") != (producer.recipe is not None)
            or (producer.arm != "composition") != (producer.run_dir is not None)
        ):
            raise ValueError(
                "explicit producer arm, integer seed and run/recipe identity required"
            )
        result.append(producer)
    expected = {("persistence", 17)} | {
        (arm, seed) for arm in ("lstm", "composition") for seed in MODEL_SEEDS
    }
    if {(p.arm, p.model_seed) for p in result} != expected or len(result) != len(
        expected
    ):
        raise ValueError(
            "one persistence seed17 and all five LSTM/composition seeds are required"
        )
    if len({p.output_dir.resolve() for p in result}) != len(result):
        raise ValueError("producer output directories must be distinct")
    runs = [p.run_dir.resolve() for p in result if p.run_dir is not None]
    recipes = [p.recipe.resolve() for p in result if p.recipe is not None]
    if len(set(runs)) != len(runs) or len(set(recipes)) != len(recipes):
        raise ValueError(
            "LSTM fits and composition recipes must have distinct canonical identities"
        )
    return result


def _receipt_files(root: Path, receipt: dict) -> dict[str, dict]:
    bindings = {}
    for item in receipt.get("files", []):
        relative = Path(item["relative_path"])
        name = str(relative)
        target = root / relative
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or name in bindings
            or not target.resolve().is_relative_to(root)
        ):
            raise ValueError("producer receipt has escaping or duplicate paths")
        binding = {
            "path": str(target),
            "size_bytes": item["size_bytes"],
            "mtime_ns": item["mtime_ns"],
        }
        _bound_file(binding)
        bindings[name] = binding
    actual = {
        str(path.relative_to(root))
        for category in ("predictions", "endpoints", "evaluations")
        for path in (root / "private" / category).rglob("*.parquet")
    }
    recorded = {
        name
        for name in bindings
        if name.startswith(
            ("private/predictions/", "private/endpoints/", "private/evaluations/")
        )
    }
    if not actual or actual != recorded:
        raise ValueError(
            "producer prediction/endpoint/evaluation inventory differs from receipt"
        )
    for name in ("status.json", "exportable/fixed-frame-summary.json"):
        if name not in bindings:
            raise ValueError("producer receipt lacks a required terminal binding")
    return bindings


def _native_receipt(
    producer: ExternalProducer,
    freeze: dict,
    source: dict,
    partition: str,
    *,
    raw_entry: dict | None = None,
) -> tuple[dict, dict, dict, dict]:
    root = producer.output_dir.resolve()
    receipt_binding = _binding(root / "run-receipt.json")
    receipt = _read_json(Path(receipt_binding["path"]))
    kind = (
        "frozen_step_ecdf_composition"
        if producer.arm == "composition"
        else "frozen_model_external"
    )
    if (
        receipt.get("schema_version") != "1.0"
        or receipt.get("producer_kind") != kind
        or receipt.get("status") != "complete"
        or receipt.get("revision_contract") != REVISION_CONTRACT
        or receipt.get("freeze_id") != freeze["freeze_id"]
        or receipt.get("partition") != partition
        or receipt.get("source_id") != source["source_id"]
        or receipt.get("source_normalization_contract")
        != source["source_normalization_contract"]
    ):
        raise ValueError(
            "native completed producer receipt identity/role/freeze differs"
        )
    if raw_entry is not None and (
        Path(receipt.get("run_dir", "")).resolve() != producer.run_dir.resolve()
        or receipt.get("model_summary_snapshot")
        != raw_entry["model_summary"]["snapshot"]
        or receipt.get("global_cohort_audit")
        != "reused frozen accepted evidence; not recomputed per participant"
    ):
        raise ValueError("raw producer run/model snapshot/global audit differs")
    counts = receipt.get("counts", {})
    if set(counts) != set(_COUNTS) or any(
        isinstance(counts[name], bool)
        or not isinstance(counts[name], int)
        or counts[name] < 0
        for name in _COUNTS
    ):
        raise ValueError("producer receipt requires exact nonnegative integer counts")
    if (
        counts["predicted_participants"] > counts["normalized_participants"]
        or counts["zero_base_exposure_participants"] > counts["normalized_participants"]
        or counts["tail_rows"] > counts["decision_rows"]
    ):
        raise ValueError("producer counts are inconsistent")
    bindings = _receipt_files(root, receipt)
    if _read_json(root / "status.json") != {
        "status": "complete",
        "freeze_id": freeze["freeze_id"],
    }:
        raise ValueError("producer terminal status differs")
    report = _read_json(root / "exportable/fixed-frame-summary.json")
    if (
        report.get("status") != "complete"
        or report.get("revision_contract") != REVISION_CONTRACT
        or report.get("freeze_id") != freeze["freeze_id"]
        or report.get("partition") != partition
        or report.get("external_already_exposed") is not True
    ):
        raise ValueError("producer summary identity/role differs")
    if raw_entry is not None:
        for name in ("private/recipe-freeze-used.json", "private/input-audit.json"):
            if name not in bindings:
                raise ValueError("raw receipt lacks frozen input/audit binding")
        audit = _read_json(root / "private/input-audit.json")
        if (
            _read_json(root / "private/recipe-freeze-used.json") != freeze
            or audit.get("source_normalization_contract")
            != source["source_normalization_contract"]
            or audit.get("accepted_global_cohort_audit")
            != source["accepted_global_cohort_audit"]
            or audit.get("global_cohort_audit") != receipt["global_cohort_audit"]
            or any(audit.get(name) != counts[name] for name in _COUNTS)
            or report.get("counts") != counts
            or report.get("thresholds_fitted_on_external") is not False
        ):
            raise ValueError(
                "raw producer input acceptance/counts/fitting boundary differs"
            )
    prediction_names = {
        Path(name).name for name in bindings if name.startswith("private/predictions/")
    }
    if len(prediction_names) != counts["predicted_participants"]:
        raise ValueError("producer prediction shard count differs from native receipt")
    for horizon in HORIZONS:
        for category in ("events", "exposure"):
            names = {
                Path(name).name
                for name in bindings
                if Path(name).parent == Path(f"private/endpoints/{category}-{horizon}")
            }
            if len(names) != counts[
                "normalized_participants"
            ] or not prediction_names.issubset(names):
                raise ValueError("independent endpoint shard population is incomplete")
    return receipt, receipt_binding, bindings, report


def _composition_policies(
    producer: ExternalProducer,
    freeze: dict,
    raw: Mapping[tuple[str, int], _AcceptedProducer],
    development_sources: tuple[str, ...],
    cuts: tuple[int, ...],
) -> tuple[dict, dict[tuple[int, int], dict]]:
    matches = [
        entry
        for entry in freeze.get("compositions", [])
        if Path(entry["recipe"]["path"]).resolve() == producer.recipe.resolve()
    ]
    if len(matches) != 1:
        raise ValueError(
            "composition recipe must be uniquely accepted in development freeze"
        )
    entry = matches[0]
    path = _bound_file(entry["recipe"], producer.recipe)
    _evidence(entry.get("accepted_development_evidence"))
    recipe = _read_json(path)
    _recheck_bindings(recipe)  # Stat compressed ECDF sidecars; never load them.
    if (
        recipe.get("schema_version") != "1.0"
        or recipe.get("revision_contract") != REVISION_CONTRACT
        or recipe.get("comparison_lstm_seed") != producer.model_seed
        or recipe.get("algebra")
        != "arithmetic_mean_of_two_right_step_member_ECDF_ranks"
        or recipe.get("members") != ["lstm", "persistence"]
        or recipe.get("source_order") != list(development_sources)
        or recipe.get("horizons") != list(HORIZONS)
        or recipe.get("reference_partition") != "calibration"
        or recipe.get("reference_population")
        != "all_context_valid_causal_calibration_decisions"
        or recipe.get("glucose_cut") != 54
        or recipe.get("nominal_fapd") != 0.5
    ):
        raise ValueError("frozen composition algebra/reference/seed/cut differs")
    for arm, seed in (("lstm", producer.model_seed), ("persistence", 17)):
        metadata = recipe["member_metadata"][arm]
        expected = raw[arm, seed].specification
        if (
            Path(metadata["run_dir"]).resolve() != expected.run_dir.resolve()
            or metadata.get("training_seed") != seed
            or metadata.get("comparison_lstm_seed") != producer.model_seed
            or metadata.get("shared_control_fit") is not (arm == "persistence")
            or metadata.get("revision_contract") != REVISION_CONTRACT
            or metadata.get("input_feature_mode") != "cgm_mask_clock"
            or metadata.get("score_columns")
            != {str(h): f"severe_hypo_{h}_probability" for h in HORIZONS}
        ):
            raise ValueError(
                "composition member lineage differs from explicit accepted producers"
            )
    policies = {(54, h): recipe["policies"][str(h)] for h in HORIZONS}
    if 70 in cuts:
        optional = entry.get("cut70_policies", {})
        if set(optional) != {str(h) for h in HORIZONS}:
            raise ValueError("cut70 composition policies were not explicitly frozen")
        for h in HORIZONS:
            record = optional[str(h)]
            evidence = _read_json(_bound_file(record["evidence"]))
            if (
                evidence.get("policy_origin") != "calibration_refit_at_cut"
                or evidence.get("base54_policy_parity_checked") is not True
                or evidence.get("policy") != record["policy"]
            ):
                raise ValueError(
                    "cut70 composition policy lacks exact accepted development evidence"
                )
            policies[70, h] = record["policy"]
    for (cut, h), policy in policies.items():
        if (
            policy.get("partition") != "calibration"
            or policy.get("glucose_cut") != cut
            or policy.get("nominal_fapd") != 0.5
            or policy.get("endpoint")
            != {
                "score_column": f"composition_{h}_score",
                "horizon_minutes": h,
                "source_order": list(development_sources),
            }
            or not np.isfinite(policy["threshold"])
        ):
            raise ValueError("composition frozen policy endpoint/role/budget differs")
    return entry, policies


def _accept_producers(
    freeze_path: Path,
    producers: list[ExternalProducer],
    partition: str,
    cuts: tuple[int, ...],
) -> tuple[dict, dict, dict[tuple[str, int], _AcceptedProducer]]:
    if partition not in _ALLOWED:
        raise ValueError("only already exposed Loop/HUPA roles are allowed")
    accepted = {}
    freeze = source = development_sources = None
    for producer in (p for p in producers if p.arm != "composition"):
        current, entry, current_source, config, summary = load_development_freeze(
            freeze_path, producer.run_dir, partition
        )
        if freeze is not None and (freeze != current or source != current_source):
            raise ValueError(
                "producer freeze/source acceptance changed during validation"
            )
        freeze, source = current, current_source
        development_sources = tuple(sorted(config["data"]["development_sources"]))
        if (
            summary.get("training_seed") != producer.model_seed
            or config["training"]["seed"] != producer.model_seed
            or (producer.arm == "lstm" and summary.get("architecture") != "lstm")
            or (
                producer.arm == "persistence"
                and summary.get("baseline") != "persistence"
            )
        ):
            raise ValueError("producer actual model family/training seed differs")
        policies = _policies(entry, summary, config)
        if any((cut, h, "0.5") not in policies for cut in cuts for h in HORIZONS):
            raise ValueError(
                "requested cut/budget was not explicitly frozen for every model"
            )
        receipt, binding, files, report = _native_receipt(
            producer, freeze, source, partition, raw_entry=entry
        )
        accepted[producer.arm, producer.model_seed] = _AcceptedProducer(
            producer,
            receipt,
            binding,
            files,
            {(cut, h): policies[cut, h, "0.5"] for cut in cuts for h in HORIZONS},
            report,
        )
    for producer in (p for p in producers if p.arm == "composition"):
        entry, policies = _composition_policies(
            producer, freeze, accepted, development_sources, cuts
        )
        receipt, binding, files, report = _native_receipt(
            producer, freeze, source, partition
        )
        expected_members = {
            arm: accepted[
                arm, producer.model_seed if arm == "lstm" else 17
            ].receipt_binding
            for arm in ("lstm", "persistence")
        }
        if "private/composition-input-receipt.json" not in files:
            raise ValueError(
                "composition receipt lacks its native input receipt binding"
            )
        audit = _read_json(
            producer.output_dir / "private/composition-input-receipt.json"
        )
        audit_text = "reused original accepted producer evidence; no new global audit"
        if (
            receipt.get("comparison_lstm_seed") != producer.model_seed
            or receipt.get("recipe") != entry["recipe"]
            or receipt.get("member_receipts") != expected_members
            or receipt.get("global_cohort_audit") != audit_text
            or receipt.get("counts")
            != accepted["lstm", producer.model_seed].receipt["counts"]
            or report.get("comparison_lstm_seed") != producer.model_seed
            or report.get("fitted_on_external") is not False
            or report.get("inherited_probability_or_regression_diagnostics")
            is not False
            or report.get("score_interpretation")
            != "rank_mean_not_calibrated_event_probability"
            or any(
                audit.get(name) != receipt.get(name)
                for name in (
                    "revision_contract",
                    "freeze_id",
                    "recipe",
                    "member_receipts",
                    "scored_rows",
                    "tail_rows",
                    "source_normalization_contract",
                    "global_cohort_audit",
                )
            )
            or receipt.get("scored_rows") != receipt["counts"]["decision_rows"]
            or receipt.get("tail_rows") != receipt["counts"]["tail_rows"]
            or report.get("scored_rows") != receipt["scored_rows"]
            or report.get("tail_rows") != receipt["tail_rows"]
        ):
            raise ValueError("composition recipe/member receipts/counts/lineage differ")
        accepted[producer.arm, producer.model_seed] = _AcceptedProducer(
            producer,
            receipt,
            binding,
            files,
            {(cut, h): policies[cut, h] for cut in cuts for h in HORIZONS},
            report,
        )
    first_counts = accepted["persistence", 17].receipt["counts"]
    if any(item.receipt["counts"] != first_counts for item in accepted.values()):
        raise ValueError(
            "external producer input populations/counts differ; no intersection allowed"
        )
    return freeze, source, accepted


def _read_ledger(
    producer: _AcceptedProducer, relative: str, columns: list[str] | None = None
) -> pd.DataFrame:
    prefix = relative + "/"
    paths = sorted(name for name in producer.bindings if name.startswith(prefix))
    if not paths or any(Path(name).parent != Path(relative) for name in paths):
        raise ValueError("required private ledger is missing or unexpectedly nested")
    frames = []
    total = 0
    for name in paths:
        parquet = pq.ParquetFile(producer.specification.output_dir / name)
        total += parquet.metadata.num_rows
        if parquet.metadata.num_rows > 1_000_000 or total > 10_000_000:
            raise ValueError("private ledger row bound exceeded")
        frames.append(parquet.read(columns=columns, use_threads=False).to_pandas())
    return pd.concat(frames, ignore_index=True)


def _identifiers(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    if set(columns).difference(frame.columns):
        raise ValueError("private ledger lacks required identity columns")
    for column in columns:
        if frame[column].isna().any():
            raise ValueError("private ledger has missing identity")
        frame[column] = frame[column].astype(str)
        if frame[column].str.strip().eq("").any():
            raise ValueError("private ledger has empty identity")
    if frame.duplicated(columns).any():
        raise ValueError("private ledger keys are not unique")
    return frame.sort_values(columns).reset_index(drop=True)


def _events(frame: pd.DataFrame) -> pd.DataFrame:
    frame = _identifiers(frame, _EVENT_IDS)
    frame["event_onset"] = pd.to_datetime(
        frame.event_onset, utc=True, errors="raise"
    ).astype("datetime64[ns, UTC]")
    if frame.event_onset.isna().any():
        raise ValueError("private event truth has missing onset")
    return frame


def _participant_ledger(
    table: pd.DataFrame, events: pd.DataFrame, exposure: pd.DataFrame, source_id: str
) -> pd.DataFrame:
    table = _identifiers(table, _IDS)
    exposure = _identifiers(exposure, _IDS)
    required = {
        *_EXPOSURE,
        "base_patient_days",
        "base_event_count",
        "detected_event_count",
        "fixed_sensitivity",
        "matched_alarm_count",
        "delivered_alarm_count",
        "unmatched_alarm_count",
        "emitted_alarm_count",
        "unevaluable_alarm_count",
        "unmatched_alarms_per_patient_day",
        "delivered_alarms_per_patient_day",
    }
    if missing := required.difference(table.columns):
        raise ValueError(
            f"participant ledger lacks required columns: {sorted(missing)}"
        )
    if missing := set(_EXPOSURE).difference(exposure.columns):
        raise ValueError(f"independent base exposure lacks columns: {sorted(missing)}")
    numeric_exposure = ["patient_days", "observation_minutes"]
    numeric_storage = all(
        (
            pd.api.types.is_integer_dtype(frame[column].dtype)
            or pd.api.types.is_float_dtype(frame[column].dtype)
        )
        and frame[column].notna().all()
        for frame in (table, exposure)
        for column in numeric_exposure
    )
    # Parquet producers store identical minutes as either integers or floats.
    # Object arrays compare exact values without rounding integers through float
    # conversion. No tolerance is allowed for an actual exposure discrepancy.
    exposure_matches = (
        table[_IDS].equals(exposure[_IDS])
        and numeric_storage
        and np.array_equal(
            table[numeric_exposure].to_numpy(dtype=object),
            exposure[numeric_exposure].to_numpy(dtype=object),
        )
    )
    if (
        set(table.source_id) != {source_id}
        or not exposure_matches
        or not set(events[_IDS].itertuples(index=False, name=None)).issubset(
            set(table[_IDS].itertuples(index=False, name=None))
        )
    ):
        raise ValueError(
            "participant source/identity/exposure differs from independent base truth"
        )
    days = pd.to_numeric(table.patient_days, errors="coerce").to_numpy(dtype=float)
    minutes = pd.to_numeric(table.observation_minutes, errors="coerce").to_numpy(
        dtype=float
    )
    if (
        not np.isfinite(days).all()
        or not np.isfinite(minutes).all()
        or (days <= 0).any()
        or (minutes <= 0).any()
        or (minutes > days * 1440).any()
        or not np.equal(days, np.floor(days)).all()
    ):
        raise ValueError(
            "base exposure requires positive whole patient days and observation minutes"
        )
    normalized_days = []
    for value, day_count in zip(table.base_patient_days, days, strict=True):
        dates = pd.DatetimeIndex(pd.to_datetime(list(value), utc=True))
        if (
            len(dates) != day_count
            or dates.hasnans
            or dates.has_duplicates
            or not dates.equals(dates.normalize())
            or not dates.is_monotonic_increasing
        ):
            raise ValueError("base patient-day identity/count is invalid")
        normalized_days.append(tuple(dates))
    table["base_patient_days"] = normalized_days
    event_counts = events.groupby(_IDS, observed=True).size()
    detected_counts = events.loc[events.detected].groupby(_IDS, observed=True).size()
    keys = list(table[_IDS].itertuples(index=False, name=None))
    for column in table.columns:
        if column.endswith("_count"):
            values = pd.to_numeric(table[column], errors="coerce").to_numpy(dtype=float)
            if (
                not np.isfinite(values).all()
                or (values < 0).any()
                or not np.equal(values, np.floor(values)).all()
            ):
                raise ValueError("participant event/alarm/decision counts are invalid")
    for column, counts in (
        ("base_event_count", event_counts),
        ("detected_event_count", detected_counts),
    ):
        if not np.array_equal(table[column], [counts.get(key, 0) for key in keys]):
            raise ValueError(
                "participant event counts disagree with independent/evaluated events"
            )
    if (
        not np.array_equal(table.matched_alarm_count, table.detected_event_count)
        or not np.array_equal(
            table.delivered_alarm_count,
            table.matched_alarm_count + table.unmatched_alarm_count,
        )
        or not np.array_equal(
            table.emitted_alarm_count,
            table.delivered_alarm_count + table.unevaluable_alarm_count,
        )
    ):
        raise ValueError("participant alarm/match counts do not reconcile")
    for metric, count in (
        ("unmatched_alarms_per_patient_day", "unmatched_alarm_count"),
        ("delivered_alarms_per_patient_day", "delivered_alarm_count"),
    ):
        if not np.allclose(table[metric], table[count] / days, rtol=0, atol=1e-12):
            raise ValueError("participant burden disagrees with fixed exposure")
    return table


def _same_number(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return bool(np.isclose(left, right, rtol=0, atol=1e-12))
    return left == right


def _load_evaluations(
    accepted: Mapping[tuple[str, int], _AcceptedProducer],
    source_id: str,
    cuts: tuple[int, ...],
) -> tuple[dict[FixedFrameKey, FixedFrameEvaluation], list[dict]]:
    evaluations = {}
    truth = {}
    descriptive = []
    for (arm, seed), producer in accepted.items():
        for horizon in HORIZONS:
            base_events = _events(
                _read_ledger(
                    producer, f"private/endpoints/events-{horizon}", _EVENT_TRUTH
                )
            )
            exposure = _identifiers(
                _read_ledger(
                    producer, f"private/endpoints/exposure-{horizon}", _EXPOSURE
                ),
                _IDS,
            )
            current = (base_events[_EVENT_TRUTH], exposure[_EXPOSURE])
            if len(exposure) != (
                producer.receipt["counts"]["predicted_participants"]
                - producer.receipt["counts"]["zero_base_exposure_participants"]
            ):
                raise ValueError(
                    "independent base participant count differs from native receipt"
                )
            if horizon not in truth:
                truth[horizon] = current
            elif any(
                not a.equals(b) for a, b in zip(current, truth[horizon], strict=True)
            ):
                raise ValueError(
                    "same-horizon Ebase/onset/Dbase/identity differs across producers"
                )
            rows = producer.report.get("policies", [])
            for cut in cuts:
                relative = f"private/evaluations/{cut}/{horizon}/0.5"
                events = _events(
                    _read_ledger(
                        producer, f"{relative}/events", [*_EVENT_TRUTH, "detected"]
                    )
                )
                if (
                    not events[_EVENT_TRUTH].equals(base_events[_EVENT_TRUTH])
                    or events.detected.isna().any()
                    or not all(
                        isinstance(value, (bool, np.bool_)) for value in events.detected
                    )
                ):
                    raise ValueError(
                        "evaluation truth/detection differs from independent Ebase"
                    )
                table = _participant_ledger(
                    _read_ledger(producer, f"{relative}/participants"),
                    events,
                    exposure,
                    source_id,
                )
                policy = producer.policies[cut, horizon]
                selected = [
                    row
                    for row in rows
                    if (
                        row.get("glucose_cut") == cut
                        and row.get("horizon_minutes") == horizon
                        and row.get("nominal_fapd") == 0.5
                    )
                ]
                if (
                    len(selected) != 1
                    or selected[0].get("threshold") != policy["threshold"]
                ):
                    raise ValueError(
                        "terminal summary must uniquely identify the frozen policy"
                    )
                aggregate = _plain(_aggregate(table))
                if any(
                    not _same_number(selected[0].get(name), value)
                    for name, value in aggregate.items()
                ):
                    raise ValueError(
                        "terminal point summary disagrees with bound participant ledger"
                    )
                descriptive.append(
                    {
                        "arm": arm,
                        "training_seed": seed,
                        "source_id": source_id,
                        "glucose_cut": cut,
                        "horizon_minutes": horizon,
                        "nominal_fapd": 0.5,
                        "threshold": policy["threshold"],
                        **aggregate,
                    }
                )
                evaluation = FixedFrameEvaluation(
                    spec=FixedFrameSpec(
                        policy["endpoint"]["score_column"], horizon, (source_id,)
                    ),
                    threshold=policy["threshold"],
                    glucose_cut=cut,
                    participants=table,
                    sources=pd.DataFrame(),
                    alarms=pd.DataFrame(),
                    events=truth[horizon][0],
                    summary=aggregate,
                )
                for comparison_seed in MODEL_SEEDS if arm == "persistence" else (seed,):
                    evaluations[FixedFrameKey(arm, cut, horizon, comparison_seed)] = (
                        evaluation
                    )
    return evaluations, descriptive


def run_external_paired_inference(
    freeze_path: Path,
    *,
    partition: str,
    producers: Sequence[ExternalProducer | Mapping],
    output_dir: Path,
    glucose_cuts: tuple[int, ...] = (54,),
) -> Path:
    """Validate the actual accepted freeze/receipts, infer, export aggregates only."""
    cuts = _cuts(glucose_cuts)
    specifications = _producer_specifications(producers)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite an external inference output")
    freeze_path = Path(freeze_path)
    freeze_binding = _binding(freeze_path)
    freeze, source, accepted = _accept_producers(
        freeze_path, specifications, partition, cuts
    )
    evaluations, descriptive = _load_evaluations(accepted, source["source_id"], cuts)
    inference = paired_external_inference(
        evaluations, source_order=(source["source_id"],), glucose_cuts=cuts
    )
    # A producer changing during the consumer is not accepted merely because its
    # ledger was internally consistent when first read.
    _bound_file(freeze_binding)
    _recheck_bindings(freeze)
    for producer in accepted.values():
        _bound_file(producer.receipt_binding)
        for binding in producer.bindings.values():
            _bound_file(binding)
        if producer.specification.recipe is not None:
            _recheck_bindings(_read_json(producer.specification.recipe))
    report = {
        "schema_version": "1.0",
        "status": "complete",
        "revision_contract": REVISION_CONTRACT,
        "freeze_id": freeze["freeze_id"],
        "partition": partition,
        "source_id": source["source_id"],
        "external_already_exposed": True,
        "fitted_on_external": False,
        "prediction_or_glucose_records_read": False,
        "nominal_fapd": 0.5,
        "glucose_cuts": list(cuts),
        "model_seeds": list(MODEL_SEEDS),
        "bootstrap_replicates": 1000,
        "bootstrap_seed": 20260810,
        "actual_fit_counts": {"persistence": 1, "lstm": len(MODEL_SEEDS)},
        "derived_composition_recipe_count": len(MODEL_SEEDS),
        "native_completed_producer_count": len(accepted),
        "shared_control_training_seed": 17,
        "descriptive_policies": descriptive,
        **inference,
    }
    # No private output is produced. No participant identifiers, event onsets,
    # alarm values, predictions, or resampling indices enter this payload.
    output_dir.mkdir(parents=True, exist_ok=False)
    _json(output_dir / "external-paired-inference.json", report)
    rows = []
    for metric, payload in inference["metrics"].items():
        for name, result in payload["contrasts"].items():
            for horizon, values in [
                ("equal_horizon_mean", result),
                *result["per_horizon"].items(),
            ]:
                rows.append(
                    {
                        "metric": metric,
                        "contrast": name,
                        "horizon_minutes": horizon,
                        "estimate": values["estimate"],
                        "ci_low": values["ci_low"],
                        "ci_high": values["ci_high"],
                        "interval_status": payload["interval_status"],
                    }
                )
    pd.DataFrame(rows).to_csv(output_dir / "external-paired-inference.csv", index=False)
    _json(
        output_dir / "status.json",
        {"status": "complete", "freeze_id": freeze["freeze_id"]},
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--partition", choices=tuple(_ALLOWED), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--glucose-cut", type=int, action="append", choices=(54, 70))
    arguments = parser.parse_args()
    run_external_paired_inference(
        arguments.freeze,
        partition=arguments.partition,
        producers=_read_json(arguments.manifest)["producers"],
        output_dir=arguments.output,
        glucose_cuts=tuple(arguments.glucose_cut or (54,)),
    )


if __name__ == "__main__":
    main()
