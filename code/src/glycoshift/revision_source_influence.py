"""Descriptive estimands and source influence from unchanged fixed-frame ledgers.

No fitting, alarm emission, event matching, bootstrap or generalization interval
occurs here. Event pooling is defined for sensitivity; burden is pooled by
exposure, as in the frozen evaluator, never divided by event count.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .fixed_frame_evaluation import (
    FixedFrameEvaluation,
    FixedFrameSpec,
    _aggregate,
    _ratio,
)
from .revision_external import _bound_file, _json, _plain
from .revision_inputs import CONFIRMATION_CONTRACT
from .revision_threshold_audit import _policy_row
from .sequences import REVISION_CONTRACT

MODEL_SEEDS = (17, 23, 42, 71, 101)
HORIZONS = (30, 60)
METHODS = ("lstm", "tcn", "persistence", "linear", "summary_logistic", "xgboost")
ORIGINAL_CUTS = (54, 60, 70, 80, 90, 100)
_IDS = ["source_id", "participant_id"]
_EVENT_IDS = [*_IDS, "event_id"]
_ALARM_IDS = [*_IDS, "context_end"]


@dataclass(frozen=True)
class SourceInfluenceKey:
    arm: str
    model_seed: int | None
    glucose_cut: float
    horizon_minutes: int


def _mean(values: Any) -> float:
    array = np.asarray(values, dtype=float)
    return float(array.mean()) if array.size else float("nan")


def _quantity(estimate: float, numerator: float, denominator: float) -> dict:
    return {
        "estimate": float(estimate), "numerator": float(numerator),
        "denominator": float(denominator),
    }


def _identities(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    if not set(columns).issubset(frame) or frame[columns].isna().any().any():
        raise ValueError(f"{name} lacks complete identity fields")
    if frame.duplicated(columns).any():
        raise ValueError(f"{name} has duplicate identities")
    for column in (value for value in columns if value != "context_end"):
        if not frame[column].map(
            lambda value: isinstance(value, str) and bool(value.strip())
        ).all():
            raise ValueError(f"{name} requires string source/participant identities")


def _validate(evaluation: FixedFrameEvaluation) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Check existing match identities and counts, without seeking new matches."""
    evaluation.spec.validate()
    if not np.isfinite((evaluation.threshold, evaluation.glucose_cut)).all():
        raise ValueError("evaluation threshold and glucose cut must be finite")
    table, events, alarms = (
        evaluation.participants.copy(), evaluation.events.copy(),
        evaluation.alarms.copy(),
    )
    _identities(table, _IDS, "participant ledger")
    _identities(events, _EVENT_IDS, "event ledger")
    _identities(alarms, _ALARM_IDS, "alarm ledger")
    sources = set(evaluation.spec.source_order)
    keys = set(table[_IDS].itertuples(index=False, name=None))
    for frame in (table, events, alarms):
        if not set(frame.source_id).issubset(sources) or not set(
            frame[_IDS].itertuples(index=False, name=None)
        ).issubset(keys):
            raise ValueError("ledger contains an undeclared source or participant")
    required_events = {"event_onset", "detected", "alarm_context_end", "lead_minutes"}
    required_alarms = {
        "cgm_last", "base_evaluable", "matched", "unmatched", "matched_event_id",
        "matched_event_onset", "lead_minutes",
    }
    if not required_events.issubset(events) or not required_alarms.issubset(alarms):
        raise ValueError("fixed-match event/alarm ledger is incomplete")
    for frame, column in ((events, "event_onset"), (alarms, "context_end")):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
        if frame[column].isna().any():
            raise ValueError("event/alarm times must be complete and valid")
    for frame, columns in ((events, ("detected",)), (alarms, (
        "base_evaluable", "matched", "unmatched",
    ))):
        for column in columns:
            if len(frame) and (
                frame[column].isna().any()
                or not pd.api.types.is_bool_dtype(frame[column])
            ):
                raise ValueError("event/alarm statuses must be complete booleans")
    glucose = pd.to_numeric(alarms.cgm_last, errors="coerce").to_numpy(float)
    if (
        not np.isfinite(glucose).all()
        or (glucose < max(54, evaluation.glucose_cut)).any()
    ):
        raise ValueError("alarm glucose is invalid for its declared gate")
    if (
        (alarms.matched & ~alarms.base_evaluable).any()
        or not np.array_equal(alarms.unmatched, alarms.base_evaluable & ~alarms.matched)
    ):
        raise ValueError("alarm matching contradicts evaluability")
    matched = alarms.loc[alarms.matched].copy()
    matched = matched.rename(columns={
        "matched_event_id": "event_id", "context_end": "matched_context_end",
        "lead_minutes": "matched_lead_minutes",
    })
    if matched.duplicated(_EVENT_IDS).any() or matched.event_id.isna().any():
        raise ValueError("fixed matches are not one-to-one with events")
    detected = events.loc[events.detected].copy()
    if set(matched[_EVENT_IDS].itertuples(index=False, name=None)) != set(
        detected[_EVENT_IDS].itertuples(index=False, name=None)
    ):
        raise ValueError("detected events and matched alarms have different identities")
    joined = detected.merge(
        matched[[*_EVENT_IDS, "matched_context_end", "matched_event_onset",
                 "matched_lead_minutes", "cgm_last"]],
        on=_EVENT_IDS, validate="one_to_one",
    )
    for column in (
        "event_onset", "alarm_context_end", "matched_context_end",
        "matched_event_onset",
    ):
        joined[column] = pd.to_datetime(joined[column], utc=True, errors="coerce")
    leads = pd.to_numeric(joined.lead_minutes, errors="coerce").to_numpy(float)
    expected_leads = (joined.event_onset - joined.matched_context_end).dt.total_seconds(
    ).to_numpy() / 60
    if (
        joined[["event_onset", "alarm_context_end", "matched_context_end",
                "matched_event_onset"]].isna().any().any()
        or not joined.event_onset.equals(joined.matched_event_onset)
        or not joined.alarm_context_end.equals(joined.matched_context_end)
        or not np.isfinite(leads).all()
        or (leads <= 0).any() or (leads > evaluation.spec.horizon_minutes).any()
        or not np.allclose(leads, expected_leads, rtol=0, atol=1e-12)
        or not np.allclose(
            leads, joined.matched_lead_minutes.to_numpy(float), rtol=0, atol=1e-12
        )
    ):
        raise ValueError("fixed event/alarm match timestamps or lead times disagree")
    table_index = pd.MultiIndex.from_frame(table[_IDS])
    for column, frame in (
        ("base_event_count", events), ("detected_event_count", detected),
        ("matched_alarm_count", alarms.loc[alarms.matched]),
        ("delivered_alarm_count", alarms.loc[alarms.base_evaluable]),
        ("unmatched_alarm_count", alarms.loc[alarms.unmatched]),
        ("emitted_alarm_count", alarms),
        ("unevaluable_alarm_count", alarms.loc[~alarms.base_evaluable]),
    ):
        counts = frame.groupby(_IDS, observed=True).size().reindex(
            table_index, fill_value=0
        ).to_numpy()
        if column not in table or not np.array_equal(table[column], counts):
            raise ValueError(f"participant {column} disagrees with fixed ledgers")
    days = pd.to_numeric(table.patient_days, errors="coerce").to_numpy(float)
    minutes = pd.to_numeric(table.observation_minutes, errors="coerce").to_numpy(float)
    if (
        not np.isfinite(days).all() or (days <= 0).any()
        or not np.equal(days, np.floor(days)).all()
        or not np.isfinite(minutes).all() or (minutes <= 0).any()
        or (minutes > days * 1440).any()
    ):
        raise ValueError("participant exposure is invalid")
    positive = table.base_event_count.to_numpy() > 0
    sensitivity = table.fixed_sensitivity.to_numpy(float)
    if not np.isnan(sensitivity[~positive]).all() or not np.allclose(
        sensitivity[positive], table.detected_event_count.to_numpy()[positive]
        / table.base_event_count.to_numpy()[positive], rtol=0, atol=1e-12,
    ):
        raise ValueError("participant sensitivity disagrees with event counts")
    for metric, count in (
        ("unmatched_alarms_per_patient_day", "unmatched_alarm_count"),
        ("delivered_alarms_per_patient_day", "delivered_alarm_count"),
    ):
        if not np.allclose(table[metric], table[count] / days, rtol=0, atol=1e-12):
            raise ValueError("participant burden disagrees with exposure and alarms")
    # Use the established aggregate helper and require its complete native schema.
    try:
        _aggregate(table)
    except KeyError as error:
        raise ValueError(
            "participant ledger lacks the complete fixed-frame schema"
        ) from error
    return table, joined


def _estimands(table: pd.DataFrame, source_order: tuple[str, ...]) -> dict:
    pooled = _aggregate(table)
    sources = [
        {"source_id": source, **_aggregate(table.loc[table.source_id.eq(source)])}
        for source in source_order
    ]
    source_sensitivity = [row["fixed_sensitivity"] for row in sources]
    positive = table.loc[table.base_event_count.gt(0)]
    sensitivity = {
        "equal_source_participant_macro": _quantity(
            _mean(source_sensitivity), np.sum(source_sensitivity), len(source_order)
        ),
        "participant_pooled": _quantity(
            pooled["fixed_sensitivity"], positive.fixed_sensitivity.sum(), len(positive)
        ),
        "event_pooled": _quantity(
            pooled["event_pooled_sensitivity"], pooled["detected_event_count"],
            pooled["base_event_count"],
        ),
    }
    burden = {}
    for name in ("unmatched", "delivered"):
        metric = f"{name}_alarms_per_patient_day"
        source_values = [row[f"participant_macro_{metric}"] for row in sources]
        burden[name] = {
            "equal_source_participant_macro_per_patient_day": _quantity(
                _mean(source_values), np.sum(source_values), len(source_order)
            ),
            "participant_pooled_per_patient_day": _quantity(
                pooled[f"participant_macro_{metric}"], table[metric].sum(), len(table)
            ),
            "exposure_pooled_per_patient_day": _quantity(
                pooled[f"pooled_{metric}"], pooled[f"{name}_alarm_count"],
                pooled["patient_days"],
            ),
        }
    burden["unmatched"]["exposure_pooled_per_24h"] = _quantity(
        pooled["pooled_unmatched_alarms_per_24h"], pooled["unmatched_alarm_count"],
        pooled["observation_minutes"] / 1440,
    )
    return {
        "source_order": list(source_order), "source_count": len(source_order),
        "participant_count": len(table),
        "event_positive_participant_count": len(positive),
        "base_event_count": pooled["base_event_count"],
        "patient_days": pooled["patient_days"],
        "observation_minutes": pooled["observation_minutes"],
        "undefined_sensitivity_sources": [
            row["source_id"] for row in sources
            if row["event_positive_participant_count"] == 0
        ],
        "undefined_burden_sources": [
            row["source_id"] for row in sources if row["participant_count"] == 0
        ],
        "sensitivity": sensitivity, "burden": burden,
        "per_source": [{
            "source_id": row["source_id"],
            **{name: row[name] for name in (
                "participant_count", "event_positive_participant_count",
                "base_event_count", "detected_event_count", "patient_days",
                "observation_minutes", "fixed_sensitivity", "event_pooled_sensitivity",
                "participant_macro_unmatched_alarms_per_patient_day",
                "participant_macro_delivered_alarms_per_patient_day",
                "pooled_unmatched_alarms_per_patient_day",
                "pooled_delivered_alarms_per_patient_day",
            )},
        } for row in sources],
        "denominator_meanings": {
            "equal_source_participant_macro": "all_declared_sources",
            "participant_pooled_sensitivity": "all_event_positive_participants",
            "event_pooled_sensitivity": "all_fixed_base_events",
            "participant_pooled_burden": "all_exposed_participants",
            "exposure_pooled_burden": "all_patient_days_or_effective_24h_days",
        },
    }


def summarize_fixed_frame_estimands(evaluation: FixedFrameEvaluation) -> dict:
    """Retain every declared source, including undefined zero-support terms."""
    table, _ = _validate(evaluation)
    return _estimands(table, evaluation.spec.source_order)


def _coverage(
    table: pd.DataFrame, matched: pd.DataFrame, source_order: tuple[str, ...],
) -> dict:
    positive = table.loc[table.base_event_count.gt(0)].copy()
    index = pd.MultiIndex.from_frame(positive[_IDS])
    rows = []
    for lead in (10, 15, 20):
        qualifying = matched.loc[
            matched.cgm_last.ge(70) & matched.lead_minutes.ge(lead)
        ]
        counts = qualifying.groupby(_IDS, observed=True).size().reindex(
            index, fill_value=0
        ).to_numpy()
        rates = pd.Series(counts / positive.base_event_count.to_numpy(), index=index)
        per_source = []
        for source in source_order:
            source_mask = positive.source_id.eq(source).to_numpy()
            per_source.append({
                "source_id": source,
                "event_positive_participant_count": int(source_mask.sum()),
                "qualifying_event_count": int(counts[source_mask].sum()),
                "base_event_count": int(positive.loc[
                    source_mask, "base_event_count"
                ].sum()),
                "participant_macro_coverage": _mean(rates.to_numpy()[source_mask]),
            })
        source_rates = [row["participant_macro_coverage"] for row in per_source]
        rows.append({
            "minimum_alarm_glucose_mg_dl": 70, "minimum_lead_minutes": lead,
            "qualifying_event_count": len(qualifying),
            "equal_source_participant_macro": _quantity(
                _mean(source_rates), np.sum(source_rates), len(source_order)
            ),
            "participant_pooled": _quantity(_mean(rates), rates.sum(), len(positive)),
            "event_pooled": _quantity(
                _ratio(len(qualifying), table.base_event_count.sum()), len(qualifying),
                table.base_event_count.sum(),
            ),
            "per_source": per_source,
        })
    return {
        "matching": "unchanged_fixed_event_alarm_matches",
        "event_denominator": "all_fixed_base_events",
        "participant_denominator": "event_positive_participants",
        "source_order": list(source_order),
        "undefined_sensitivity_sources": [
            source for source in source_order if not positive.source_id.eq(source).any()
        ],
        "coverage": rows,
    }


def summarize_fixed_match_coverage(evaluation: FixedFrameEvaluation) -> dict:
    """Joint glucose >=70 AND lead >=10/15/20 coverage on the original matches."""
    table, matched = _validate(evaluation)
    return _coverage(table, matched, evaluation.spec.source_order)


def source_influence_report(
    evaluations: Mapping[SourceInfluenceKey, FixedFrameEvaluation], *,
    source_order: tuple[str, ...], model_seed_order: tuple[int, ...] = MODEL_SEEDS,
    horizon_order: tuple[int, ...] = HORIZONS, control_seed: int | None = None,
) -> dict:
    """Describe the fixed principal contrast and each changed LOO estimand.

    Keys may also be existing AnalysisKey/FixedFrameKey objects with the same
    named fields. Require exactly one persistence control seed (None by default,
    explicitly 17 for cross-fitting), never five aliases counted as five fits.
    Additional arms/cuts are described separately; the principal contrast uses
    only LSTM/persistence at cuts 54 and 70. Smaller complete seed/horizon sets
    are supported for synthetic oracles, not replacements of the frozen recipe.
    """
    for name, values, allowed in (
        ("source_order", source_order, None),
        ("model_seed_order", model_seed_order, MODEL_SEEDS),
        ("horizon_order", horizon_order, HORIZONS),
    ):
        if not values or len(values) != len(set(values)) or (
            allowed is not None and not set(values).issubset(allowed)
        ):
            raise ValueError(f"invalid {name}")
    if (
        control_seed is not None and (
            isinstance(control_seed, bool) or not isinstance(control_seed, int)
            or control_seed != 17
        )
    ):
        raise ValueError("control_seed must explicitly identify None or seed17")
    canonical = {}
    for key, evaluation in evaluations.items():
        identity = SourceInfluenceKey(
            key.arm, key.model_seed, key.glucose_cut, key.horizon_minutes
        )
        if identity in canonical:
            raise ValueError("duplicate canonical evaluation identity")
        if (
            evaluation.spec.source_order != source_order
            or evaluation.spec.horizon_minutes != key.horizon_minutes
            or evaluation.glucose_cut != key.glucose_cut
        ):
            raise ValueError("evaluation identity or declared source order differs")
        canonical[identity] = evaluation
    primary = {
        key: value for key, value in canonical.items()
        if key.arm in ("lstm", "persistence") and key.glucose_cut in (54, 70)
        and key.horizon_minutes in horizon_order
    }
    expected = {
        SourceInfluenceKey(arm, seed, cut, horizon)
        for arm, seeds in (("lstm", model_seed_order), ("persistence", (control_seed,)))
        for seed in seeds for cut in (54, 70) for horizon in horizon_order
    }
    if set(primary) != expected:
        raise ValueError("primary requires complete LSTM seeds and one shared control")
    tables, condition_rows, truth_by_horizon = {}, [], {}
    for key, evaluation in canonical.items():
        table, matched = _validate(evaluation)
        tables[key] = table
        participant_truth = table[[
            *_IDS, "base_event_count", "patient_days", "observation_minutes",
            "base_patient_days",
        ]].sort_values(_IDS).reset_index(drop=True)
        event_truth = evaluation.events[[*_EVENT_IDS, "event_onset"]].sort_values(
            _EVENT_IDS
        ).reset_index(drop=True)
        current_truth = (participant_truth, event_truth)
        if key.horizon_minutes in truth_by_horizon:
            try:
                for current, reference in zip(
                    current_truth, truth_by_horizon[key.horizon_minutes], strict=True
                ):
                    pd.testing.assert_frame_equal(
                        current, reference, check_dtype=False, check_exact=True,
                        check_categorical=False,
                    )
            except AssertionError:
                raise ValueError(
                    "same-horizon event/exposure truth differs across comparisons"
                ) from None
        truth_by_horizon[key.horizon_minutes] = current_truth
        condition_rows.append({
            "arm": key.arm, "model_seed": key.model_seed,
            "glucose_cut": key.glucose_cut, "horizon_minutes": key.horizon_minutes,
            "threshold": evaluation.threshold,
            "estimands": _estimands(table, source_order),
            "fixed_match_joint_coverage": _coverage(table, matched, source_order),
        })
    effects = np.empty((len(source_order), len(horizon_order), len(model_seed_order)))
    for s_index, source in enumerate(source_order):
        for h_index, horizon in enumerate(horizon_order):
            for m_index, seed in enumerate(model_seed_order):
                value = 0.0
                for arm, actual_seed, cut, coefficient in (
                    ("lstm", seed, 70, 1), ("persistence", control_seed, 70, -1),
                    ("lstm", seed, 54, -1), ("persistence", control_seed, 54, 1),
                ):
                    table = tables[SourceInfluenceKey(arm, actual_seed, cut, horizon)]
                    positive = table.loc[
                        table.source_id.eq(source) & table.base_event_count.gt(0)
                    ]
                    value += coefficient * _mean(positive.fixed_sensitivity)
                effects[s_index, h_index, m_index] = value
    full = _mean(effects)
    undefined = [
        source for index, source in enumerate(source_order)
        if not np.isfinite(effects[index]).all()
    ]
    return {
        "analysis": "descriptive_source_influence",
        "primary_contrast": "(LSTM-current_glucose)_70_minus_(LSTM-current_glucose)_54",
        "source_order": list(source_order), "model_seeds": list(model_seed_order),
        "horizons": list(horizon_order),
        "control": {"arm": "persistence", "model_seed": control_seed,
                    "unique_fit_count": 1},
        "aggregation": "equal_source_then_equal_horizon_and_model_seed",
        "uncertainty": "descriptive_no_generalization_ci",
        "conditions": condition_rows,
        "full_contrast": {"estimate": full, "undefined_sources": undefined},
        "per_source": [{
            "source_id": source, "estimate": _mean(effects[index]),
            "contribution_to_full": _mean(effects[index]) / len(source_order),
            "source_weight": 1 / len(source_order),
            "per_horizon_seed": effects[index].tolist(),
        } for index, source in enumerate(source_order)],
        "leave_one_source_out": [{
            "excluded_source": source,
            "retained_source_order": [
                value for value in source_order if value != source
            ],
            "estimand_changed": True,
            "estimate": _mean(np.delete(effects, index, axis=0)),
            "difference_from_full": _mean(np.delete(effects, index, axis=0)) - full,
            "undefined_sources": [value for value in undefined if value != source],
        } for index, source in enumerate(source_order)],
    }


def _snapshot(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError("analysis input must be a regular non-symlink file")
    stat = path.stat()
    return {"path": str(path.resolve()), "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fit_identity(row: Mapping) -> tuple[str, int | None]:
    arm, seed = row["arm"], row["model_seed"]
    if (
        arm not in METHODS or isinstance(seed, bool)
        or (seed is not None and not isinstance(seed, int))
    ):
        raise ValueError("producer method/seed identity is invalid")
    return arm, seed


def _condition_identity(row: Mapping) -> tuple[str, int | None, int, int]:
    arm, seed = _fit_identity(row)
    cut, horizon = row["glucose_cut"], row["horizon_minutes"]
    if (
        isinstance(cut, bool) or not isinstance(cut, int)
        or isinstance(horizon, bool) or not isinstance(horizon, int)
    ):
        raise ValueError("analysis cut/horizon identity must use integer minutes/units")
    return arm, seed, cut, horizon


def prepare_source_influence_config(
    analysis_config_path: Path, output_dir: Path,
) -> dict:
    """Snapshot standard completed-analysis paths once, without reading raw data.

    Save the returned private config and run this module with ``--config PATH``.
    Its stat bindings are checked again before loading and before completion.
    """
    analysis_config_path = Path(analysis_config_path)
    original = _read_json(_bound_file(_snapshot(analysis_config_path)))
    prepared, analysis = (
        Path(original["prepared_dir"]).resolve(), Path(original["output_dir"]).resolve()
    )
    paths = {
        "analysis_config": analysis_config_path,
        "prepared_audit": prepared / "input-audit.json",
        "analysis_status": analysis / "analysis-status.json",
        "fit_ledger": analysis / "private/unique-fit-ledger.json",
        "cut_grid_summary": analysis / "exportable/cut-grid-summary.json",
        **{
            f"selection_{category}_{horizon}": prepared / "endpoints/selection"
            / f"{category}-{horizon}.parquet"
            for category in ("events", "exposure") for horizon in HORIZONS
        },
    }
    producers, conditions = [], []
    for entry in original["arms"]:
        arm, seed = _fit_identity(entry)
        directory = Path(entry["run_dir"]).resolve()
        producers.append({
            "arm": arm, "model_seed": seed, "run_dir": str(directory),
            "files": {name: _snapshot(directory / filename) for name, filename in (
                ("model_summary", "model_summary.json"),
                ("config", "config_resolved.yaml"),
                ("cohort_audit", "cohort_audit.json"),
            )},
        })
        for cut in (54, 70):
            for horizon in HORIZONS:
                root = analysis / "private" / arm / (
                    "shared" if seed is None else str(seed)
                ) / "refitted" / str(cut) / str(horizon)
                conditions.append({
                    "arm": arm, "model_seed": seed, "glucose_cut": cut,
                    "horizon_minutes": horizon,
                    "files": {name: _snapshot(root / (
                        f"{name}.json" if name in ("policy", "summary")
                        else f"{name}.parquet"
                    )) for name in (
                        "policy", "summary", "participants", "sources", "events",
                        "alarms",
                    )},
                })
    return {
        "revision_contract": REVISION_CONTRACT,
        "ascertainment_contract": CONFIRMATION_CONTRACT,
        "prepared_dir": str(prepared), "analysis_dir": str(analysis),
        "output_dir": str(Path(output_dir).resolve()),
        "input_bindings": {name: _snapshot(path) for name, path in paths.items()},
        "producers": producers, "conditions": conditions,
    }


def _assert_same_frame(
    left: pd.DataFrame, right: pd.DataFrame, keys: list[str],
) -> None:
    try:
        pd.testing.assert_frame_equal(
            left.sort_values(keys).reset_index(drop=True),
            right.sort_values(keys).reset_index(drop=True),
            check_dtype=False, check_exact=True, check_categorical=False,
        )
    except AssertionError:
        raise ValueError(
            "analysis ledger truth differs from corrected prepared endpoints"
        ) from None


def run_source_influence(config: Mapping[str, Any]) -> Path:
    """Accept all fourteen fits/56 requested conditions, then export aggregates.

    The config is produced by ``prepare_source_influence_config``. Model/config,
    cohort, policy, summary and every complete ledger are individually bound.
    Only metadata and participant/event/alarm evaluation ledgers are decoded.
    """
    import yaml

    if (
        config.get("revision_contract") != REVISION_CONTRACT
        or config.get("ascertainment_contract") != CONFIRMATION_CONTRACT
    ):
        raise ValueError("source influence requires the complete-confirmation contract")
    output = Path(config["output_dir"]).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("refusing to overwrite non-empty source influence output")
    analysis, prepared = (
        Path(config["analysis_dir"]).resolve(), Path(config["prepared_dir"]).resolve()
    )
    bindings = []

    def accepted(binding: Mapping, expected: Path | None = None) -> Path:
        path = _bound_file(binding, expected)
        bindings.append(dict(binding))
        return path

    expected_inputs = {
        "prepared_audit": prepared / "input-audit.json",
        "analysis_status": analysis / "analysis-status.json",
        "fit_ledger": analysis / "private/unique-fit-ledger.json",
        "cut_grid_summary": analysis / "exportable/cut-grid-summary.json",
        **{
            f"selection_{category}_{horizon}": prepared / "endpoints/selection"
            / f"{category}-{horizon}.parquet"
            for category in ("events", "exposure") for horizon in HORIZONS
        },
    }
    if set(config["input_bindings"]) != {*expected_inputs, "analysis_config"}:
        raise ValueError("source influence input bindings are incomplete or unexpected")
    inputs = {
        name: accepted(value, expected_inputs.get(name))
        for name, value in config["input_bindings"].items()
    }
    original = _read_json(inputs["analysis_config"])
    audit = _read_json(inputs["prepared_audit"])
    index = _read_json(inputs["cut_grid_summary"])
    fits = _read_json(inputs["fit_ledger"])["fits"]
    expected_fits = {
        (arm, seed) for arm in METHODS
        for seed in (MODEL_SEEDS if arm in ("lstm", "tcn") else (None,))
    }
    source_order = tuple(audit["source_order"])
    if (
        _read_json(inputs["analysis_status"]).get("status") != "complete"
        or original.get("revision_contract") != REVISION_CONTRACT
        or Path(original["output_dir"]).resolve() != analysis
        or Path(original["prepared_dir"]).resolve() != prepared
        or original.get("threshold_candidate_count", 2001) != 2001
        or audit.get("status") != "passed"
        or audit.get("revision_contract") != REVISION_CONTRACT
        or audit.get("ascertainment_contract") != CONFIRMATION_CONTRACT
        or audit.get("timeline_storage") != "original_partitions"
        or index.get("status") != "complete"
        or index.get("revision_contract") != REVISION_CONTRACT
        or tuple(index.get("source_order", ())) != source_order
        or tuple(index.get("cuts", ())) != ORIGINAL_CUTS
        or tuple(index.get("horizons", ())) != HORIZONS
        or tuple(index.get("model_seeds", ())) != MODEL_SEEDS
        or index.get("nominal_fapd") != 0.5
        or index.get("evaluation_partition") != "selection"
        or index.get("threshold_partition") != "calibration"
        or index.get("unique_fit_count") != len(expected_fits)
        or index.get("primary_control_unique_fit_count") != 1
        or index.get("primary_lstm_unique_fit_count") != 5
    ):
        raise ValueError("corrected preparation or completed analysis identity differs")
    recorded_fits = {}
    for rows in (original["arms"], fits, config["producers"]):
        current = {_fit_identity(row): Path(row["run_dir"]).resolve() for row in rows}
        if (
            len(rows) != len(current) or set(current) != expected_fits
            or len(set(current.values())) != len(current)
            or (recorded_fits and current != recorded_fits)
        ):
            raise ValueError("six methods/deep seeds or producer identities differ")
        recorded_fits = current
    for root in (prepared, analysis, *recorded_fits.values()):
        if output == root or output.is_relative_to(root) or root.is_relative_to(output):
            raise ValueError("source influence output must be isolated from its inputs")
    model_summaries = {}
    for producer in config["producers"]:
        identity = _fit_identity(producer)
        arm, seed = identity
        directory = recorded_fits[identity]
        if set(producer["files"]) != {"model_summary", "config", "cohort_audit"}:
            raise ValueError("producer metadata bindings are incomplete")
        summary = _read_json(accepted(
            producer["files"]["model_summary"], directory / "model_summary.json"
        ))
        cohort = _read_json(accepted(
            producer["files"]["cohort_audit"], directory / "cohort_audit.json"
        ))
        effective = yaml.safe_load(accepted(
            producer["files"]["config"], directory / "config_resolved.yaml"
        ).read_text(encoding="utf-8"))
        actual_seed = seed if arm in ("lstm", "tcn") else 17
        if (
            cohort != audit
            or cohort.get("ascertainment_contract") != CONFIRMATION_CONTRACT
            or effective.get("protocol", {}).get("ascertainment_contract")
            != CONFIRMATION_CONTRACT
            or effective.get("training", {}).get("revision_contract")
            != REVISION_CONTRACT
            or effective.get("training", {}).get("seed") != actual_seed
            or effective.get("model", {}).get("input_feature_mode") != "cgm_mask_clock"
            or tuple(sorted(effective.get("data", {}).get("development_sources", [])))
            != source_order
            or Path(effective.get("output", {}).get("run_dir", "")).resolve()
            != directory
            or summary.get("revision_contract") != REVISION_CONTRACT
            or summary.get("training_seed") != actual_seed
            or summary.get("input_feature_mode") != "cgm_mask_clock"
            or summary.get("architecture" if arm in ("lstm", "tcn") else "baseline")
            != arm
            or any(
                summary.get("tasks", {}).get(f"hypo_{h}", {}).get(
                    "threshold_partition"
                ) != "calibration"
                or summary["tasks"][f"hypo_{h}"].get("temperature_fit_partition")
                != "calibration"
                or summary["tasks"][f"hypo_{h}"].get("target_column")
                != f"severe_hypo_{h}_label"
                for h in HORIZONS
            )
        ):
            raise ValueError("producer config/cohort/model is not corrected original")
        model_summaries[identity] = summary
    indexed = {}
    for row in index["analyses"]:
        identity = (*_condition_identity(row), row["mode"])
        if identity in indexed:
            raise ValueError("duplicate indexed analysis condition")
        indexed[identity] = row
    expected_index = {
        (arm, seed, cut, horizon, mode) for arm, seed in expected_fits
        for cut in ORIGINAL_CUTS for horizon in HORIZONS
        for mode in ("refitted", "gate_only")
    }
    if set(indexed) != expected_index:
        raise ValueError("completed cut grid has missing or unexpected conditions")
    expected_conditions = {
        (arm, seed, cut, horizon) for arm, seed in expected_fits
        for cut in (54, 70) for horizon in HORIZONS
    }
    conditions = config["conditions"]
    if (
        len(conditions) != len(expected_conditions)
        or {_condition_identity(row) for row in conditions} != expected_conditions
    ):
        raise ValueError("runner requires exactly 56 unique refitted 54/70 conditions")
    prepared_truth = {}
    for horizon in HORIZONS:
        tables = []
        for category in ("events", "exposure"):
            path = inputs[f"selection_{category}_{horizon}"]
            audited = Path(audit["endpoint_paths"][f"selection/{horizon}"][category])
            if audited.resolve() != path:
                raise ValueError("prepared selection endpoint differs from its audit")
            with pq.ParquetFile(path) as parquet:
                table = parquet.read(use_threads=False).to_pandas()
            if "partition" in table and not table.partition.eq("selection").all():
                raise ValueError("prepared endpoints contain a non-selection role")
            tables.append(table)
        prepared_truth[horizon] = tables
    evaluations = {}
    for condition in conditions:
        arm, seed, cut, horizon = _condition_identity(condition)
        root = analysis / "private" / arm / (
            "shared" if seed is None else str(seed)
        ) / "refitted" / str(cut) / str(horizon)
        files = condition["files"]
        if set(files) != {
            "policy", "summary", "participants", "sources", "events", "alarms"
        }:
            raise ValueError("complete condition ledger bindings are required")
        paths = {
            name: accepted(binding, root / (
                f"{name}.json" if name in ("policy", "summary") else f"{name}.parquet"
            )) for name, binding in files.items()
        }
        wrapper, summary = _read_json(paths["policy"]), _read_json(paths["summary"])
        row = indexed[arm, seed, cut, horizon, "refitted"]
        base = model_summaries[arm, seed]["tasks"][f"hypo_{horizon}"][
            "fixed_frame_policies"
        ]["0.5"]
        try:
            _policy_row(row, wrapper, list(source_order), 2001, base)
        except KeyError as error:
            raise ValueError(
                "retained calibration policy schema is incomplete"
            ) from error
        loaded = {}
        for name in ("participants", "sources", "events", "alarms"):
            with pq.ParquetFile(paths[name]) as parquet:
                loaded[name] = parquet.read(use_threads=False).to_pandas()
        policy = wrapper["policy"]
        evaluation = FixedFrameEvaluation(
            spec=FixedFrameSpec(
                policy["endpoint"]["score_column"], horizon, source_order
            ), threshold=policy["threshold"], glucose_cut=cut, summary=summary,
            **loaded,
        )
        table, _ = _validate(evaluation)
        events, exposure = prepared_truth[horizon]
        truth_columns = [*_EVENT_IDS, "event_onset"]
        _assert_same_frame(
            loaded["events"][truth_columns], events[truth_columns], _EVENT_IDS
        )
        exposure_columns = [*_IDS, "patient_days", "observation_minutes"]
        _assert_same_frame(table[exposure_columns], exposure[exposure_columns], _IDS)
        source_rows = [{"source_id": source, **_aggregate(table.loc[
            table.source_id.eq(source)
        ])} for source in source_order]
        actual_sources = loaded["sources"].to_dict("records")
        if _plain(source_rows) != _plain(actual_sources) or any(
            name not in row or _plain(row[name]) != _plain(value)
            for name, value in summary.items()
        ):
            raise ValueError("source/indexed summary differs from participant ledgers")
        points = _estimands(table, source_order)
        for name, value in {
            "participant_count": len(table),
            "base_event_count": points["base_event_count"],
            "participant_pooled_sensitivity": points["sensitivity"][
                "participant_pooled"
            ]["estimate"],
            "equal_source_participant_macro_fixed_sensitivity": points["sensitivity"][
                "equal_source_participant_macro"
            ]["estimate"],
            "equal_source_participant_macro_unmatched_alarms_per_patient_day": (
                points["burden"]["unmatched"][
                    "equal_source_participant_macro_per_patient_day"
                ]["estimate"]
            ),
        }.items():
            if name not in summary or _plain(summary[name]) != _plain(value):
                raise ValueError("terminal estimand differs from ledger quantities")
        evaluations[SourceInfluenceKey(arm, seed, cut, horizon)] = evaluation
    report = source_influence_report(evaluations, source_order=source_order)
    report.update({
        "revision_contract": REVISION_CONTRACT,
        "ascertainment_contract": CONFIRMATION_CONTRACT,
        "status": "complete", "methods": list(METHODS), "unique_fit_count": 14,
        "condition_count": 56, "evaluation_partition": "selection",
        "threshold_partition": "calibration", "nominal_fapd": 0.5,
    })
    estimand_rows, coverage_rows = [], []
    for condition in report["conditions"]:
        identity = {name: condition[name] for name in (
            "arm", "model_seed", "glucose_cut", "horizon_minutes"
        )}
        for name, quantity in condition["estimands"]["sensitivity"].items():
            estimand_rows.append({**identity, "metric": "sensitivity",
                                  "estimand": name, **quantity})
        for metric, quantities in condition["estimands"]["burden"].items():
            for name, quantity in quantities.items():
                estimand_rows.append({**identity, "metric": f"{metric}_alarm_burden",
                                      "estimand": name, **quantity})
        for coverage in condition["fixed_match_joint_coverage"]["coverage"]:
            for name in (
                "equal_source_participant_macro", "participant_pooled", "event_pooled"
            ):
                coverage_rows.append({
                    **identity, "minimum_alarm_glucose_mg_dl": 70,
                    "minimum_lead_minutes": coverage["minimum_lead_minutes"],
                    "estimand": name, **coverage[name],
                })
    for binding in bindings:
        _bound_file(binding)
    output.mkdir(parents=True, exist_ok=True)
    private, exportable = output / "private", output / "exportable"
    private.mkdir(exist_ok=True)
    exportable.mkdir(exist_ok=True)
    _json(private / "input-config.json", dict(config))
    _json(exportable / "source-influence.json", report)
    pd.DataFrame(estimand_rows).to_csv(exportable / "estimands.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(exportable / "joint-coverage.csv", index=False)
    pd.DataFrame(report["leave_one_source_out"]).drop(columns=[
        "retained_source_order", "undefined_sources"
    ]).to_csv(exportable / "leave-one-source-out.csv", index=False)
    pd.DataFrame(report["per_source"]).drop(columns=["per_horizon_seed"]).to_csv(
        exportable / "source-contributions.csv", index=False
    )
    for binding in bindings:
        _bound_file(binding)
    _json(private / "receipt.json", {
        "status": "complete", "revision_contract": REVISION_CONTRACT,
        "ascertainment_contract": CONFIRMATION_CONTRACT,
        "input_bindings": bindings, "condition_count": len(evaluations),
        "unique_fit_count": len(recorded_fits),
        "raw_cgm_or_weights_read": False, "policies_refit": False,
        "outputs": [_snapshot(path) for path in sorted(exportable.iterdir())],
    })
    _json(output / "status.json", {"status": "complete", "condition_count": 56})
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    run_source_influence(_read_json(arguments.config))


if __name__ == "__main__":
    main()
