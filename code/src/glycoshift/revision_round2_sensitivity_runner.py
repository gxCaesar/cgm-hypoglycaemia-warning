"""Consume accepted corrected-original cut-grid ledgers without fitting policies.

CLI from code/: PYTHONPATH=src python -m
glycoshift.revision_round2_sensitivity_runner --config accepted.json

The JSON extends revision_event_sensitivity's accepted-input schema with a bound
analysis_summary (cut-grid-summary.json) and each run's bound cohort_audit. It
requires all fourteen fits across six methods, five seeds per deep architecture,
and exactly the refitted 54/70 by 30/60-minute conditions at budget 0.5. Private
time axes, catalogues and rematched ledgers remain beside the approved inputs.
Only aggregate exportable/ files may be selected for a later curated pull.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .events import Event
from .revision_event_sensitivity import (
    _MATCH,
    _TRUTH,
    HORIZONS,
    _aggregate_participants,
    _equal,
    _event_frame,
    _identity,
    _primary_catalogue,
    _read_key,
    _read_participant,
    _validate_protocol,
    base_catalogue,
    build_variant_catalogue,
    rematch_emitted_alarms,
    validate_event_sensitivity_inputs,
)
from .revision_external import _bound_file, _json, _parquet, _recheck_bindings
from .revision_inputs import (
    CONFIRMATION_CONTRACT,
    build_participant_replay,
    validate_confirmation_masks,
)
from .revision_round2_sensitivity import (
    CatalogueRematch,
    ElapsedSpanEvent,
    build_elapsed_span_catalogue,
    rematch_catalogues_on_common_frame,
    summarize_common_source_support,
)
from .sequences import REVISION_CONTRACT

MODEL_SEEDS = (17, 23, 42, 71, 101)
METHODS = ("persistence", "linear", "summary_logistic", "xgboost", "lstm", "tcn")
_IDS = ["source_id", "participant_id"]
_CONDITIONS = ["arm", "model_seed", "horizon_minutes", "glucose_cut"]
_PAIRS = {
    "observed_only": ("primary", "observed_only"),
    "elapsed_span": ("primary", "elapsed_span"),
}


def _signature(arm, seed, horizon, cut):
    return arm, None if pd.isna(seed) else int(seed), int(horizon), int(cut)


def _validate_round2(config, inputs):
    if (
        inputs.audit.get("ascertainment_contract") != CONFIRMATION_CONTRACT
        or inputs.replay_config["protocol"].get("ascertainment_contract")
        != CONFIRMATION_CONTRACT
        or inputs.audit.get("timeline_storage", "original_partitions")
        != "original_partitions"
        or "crossfit_contract" in inputs.audit
    ):
        raise ValueError("round2 sensitivity requires corrected-original preparation")
    if not inputs.replay_config["project"].get("synthetic", False) and (
        len(inputs.sources) != 11
        or inputs.audit.get("original_participants") != 1020
        or len(inputs.selected) != 153
    ):
        raise ValueError("round2 sensitivity requires the complete original population")
    summary_path = _bound_file(config["analysis_summary"])
    summary = json.loads(summary_path.read_text())
    if (
        summary.get("status") != "complete"
        or summary.get("revision_contract") != REVISION_CONTRACT
        or tuple(summary.get("source_order", ())) != inputs.sources
        or tuple(summary.get("model_seeds", ())) != MODEL_SEEDS
        or summary.get("nominal_fapd") != 0.5
        or summary.get("evaluation_partition") != "selection"
        or summary.get("threshold_partition") != "calibration"
    ):
        raise ValueError(
            "bound original cut-grid summary is incomplete or incompatible"
        )
    expected_fits = {
        (arm, seed)
        for arm in METHODS
        for seed in (MODEL_SEEDS if arm in {"lstm", "tcn"} else (None,))
    }
    if {(run["arm"], run["model_seed"]) for run in inputs.runs} != expected_fits:
        raise ValueError("round2 requires all six methods and fourteen actual fits")
    expected_conditions = {
        (horizon, cut, 0.5) for horizon in HORIZONS for cut in (54, 70)
    }
    analysis_root = summary_path.parent.parent
    if summary_path != analysis_root / "exportable/cut-grid-summary.json":
        raise ValueError("cut-grid summary escapes its producing analysis layout")
    available_rows = [
        _signature(
            row["arm"], row["model_seed"], row["horizon_minutes"], row["glucose_cut"]
        )
        for row in summary["analyses"]
        if row["mode"] == "refitted" and row["glucose_cut"] in (54, 70)
    ]
    available = set(available_rows)
    required = {
        (arm, seed, horizon, cut)
        for arm, seed in expected_fits
        for horizon in HORIZONS
        for cut in (54, 70)
    }
    if available != required or len(available_rows) != len(required):
        raise ValueError("cut-grid summary lacks the complete refitted condition set")
    protected = [inputs.prepared, inputs.normalized_root, analysis_root]
    for run in inputs.runs:
        run_root = Path(run["run_dir"]).resolve()
        protected.append(run_root)
        cohort = json.loads(
            _bound_file(run["cohort_audit"], run_root / "cohort_audit.json").read_text()
        )
        if (
            cohort.get("ascertainment_contract") != CONFIRMATION_CONTRACT
            or cohort.get("revision_contract") != REVISION_CONTRACT
            or "crossfit_contract" in cohort
            or any(
                cohort.get(field) != inputs.audit.get(field)
                for field in ("normalized_path", "legacy_windows", "source_order")
            )
        ):
            raise ValueError(
                "producing fit is not bound to the corrected-original pool"
            )
        conditions = run["conditions"]
        actual = {
            (item["horizon_minutes"], item["glucose_cut"], item["nominal_fapd"])
            for item in conditions
        }
        if actual != expected_conditions:
            raise ValueError(
                "round2 needs exactly cuts 54/70, both horizons, budget 0.5"
            )
        if Path(run["ledger_root"]).resolve() != analysis_root / "private":
            raise ValueError(
                "round2 ledgers must belong to the bound cut-grid analysis"
            )
        for condition in conditions:
            if condition.get("mode") != "refitted":
                raise ValueError("round2 requires already-refitted original policies")
            directory = (
                analysis_root
                / "private"
                / run["arm"]
                / ("shared" if run["model_seed"] is None else str(run["model_seed"]))
                / "refitted"
                / str(condition["glucose_cut"])
                / str(condition["horizon_minutes"])
            )
            for category in ("alarms", "events", "participants"):
                _bound_file(condition[category], directory / f"{category}.parquet")
    if any(
        inputs.output.is_relative_to(root) or root.is_relative_to(inputs.output)
        for root in protected
    ):
        raise ValueError("round2 output overlaps a retained input or producing run")


def _catalogue_table(events, key):
    table = _event_frame(events, key)
    table["kind"] = pd.Series([event.kind for event in events], dtype="string")
    if events and isinstance(events[0], ElapsedSpanEvent):
        table["onset_confirmed_at"] = pd.to_datetime(
            [event.onset_confirmed_at for event in events], utc=True
        )
        table["right_censored"] = [event.right_censored for event in events]
        table["censor_reason"] = pd.Series(
            [event.censor_reason for event in events], dtype="string"
        )
    return table


def _read_catalogue(path, *, elapsed=False):
    rows = pq.ParquetFile(path).read().to_pandas().to_dict("records")
    return [
        ElapsedSpanEvent(
            row["event_id"],
            row["kind"],
            row["event_onset"],
            row["event_end"],
            row["onset_confirmed_at"],
            bool(row["right_censored"]),
            None if pd.isna(row["censor_reason"]) else row["censor_reason"],
        )
        if elapsed
        else Event(row["event_id"], row["kind"], row["event_onset"], row["event_end"])
        for row in rows
    ]


def _prepare_catalogues(inputs, private):
    for ordinal, entry in enumerate(inputs.entries):
        key = (str(entry["source_id"]), str(entry["participant_id"]))
        timeline = pq.ParquetFile(entry["timeline_path"]).read().to_pandas()
        _identity(timeline, key)
        if (
            len(timeline) != entry["stored_rows"]
            or not timeline.partition.eq("selection").all()
        ):
            raise ValueError("prepared timeline row count/role differs from manifest")
        validate_confirmation_masks(timeline, CONFIRMATION_CONTRACT)
        normalized = _read_participant(
            [Path(path).resolve() for path in entry["normalized_paths"]],
            key,
            inputs.normalized_root,
        )
        replay = build_participant_replay(
            normalized, inputs.replay_config, partition="selection"
        )
        _equal(
            replay.timeline,
            timeline,
            sorted(set(replay.timeline) | set(timeline)),
            "rebuilt corrected primary timeline",
        )
        for horizon in HORIZONS:
            for category, expected in (
                ("events", replay.events[horizon]),
                ("exposure", replay.exposure[horizon]),
            ):
                actual = _read_key(inputs.endpoints[horizon, category], key)
                _equal(actual, expected, list(expected), "rebuilt primary " + category)
        catalogues = {
            "primary": _primary_catalogue(normalized),
            "observed_only": build_variant_catalogue(normalized, "observed_only"),
            "elapsed_span": build_elapsed_span_catalogue(normalized),
        }
        directory = private / "catalogues" / f"participant-{ordinal:05d}"
        directory.mkdir(parents=True)
        _parquet(directory / "time-axis.parquet", normalized[[*_IDS, "timestamp"]])
        for name, events in catalogues.items():
            _parquet(directory / f"{name}.parquet", _catalogue_table(events, key))
        if (ordinal + 1) % 25 == 0:
            print(
                json.dumps(
                    {
                        "stage": "catalogues",
                        "participants": ordinal + 1,
                        "total": len(inputs.entries),
                    }
                ),
                flush=True,
            )


def _accepted_primary(inputs, timeline, key, condition):
    horizon, cut = condition["horizon_minutes"], condition["glucose_cut"]
    events = _read_key(inputs.endpoints[horizon, "events"], key)
    exposure = _read_key(inputs.endpoints[horizon, "exposure"], key)
    accepted_events = _read_key(Path(condition["events"]["path"]), key)
    participants = _read_key(Path(condition["participants"]["path"]), key)
    emitted = _read_key(Path(condition["alarms"]["path"]), key)
    _equal(accepted_events, events, _TRUTH, "accepted primary event catalogue")
    alarms, ledger, metrics = rematch_emitted_alarms(
        timeline, events, exposure, emitted, horizon=horizon, glucose_cut=cut
    )
    _equal(
        alarms, emitted, [*_IDS, "context_end", *_MATCH], "accepted primary matching"
    )
    _equal(
        metrics, participants, list(metrics), "accepted primary participant allocation"
    )
    return emitted, exposure, CatalogueRematch(alarms, ledger, metrics)


def _check_truth(metrics):
    for _, horizon in metrics.groupby("horizon_minutes", observed=True):
        reference = None
        for _, condition in horizon.groupby(_CONDITIONS, observed=True, dropna=False):
            truth = condition.set_index(_IDS)[
                ["base_event_count", "patient_days", "observation_minutes"]
            ].sort_index()
            if reference is None:
                reference = truth
            else:
                _equal(
                    truth.reset_index(),
                    reference.reset_index(),
                    list(truth.reset_index()),
                    "method/cut/seed-invariant catalogue truth",
                )


def _concat_ledgers(tables):
    """Promote empty/null Arrow columns explicitly without Pandas dtype inference."""
    return pa.concat_tables(
        [pa.Table.from_pandas(table, preserve_index=False) for table in tables],
        promote_options="permissive",
    ).to_pandas()


def _export_pair(name, metrics, sources, exportable):
    primary, alternative = metrics["primary"], metrics[name]
    for table in metrics.values():
        _check_truth(table)
    support = summarize_common_source_support(
        primary.loc[primary.arm.isin(("lstm", "persistence"))],
        alternative.loc[alternative.arm.isin(("lstm", "persistence"))],
        source_order=sources,
    )
    common = tuple(support.audit["common_source_order"])
    folder = exportable / f"primary-vs-{name}"
    folder.mkdir()
    for category in ("source_support", "population_audit", "points", "contrasts"):
        table = getattr(support, category).copy()
        table.rename(
            columns={column: column.replace("observed_only", name) for column in table},
            inplace=True,
        )
        if "catalogue" in table:
            table["catalogue"] = table.catalogue.replace({"observed_only": name})
        _json(folder / f"{category}.json", table.to_dict("records"))
    _json(folder / "support-audit.json", support.audit)
    rows = []
    for catalogue, table in metrics.items():
        for condition, part in table.groupby(_CONDITIONS, observed=True, dropna=False):
            descriptor = dict(zip(_CONDITIONS, condition, strict=True))
            for scope, source_order in (
                ("full_sources", sources),
                ("common_support", common),
            ):
                admitted = part.loc[part.source_id.isin(source_order)]
                if source_order:
                    _, summary = _aggregate_participants(admitted, source_order)
                else:
                    summary = {
                        "source_count": 0,
                        "status": "undefined_no_common_sources",
                        "equal_source_participant_macro_fixed_sensitivity": float(
                            "nan"
                        ),
                    }
                rows.append(
                    {**descriptor, "scope": scope, "catalogue": catalogue, **summary}
                )
    _json(folder / "all-method-aggregates.json", rows)


def run_round2_sensitivity(config: dict) -> Path:
    inputs = validate_event_sensitivity_inputs(config)
    _validate_round2(config, inputs)
    _recheck_bindings(config)
    private, exportable = inputs.output / "private", inputs.output / "exportable"
    private.mkdir(parents=True)
    exportable.mkdir()
    collected = {
        name: {variant: [] for variant in variants} for name, variants in _PAIRS.items()
    }
    try:
        _prepare_catalogues(inputs, private)
        for run_index, run in enumerate(inputs.runs):
            for condition in run["conditions"]:
                horizon, cut = condition["horizon_minutes"], condition["glucose_cut"]
                descriptor = {
                    "arm": run["arm"],
                    "model_seed": run["model_seed"],
                    "horizon_minutes": horizon,
                    "glucose_cut": cut,
                }
                ledgers = {
                    name: {
                        variant: {
                            category: []
                            for category in ("alarms", "events", "participants")
                        }
                        for variant in variants
                    }
                    for name, variants in _PAIRS.items()
                }
                for ordinal, entry in enumerate(inputs.entries):
                    key = (str(entry["source_id"]), str(entry["participant_id"]))
                    timeline = pq.ParquetFile(entry["timeline_path"]).read().to_pandas()
                    validate_confirmation_masks(timeline, CONFIRMATION_CONTRACT)
                    emitted, exposure, primary = _accepted_primary(
                        inputs, timeline, key, condition
                    )
                    directory = private / "catalogues" / f"participant-{ordinal:05d}"
                    catalogues = {
                        name: _read_catalogue(
                            directory / f"{name}.parquet",
                            elapsed=name == "elapsed_span",
                        )
                        for name in ("primary", "observed_only", "elapsed_span")
                    }
                    observed = _event_frame(
                        base_catalogue(catalogues["observed_only"], timeline, horizon),
                        key,
                    )
                    result = rematch_emitted_alarms(
                        timeline,
                        observed,
                        exposure,
                        emitted,
                        horizon=horizon,
                        glucose_cut=cut,
                    )
                    _, elapsed_results = rematch_catalogues_on_common_frame(
                        pq.ParquetFile(directory / "time-axis.parquet")
                        .read()
                        .to_pandas(),
                        timeline,
                        {name: catalogues[name] for name in _PAIRS["elapsed_span"]},
                        emitted,
                        horizon=horizon,
                        glucose_cut=cut,
                    )
                    pair_results = {
                        "observed_only": {
                            "primary": primary,
                            "observed_only": CatalogueRematch(*result),
                        },
                        "elapsed_span": elapsed_results,
                    }
                    for pair, variants in pair_results.items():
                        for variant, evaluation in variants.items():
                            for category in ledgers[pair][variant]:
                                table = getattr(evaluation, category)
                                ledgers[pair][variant][category].append(table)
                            collected[pair][variant].append(
                                evaluation.participants.assign(**descriptor)
                            )
                for pair, variants in ledgers.items():
                    for variant, categories in variants.items():
                        directory = (
                            private
                            / "evaluations"
                            / pair
                            / variant
                            / f"run-{run_index:03d}"
                            / str(cut)
                            / str(horizon)
                        )
                        directory.mkdir(parents=True)
                        for category, tables in categories.items():
                            _parquet(
                                directory / f"{category}.parquet",
                                _concat_ledgers(tables),
                            )
                print(
                    json.dumps(
                        {"stage": "conditions", "run_index": run_index, **descriptor}
                    ),
                    flush=True,
                )
        for name, variants in collected.items():
            metrics = {
                variant: _concat_ledgers(tables)
                for variant, tables in variants.items()
            }
            _export_pair(name, metrics, inputs.sources, exportable)
        _recheck_bindings(config)
        _validate_protocol(inputs.replay_config, inputs.sources)
        _json(private / "accepted-inputs-used.json", config)
        _json(
            exportable / "summary.json",
            {
                "status": "complete",
                "revision_contract": REVISION_CONTRACT,
                "ascertainment_contract": CONFIRMATION_CONTRACT,
                "source_order": inputs.sources,
                "fit_count": len(inputs.runs),
                "condition_count": sum(len(run["conditions"]) for run in inputs.runs),
                "comparisons": {
                    "primary-vs-observed_only": "accepted_primary_frame",
                    "primary-vs-elapsed_span": "common_60_75_minute_frame",
                },
                "fitted_or_reemitted": False,
            },
        )
        _json(inputs.output / "status.json", {"status": "complete"})
    except Exception as error:
        _json(
            inputs.output / "status.json",
            {"status": "failed", "exception_type": type(error).__name__},
        )
        raise
    return inputs.output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run_round2_sensitivity(json.loads(args.config.read_text()))


if __name__ == "__main__":
    main()
