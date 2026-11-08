"""Prospective development cut-grid analysis for accepted corrected runs.

Private participant/event/alarm ledgers stay beside the approved remote inputs.
Only ``exportable/`` aggregate JSON/CSV files are candidates for a curated pull.
The module never trains, recalibrates temperatures, selects a winning model,
opens external cohorts, computes hashes, or changes the prepared population.

Run ``python -m glycoshift.revision_analysis --config analysis.json``. The JSON
requires ``revision_contract``, ``prepared_dir``, ``output_dir`` and ``arms``.
Each arm entry has ``arm``, ``model_seed`` and ``run_dir``. Exactly five ``lstm``
entries use seeds 17/23/42/71/101; one ``persistence`` entry has a null seed. Other
arms are descriptive. ``threshold_candidate_count`` defaults to, and must be,
2001. Every run must retain its original fixed-frame event/participant ledgers.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .fixed_frame_evaluation import (
    FixedFrameEvaluation,
    FixedFrameKey,
    FixedFrameSpec,
    evaluate_fixed_frame,
    fit_fixed_frame_alarm_policy,
    joint_fixed_frame_bootstrap,
)
from .revision_alarm_summary import summarize_alarm_glucose
from .sequences import REVISION_CONTRACT

CUTS = (54, 60, 70, 80, 90, 100)
HORIZONS = (30, 60)
MODEL_SEEDS = (17, 23, 42, 71, 101)
_PARTITIONS = ("calibration", "selection")
_KEYS = ["source_id", "participant_id", "context_end"]
_BINDING = [
    "patient_day",
    "cgm_last",
    "anchor_cgm_observed",
    "anchor_cgm_interpolated",
    "anchor_cgm_censored",
    "base_evaluable_30",
    "base_evaluable_60",
    "severe_hypo_30_label",
    "severe_hypo_60_label",
]


@dataclass(frozen=True)
class AnalysisKey:
    arm: str
    model_seed: int | None
    glucose_cut: int
    horizon_minutes: int


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_jsonable(value), indent=2, allow_nan=False) + "\n")


def _role(frame: pd.DataFrame, partition: str) -> None:
    if (
        partition not in _PARTITIONS
        or frame.empty
        or "partition" not in frame
        or not frame["partition"].eq(partition).all()
    ):
        raise ValueError(f"analysis requires only the {partition} partition")


def _participants(frame: pd.DataFrame) -> set[tuple[str, str]]:
    return set(
        frame[["source_id", "participant_id"]]
        .drop_duplicates()
        .astype(str)
        .itertuples(index=False, name=None)
    )


def _timeline(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    timeline = frame.copy(deep=False)
    timeline["base_evaluable"] = frame[f"base_evaluable_{horizon}"]
    return timeline


def _base_policy_parity(
    recorded: Mapping[str, Any], recomputed: Mapping[str, Any]
) -> None:
    """Verify, rather than silently replace, the previously fitted base policy."""

    exact = (
        "partition",
        "endpoint",
        "glucose_cut",
        "nominal_fapd",
        "threshold",
        "threshold_count",
        "candidate_rule",
        "tie_rule",
        "participant_count",
        "base_event_count",
        "patient_days",
        "observation_minutes",
    )
    for field in exact:
        if field not in recorded or _jsonable(recorded[field]) != _jsonable(
            recomputed[field]
        ):
            raise ValueError(f"frozen base54 policy parity failed: {field}")
    for field in (
        "achieved_equal_source_participant_macro_fapd",
        "achieved_equal_source_participant_macro_fixed_sensitivity",
    ):
        if field not in recorded or not np.isclose(
            recorded[field], recomputed[field], rtol=0, atol=1e-12
        ):
            raise ValueError(f"frozen base54 policy parity failed: {field}")


def analyze_arm(
    calibration: pd.DataFrame,
    selection: pd.DataFrame,
    endpoints: Mapping[tuple[str, int], tuple[pd.DataFrame, pd.DataFrame]],
    *,
    tasks: Mapping[str, Any],
    source_order: tuple[str, ...],
    candidate_count: int = 2001,
) -> Iterator[tuple[int, int, str, dict[str, Any], FixedFrameEvaluation]]:
    """Yield refitted and frozen-base gate-only evaluations without dropping rows."""

    _role(calibration, "calibration")
    _role(selection, "selection")
    if _participants(calibration).intersection(_participants(selection)):
        raise ValueError("calibration and selection participants overlap")
    for partition in _PARTITIONS:
        for horizon in HORIZONS:
            for table in endpoints[partition, horizon]:
                if "partition" not in table or not table.partition.eq(partition).all():
                    raise ValueError("endpoint partition differs from its bound role")
    for horizon in HORIZONS:
        task = tasks[f"hypo_{horizon}"]
        if (
            task.get("threshold_partition") != "calibration"
            or task.get("temperature_fit_partition") != "calibration"
            or task.get("target_column") != f"severe_hypo_{horizon}_label"
        ):
            raise ValueError(
                "run target or temperature/threshold fitting role mismatch"
            )
        recorded = task["fixed_frame_policies"]["0.5"]
        spec = FixedFrameSpec(
            f"severe_hypo_{horizon}_probability", horizon, source_order
        )
        cal, selected = _timeline(calibration, horizon), _timeline(selection, horizon)
        cal_events, cal_exposure = endpoints["calibration", horizon]
        events, exposure = endpoints["selection", horizon]
        base = fit_fixed_frame_alarm_policy(
            cal,
            cal_events,
            cal_exposure,
            spec=spec,
            partition="calibration",
            glucose_cut=54,
            nominal_fapd=0.5,
            candidate_count=candidate_count,
        )
        _base_policy_parity(recorded, base.as_dict())
        for cut in CUTS:
            policy = (
                base
                if cut == 54
                else fit_fixed_frame_alarm_policy(
                    cal,
                    cal_events,
                    cal_exposure,
                    spec=spec,
                    partition="calibration",
                    glucose_cut=cut,
                    nominal_fapd=0.5,
                    candidate_count=candidate_count,
                )
            )
            evaluated = evaluate_fixed_frame(
                selected,
                events,
                exposure,
                spec=spec,
                threshold=float(policy.threshold),
                glucose_cut=cut,
            )
            yield (
                horizon,
                cut,
                "refitted",
                {
                    "policy": recorded if cut == 54 else policy.as_dict(),
                    "policy_origin": "verified_frozen_base54"
                    if cut == 54
                    else "calibration_refit_at_cut",
                    "applied_glucose_cut": cut,
                    "base54_policy_parity_checked": True,
                },
                evaluated,
            )
            gate = (
                evaluated
                if cut == 54
                else evaluate_fixed_frame(
                    selected,
                    events,
                    exposure,
                    spec=spec,
                    threshold=float(recorded["threshold"]),
                    glucose_cut=cut,
                )
            )
            yield (
                horizon,
                cut,
                "gate_only",
                {
                    "policy": dict(recorded),
                    "policy_origin": "frozen_base54_gate_only",
                    "applied_glucose_cut": cut,
                    "base54_policy_parity_checked": True,
                },
                gate,
            )


def joint_primary_inference(
    evaluations: Mapping[AnalysisKey, FixedFrameEvaluation],
    *,
    source_order: tuple[str, ...],
) -> dict[str, Any]:
    """Fixed LSTM/control contrasts; a single control fit is shared over seed keys."""

    primary = {
        key: value
        for key, value in evaluations.items()
        if key.arm in {"lstm", "persistence"} and key.glucose_cut in (54, 70)
    }
    expected = {
        AnalysisKey(arm, seed, cut, horizon)
        for arm, seeds in (("lstm", MODEL_SEEDS), ("persistence", (None,)))
        for seed in seeds
        for cut in (54, 70)
        for horizon in HORIZONS
    }
    if set(primary) != expected:
        raise ValueError(
            "primary requires exactly five LSTM seeds and one shared null-seed control"
        )
    expanded = {
        FixedFrameKey(arm, cut, horizon, seed): primary[
            AnalysisKey(arm, seed if arm == "lstm" else None, cut, horizon)
        ]
        for arm in ("lstm", "persistence")
        for cut in (54, 70)
        for horizon in HORIZONS
        for seed in MODEL_SEEDS
    }
    contrasts = {
        "lstm_minus_persistence_at_54": {("lstm", 54): 1, ("persistence", 54): -1},
        "lstm_minus_persistence_at_70": {("lstm", 70): 1, ("persistence", 70): -1},
        "interaction_70_minus_54": {
            ("lstm", 70): 1,
            ("persistence", 70): -1,
            ("lstm", 54): -1,
            ("persistence", 54): 1,
        },
    }
    return {
        "primary_pair": ["lstm", "persistence"],
        "primary_contrast": "interaction_70_minus_54",
        "shared_control": {
            "unique_fit_count": 1,
            "model_seed": None,
            "bootstrap_alias_seeds": list(MODEL_SEEDS),
            "aliases_are_independent_fits": False,
        },
        "lstm_unique_fit_count": 5,
        "sensitivity": joint_fixed_frame_bootstrap(
            expanded,
            contrasts=contrasts,
            source_order=source_order,
            replicates=1000,
            bootstrap_seed=20260810,
        ),
        "unmatched_alarm_burden": joint_fixed_frame_bootstrap(
            expanded,
            contrasts=contrasts,
            source_order=source_order,
            replicates=1000,
            bootstrap_seed=20260810,
            metric="unmatched_alarms_per_patient_day",
        ),
    }


def _accepted_path(raw: str, root: Path) -> Path:
    path = Path(raw)
    if path.is_symlink() or not path.is_file():
        raise ValueError("prepared input must be a regular non-symlink file")
    path = path.resolve()
    if not path.is_relative_to(root):
        raise ValueError("prepared input escapes the accepted bundle")
    return path


def _load_prepared(prepared: Path):
    audit = json.loads((prepared / "input-audit.json").read_text())
    if (
        audit.get("status") != "passed"
        or audit.get("revision_contract") != REVISION_CONTRACT
    ):
        raise ValueError("analysis requires accepted revision preparation")
    sources = tuple(audit["source_order"])
    if not sources or len(set(sources)) != len(sources):
        raise ValueError("prepared source order must contain unique sources")
    manifest = json.loads((prepared / "participant-manifest.private.json").read_text())[
        "participants"
    ]
    entries = [entry for entry in manifest if entry["partition"] in _PARTITIONS]
    keys = [(str(x["source_id"]), str(x["participant_id"])) for x in entries]
    if not entries or len(keys) != len(set(keys)):
        raise ValueError("prepared evaluation participants are empty or overlap roles")
    endpoints = {}
    for partition in _PARTITIONS:
        expected = {
            key
            for key, entry in zip(keys, entries, strict=True)
            if entry["partition"] == partition
        }
        if {key[0] for key in expected} != set(sources):
            raise ValueError("prepared evaluation partition source set differs")
        for horizon in HORIZONS:
            tables = []
            for category in ("events", "exposure"):
                path = (
                    prepared / "endpoints" / partition / f"{category}-{horizon}.parquet"
                )
                bound = audit["endpoint_paths"][f"{partition}/{horizon}"][category]
                if _accepted_path(bound, prepared) != path.resolve():
                    raise ValueError("endpoint path differs from preparation audit")
                table = pq.read_table(path, use_threads=False).to_pandas()
                if "partition" in table and not table.partition.eq(partition).all():
                    raise ValueError("prepared endpoint contains another partition")
                table["partition"] = partition
                actual = _participants(table)
                if (
                    actual != expected
                    if category == "exposure"
                    else not actual.issubset(expected)
                ):
                    raise ValueError(
                        "prepared endpoint participants differ from manifest"
                    )
                tables.append(table)
            endpoints[partition, horizon] = tuple(tables)
    return sources, entries, endpoints


def _validate_bindings(
    frame: pd.DataFrame, entries: list[dict], prepared: Path
) -> None:
    if set(frame["partition"]) != set(_PARTITIONS):
        raise ValueError("predictions must contain calibration and selection only")
    groups = frame.groupby(["source_id", "participant_id"], observed=True).indices
    expected_keys = {(str(x["source_id"]), str(x["participant_id"])) for x in entries}
    if set(groups) != expected_keys:
        raise ValueError(
            "prediction participants differ from prepared evaluation roles"
        )
    for entry in entries:
        key = (str(entry["source_id"]), str(entry["participant_id"]))
        part = frame.iloc[groups[key]]
        _role(part, entry["partition"])
        path = _accepted_path(entry["timeline_path"], prepared)
        expected = (
            pq.ParquetFile(path)
            .read(columns=[*_KEYS, "partition", *_BINDING], use_threads=False)
            .to_pandas()
        )
        if len(expected) != entry["stored_rows"] or len(part) != len(expected):
            raise ValueError("prediction timeline is incomplete or duplicated")
        if not (
            expected.source_id.astype(str).eq(key[0])
            & expected.participant_id.astype(str).eq(key[1])
            & expected.partition.eq(entry["partition"])
        ).all():
            raise ValueError("prepared shard identity differs from manifest")
        actual = part.copy()
        for table in (actual, expected):
            table["context_end"] = pd.to_datetime(table.context_end, utc=True)
            if table.context_end.isna().any() or table.context_end.duplicated().any():
                raise ValueError("prediction or prepared decision keys are invalid")
            table.sort_values("context_end", inplace=True)
            table.reset_index(drop=True, inplace=True)
        try:
            pd.testing.assert_frame_equal(
                actual[["context_end", *_BINDING]],
                expected[["context_end", *_BINDING]],
                check_dtype=False,
                check_exact=True,
                check_categorical=False,
            )
        except AssertionError:
            raise ValueError(
                "prediction metadata/truth differs from prepared timeline"
            ) from None


def _validate_run_truth(directory: Path, endpoints: Mapping) -> None:
    """Bind the model's original reported event IDs and exposure, not just totals."""

    ids = ["source_id", "participant_id"]
    event_columns = [*ids, "event_id", "event_onset"]
    exposure_columns = [*ids, "patient_days", "observation_minutes", "base_event_count"]
    for (partition, horizon), (events, exposure) in endpoints.items():
        root = directory / "fixed-frame" / partition / str(horizon) / "0.5"
        old_events = (
            pq.ParquetFile(root / "events.parquet")
            .read(columns=event_columns, use_threads=False)
            .to_pandas()
        )
        old_exposure = (
            pq.ParquetFile(root / "participants.parquet")
            .read(columns=exposure_columns, use_threads=False)
            .to_pandas()
        )
        current_events = events[event_columns].copy()
        current_exposure = exposure.copy()
        for table in (old_events, current_events, old_exposure, current_exposure):
            for column in ids:
                table[column] = table[column].astype(str)
        for table in (old_events, current_events):
            table["event_id"] = table["event_id"].astype(str)
            table["event_onset"] = pd.to_datetime(table.event_onset, utc=True)
            table.sort_values([*ids, "event_id"], inplace=True)
            table.reset_index(drop=True, inplace=True)
        counts = (
            current_events.groupby(ids, observed=True).size().rename("base_event_count")
        )
        current_exposure = current_exposure.merge(counts, on=ids, how="left")
        current_exposure["base_event_count"] = current_exposure.base_event_count.fillna(
            0
        )
        for table in (old_exposure, current_exposure):
            table.sort_values(ids, inplace=True)
            table.reset_index(drop=True, inplace=True)
        for label, old, current, columns in (
            ("event", old_events, current_events, event_columns),
            ("exposure", old_exposure, current_exposure, exposure_columns),
        ):
            try:
                pd.testing.assert_frame_equal(
                    old[columns],
                    current[columns],
                    check_dtype=False,
                    check_exact=True,
                    check_categorical=False,
                )
            except AssertionError:
                raise ValueError(
                    f"frozen run {label} truth differs from prepared endpoints"
                ) from None


def _entries(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    entries = [dict(entry) for entry in config["arms"]]
    keys, directories = [], []
    for entry in entries:
        arm, seed = entry["arm"], entry["model_seed"]
        if (
            not isinstance(arm, str)
            or not arm
            or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in arm)
        ):
            raise ValueError("arm must be a simple lowercase name")
        if seed is not None and (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed not in MODEL_SEEDS
        ):
            raise ValueError("model_seed must be null or a prespecified seed")
        directory = Path(entry["run_dir"]).resolve()
        summary = json.loads((directory / "model_summary.json").read_text())
        if summary.get("revision_contract") != REVISION_CONTRACT:
            raise ValueError("analysis cannot accept a historical or uncorrected run")
        if seed is not None and summary.get("training_seed") != seed:
            raise ValueError("run training seed differs from explicit arm entry")
        if arm == "lstm" and summary.get("architecture") != "lstm":
            raise ValueError("primary LSTM entry has a different architecture")
        entry.update(run_dir=directory, summary=summary)
        keys.append((arm, seed))
        directories.append(directory)
    if len(keys) != len(set(keys)) or len(directories) != len(set(directories)):
        raise ValueError(
            "duplicate fit/alias entries must not be counted independently"
        )
    if {seed for arm, seed in keys if arm == "lstm"} != set(MODEL_SEEDS):
        raise ValueError("primary requires the complete five-seed LSTM set")
    if [seed for arm, seed in keys if arm == "persistence"] != [None]:
        raise ValueError("persistence requires exactly one shared null-seed fit")
    return entries


def run_revision_analysis(config: Mapping[str, Any]) -> Path:
    if config.get("revision_contract") != REVISION_CONTRACT:
        raise ValueError("explicit revision_contract is required")
    candidate_count = config.get("threshold_candidate_count", 2001)
    if candidate_count != 2001:
        raise ValueError("analysis requires the original 2001 threshold candidates")
    entries = _entries(config)
    prepared = Path(config["prepared_dir"]).resolve()
    source_order, manifest, endpoints = _load_prepared(prepared)
    output = Path(config["output_dir"]).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("refusing to overwrite non-empty analysis output")
    output.mkdir(parents=True, exist_ok=True)
    private, exportable = output / "private", output / "exportable"
    private.mkdir(exist_ok=True)
    exportable.mkdir(exist_ok=True)
    primary, rows = {}, []
    alarm_glucose_rows = []
    fit_ledger = []
    try:
        for entry in entries:
            arm, seed, directory = entry["arm"], entry["model_seed"], entry["run_dir"]
            path = directory / "predictions.parquet"
            file = pq.ParquetFile(path)
            needed = [
                *_KEYS,
                "partition",
                *_BINDING,
                *(f"severe_hypo_{h}_probability" for h in HORIZONS),
            ]
            if not set(needed).issubset(file.schema_arrow.names):
                raise ValueError("corrected predictions lack complete replay columns")
            # Read only identity/role columns first; never load training shards.
            identities = file.read(
                columns=["source_id", "participant_id", "partition"], use_threads=False
            ).to_pandas()
            if set(identities.partition) != set(_PARTITIONS):
                raise ValueError("predictions have non-development or missing roles")
            expected_roles = {
                (
                    str(entry["source_id"]),
                    str(entry["participant_id"]),
                    entry["partition"],
                )
                for entry in manifest
            }
            actual_roles = set(
                identities.drop_duplicates()
                .astype(str)
                .itertuples(index=False, name=None)
            )
            if actual_roles != expected_roles:
                raise ValueError(
                    "prediction identities or roles differ from prepared manifest"
                )
            del identities
            _validate_run_truth(directory, endpoints)
            frame = file.read(columns=needed, use_threads=False).to_pandas()
            file.close()
            _validate_bindings(frame, manifest, prepared)
            fit_ledger.append(
                {
                    "arm": arm,
                    "model_seed": seed,
                    "run_dir": str(directory),
                    "prediction_rows": len(frame),
                    "shared_over_bootstrap_seeds": list(MODEL_SEEDS)
                    if arm == "persistence"
                    else [],
                }
            )
            calibration = frame.loc[frame.partition.eq("calibration")]
            selection = frame.loc[frame.partition.eq("selection")]
            for horizon, cut, mode, policy, evaluation in analyze_arm(
                calibration,
                selection,
                endpoints,
                tasks=entry["summary"]["tasks"],
                source_order=source_order,
                candidate_count=candidate_count,
            ):
                target = (
                    private
                    / arm
                    / ("shared" if seed is None else str(seed))
                    / mode
                    / str(cut)
                    / str(horizon)
                )
                target.mkdir(parents=True, exist_ok=True)
                _write_json(target / "policy.json", policy)
                for name in ("participants", "sources", "events", "alarms"):
                    getattr(evaluation, name).to_parquet(
                        target / f"{name}.parquet", index=False
                    )
                _write_json(target / "summary.json", evaluation.summary)
                alarm_glucose_rows.extend(
                    {
                        "arm": arm,
                        "model_seed": seed,
                        "horizon_minutes": horizon,
                        "glucose_cut": cut,
                        "mode": mode,
                        **row,
                    }
                    for row in summarize_alarm_glucose(
                        evaluation.alarms, source_order=source_order
                    )
                )
                rows.append(
                    {
                        "arm": arm,
                        "model_seed": seed,
                        "horizon_minutes": horizon,
                        "glucose_cut": cut,
                        "mode": mode,
                        "threshold": evaluation.threshold,
                        **evaluation.summary,
                    }
                )
                if (
                    mode == "refitted"
                    and arm in {"lstm", "persistence"}
                    and cut in (54, 70)
                ):
                    primary[AnalysisKey(arm, seed, cut, horizon)] = replace(
                        evaluation, alarms=evaluation.alarms.iloc[:0].copy()
                    )
            del frame, calibration, selection, evaluation
        inference = joint_primary_inference(primary, source_order=source_order)
        _write_json(private / "unique-fit-ledger.json", {"fits": fit_ledger})
        _write_json(exportable / "primary-inference.json", inference)
        _write_json(exportable / "alarm-glucose-summary.json", alarm_glucose_rows)
        pd.DataFrame(alarm_glucose_rows).to_csv(
            exportable / "alarm-glucose-summary.csv", index=False
        )
        summary = {
            "revision_contract": REVISION_CONTRACT,
            "status": "complete",
            "source_order": source_order,
            "cuts": CUTS,
            "horizons": HORIZONS,
            "model_seeds": MODEL_SEEDS,
            "nominal_fapd": 0.5,
            "evaluation_partition": "selection",
            "threshold_partition": "calibration",
            "unique_fit_count": len(entries),
            "primary_control_unique_fit_count": 1,
            "primary_lstm_unique_fit_count": 5,
            "analyses": rows,
        }
        _write_json(exportable / "cut-grid-summary.json", summary)
        pd.DataFrame(
            [
                {
                    key: json.dumps(_jsonable(value))
                    if isinstance(value, (list, tuple, dict))
                    else value
                    for key, value in row.items()
                }
                for row in rows
            ]
        ).to_csv(exportable / "cut-grid-summary.csv", index=False)
        _write_json(
            output / "analysis-status.json",
            {"status": "complete", "exportable": "aggregate files only"},
        )
    except Exception as error:
        _write_json(
            output / "analysis-status.json",
            {"status": "failed", "exception_type": type(error).__name__},
        )
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run_revision_analysis(json.loads(args.config.read_text()))


if __name__ == "__main__":
    main()
