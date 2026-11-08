"""Frozen, participant-streamed replay of already exposed Loop/HUPA sources.

This module consumes an explicit development freeze; it never creates one.
It does not fit models, scalers, temperatures, mappings or alarm thresholds.
The original normalized source role and global-cohort audits must have been
accepted before the freeze. Participant checks here do not redo global dedup.

CLI: ``python -m glycoshift.revision_external --freeze freeze.json --run-dir RUN
--partition external_loop --output NEW_OUTPUT``. Private predictions/endpoints
and ledgers stay remote. Only ``exportable/`` aggregates may be curated for pull.

Freeze schema: revision_contract, status="development_recipe_frozen", freeze_id,
accepted_development_evidence=[file bindings], runs=[run entries], partitions.
A file binding is {path,size_bytes,mtime_ns}. Each run entry contains run_dir,
config={binding,snapshot}, model_summary={binding,snapshot}, model_artifacts,
and cut_policies[cut][horizon][budget]={policy,evidence?}. Base54 policies must
match the saved summary at budgets .5/1.0. Optional cut70 is budget .5 only and
requires its exact development-analysis policy file binding as evidence.
Each partition contains source_id, input_path, source_manifest binding,
required_origin, normalized_inputs bindings, accepted_normalization_evidence,
accepted_global_cohort_audit, and source_normalization_contract with explicit
lower_censoring_absent_means_false and nonempty source_preprocessing_limits.
No digest is calculated. File bindings are checked before and after execution.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from sklearn.pipeline import Pipeline

from .calibration import apply_temperature
from .config import load_config, resolve_path
from .deep_training import (
    _device,
    _load_frozen_model,
    _predict_dataset,
    _restore_regression_scale,
)
from .fast_sequences import FastWindowSequenceDataset
from .fixed_frame_evaluation import FixedFrameSpec, _aggregate, evaluate_fixed_frame
from .io import _validate_manifest
from .pipeline import (
    _continuous_baseline_forecast,
    _evaluation_source,
    _predict_probability,
)
from .revision_alarm_summary import summarize_alarm_glucose
from .revision_baselines import BASELINES, _task_features, build_revision_features
from .revision_inputs import ReplayInputs, build_participant_replay
from .schema import validate_frame
from .sequences import REVISION_CONTRACT, TASKS, SequenceScaler

_ALLOWED = {
    "external_loop": "Loop Observational Study",
    "stress_hupa_ucm": "HUPA-UCM",
}
_HORIZONS = (30, 60)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_plain(payload), indent=2, allow_nan=False) + "\n")


def _parquet(path: Path, frame: pd.DataFrame) -> None:
    """Keep empty event/alarm shards readable as one typed Parquet dataset."""
    frame = frame.copy(deep=False)
    identifiers = {"source_id", "participant_id", "event_id", "matched_event_id"}
    for column in frame:
        if column in identifiers or column.endswith("_event_id"):
            frame[column] = frame[column].astype("string")
    if "patient_day" in frame:
        dates = {
            value: pd.Timestamp(pd.Timestamp(value).date(), tz="UTC")
            for value in frame.patient_day.unique()
        }
        frame["patient_day"] = pd.to_datetime(frame.patient_day.map(dates), utc=True)
    table = pa.Table.from_pandas(frame, preserve_index=False)
    times = {
        "context_start",
        "context_end",
        "patient_day",
        "event_onset",
        "alarm_context_end",
        "matched_event_onset",
    }
    booleans = {
        "base_evaluable",
        "matched",
        "unmatched",
        "reachable",
        "detected",
        "late_detection",
        "missed",
    }
    fields = []
    for field in table.schema:
        name, kind = field.name, field.type
        if name in identifiers or name.endswith("_event_id"):
            kind = pa.string()
        elif name in times:
            kind = pa.timestamp("ns", tz="UTC")
        elif name == "matched_lead_minutes":
            kind = pa.list_(pa.float64())
        elif name == "base_patient_days":
            kind = pa.list_(pa.timestamp("ns", tz="UTC"))
        elif pa.types.is_null(kind):
            if name in booleans or (
                name.startswith("coverage_") and not name.endswith("_count")
            ):
                kind = pa.bool_()
            elif name.endswith("_count") or name == "patient_days":
                kind = pa.int64()
            else:
                kind = pa.float64()
        fields.append(pa.field(name, kind))
    pq.write_table(table.cast(pa.schema(fields)), path)


def _write_run_receipt(
    output: Path,
    *,
    run_dir: Path,
    freeze_id: str,
    partition: str,
    source: dict,
    summary: dict,
    counts: dict,
) -> None:
    """Bind the selected producer only after all scored output writers close."""
    paths = [
        *sorted((output / "private/predictions").glob("*.parquet")),
        *sorted((output / "private/endpoints").rglob("*.parquet")),
        *sorted((output / "private/evaluations").rglob("*.parquet")),
        output / "private/input-audit.json",
        output / "private/recipe-freeze-used.json",
        output / "exportable/fixed-frame-summary.json",
        output / "exportable/fixed-frame-summary.csv",
        output / "exportable/alarm-glucose-summary.json",
        output / "exportable/alarm-glucose-summary.csv",
        output / "status.json",
    ]
    bindings = []
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError("completed external output is not a regular file")
        stat = path.stat()
        bindings.append(
            {
                "relative_path": str(path.relative_to(output)),
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    receipt = {
        "schema_version": "1.0",
        "producer_kind": "frozen_model_external",
        "status": "complete",
        "revision_contract": REVISION_CONTRACT,
        "freeze_id": freeze_id,
        "partition": partition,
        "source_id": source["source_id"],
        "run_dir": str(run_dir.resolve()),
        "model_summary_snapshot": summary,
        "source_normalization_contract": source["source_normalization_contract"],
        "global_cohort_audit": (
            "reused frozen accepted evidence; not recomputed per participant"
        ),
        "counts": counts,
        "binding_scope": (
            "complete prediction/independent-endpoint/evaluation-ledger datasets "
            "and terminal summaries"
        ),
        "files": bindings,
    }
    _publish_run_receipt(output, receipt)


def _publish_run_receipt(output: Path, receipt: dict) -> None:
    """Publish a native receipt only after every output file has closed."""
    destination = output / "run-receipt.json"
    if destination.exists():
        raise FileExistsError("refusing to overwrite a producer receipt")
    temporary = output / "run-receipt.pending.json"
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(_plain(receipt), handle, indent=2, allow_nan=False)
        handle.write("\n")
    temporary.replace(destination)


def _export_alarm_glucose(
    output: Path,
    *,
    source_order: tuple[str, ...],
    conditions: list[tuple],
) -> None:
    """Compute exact alarm-weighted quantiles from complete private ledgers."""
    columns = [
        "source_id",
        "participant_id",
        "context_end",
        "cgm_last",
        "base_evaluable",
        "matched",
        "unmatched",
    ]
    rows = []
    for cut, horizon, budget in conditions:
        root = (
            output
            / "private/evaluations"
            / str(cut)
            / str(horizon)
            / str(budget)
            / "alarms"
        )
        paths = sorted(root.glob("*.parquet"))
        if not paths:
            raise ValueError(
                "complete private alarm ledgers are required for CGM summary"
            )
        parts = [
            pq.ParquetFile(path).read(columns=columns, use_threads=False).to_pandas()
            for path in paths
        ]
        alarms = pd.concat(parts, ignore_index=True)
        for summary in summarize_alarm_glucose(alarms, source_order=source_order):
            rows.append(
                {
                    "glucose_cut": cut,
                    "horizon_minutes": horizon,
                    "nominal_fapd": float(budget),
                    **summary,
                }
            )
        del parts, alarms
    _json(
        output / "exportable/alarm-glucose-summary.json",
        {
            "aggregation": "exact_alarm_weighted_values_from_complete_ledgers",
            "participant_quantiles_averaged": False,
            "rows": rows,
        },
    )
    pd.DataFrame(rows).to_csv(
        output / "exportable/alarm-glucose-summary.csv", index=False
    )


def _bound_file(binding: Mapping[str, Any], expected: Path | None = None) -> Path:
    path = Path(binding["path"])
    if path.is_symlink() or not path.is_file():
        raise ValueError("freeze binding must be a regular non-symlink file")
    path = path.resolve()
    if expected is not None and path != expected.resolve():
        raise ValueError("freeze binding points at a different artifact")
    stat = path.stat()
    if stat.st_size != binding["size_bytes"] or stat.st_mtime_ns != binding["mtime_ns"]:
        raise ValueError("frozen artifact size/mtime binding changed")
    return path


def _evidence(bindings: list[dict]) -> None:
    if not isinstance(bindings, list) or not bindings:
        raise ValueError("freeze requires actual accepted evidence files")
    for binding in bindings:
        _bound_file(binding)


def _recheck_bindings(value: Any) -> None:
    if isinstance(value, Mapping):
        if {"path", "size_bytes", "mtime_ns"}.issubset(value):
            _bound_file(value)
        else:
            for item in value.values():
                _recheck_bindings(item)
    elif isinstance(value, list):
        for item in value:
            _recheck_bindings(item)


def load_development_freeze(
    freeze_path: Path,
    run_dir: Path,
    partition: str,
) -> tuple[dict, dict, dict, dict, dict]:
    """Validate snapshots/evidence without inventing a development acceptance."""
    if partition not in _ALLOWED:
        raise ValueError("only already exposed Loop/HUPA partitions are allowed")
    freeze = json.loads(freeze_path.read_text())
    if (
        freeze.get("revision_contract") != REVISION_CONTRACT
        or freeze.get("status") != "development_recipe_frozen"
        or not str(freeze.get("freeze_id", "")).strip()
    ):
        raise ValueError("explicit accepted development recipe freeze is required")
    _evidence(freeze.get("accepted_development_evidence"))
    replay_contracts = []
    for run in freeze["runs"]:
        frozen_config = run["config"]["snapshot"]
        data = frozen_config["data"]
        replay_contracts.append(
            {
                "protocol": frozen_config["protocol"],
                "minimum_days": data["minimum_days"],
                "minimum_day_coverage": data["minimum_day_coverage"],
                "development_sources": data["development_sources"],
                "external_source": data["external_source"],
                "stress_sources": data["stress_sources"],
                "cuts": sorted(run["cut_policies"]),
            }
        )
    if not replay_contracts or any(
        value != replay_contracts[0] for value in replay_contracts[1:]
    ):
        raise ValueError(
            "frozen runs differ in source/protocol/QC/cut timeline contract"
        )
    matches = [
        entry
        for entry in freeze["runs"]
        if Path(entry["run_dir"]).resolve() == run_dir.resolve()
    ]
    if len(matches) != 1:
        raise ValueError("run is not uniquely listed in the accepted freeze")
    entry = matches[0]
    config_file = _bound_file(
        entry["config"]["binding"], run_dir / "config_resolved.yaml"
    )
    config = load_config(config_file)
    snapshot = {key: value for key, value in config.items() if key != "_config_path"}
    if _plain(snapshot) != entry["config"]["snapshot"]:
        raise ValueError("frozen effective configuration snapshot changed")
    summary_file = _bound_file(
        entry["model_summary"]["binding"], run_dir / "model_summary.json"
    )
    summary = json.loads(summary_file.read_text())
    if summary != entry["model_summary"]["snapshot"]:
        raise ValueError("frozen model summary/policy/temperature snapshot changed")
    if (
        summary.get("revision_contract") != REVISION_CONTRACT
        or summary.get("input_feature_mode") != "cgm_mask_clock"
        or config["model"].get("input_feature_mode") != "cgm_mask_clock"
        or tuple(config["protocol"]["horizons_minutes"]) != _HORIZONS
    ):
        raise ValueError(
            "external replay requires the corrected CGM/mask/clock contract"
        )
    if _evaluation_source(config, partition) != _ALLOWED[partition]:
        raise ValueError("configured external source is unknown or confirmatory")
    if partition not in freeze["partitions"]:
        raise ValueError("external partition is not listed in the accepted freeze")
    source = freeze["partitions"][partition]
    if source["source_id"] != _ALLOWED[partition]:
        raise ValueError("freeze contains an unknown or confirmatory external source")
    _evidence(source.get("accepted_normalization_evidence"))
    _evidence(source.get("accepted_global_cohort_audit"))
    _evidence(source.get("normalized_inputs"))
    contract = source["source_normalization_contract"]
    if (
        not isinstance(contract.get("lower_censoring_absent_means_false"), bool)
        or not str(contract.get("source_preprocessing_limits", "")).strip()
    ):
        raise ValueError(
            "explicit verified normalization and preprocessing limits required"
        )
    configured = config["data"]["evaluation_inputs"][partition]
    if (
        configured["expected_source"] != source["source_id"]
        or configured["required_origin"] != source["required_origin"]
    ):
        raise ValueError(
            "frozen source role/origin differs from original configuration"
        )
    manifest = _bound_file(source["source_manifest"])
    _validate_manifest(
        manifest,
        expected_sources={source["source_id"]},
        role=partition,
        required_origin=source["required_origin"],
    )
    development = resolve_path(config, config["data"].get("input_path"))
    development = None if development is None else development.resolve()
    input_path = Path(source["input_path"]).resolve()
    if development is not None and (
        input_path == development
        or input_path.is_relative_to(development)
        or development.is_relative_to(input_path)
    ):
        raise ValueError("external normalized inputs must be isolated from development")
    architecture, baseline = summary.get("architecture"), summary.get("baseline")
    required = (
        {run_dir / "ckpts/best.pt", run_dir / "scaler.json"}
        if architecture in {"lstm", "tcn"}
        else {run_dir / "baseline_models.joblib"}
        if baseline in BASELINES
        else set()
    )
    if not required:
        raise ValueError("unsupported corrected frozen model family")
    artifacts = {_bound_file(binding) for binding in entry["model_artifacts"]}
    if artifacts != {path.resolve() for path in required}:
        raise ValueError("frozen model artifact set differs from the required family")
    _policies(entry, summary, config)
    return freeze, entry, source, config, summary


def _policies(entry: dict, summary: dict, config: dict) -> dict:
    cuts = entry["cut_policies"]
    if set(cuts).difference({"54", "70"}) or "54" not in cuts:
        raise ValueError("only frozen development cuts 54/70 may be evaluated")
    selected = {}
    for cut, horizons in cuts.items():
        if set(horizons) != {"30", "60"}:
            raise ValueError("frozen cut policies require both horizons")
        for horizon_text, budgets in horizons.items():
            expected = {"0.5", "1.0"} if cut == "54" else {"0.5"}
            if set(budgets) != expected:
                raise ValueError("base54 requires .5/1.0; cut70 requires .5 only")
            horizon = int(horizon_text)
            task = summary["tasks"][f"hypo_{horizon}"]
            if (
                task["target_column"] != f"severe_hypo_{horizon}_label"
                or task["temperature_fit_partition"] != "calibration"
                or task["threshold_partition"] != "calibration"
            ):
                raise ValueError("frozen task target or calibration role differs")
            for budget, record in budgets.items():
                policy = record["policy"]
                if cut == "54":
                    if _plain(policy) != _plain(task["fixed_frame_policies"][budget]):
                        raise ValueError(
                            "base54 policy differs from frozen training policy"
                        )
                else:
                    if not isinstance(record.get("evidence"), Mapping):
                        raise ValueError(
                            "cut70 lacks frozen development policy evidence"
                        )
                    evidence = json.loads(_bound_file(record["evidence"]).read_text())
                    if evidence.get(
                        "policy_origin"
                    ) != "calibration_refit_at_cut" or not evidence.get(
                        "base54_policy_parity_checked"
                    ):
                        raise ValueError(
                            "cut70 lacks accepted development policy evidence"
                        )
                    if _plain(policy) != evidence["policy"]:
                        raise ValueError(
                            "cut70 policy differs from its frozen evidence"
                        )
                endpoint = policy["endpoint"]
                if (
                    policy["partition"] != "calibration"
                    or policy["glucose_cut"] != int(cut)
                    or policy["nominal_fapd"] != float(budget)
                    or endpoint["horizon_minutes"] != horizon
                    or endpoint["score_column"] != f"severe_hypo_{horizon}_probability"
                    or tuple(endpoint["source_order"])
                    != tuple(sorted(config["data"]["development_sources"]))
                    or not np.isfinite(policy["threshold"])
                ):
                    raise ValueError("frozen policy endpoint/role/cut/budget mismatch")
                selected[int(cut), horizon, budget] = policy
    for task in TASKS:
        specification = summary["tasks"][task]
        temperature = specification["temperature"]
        kind, horizon = task.split("_")
        expected_target = (
            f"severe_hypo_{horizon}_label" if kind == "hypo" else f"{task}_label"
        )
        if (
            specification["temperature_fit_partition"] != "calibration"
            or specification["target_column"] != expected_target
            or not np.isfinite(temperature)
            or temperature <= 0
        ):
            raise ValueError("task temperatures must be frozen calibration fits")
    return selected


def _participant_stream(
    source: dict, *, batch_rows: int, max_rows: int
) -> Iterator[pd.DataFrame]:
    dataset = ds.dataset(source["input_path"], format="parquet", partitioning="hive")
    fragments = {
        str(Path(fragment.path).resolve()) for fragment in dataset.get_fragments()
    }
    if fragments != {
        str(Path(binding["path"]).resolve()) for binding in source["normalized_inputs"]
    }:
        raise ValueError(
            "normalized dataset files differ from the accepted freeze inventory"
        )
    expected_source = source["source_id"]
    # Inspect identities alone before any CGM/action decoding or model inference.
    for batch in dataset.scanner(
        columns=["source_id"], batch_size=batch_rows, use_threads=False
    ).to_batches():
        identities = batch.column(0).to_pylist()
        if any(value is None or str(value) != expected_source for value in identities):
            raise ValueError(
                "normalized input contains unknown or confirmatory sources"
            )
    scanner = dataset.scanner(
        filter=ds.field("source_id") == expected_source,
        batch_size=batch_rows,
        use_threads=False,
    )
    current, pieces, count, seen = None, [], 0, set()
    for batch in scanner.to_batches():
        frame = batch.to_pandas()
        if frame["participant_id"].isna().any():
            raise ValueError("normalized participant identity is missing")
        ids = frame["participant_id"].astype(str).to_numpy()
        if not len(ids):
            continue
        edges = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1, len(ids)]
        for start, stop in zip(edges[:-1], edges[1:], strict=True):
            key = str(ids[start])
            if key != current:
                if current is not None:
                    yield pd.concat(pieces, ignore_index=True)
                if key in seen or len(seen) >= 100_000:
                    raise ValueError("participant contiguity/group bound violated")
                seen.add(key)
                current, pieces, count = key, [], 0
            count += int(stop - start)
            if count > max_rows:
                raise ValueError("participant row bound exceeded")
            pieces.append(frame.iloc[start:stop])
    if current is not None:
        yield pd.concat(pieces, ignore_index=True)


class FrozenPredictor:
    def __init__(self, run_dir: Path, config: dict, summary: dict):
        self.config, self.summary = config, summary
        self.device = _device(config)
        self.models, self.scaler = None, None
        self.model = None
        if summary.get("architecture") in {"lstm", "tcn"}:
            self.model, loaded = _load_frozen_model(run_dir, device=self.device)
            if loaded != summary:
                raise ValueError("model summary changed while loading frozen weights")
            self.scaler = SequenceScaler.from_dict(
                json.loads((run_dir / "scaler.json").read_text())
            )
        else:
            self.models = joblib.load(run_dir / "baseline_models.joblib")
            if set(self.models) != set(TASKS):
                raise ValueError("frozen classical model heads differ from training")

    def predict(self, normalized: pd.DataFrame, replay: ReplayInputs) -> pd.DataFrame:
        windows = replay.timeline
        if windows.empty:
            return windows.copy()
        if self.model is not None:
            context = int(
                float(self.config["protocol"]["context_hours"])
                * 60
                / int(self.config["protocol"]["sample_minutes"])
            )
            dataset = FastWindowSequenceDataset(
                normalized,
                windows,
                scaler=self.scaler,
                context_samples=context,
                input_feature_mode="cgm_mask_clock",
                revision_contract=REVISION_CONTRACT,
            )
            prediction = _predict_dataset(
                self.model,
                dataset,
                device=self.device,
                batch_size=int(self.config["training"]["batch_size"]),
                num_workers=0,
                memory_stride_minutes=int(
                    self.config["model"]["memory_stride_minutes"]
                ),
            )
            _restore_regression_scale(prediction, self.scaler)
        else:
            features = build_revision_features(normalized, windows, self.config)
            prediction = windows.copy()
            for task in TASKS:
                artifact = self.models[task]
                if artifact["baseline"] != self.summary["baseline"]:
                    raise ValueError("joblib baseline differs from the frozen summary")
                x = _task_features(
                    windows,
                    features,
                    baseline=artifact["baseline"],
                    task=task,
                    sample_minutes=int(self.config["protocol"]["sample_minutes"]),
                )
                if (
                    list(x.columns) != artifact["features"]
                    or artifact["features"] != self.summary["tasks"][task]["features"]
                ):
                    raise ValueError("classical frozen feature schema changed")
                model = artifact["model"]
                probability = (
                    _predict_probability(model, x)
                    if isinstance(model, Pipeline)
                    else model.predict_proba(x)[:, 1]
                )
                raw = np.asarray(probability, float)
                if not np.isfinite(raw).all() or np.any((raw < 0) | (raw > 1)):
                    raise ValueError("frozen classical probabilities are invalid")
                clipped = np.clip(raw, 1e-6, 1 - 1e-6)
                prediction[f"{task}_logit"] = np.log(clipped / (1 - clipped))
            if self.summary.get("continuous_prediction_available"):
                for horizon in _HORIZONS:
                    prediction[f"cgm_{horizon}_prediction"] = (
                        _continuous_baseline_forecast(
                            prediction,
                            baseline=self.summary["baseline"],
                            horizon=horizon,
                            sample_minutes=int(
                                self.config["protocol"]["sample_minutes"]
                            ),
                        )
                    )
        for horizon in _HORIZONS:
            column = f"cgm_{horizon}_prediction"
            if (
                column in prediction
                and not np.isfinite(prediction[column].to_numpy(float)).all()
            ):
                raise ValueError(
                    "frozen model produced non-finite continuous predictions"
                )
        for task in TASKS:
            logits = prediction[f"{task}_logit"].to_numpy(float)
            if not np.isfinite(logits).all():
                raise ValueError("frozen model produced non-finite logits")
            prediction[f"{task}_raw_probability"] = apply_temperature(logits, 1.0)
            prediction[f"{task}_probability"] = apply_temperature(
                logits, float(self.summary["tasks"][task]["temperature"])
            )
        for horizon in _HORIZONS:
            prediction[f"severe_hypo_{horizon}_raw_logit"] = prediction[
                f"hypo_{horizon}_logit"
            ]
            prediction[f"severe_hypo_{horizon}_raw_probability"] = prediction[
                f"hypo_{horizon}_raw_probability"
            ]
            prediction[f"severe_hypo_{horizon}_probability"] = prediction[
                f"hypo_{horizon}_probability"
            ]
        if len(prediction) != len(windows) or not pd.to_datetime(
            prediction.context_end, utc=True
        ).equals(pd.to_datetime(windows.context_end, utc=True)):
            raise ValueError("frozen prediction decision alignment changed")
        return prediction


def run_revision_external(
    freeze_path: Path,
    *,
    run_dir: Path,
    partition: str,
    output_dir: Path,
    batch_rows: int = 65_536,
    max_participant_rows: int = 1_000_000,
) -> Path:
    if min(batch_rows, max_participant_rows) <= 0:
        raise ValueError("stream resource bounds must be positive")
    freeze_path, run_dir, output_dir = (
        freeze_path.resolve(),
        run_dir.resolve(),
        output_dir.resolve(),
    )
    freeze_stat = freeze_path.stat()
    freeze, entry, source, config, summary = load_development_freeze(
        freeze_path, run_dir, partition
    )
    policies = _policies(entry, summary, config)
    normalized_path = Path(source["input_path"]).resolve()
    if (
        output_dir == normalized_path
        or output_dir.is_relative_to(normalized_path)
        or normalized_path.is_relative_to(output_dir)
    ):
        raise ValueError(
            "external output must be isolated from normalized source inputs"
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite an external replay")
    output_dir.mkdir(parents=True, exist_ok=True)
    private, exportable = output_dir / "private", output_dir / "exportable"
    private.mkdir()
    exportable.mkdir()
    _json(private / "recipe-freeze-used.json", freeze)
    participants = {key: [] for key in policies}
    counts = {
        "normalized_participants": 0,
        "predicted_participants": 0,
        "zero_base_exposure_participants": 0,
        "decision_rows": 0,
        "tail_rows": 0,
        "raw_rows": 0,
    }
    contract = source["source_normalization_contract"]
    try:
        predictor = FrozenPredictor(run_dir, config, summary)
        for ordinal, frame in enumerate(
            _participant_stream(
                source, batch_rows=batch_rows, max_rows=max_participant_rows
            )
        ):
            counts["normalized_participants"] += 1
            counts["raw_rows"] += len(frame)
            validate_frame(frame)
            frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True).dt.as_unit(
                "ns"
            )
            if (
                "cgm_lower_censored" not in frame
                and contract["lower_censoring_absent_means_false"]
            ):
                frame["cgm_lower_censored"] = False
            replay = build_participant_replay(
                frame,
                config,
                partition=partition,
                lower_censoring_absent_means_false=contract[
                    "lower_censoring_absent_means_false"
                ],
            )
            prediction = predictor.predict(frame, replay)
            for category, table in (
                ("predictions", prediction),
                *(
                    (f"endpoints/{name}-{horizon}", tables[horizon])
                    for name, tables in (
                        ("events", replay.events),
                        ("exposure", replay.exposure),
                    )
                    for horizon in _HORIZONS
                ),
            ):
                if category == "predictions" and table.empty:
                    continue
                directory = private / category
                directory.mkdir(parents=True, exist_ok=True)
                _parquet(directory / f"part-{ordinal:05d}.parquet", table)
            if prediction.empty:
                del prediction, replay, frame
                continue
            counts["predicted_participants"] += 1
            counts["decision_rows"] += len(prediction)
            counts["tail_rows"] += int((~prediction.base_evaluable).sum())
            if counts["predicted_participants"] == 1:
                print(
                    json.dumps(
                        {
                            "event": "first_finite_prediction",
                            "rows": len(prediction),
                            "scored_heads": len(TASKS),
                        }
                    ),
                    flush=True,
                )
            elif counts["normalized_participants"] % 25 == 0:
                print(
                    json.dumps({"event": "participant_progress", **counts}), flush=True
                )
            if any(replay.exposure[h].empty for h in _HORIZONS):
                if not all(replay.exposure[h].empty for h in _HORIZONS):
                    raise ValueError("base participant differs between horizons")
                counts["zero_base_exposure_participants"] += 1
                del prediction, replay, frame
                continue
            for (cut, horizon, budget), policy in policies.items():
                timeline = prediction.copy(deep=False)
                timeline["base_evaluable"] = prediction[f"base_evaluable_{horizon}"]
                evaluated = evaluate_fixed_frame(
                    timeline,
                    replay.events[horizon],
                    replay.exposure[horizon],
                    spec=FixedFrameSpec(
                        f"severe_hypo_{horizon}_probability",
                        horizon,
                        (source["source_id"],),
                    ),
                    threshold=float(policy["threshold"]),
                    glucose_cut=cut,
                )
                participants[cut, horizon, budget].append(evaluated.participants)
                root = private / "evaluations" / str(cut) / str(horizon) / budget
                for name in ("participants", "sources", "events", "alarms"):
                    destination = root / name
                    destination.mkdir(parents=True, exist_ok=True)
                    _parquet(
                        destination / f"part-{ordinal:05d}.parquet",
                        getattr(evaluated, name),
                    )
            del frame, prediction, replay, timeline, evaluated
        summaries = []
        for key, tables in participants.items():
            if not tables:
                raise ValueError("no base-evaluable external participants")
            table = pd.concat(tables, ignore_index=True)
            if table.duplicated(["source_id", "participant_id"]).any():
                raise ValueError("external participant metrics are duplicated")
            metrics = _aggregate(table)
            cut, horizon, budget = key
            metrics.update(
                {
                    "glucose_cut": cut,
                    "horizon_minutes": horizon,
                    "nominal_fapd": float(budget),
                    "threshold": policies[key]["threshold"],
                    "equal_source_participant_macro_fixed_sensitivity": metrics[
                        "fixed_sensitivity"
                    ],
                    "source_id": source["source_id"],
                    "source_count": 1,
                }
            )
            summaries.append(metrics)
        _json(
            private / "input-audit.json",
            {
                **counts,
                "source_normalization_contract": contract,
                "prediction_batch_size": int(config["training"]["batch_size"]),
                "prediction_loader_workers": 0,
                "device": str(predictor.device),
                "global_cohort_audit": (
                    "reused frozen accepted evidence; not recomputed per participant"
                ),
                "accepted_global_cohort_audit": source["accepted_global_cohort_audit"],
            },
        )
        result = {
            "status": "complete",
            "revision_contract": REVISION_CONTRACT,
            "freeze_id": freeze["freeze_id"],
            "partition": partition,
            "external_already_exposed": True,
            "thresholds_fitted_on_external": False,
            "source_preprocessing_limits": contract["source_preprocessing_limits"],
            "counts": counts,
            "policies": summaries,
        }
        _json(exportable / "fixed-frame-summary.json", result)
        pd.DataFrame(
            [
                {
                    key: json.dumps(_plain(value))
                    if isinstance(value, (list, dict, tuple))
                    else value
                    for key, value in row.items()
                }
                for row in summaries
            ]
        ).to_csv(exportable / "fixed-frame-summary.csv", index=False)
        _export_alarm_glucose(
            output_dir, source_order=(source["source_id"],), conditions=list(policies)
        )
        # Recheck after all ledger consumers finish, before publishing completion.
        _recheck_bindings(freeze)
        if summary != entry["model_summary"]["snapshot"]:
            raise ValueError("frozen in-memory calibration changed during replay")
        final_stat = freeze_path.stat()
        if (final_stat.st_size, final_stat.st_mtime_ns) != (
            freeze_stat.st_size,
            freeze_stat.st_mtime_ns,
        ):
            raise ValueError("development freeze changed during external replay")
        _json(
            output_dir / "status.json",
            {"status": "complete", "freeze_id": freeze["freeze_id"]},
        )
        _write_run_receipt(
            output_dir,
            run_dir=run_dir,
            freeze_id=freeze["freeze_id"],
            partition=partition,
            source=source,
            summary=summary,
            counts=counts,
        )
    except Exception as error:
        _json(
            output_dir / "status.json",
            {"status": "failed", "exception_type": type(error).__name__},
        )
        raise
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--partition", choices=tuple(_ALLOWED), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_revision_external(
        args.freeze,
        run_dir=args.run_dir,
        partition=args.partition,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
