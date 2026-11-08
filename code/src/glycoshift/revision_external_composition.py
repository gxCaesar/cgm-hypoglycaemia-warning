"""Exact-key postprocessing of two already-scored, frozen external members.

From code/, run ``PYTHONPATH=src python -m glycoshift.revision_external_composition
--freeze F --composition RECIPE --lstm-output L --persistence-output P
--partition external_loop --output O``.
Only already exposed Loop/HUPA roles are accepted. No model, mapping, temperature
or policy is fitted. A rank mean has no inherited probability/regression metrics.

The accepted development freeze must contain a unique ``compositions`` entry:
``{recipe: {path,size_bytes,mtime_ns}, accepted_development_evidence: [bindings]}``.
It references the existing real recipe; this consumer creates no new freeze.
Optional ``cut70_policies`` has exactly keys "30"/"60", each {policy,evidence};
the evidence file must record that same policy, policy_origin=
"calibration_refit_at_cut" and base54_policy_parity_checked=true. Base54/.5 uses
the recipe's original policy. Cut70/.5 is never inferred from member policies.

Both external output roots require their native completed run-receipt.json,
exact producer/model snapshots and complete prediction/endpoint file bindings.
Private shards are read one participant at a time; only aggregate JSON/CSV files
under exportable/ are candidates for transfer. Source/global-QC acceptance is
reused from the producer, never claimed as a new global audit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .calibration import apply_temperature
from .fixed_frame_evaluation import FixedFrameSpec, _aggregate, evaluate_fixed_frame
from .revision_composition import (
    HORIZONS,
    MEMBERS,
    compose_frozen_scores,
    load_frozen_composition,
)
from .revision_external import (
    _ALLOWED,
    _bound_file,
    _evidence,
    _export_alarm_glucose,
    _json,
    _parquet,
    _plain,
    _publish_run_receipt,
    _recheck_bindings,
    load_development_freeze,
)
from .sequences import REVISION_CONTRACT

_IDS = ["source_id", "participant_id"]
_KEYS = [*_IDS, "context_end"]
_METADATA = [
    *_KEYS,
    "partition",
    "patient_day",
    "cgm_last",
    "base_evaluable",
    "anchor_cgm_observed",
    "anchor_cgm_interpolated",
    "anchor_cgm_censored",
    *(f"base_evaluable_{h}" for h in HORIZONS),
    *(
        f"severe_hypo_{h}_{suffix}"
        for h in HORIZONS
        for suffix in ("label", "event_id", "event_onset")
    ),
]
_SCORES = [
    name
    for h in HORIZONS
    for name in (
        f"hypo_{h}_logit",
        f"hypo_{h}_probability",
        f"severe_hypo_{h}_probability",
    )
]


def _composition_recipe(freeze: dict, path: Path):
    matches = [
        entry
        for entry in freeze.get("compositions", [])
        if Path(entry["recipe"]["path"]).resolve() == path.resolve()
    ]
    if len(matches) != 1:
        raise ValueError(
            "composition recipe is not uniquely accepted in the development freeze"
        )
    entry = matches[0]
    _bound_file(entry["recipe"], path)
    _evidence(entry.get("accepted_development_evidence"))
    frozen = load_frozen_composition(path)
    policies = {(54, h): dict(frozen.policies[h]) for h in HORIZONS}
    if "cut70_policies" in entry:
        if set(entry["cut70_policies"]) != {str(h) for h in HORIZONS}:
            raise ValueError("cut70 composition policies require both horizons")
        for horizon in HORIZONS:
            record = entry["cut70_policies"][str(horizon)]
            evidence = json.loads(_bound_file(record["evidence"]).read_text())
            policy = record["policy"]
            if (
                evidence.get("policy_origin") != "calibration_refit_at_cut"
                or evidence.get("base54_policy_parity_checked") is not True
                or _plain(evidence.get("policy")) != _plain(policy)
            ):
                raise ValueError(
                    "cut70 composition policy lacks exact development evidence"
                )
            endpoint = policy["endpoint"]
            if (
                policy["partition"] != "calibration"
                or policy["glucose_cut"] != 70
                or policy["nominal_fapd"] != 0.5
                or endpoint["horizon_minutes"] != horizon
                or endpoint["score_column"] != f"composition_{horizon}_score"
                or tuple(endpoint["source_order"]) != frozen.source_order
                or not np.isfinite(policy["threshold"])
            ):
                raise ValueError("cut70 composition endpoint/role/budget differs")
            policies[70, horizon] = dict(policy)
    return entry, frozen, policies


def _read_receipt(
    root: Path, freeze: dict, run_entry: dict, source: dict, partition: str
):
    path = root / "run-receipt.json"
    if path.is_symlink() or not path.is_file():
        raise ValueError("native completed external producer receipt is required")
    stat = path.stat()
    receipt_binding = {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    receipt = json.loads(path.read_text())
    if (
        receipt.get("schema_version") != "1.0"
        or receipt.get("status") != "complete"
        or receipt.get("revision_contract") != REVISION_CONTRACT
        or receipt.get("freeze_id") != freeze["freeze_id"]
        or receipt.get("partition") != partition
        or receipt.get("source_id") != source["source_id"]
        or Path(receipt.get("run_dir", "")).resolve()
        != Path(run_entry["run_dir"]).resolve()
        or receipt.get("model_summary_snapshot")
        != run_entry["model_summary"]["snapshot"]
        or receipt.get("source_normalization_contract")
        != source["source_normalization_contract"]
        or receipt.get("global_cohort_audit")
        != "reused frozen accepted evidence; not recomputed per participant"
    ):
        raise ValueError(
            "external producer identity/role/recipe/member snapshot differs"
        )
    bindings = {}
    for binding in receipt["files"]:
        relative = Path(binding["relative_path"])
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or str(relative) in bindings
        ):
            raise ValueError("producer receipt has escaping or duplicate file paths")
        target = root / relative
        if not target.resolve().is_relative_to(root):
            raise ValueError("producer file binding escapes its output root")
        _bound_file(
            {
                "path": str(target),
                "size_bytes": binding["size_bytes"],
                "mtime_ns": binding["mtime_ns"],
            }
        )
        bindings[str(relative)] = binding
    for required in (
        "private/recipe-freeze-used.json",
        "private/input-audit.json",
        "status.json",
        "exportable/fixed-frame-summary.json",
    ):
        if required not in bindings:
            raise ValueError("producer receipt lacks a required terminal binding")
    if json.loads((root / "private/recipe-freeze-used.json").read_text()) != freeze:
        raise ValueError("producer used a different development freeze snapshot")
    status = json.loads((root / "status.json").read_text())
    if status != {"status": "complete", "freeze_id": freeze["freeze_id"]}:
        raise ValueError("external producer terminal status is not complete")
    audit = json.loads((root / "private/input-audit.json").read_text())
    if (
        audit["source_normalization_contract"]
        != source["source_normalization_contract"]
        or audit["accepted_global_cohort_audit"]
        != source["accepted_global_cohort_audit"]
        or "not recomputed" not in audit["global_cohort_audit"]
    ):
        raise ValueError("external normalization/global-QC acceptance differs")
    report = json.loads((root / "exportable/fixed-frame-summary.json").read_text())
    if (
        any(audit.get(name) != value for name, value in receipt["counts"].items())
        or report.get("counts") != receipt["counts"]
        or report.get("partition") != partition
        or report.get("freeze_id") != freeze["freeze_id"]
    ):
        raise ValueError("producer counts or summary differ from its native receipt")
    prediction_files = sorted((root / "private/predictions").rglob("*.parquet"))
    endpoint_files = sorted((root / "private/endpoints").rglob("*.parquet"))
    evaluation_files = sorted((root / "private/evaluations").rglob("*.parquet"))
    categories = {f"{name}-{h}" for name in ("events", "exposure") for h in HORIZONS}
    if any(
        path.parent != root / "private/predictions" for path in prediction_files
    ) or any(
        path.parent.parent != root / "private/endpoints"
        or path.parent.name not in categories
        for path in endpoint_files
    ):
        raise ValueError(
            "producer output contains unexpected nested prediction/endpoint shards"
        )
    actual = {
        str(p.relative_to(root))
        for p in prediction_files + endpoint_files + evaluation_files
    }
    recorded = {
        name
        for name in bindings
        if name.startswith(
            ("private/predictions/", "private/endpoints/", "private/evaluations/")
        )
    }
    if not prediction_files or actual != recorded:
        raise ValueError(
            "external prediction/endpoint/evaluation files differ "
            "from producer inventory"
        )
    index, row_count = {}, 0
    for file in prediction_files:
        parquet = pq.ParquetFile(file)
        if parquet.metadata.num_rows > 1_000_000:
            raise ValueError("participant prediction row bound exceeded")
        table = parquet.read(columns=_IDS, use_threads=False)
        key = []
        for column in _IDS:
            values = table.column(column)
            unique = values.unique()
            if values.null_count or len(unique) != 1:
                raise ValueError(
                    "prediction shard must contain one complete participant"
                )
            key.append(str(unique[0].as_py()))
        key = tuple(key)
        if key in index or key[0] != source["source_id"] or len(index) >= 100_000:
            raise ValueError(
                "duplicate/unknown participant or participant group bound exceeded"
            )
        index[key] = file
        row_count += parquet.metadata.num_rows
    counts = receipt["counts"]
    if (
        len(index) != counts["predicted_participants"]
        or row_count != counts["decision_rows"]
    ):
        raise ValueError("producer decision/participant counts do not reconcile")
    names = {file.name for file in prediction_files}
    for category in ("events", "exposure"):
        for horizon in HORIZONS:
            files = list(
                (root / f"private/endpoints/{category}-{horizon}").glob("*.parquet")
            )
            if len(files) != counts["normalized_participants"] or not names.issubset(
                {file.name for file in files}
            ):
                raise ValueError("independent endpoint shard population is incomplete")
            for file in files:
                if file.name not in names and pq.ParquetFile(file).metadata.num_rows:
                    raise ValueError(
                        "nonempty base truth lacks its complete decision timeline"
                    )
    return receipt, receipt_binding, index


def _member_frames(paths: dict, summaries: dict, partition: str):
    frames = {}
    for member in MEMBERS:
        file = pq.ParquetFile(paths[member])
        if not set(_METADATA + _SCORES).issubset(file.schema_arrow.names):
            raise ValueError(
                "external member lacks complete decision/truth/score columns"
            )
        frame = file.read(columns=[*_METADATA, *_SCORES], use_threads=False).to_pandas()
        if not frame.partition.eq(partition).all():
            raise ValueError("external prediction partition differs from its receipt")
        if not np.array_equal(frame.base_evaluable, frame.base_evaluable_60):
            raise ValueError(
                "base evaluability differs from the frozen maximum-horizon frame"
            )
        for horizon in HORIZONS:
            task = f"hypo_{horizon}"
            expected = apply_temperature(
                frame[f"{task}_logit"].to_numpy(float),
                float(summaries[member]["tasks"][task]["temperature"]),
            )
            probability = frame[f"severe_hypo_{horizon}_probability"].to_numpy(float)
            if not np.array_equal(
                probability, frame[f"{task}_probability"].to_numpy(float)
            ) or not np.allclose(probability, expected, rtol=0, atol=1e-12):
                raise ValueError(
                    "external severe/hypo score/temperature binding differs"
                )
        frames[member] = frame
    return frames


def _paired_truth(roots: dict, paths: dict, key: tuple, horizon: int):
    tables = {}
    for member in MEMBERS:
        values = []
        for category in ("events", "exposure"):
            file = (
                roots[member]
                / f"private/endpoints/{category}-{horizon}"
                / paths[member].name
            )
            table = pq.read_table(file, use_threads=False).to_pandas()
            columns = (
                [*_IDS, "event_id", "event_onset"]
                if category == "events"
                else [*_IDS, "patient_days", "observation_minutes"]
            )
            if not set(columns).issubset(table):
                raise ValueError("independent endpoint lacks required truth columns")
            table = table[columns].copy()
            for column in _IDS:
                table[column] = table[column].astype(str)
            if (
                len(table)
                and not (
                    table.source_id.eq(key[0]) & table.participant_id.eq(key[1])
                ).all()
            ):
                raise ValueError(
                    "independent endpoint participant differs from prediction"
                )
            if category == "events":
                table["event_id"] = table.event_id.astype(str)
                table["event_onset"] = pd.to_datetime(table.event_onset, utc=True)
                table.sort_values([*_IDS, "event_id"], inplace=True)
            else:
                table.sort_values(_IDS, inplace=True)
            values.append(table.reset_index(drop=True))
        tables[member] = values
    for index in (0, 1):
        try:
            pd.testing.assert_frame_equal(
                tables["lstm"][index],
                tables["persistence"][index],
                check_dtype=False,
                check_exact=True,
                check_categorical=False,
            )
        except AssertionError:
            raise ValueError(
                "independent base event/exposure truth differs between members"
            ) from None
    return tables["lstm"]


def run_external_composition(
    freeze_path: Path,
    composition_path: Path,
    *,
    lstm_output: Path,
    persistence_output: Path,
    partition: str,
    output_dir: Path,
) -> Path:
    if partition not in _ALLOWED:
        raise ValueError("only already exposed Loop/HUPA composition is allowed")
    freeze_path, composition_path, output_dir = (
        freeze_path.resolve(),
        composition_path.resolve(),
        output_dir.resolve(),
    )
    freeze_stat = freeze_path.stat()
    freeze = json.loads(freeze_path.read_text())
    if (
        freeze.get("revision_contract") != REVISION_CONTRACT
        or freeze.get("status") != "development_recipe_frozen"
    ):
        raise ValueError("real accepted development freeze is required")
    entry, frozen, policies = _composition_recipe(freeze, composition_path)
    roots = {"lstm": lstm_output.resolve(), "persistence": persistence_output.resolve()}
    if len(set(roots.values())) != 2:
        raise ValueError("composition requires two distinct external producer outputs")
    receipts, receipt_bindings, indices, summaries = {}, {}, {}, {}
    source = None
    for member in MEMBERS:
        metadata = frozen.member_metadata[member]
        _, run_entry, current_source, config, summary = load_development_freeze(
            freeze_path, Path(metadata["run_dir"]), partition
        )
        if source is None:
            source = current_source
        elif current_source != source:
            raise ValueError(
                "external members have different normalization/QC source acceptance"
            )
        if (
            metadata.get("revision_contract") != REVISION_CONTRACT
            or metadata.get("input_feature_mode") != "cgm_mask_clock"
            or metadata.get("training_seed") != summary.get("training_seed")
            or summary.get("training_seed") != config["training"]["seed"]
            or metadata.get("comparison_lstm_seed") != frozen.lstm_seed
            or metadata.get("shared_control_fit") != (member == "persistence")
            or tuple(sorted(config["data"]["development_sources"]))
            != frozen.source_order
            or (
                member == "lstm"
                and (
                    summary.get("architecture") != "lstm"
                    or summary.get("training_seed") != frozen.lstm_seed
                )
            )
            or (member == "persistence" and summary.get("baseline") != "persistence")
            or metadata.get("score_columns")
            != {str(h): f"severe_hypo_{h}_probability" for h in HORIZONS}
        ):
            raise ValueError(
                "frozen composition member snapshot differs from accepted producer"
            )
        receipts[member], receipt_bindings[member], indices[member] = _read_receipt(
            roots[member], freeze, run_entry, source, partition
        )
        summaries[member] = summary
    if receipts["lstm"]["counts"] != receipts["persistence"]["counts"] or set(
        indices["lstm"]
    ) != set(indices["persistence"]):
        raise ValueError(
            "external member populations/counts differ; no intersection allowed"
        )
    for root in [*roots.values(), Path(source["input_path"]).resolve()]:
        if (
            output_dir == root
            or output_dir.is_relative_to(root)
            or root.is_relative_to(output_dir)
        ):
            raise ValueError(
                "composition output must be isolated from all source inputs"
            )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite external composition output")
    output_dir.mkdir(parents=True, exist_ok=True)
    private, exportable = output_dir / "private", output_dir / "exportable"
    private.mkdir()
    exportable.mkdir()
    metrics = {condition: [] for condition in policies}
    scored_rows, tail_rows = 0, 0
    try:
        for ordinal, key in enumerate(sorted(indices["lstm"])):
            paths = {member: indices[member][key] for member in MEMBERS}
            frames = _member_frames(paths, summaries, partition)
            combined = compose_frozen_scores(frozen, frames)
            if combined.empty or not combined.partition.eq(partition).all():
                raise ValueError(
                    "composed decision population is empty or has another role"
                )
            if any(
                name.endswith(
                    ("_probability", "_prediction", "_prediction_scaled", "_logit")
                )
                for name in combined
            ):
                raise AssertionError(
                    "rank mean inherited a member probability/regression head"
                )
            scored_rows += len(combined)
            tail_rows += int((~combined.base_evaluable).sum())
            destination = private / "predictions"
            destination.mkdir(exist_ok=True)
            _parquet(destination / f"part-{ordinal:05d}.parquet", combined)
            for horizon in HORIZONS:
                events, exposure = _paired_truth(roots, paths, key, horizon)
                for name, table in (("events", events), ("exposure", exposure)):
                    destination = private / f"endpoints/{name}-{horizon}"
                    destination.mkdir(parents=True, exist_ok=True)
                    _parquet(destination / f"part-{ordinal:05d}.parquet", table)
                if exposure.empty:
                    if not events.empty:
                        raise ValueError("base events exist outside base exposure")
                    continue
                for (cut, h), policy in policies.items():
                    if h != horizon:
                        continue
                    timeline = combined.copy(deep=False)
                    timeline["base_evaluable"] = combined[f"base_evaluable_{horizon}"]
                    evaluated = evaluate_fixed_frame(
                        timeline,
                        events,
                        exposure,
                        spec=FixedFrameSpec(
                            f"composition_{horizon}_score",
                            horizon,
                            (source["source_id"],),
                        ),
                        threshold=float(policy["threshold"]),
                        glucose_cut=cut,
                    )
                    metrics[cut, horizon].append(evaluated.participants)
                    for name in ("participants", "sources", "alarms", "events"):
                        destination = (
                            private / f"evaluations/{cut}/{horizon}/0.5/{name}"
                        )
                        destination.mkdir(parents=True, exist_ok=True)
                        _parquet(
                            destination / f"part-{ordinal:05d}.parquet",
                            getattr(evaluated, name),
                        )
            del frames, combined
        if (
            scored_rows != receipts["lstm"]["counts"]["decision_rows"]
            or tail_rows != receipts["lstm"]["counts"]["tail_rows"]
        ):
            raise ValueError("composition lost or changed complete/tail decision rows")
        rows = []
        for condition, tables in metrics.items():
            if not tables:
                raise ValueError(
                    "composition has no base-evaluable external participants"
                )
            table = pd.concat(tables, ignore_index=True)
            if table.duplicated(_IDS).any():
                raise ValueError(
                    "composition participant metrics contain duplicate identities"
                )
            summary = _aggregate(table)
            cut, horizon = condition
            rows.append(
                {
                    "source_id": source["source_id"],
                    "source_count": 1,
                    "glucose_cut": cut,
                    "horizon_minutes": horizon,
                    "nominal_fapd": 0.5,
                    "threshold": policies[condition]["threshold"],
                    **summary,
                    "equal_source_participant_macro_fixed_sensitivity": summary[
                        "fixed_sensitivity"
                    ],
                }
            )
        _json(
            private / "composition-input-receipt.json",
            {
                "revision_contract": REVISION_CONTRACT,
                "freeze_id": freeze["freeze_id"],
                "recipe": entry["recipe"],
                "member_receipts": receipt_bindings,
                "source_normalization_contract": source[
                    "source_normalization_contract"
                ],
                "global_cohort_audit": (
                    "reused original accepted producer evidence; no new global audit"
                ),
                "scored_rows": scored_rows,
                "tail_rows": tail_rows,
            },
        )
        payload = {
            "status": "complete",
            "revision_contract": REVISION_CONTRACT,
            "freeze_id": freeze["freeze_id"],
            "partition": partition,
            "external_already_exposed": True,
            "comparison_lstm_seed": frozen.lstm_seed,
            "score_interpretation": "rank_mean_not_calibrated_event_probability",
            "inherited_probability_or_regression_diagnostics": False,
            "fitted_on_external": False,
            "scored_rows": scored_rows,
            "tail_rows": tail_rows,
            "policies": rows,
        }
        _json(exportable / "fixed-frame-summary.json", payload)
        pd.DataFrame(
            [
                {
                    name: json.dumps(_plain(value))
                    if isinstance(value, (list, tuple, dict))
                    else value
                    for name, value in row.items()
                }
                for row in rows
            ]
        ).to_csv(exportable / "fixed-frame-summary.csv", index=False)
        _export_alarm_glucose(
            output_dir,
            source_order=(source["source_id"],),
            conditions=[(cut, horizon, "0.5") for cut, horizon in policies],
        )
        # Recheck after all ledger consumers finish, before publishing completion.
        _recheck_bindings(freeze)
        # Recheck bindings without allocating a second full ECDF copy.
        _recheck_bindings(json.loads(composition_path.read_text()))
        for member in MEMBERS:
            _bound_file(receipt_bindings[member])
            for binding in receipts[member]["files"]:
                _bound_file(
                    {
                        "path": str(roots[member] / binding["relative_path"]),
                        "size_bytes": binding["size_bytes"],
                        "mtime_ns": binding["mtime_ns"],
                    }
                )
        if (freeze_path.stat().st_size, freeze_path.stat().st_mtime_ns) != (
            freeze_stat.st_size,
            freeze_stat.st_mtime_ns,
        ):
            raise ValueError("accepted development freeze changed during composition")
        _json(
            output_dir / "status.json",
            {"status": "complete", "freeze_id": freeze["freeze_id"]},
        )
        paths = [
            *sorted(private.rglob("*.parquet")),
            private / "composition-input-receipt.json",
            *sorted(exportable.glob("*.json")),
            *sorted(exportable.glob("*.csv")),
            output_dir / "status.json",
        ]
        files = []
        for path in paths:
            if path.is_symlink() or not path.is_file():
                raise ValueError("composition output must be a regular file")
            stat = path.stat()
            files.append(
                {
                    "relative_path": str(path.relative_to(output_dir)),
                    "size_bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
        _publish_run_receipt(
            output_dir,
            {
                "schema_version": "1.0",
                "producer_kind": "frozen_step_ecdf_composition",
                "status": "complete",
                "revision_contract": REVISION_CONTRACT,
                "freeze_id": freeze["freeze_id"],
                "partition": partition,
                "source_id": source["source_id"],
                "comparison_lstm_seed": frozen.lstm_seed,
                "recipe": entry["recipe"],
                "member_receipts": receipt_bindings,
                "scored_rows": scored_rows,
                "tail_rows": tail_rows,
                "source_normalization_contract": source[
                    "source_normalization_contract"
                ],
                "global_cohort_audit": (
                    "reused original accepted producer evidence; no new global audit"
                ),
                "counts": {
                    **receipts["lstm"]["counts"],
                    "decision_rows": scored_rows,
                    "tail_rows": tail_rows,
                },
                "binding_scope": (
                    "complete prediction/independent-endpoint/evaluation-ledger "
                    "datasets, input receipt and terminal summaries"
                ),
                "files": files,
            },
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
    parser.add_argument("--composition", type=Path, required=True)
    parser.add_argument("--lstm-output", type=Path, required=True)
    parser.add_argument("--persistence-output", type=Path, required=True)
    parser.add_argument("--partition", choices=tuple(_ALLOWED), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_external_composition(
        args.freeze,
        args.composition,
        lstm_output=args.lstm_output,
        persistence_output=args.persistence_output,
        partition=args.partition,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
