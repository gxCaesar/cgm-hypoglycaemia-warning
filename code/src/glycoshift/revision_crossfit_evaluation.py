"""Freeze inner policies before evaluating each participant's outer fold.

This module applies already fitted models. It never fits a model, scaler,
temperature or ECDF on outer participants. Thresholds for cuts 54/70 are fitted
only on the inner calibration timeline, before outer values are loaded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from .calibration import apply_temperature
from .config import load_config
from .fixed_frame_evaluation import (
    FixedFrameSpec,
    _aggregate,
    evaluate_fixed_frame,
    fit_fixed_frame_alarm_policy,
)
from .revision_crossfit_prepare import CROSSFIT_CONTRACT
from .revision_external import (
    FrozenPredictor,
    _bound_file,
    _json,
    _parquet,
    _plain,
    _recheck_bindings,
)
from .revision_inputs import CONFIRMATION_CONTRACT
from .revision_runner import load_crossfit_inner_bundle, load_crossfit_outer_bundle
from .revision_training import RevisionTrainingContext
from .sequences import REVISION_CONTRACT, TASKS

KEYS = ["source_id", "participant_id"]
HORIZONS = (30, 60)
CUTS = (54, 70)


def _binding(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError("crossfit input must be a regular non-symlink file")
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _policy_parity(recorded: dict, recomputed: dict) -> None:
    for field, value in recomputed.items():
        if field not in recorded:
            raise ValueError(f"recorded inner policy lacks {field}")
        old = recorded[field]
        if isinstance(value, float):
            # JSON receipts may carry null for an explicitly undefined estimand.
            old = np.nan if old is None else old
            equal = np.isclose(old, value, rtol=0, atol=1e-12, equal_nan=True)
        else:
            equal = _plain(old) == _plain(value)
        if not equal:
            raise ValueError(f"refitted base54 policy differs from training: {field}")


def freeze_inner_policies(
    prepared: Path, run_dir: Path, *, outer_fold: int, output: Path
) -> dict:
    """Bind the fit and calibration policies before opening any outer values."""
    if output.exists():
        raise FileExistsError("refusing to overwrite a crossfit policy freeze")
    run_dir, prepared = run_dir.resolve(), prepared.resolve()
    files = {
        name: _binding(run_dir / name)
        for name in (
            "config_resolved.yaml",
            "model_summary.json",
            "cohort_audit.json",
            "predictions.parquet",
            "training_summary.json",
        )
    }
    config = load_config(run_dir / "config_resolved.yaml")
    summary = json.loads((run_dir / "model_summary.json").read_text())
    audit = json.loads((run_dir / "cohort_audit.json").read_text())
    if (
        audit.get("crossfit_contract") != CROSSFIT_CONTRACT
        or audit.get("outer_fold") != outer_fold
        or audit.get("crossfit_prepared_dir") != str(prepared)
        or config["protocol"].get("ascertainment_contract") != CONFIRMATION_CONTRACT
        or summary.get("revision_contract") != REVISION_CONTRACT
        or summary.get("input_feature_mode") != "cgm_mask_clock"
    ):
        raise ValueError("fit is not bound to this corrected outer-fold complement")
    artifacts = (
        ("ckpts/best.pt", "scaler.json")
        if summary.get("architecture") in {"lstm", "tcn"}
        else ("baseline_models.joblib",)
        if summary.get("baseline")
        in {"persistence", "linear", "summary_logistic", "xgboost"}
        else ()
    )
    if not artifacts:
        raise ValueError("unknown fitted crossfit model")
    files.update({name: _binding(run_dir / name) for name in artifacts})
    for name in (
        "crossfit-audit.json",
        "shared/input-audit.json",
        "shared/participant-manifest.private.json",
        f"folds/fold-{outer_fold}/participant-roles.private.json",
    ):
        files[f"prepared/{name}"] = _binding(prepared / name)
    (_, _, _), replays = load_crossfit_inner_bundle(
        prepared, config, outer_fold=outer_fold, sequences=False
    )
    sources = tuple(sorted(map(str, config["data"]["development_sources"])))
    context = RevisionTrainingContext(
        replays, source_order=sources, sensitivity_source_policy="supported_sources"
    )
    predictions = (
        ds.dataset(run_dir / "predictions.parquet")
        .to_table(filter=ds.field("partition") == "calibration", use_threads=False)
        .to_pandas()
    )
    context._bind(predictions, "calibration")
    for task in TASKS:
        record = summary["tasks"][task]
        if record.get("temperature_fit_partition") != "calibration":
            raise ValueError("inner temperature has an invalid fitting role")
        expected = apply_temperature(
            predictions[f"{task}_logit"].to_numpy(float), record["temperature"]
        )
        if not np.array_equal(
            expected, predictions[f"{task}_probability"].to_numpy(float)
        ):
            raise ValueError("stored inner calibration probabilities changed")
    policies = {}
    for horizon in HORIZONS:
        score = f"severe_hypo_{horizon}_probability"
        if not np.array_equal(
            predictions[score], predictions[f"hypo_{horizon}_probability"]
        ):
            raise ValueError("severe probability differs from the fitted task")
        for cut in CUTS:
            policy = fit_fixed_frame_alarm_policy(
                context._timeline(predictions, horizon),
                replays["calibration"].events[horizon],
                replays["calibration"].exposure[horizon],
                spec=FixedFrameSpec(score, horizon, sources),
                partition="calibration",
                glucose_cut=cut,
                nominal_fapd=0.5,
                candidate_count=int(
                    config["evaluation"].get("threshold_candidate_count", 2001)
                ),
                sensitivity_source_policy="supported_sources",
            ).as_dict()
            if cut == 54:
                _policy_parity(
                    summary["tasks"][f"hypo_{horizon}"]["fixed_frame_policies"]["0.5"],
                    policy,
                )
            policies[f"{cut}/{horizon}"] = policy
    result = {
        "crossfit_contract": CROSSFIT_CONTRACT,
        "outer_fold": outer_fold,
        "prepared_dir": str(prepared),
        "run_dir": str(run_dir),
        "bindings": files,
        "policies": policies,
        "summary": summary,
        "source_order": sources,
        "policy_fit_role": "calibration",
        "outer_values_opened": False,
        "base54_policy_parity_checked": True,
    }
    _recheck_bindings(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    _json(output, result)
    return result


def run_crossfit_evaluation(
    prepared: Path, run_dir: Path, *, outer_fold: int, output: Path
) -> Path:
    """Freeze, predict once and apply all four fixed cut/horizon policies."""
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("refusing to overwrite an outer evaluation")
    output.mkdir(parents=True, exist_ok=True)
    private = output / "private"
    private.mkdir()
    try:
        freeze_path = private / "policy-freeze.json"
        freeze = freeze_inner_policies(
            prepared, run_dir, outer_fold=outer_fold, output=freeze_path
        )
        freeze_binding = _binding(freeze_path)
        config = load_config(_bound_file(freeze["bindings"]["config_resolved.yaml"]))
        frame, replay, audit = load_crossfit_outer_bundle(
            prepared, config, outer_fold=outer_fold, sequences=True
        )
        if frame is None or audit.get("role") != "outer_evaluation":
            raise ValueError("outer inference requires its isolated value bundle")
        predictor = FrozenPredictor(Path(freeze["run_dir"]), config, freeze["summary"])
        predictions = predictor.predict(frame, replay)
        if not predictions.partition.eq("outer_evaluation").all():
            raise ValueError("outer prediction contains an inner role")
        _parquet(private / "predictions.parquet", predictions)
        rows = []
        for horizon in HORIZONS:
            exposure = replay.exposure[horizon]
            keys = pd.MultiIndex.from_frame(exposure[KEYS])
            present = pd.MultiIndex.from_frame(predictions[KEYS]).isin(keys)
            timeline = predictions.loc[present].copy(deep=False)
            timeline = timeline.assign(
                base_evaluable=timeline[f"base_evaluable_{horizon}"]
            )
            sources = tuple(
                s for s in freeze["source_order"] if s in set(exposure.source_id)
            )
            if not sources:
                raise ValueError("outer horizon has no base exposure")
            for cut in CUTS:
                policy = freeze["policies"][f"{cut}/{horizon}"]
                evaluation = evaluate_fixed_frame(
                    timeline,
                    replay.events[horizon],
                    exposure,
                    spec=FixedFrameSpec(
                        f"severe_hypo_{horizon}_probability", horizon, sources
                    ),
                    threshold=policy["threshold"],
                    glucose_cut=cut,
                )
                destination = private / "evaluations" / str(cut) / str(horizon)
                destination.mkdir(parents=True)
                for name in ("participants", "events", "alarms", "sources"):
                    _parquet(destination / f"{name}.parquet", getattr(evaluation, name))
                rows.append(
                    {
                        "outer_fold": outer_fold,
                        "glucose_cut": cut,
                        "horizon_minutes": horizon,
                        **evaluation.summary,
                    }
                )
        _recheck_bindings(freeze)
        _bound_file(freeze_binding)
        _json(private / "outer-input-audit.json", audit)
        exportable = output / "exportable"
        exportable.mkdir()
        _json(
            exportable / "fold-summary.json",
            {
                "crossfit_contract": CROSSFIT_CONTRACT,
                "rows": rows,
                "interpretation": (
                    "fold-specific diagnostics, not equal-weight fold aggregation"
                ),
            },
        )
        _json(
            output / "receipt.json",
            {
                "status": "complete",
                "crossfit_contract": CROSSFIT_CONTRACT,
                "outer_fold": outer_fold,
                "run_dir": str(run_dir.resolve()),
                "prepared_dir": str(prepared.resolve()),
                "policy_freeze": freeze_binding,
                "outputs": [
                    _binding(path) for path in sorted(private.rglob("*.parquet"))
                ],
                "outer_refitting": False,
            },
        )
    except Exception as error:
        _json(output / "failure.json", {"status": "failed", "error": str(error)})
        raise
    return output


def aggregate_oof_participants(
    table: pd.DataFrame, source_order: tuple[str, ...]
) -> dict:
    """Aggregate concatenated participant contributions, never fold means."""
    if table.empty or table.duplicated(KEYS).any():
        raise ValueError("OOF participant contributions must be nonempty and unique")
    if not set(table.source_id.astype(str)).issubset(source_order):
        raise ValueError("OOF contributions contain an unauthorized source")
    sources = []
    for source in source_order:
        block = table.loc[table.source_id.astype(str).eq(source)]
        sources.append({"source_id": source, **_aggregate(block)})
    undefined = [row["source_id"] for row in sources if row["base_event_count"] == 0]
    missing = [row["source_id"] for row in sources if row["participant_count"] == 0]
    return {
        "source_order": source_order,
        "sources": sources,
        "undefined_sensitivity_sources": undefined,
        "zero_exposure_sources": missing,
        "equal_source_participant_macro_fixed_sensitivity": (
            float("nan")
            if undefined
            else float(np.mean([r["fixed_sensitivity"] for r in sources]))
        ),
        "equal_source_participant_macro_unmatched_alarms_per_patient_day": (
            float("nan")
            if missing
            else float(
                np.mean(
                    [
                        r["participant_macro_unmatched_alarms_per_patient_day"]
                        for r in sources
                    ]
                )
            )
        ),
        "participant_pooled": _aggregate(table),
        "uncertainty": "source/fold/model-seed variation; no overall generalization CI",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, choices=range(5), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_crossfit_evaluation(
        args.prepared, args.run_dir, outer_fold=args.outer_fold, output=args.output
    )


if __name__ == "__main__":
    main()
