"""Export retained U05 policy evidence without loading predictions or refitting.

Usage: ``python -m glycoshift.revision_threshold_audit --config analysis.json
--output-dir new-audit-directory``. The config's existing output_dir is the
accepted analysis input. This exporter reads aggregate JSON only. Missing
candidate sweeps remain missing; no participant ledger or timeline is opened.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
from importlib.metadata import version
from pathlib import Path

_IDENTITY = ("arm", "model_seed", "horizon_minutes", "glucose_cut", "mode")
_POLICY_METRICS = {
    "policy_calibration_sensitivity": (
        "achieved_equal_source_participant_macro_fixed_sensitivity"
    ),
    "policy_calibration_achieved_fapd": (
        "achieved_equal_source_participant_macro_fapd"
    ),
}
_SELECTION_FIELDS = (
    "participant_count", "event_positive_participant_count", "base_event_count",
    "patient_days", "observation_minutes", "eligible_decision_count",
    "eligible_evaluable_decision_count", "emitted_alarm_count",
    "delivered_alarm_count", "unevaluable_alarm_count", "matched_alarm_count",
    "unmatched_alarm_count", "equal_source_participant_macro_fixed_sensitivity",
    "equal_source_participant_macro_unmatched_alarms_per_patient_day",
)
_CANDIDATE_FIELDS = (
    *_IDENTITY, "candidate_index", "threshold", "is_sentinel",
    "calibration_sensitivity", "calibration_achieved_fapd", "feasible", "selected",
)


def _read(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required aggregate is not a regular file: {path}")
    return json.loads(path.read_text())


def _key(row):
    return tuple(row[field] for field in _IDENTITY)


def _fit_key(row):
    return row["arm"], row["model_seed"]


def _finite(value, field):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"policy {field} is not numeric")
    if not math.isfinite(value):
        raise ValueError(f"policy {field} is not finite")
    return value


def _write_csv(path, rows, fields):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _policy_row(index, wrapper, source_order, requested_count, base):
    """Bind one policy to its indexed application and the retained base policy."""
    arm, seed, horizon, cut, mode = _key(index)
    policy = wrapper["policy"]
    expected_cut = cut if mode == "refitted" else 54
    origin = (
        "frozen_base54_gate_only" if mode == "gate_only" else
        "verified_frozen_base54" if cut == 54 else "calibration_refit_at_cut"
    )
    if (
        wrapper.get("applied_glucose_cut") != cut
        or wrapper.get("policy_origin") != origin
        or wrapper.get("base54_policy_parity_checked") is not True
        or policy.get("partition") != "calibration"
        or policy.get("glucose_cut") != expected_cut
        or policy.get("nominal_fapd") != 0.5
        or policy.get("endpoint") != {
            "score_column": f"severe_hypo_{horizon}_probability",
            "horizon_minutes": horizon,
            "source_order": source_order,
        }
        or policy.get("threshold") != index["threshold"]
    ):
        raise ValueError(f"policy/index binding failed: {_key(index)}")
    if (mode == "gate_only" or cut == 54) and policy != base:
        raise ValueError(f"frozen base54 policy differs: {_key(index)}")
    threshold = _finite(policy["threshold"], "threshold")
    sensitivity, burden = (
        _finite(policy[name], name) for name in _POLICY_METRICS.values()
    )
    count = policy["threshold_count"]
    if (
        not 0 <= sensitivity <= 1 or not 0 <= burden <= 0.5
        or isinstance(count, bool) or not isinstance(count, int)
        or not 1 <= count <= requested_count + 1
        or policy["tie_rule"] != "maximum_fixed_sensitivity_then_highest_threshold"
    ):
        raise ValueError(f"invalid retained policy metrics or rules: {_key(index)}")
    no_eligible = policy["candidate_rule"] == (
        "no_eligible_decisions_no_alarm_sentinel"
    )
    if policy["candidate_rule"] not in (
        f"{requested_count}_online_eligible_quantiles_plus_no_alarm_sentinel",
        "no_eligible_decisions_no_alarm_sentinel",
    ) or (no_eligible and (count != 1 or sensitivity != 0 or burden != 0)):
        raise ValueError(f"invalid retained candidate rule: {_key(index)}")
    same_cut = expected_cut == cut
    row = {
        **dict(zip(_IDENTITY, (arm, seed, horizon, cut, mode), strict=True)),
        "policy_glucose_cut": expected_cut,
        "applied_glucose_cut": cut,
        "policy_origin": origin,
        "threshold_partition": "calibration",
        "evaluation_partition": "selection",
        "nominal_fapd": policy["nominal_fapd"],
        "threshold": threshold,
        "requested_quantile_count": requested_count,
        "reported_threshold_count": count,
        "threshold_count_evidence": "retained_policy_not_candidate_table",
        "candidate_rule": policy["candidate_rule"],
        "tie_rule": policy["tie_rule"],
        "policy_calibration_sensitivity": sensitivity,
        "policy_calibration_achieved_fapd": burden,
        "policy_calibration_budget_slack": policy["nominal_fapd"] - burden,
        "applied_cut_calibration_sensitivity": sensitivity if same_cut else None,
        "applied_cut_calibration_achieved_fapd": burden if same_cut else None,
        "applied_cut_calibration_budget_slack": (
            policy["nominal_fapd"] - burden if same_cut else None
        ),
        "applied_cut_calibration_evidence": (
            "retained_fitted_policy" if same_cut else "missing_gate_only_replay"
        ),
        "selected_is_sentinel": True if no_eligible else None,
        "sentinel_value": threshold if no_eligible else None,
        "policy_calibration_online_eligible_row_count": 0 if no_eligible else None,
        "policy_calibration_zero_emission_reason": (
            "no_online_eligible_rows_per_retained_rule" if no_eligible else
            "not_identifiable_without_retained_calibration_emission_evidence"
        ),
        "candidate_rows_exported": 0,
        "lower_feasible_threshold": None,
        "lower_feasible_calibration_sensitivity": None,
        "lower_feasible_calibration_achieved_fapd": None,
        "higher_feasible_threshold": None,
        "higher_feasible_calibration_sensitivity": None,
        "higher_feasible_calibration_achieved_fapd": None,
    }
    for field in (
        "participant_count", "base_event_count", "patient_days", "observation_minutes"
    ):
        row[f"policy_calibration_{field}"] = policy[field]
    return row


def export_threshold_audit(config_path: Path, output_dir: Path) -> Path:
    """Validate retained bindings, then write a new aggregate-only audit tree."""
    if output_dir.exists():
        raise FileExistsError(output_dir)
    config_path = config_path.resolve()
    config = _read(config_path)
    analysis = Path(config["output_dir"]).resolve()
    status = _read(analysis / "analysis-status.json")
    summary = _read(analysis / "exportable/cut-grid-summary.json")
    ledger = _read(analysis / "private/unique-fit-ledger.json")["fits"]
    requested_count = config.get("threshold_candidate_count", 2001)
    if (
        status.get("status") != "complete" or summary.get("status") != "complete"
        or summary.get("revision_contract") != config["revision_contract"]
        or summary.get("threshold_partition") != "calibration"
        or summary.get("evaluation_partition") != "selection"
        or summary.get("nominal_fapd") != 0.5 or requested_count != 2001
    ):
        raise ValueError("analysis status, contract or fitting role is not accepted")
    sources = summary["source_order"]
    if not sources or len(sources) != len(set(sources)):
        raise ValueError("invalid analysis source order")
    entries = config["arms"]
    fits = {_fit_key(entry): Path(entry["run_dir"]).resolve() for entry in entries}
    actual = {_fit_key(entry): Path(entry["run_dir"]).resolve() for entry in ledger}
    if (
        not fits or len(fits) != len(entries) or len(actual) != len(ledger)
        or len(set(fits.values())) != len(fits)
        or fits != actual or summary["unique_fit_count"] != len(fits)
    ):
        raise ValueError("config and retained unique-fit ledger differ")
    expected = {
        (*fit, horizon, cut, mode)
        for fit in fits for horizon in summary["horizons"]
        for cut in summary["cuts"] for mode in ("gate_only", "refitted")
    }
    indexes = {_key(row): row for row in summary["analyses"]}
    if set(indexes) != expected or len(indexes) != len(summary["analyses"]):
        raise ValueError("analysis index has missing, unexpected or duplicate policies")
    models = {}
    for fit, directory in fits.items():
        model = _read(directory / "model_summary.json")
        if model["revision_contract"] != config["revision_contract"]:
            raise ValueError("model summary revision contract differs")
        if fit[1] is not None and model["training_seed"] != fit[1]:
            raise ValueError("model summary seed differs from fit index")
        models[fit] = model
    rows, missing = [], []
    for key, index in indexes.items():
        arm, seed, horizon, cut, mode = key
        if (
            not isinstance(arm, str) or not arm
            or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in arm)
            or horizon not in (30, 60) or cut not in (54, 60, 70, 80, 90, 100)
        ):
            raise ValueError("invalid indexed policy identity")
        directory = (
            analysis / "private" / arm / ("shared" if seed is None else str(seed))
            / mode / str(cut) / str(horizon)
        )
        wrapper = _read(directory / "policy.json")
        selected = _read(directory / "summary.json")
        indexed_summary = {
            k: v for k, v in index.items() if k not in (*_IDENTITY, "threshold")
        }
        if selected != indexed_summary:
            raise ValueError(f"private selection summary differs from index: {key}")
        task = models[arm, seed]["tasks"][f"hypo_{horizon}"]
        if (
            task["threshold_partition"] != "calibration"
            or task["temperature_fit_partition"] != "calibration"
        ):
            raise ValueError("retained model fitting role differs")
        row = _policy_row(
            index, wrapper, sources, requested_count,
            task["fixed_frame_policies"]["0.5"],
        )
        candidate_files = sorted(path.name for path in directory.glob("*candidate*"))
        row["candidate_table_status"] = (
            "not_saved" if not candidate_files else "present_schema_not_bound"
        )
        for field in _SELECTION_FIELDS:
            row[f"selection_{field}"] = selected[field]
        row["selection_zero_emissions_observed"] = selected["emitted_alarm_count"] == 0
        rows.append(row)
        missing.append({
            **{field: row[field] for field in _IDENTITY},
            "candidate_table_status": row["candidate_table_status"],
            "candidate_filenames": candidate_files,
            "missing": [
                "complete_candidate_sweep", "adjacent_feasible_candidates",
                "numeric_sentinel_verification", "calibration_emission_ledger",
            ],
        })
    output_dir.mkdir(parents=True)
    _write_csv(
        output_dir / "threshold-policy-operating-points.csv", rows, list(rows[0])
    )
    _write_csv(output_dir / "threshold-policy-candidates.csv", [], _CANDIDATE_FIELDS)
    (output_dir / "threshold-policy-operating-points.json").write_text(
        json.dumps(rows, indent=2, allow_nan=False) + "\n"
    )
    (output_dir / "missing-evidence.json").write_text(
        json.dumps(missing, indent=2, allow_nan=False) + "\n"
    )
    receipt = {
        "status": "completed_with_missing_candidate_evidence",
        "revision_contract": config["revision_contract"],
        "input_config": str(config_path), "input_analysis_dir": str(analysis),
        "unique_fits_scanned": len(fits), "policy_rows_exported": len(rows),
        "candidate_rows_exported": 0,
        "candidate_count_source": "retained_policy_reports_only",
        "binding_checks": [
            "config_fit_ledger", "complete_index_scope", "policy_index_threshold",
            "endpoint_source_horizon_cut", "frozen_base54_policy", "selection_summary",
        ],
        "raw_predictions_read": False, "participant_ledgers_read": False,
        "thresholds_refit": False, "candidate_sweeps_recomputed": False,
        "python_version": platform.python_version(),
        "package_versions": {name: version(name) for name in ("numpy", "pandas")},
    }
    (output_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n"
    )
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    export_threshold_audit(args.config, args.output_dir)


if __name__ == "__main__":
    main()
