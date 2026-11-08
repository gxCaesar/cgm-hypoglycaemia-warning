#!/usr/bin/env python3
"""Candidate Figure 8: fixed development rank pair and conditional external CIs.

Only accepted aggregates and mapping diagnostics are consumed. Development seed
points are descriptive. External whiskers are the existing paired participant
percentile intervals, never uncertainty estimated from five model seeds.
The existing whole-set plan and three-publisher figure study are reused.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "code/scripts/figures"))
import glycofig as G  # noqa: E402

CONTRACT = "2026-11-04-fixed-frame"
SEEDS = (17, 23, 42, 71, 101)
HORIZONS = (30, 60)
SOURCES = ("AZT1D", "BrisT1D", "CTR3", "DCLP3", "DCLP5", "Flair", "IOBP2",
           "PEDAP", "ReplaceBG", "Shanghai T1DM", "T1D-UOM")
ARMS = ("persistence", "lstm", "composition")
IDENTITIES = (("persistence", None), *((a, s) for a in ("lstm", "composition") for s in SEEDS))
LABELS = {"persistence": "Current CGM", "lstm": "LSTM", "composition": "Fixed rank mean"}
COLOURS = {"persistence": G.NOMINAL["rose"], "lstm": G.NOMINAL["blue"], "composition": G.INK}
MARKERS = {"persistence": "o", "lstm": "v", "composition": "D"}
SENSITIVITY = "equal_source_participant_macro_fixed_sensitivity"
BURDEN = "equal_source_participant_macro_unmatched_alarms_per_patient_day"
SOURCE_BURDEN = "participant_macro_unmatched_alarms_per_patient_day"
TRUTH = ("participant_count", "event_positive_participant_count", "base_event_count", "patient_days", "observation_minutes")
ADDITIVE = ("participant_count", "event_positive_participant_count", "base_event_count", "detected_event_count",
            "patient_days", "observation_minutes", "emitted_alarm_count", "delivered_alarm_count",
            "unevaluable_alarm_count", "matched_alarm_count", "unmatched_alarm_count")
MAP_COUNTS = ("threshold_crossing_disagreements", "online_eligible_crossing_disagreements", "common_emitted_alarms",
              "step_only_emitted_alarms", "linear_only_emitted_alarms", "common_emissions_with_match_change",
              "base_events_with_detection_change", "base_events_with_matched_alarm_change")
COHORTS = {"stress_hupa_ucm": "HUPA-UCM", "external_loop": "Loop Observational Study"}
COMPARISONS = (("lstm", "persistence"), ("composition", "persistence"), ("composition", "lstm"))
NAMES = tuple(f"{a}_at_54" for a in ARMS) + tuple(f"{a}_minus_{b}_at_54" for a, b in COMPARISONS)
EXTERNAL_METRICS = ("fixed_sensitivity", "unmatched_alarms_per_patient_day", "delivered_alarms_per_patient_day")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    require(path.is_file() and not path.is_symlink(), f"missing/unsafe aggregate: {path}")
    return json.loads(path.read_text())


def number(value, label, *, undefined=False):
    if value is None:
        require(undefined, f"undefined {label}")
        return None
    require(type(value) in (int, float) and math.isfinite(value), f"invalid {label}")
    return float(value)


def near(a, b, label):
    require(math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-10), f"aggregate mismatch: {label}")


def binding(path):
    stat = path.stat()
    return {"path": str(path.resolve()), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def read_csv(path):
    with path.open(newline="") as stream:
        return [{**row, "source": str(path.resolve()), "csv_line": line}
                for line, row in enumerate(csv.DictReader(stream), start=2)]


def numeric_row(raw, *, global_row):
    row = dict(raw)
    fields = (*ADDITIVE, SENSITIVITY, BURDEN) if global_row else (*ADDITIVE, "fixed_sensitivity", SOURCE_BURDEN)
    for field in fields:
        row[field] = number(float(raw[field]) if isinstance(raw[field], str) else raw[field], field)
    require(0 <= row[SENSITIVITY if global_row else "fixed_sensitivity"] <= 1
            and row[BURDEN if global_row else SOURCE_BURDEN] >= 0, "invalid development rate")
    for field in ADDITIVE:
        require(row[field] >= 0, f"invalid count/exposure: {field}")
    require(row["participant_count"] > 0 and row["patient_days"] > 0 and row["observation_minutes"] > 0
            and 0 < row["event_positive_participant_count"] <= min(row["participant_count"], row["base_event_count"])
            and row["detected_event_count"] <= row["base_event_count"], "development event-positive population differs")
    if global_row:
        require(float(raw["source_count"]) == 11, "development source denominator differs")
    near(row["delivered_alarm_count"], row["matched_alarm_count"] + row["unmatched_alarm_count"], "alarm matches")
    near(row["emitted_alarm_count"], row["delivered_alarm_count"] + row["unevaluable_alarm_count"], "emitted alarms")
    return row


def load_development(conditions, sources, composition_summaries, development_acceptance, development_audit, mapping_dir):
    accepted, audit = read_json(development_acceptance), read_json(development_audit)
    require(accepted.get("status") == "accepted" and accepted.get("owner") == "primary"
            and accepted.get("revision_contract") == CONTRACT, "primary development acceptance is required")
    require(audit.get("status") == "structural_numeric_checks_passed" and audit.get("revision_contract") == CONTRACT
            and audit.get("unique_fits_checked") == 14 and audit.get("main_conditions_checked") == 336
            and audit.get("composition_tables_checked") == 20 and audit.get("composition_seeds_checked") == list(SEEDS)
            and audit.get("source_order") == list(SOURCES), "accepted development coverage differs")
    require(development_audit.stat().st_size == accepted["development_audit"]["size_bytes"], "development audit copy size differs")
    globals_, source_rows = {}, {}
    for raw in read_csv(conditions):
        arm, seed = raw["arm"], int(raw["model_seed"]) if raw["model_seed"] else None
        if arm not in ("persistence", "lstm") or raw["partition"] != "selection" or int(raw["glucose_cut"]) != 54 or raw["mode"] != "refitted":
            continue
        key = arm, seed, "selection", int(raw["horizon_minutes"])
        require(key not in globals_ and float(raw["nominal_fapd"]) == 0.5, "duplicate/wrong development member condition")
        globals_[key] = numeric_row(raw, global_row=True)
    compositions = read_json(composition_summaries)
    wanted_comp = {("composition", s, p, h) for s in SEEDS for p in ("calibration", "selection") for h in HORIZONS}
    require(len(compositions) == 20, "all twenty accepted composition summaries are required")
    for index, raw in enumerate(compositions):
        key = raw["arm"], raw["model_seed"], raw["partition"], raw["horizon_minutes"]
        require(key in wanted_comp and key not in globals_ and raw["glucose_cut"] == 54 and raw["mode"] == "frozen_composition"
                and raw["global_summary_status"] == "reconstructed_existing_estimator", "composition identity/estimator differs")
        globals_[key] = {**numeric_row(raw, global_row=True), "source": str(composition_summaries.resolve()), "json_pointer": f"/{index}"}
    wanted_globals = wanted_comp | {(a, s, "selection", h) for a, s in IDENTITIES if a != "composition" for h in HORIZONS}
    require(set(globals_) == wanted_globals, "development five-seed/shared-member coverage differs")
    for raw in read_csv(sources):
        arm, seed, role = raw["arm"], int(raw["model_seed"]) if raw["model_seed"] else None, raw["partition"]
        if arm not in ARMS or int(raw["glucose_cut"]) != 54:
            continue
        if (arm == "composition" and raw["mode"] != "frozen_composition") or (arm != "composition" and (role != "selection" or raw["mode"] != "refitted")):
            continue
        key = arm, seed, role, int(raw["horizon_minutes"]), raw["source_id"]
        require(key[:4] in wanted_globals and key[4] in SOURCES and key not in source_rows
                and float(raw["nominal_fapd"]) == 0.5, "development source identity/budget differs")
        source_rows[key] = numeric_row(raw, global_row=False)
    require(set(source_rows) == {(*k, source) for k in wanted_globals for source in SOURCES}, "development source grid is incomplete")
    for key, aggregate in globals_.items():
        rows = [source_rows[*key, source] for source in SOURCES]
        near(aggregate[SENSITIVITY], np.mean([r["fixed_sensitivity"] for r in rows]), "source-macro sensitivity")
        near(aggregate[BURDEN], np.mean([r[SOURCE_BURDEN] for r in rows]), "source-macro burden")
        for field in ADDITIVE:
            near(aggregate[field], sum(r[field] for r in rows), field)
        if key[2] == "selection":
            reference = globals_["persistence", None, "selection", key[3]]
            for field in TRUTH:
                near(aggregate[field], reference[field], f"fixed population {field}")
            for source in SOURCES:
                for field in TRUTH:
                    near(source_rows[*key, source][field], source_rows["persistence", None, "selection", key[3], source][field], f"source population {field}")
    mappings, paths = [], [conditions, sources, composition_summaries, development_acceptance, development_audit]
    for seed in SEEDS:
        path = mapping_dir / f"seed{seed}-development-mapping-audit.json"
        mapping = read_json(path)
        remote = [b for b in audit["input_bindings"] if b["path"].endswith(f"/composition-seed{seed}-r1/development-mapping-audit.json")]
        require(len(remote) == 1 and remote[0]["size_bytes"] == path.stat().st_size, "mapping differs from accepted source binding")
        require(mapping.get("scope") == "development_only_mapping_diagnostic_no_threshold_refit"
                and mapping.get("equivalence_claim") == "not_inferred_from_aggregate_parity"
                and set(mapping["horizons"]) == {"30", "60"}, "mapping diagnostic scope differs")
        for h in HORIZONS:
            entry = mapping["horizons"][str(h)]
            require(entry["threshold_rule"] == "saved_step_threshold_and_calibration_affine_counterpart"
                    and set(entry["partitions"]) == {"calibration", "selection"}, "mapping threshold/role coverage differs")
            number(entry["step_threshold"], "saved step threshold")
            number(entry["linear_threshold"], "affine counterpart threshold")
            for role, raw in entry["partitions"].items():
                require(type(raw["rows"]) is int and raw["rows"] > 0, "mapping decision population is empty")
                for field in MAP_COUNTS:
                    require(type(raw[field]) is int and raw[field] >= 0, f"invalid mapping count: {field}")
                require(number(raw["score_max_abs_difference"], "mapping score difference") >= 0,
                        "mapping score difference is invalid")
                mappings.append({"model_seed": seed, "horizon_minutes": h, "partition": role, **raw,
                                 "source": str(path.resolve()), "json_pointer": f"/horizons/{h}/partitions/{role}",
                                 "accepted_remote_binding": remote[0]})
        paths.append(path)
    return {"globals": globals_, "sources": source_rows, "mappings": mappings,
            "input_bindings": [binding(p) for p in paths],
            "coverage": {"member_fits": 6, "derived_recipes": 5, "composition_summaries": 20,
                         "development_global_objects": len(globals_), "source_objects": len(source_rows), "mapping_objects": len(mappings)}}


def interval(row, *, may_be_undefined):
    point = number(row["estimate"], "external point", undefined=may_be_undefined)
    low, high = number(row["ci_low"], "external CI", undefined=True), number(row["ci_high"], "external CI", undefined=True)
    require((low is None) == (high is None), "external CI is half missing")
    if low is not None:
        require(point is not None and low <= high, "external percentile interval is reversed/undefined")
    return point, low, high


def load_external(path, partition):
    report = read_json(path)
    require(report.get("schema_version") == "1.0" and report.get("status") == "complete"
            and report.get("revision_contract") == CONTRACT and report.get("freeze_id")
            and report.get("partition") == partition and report.get("source_id") == COHORTS[partition]
            and report.get("external_already_exposed") is True and report.get("fitted_on_external") is False
            and report.get("prediction_or_glucose_records_read") is False and report.get("nominal_fapd") == 0.5
            and report.get("glucose_cuts") == [54] and report.get("model_seeds") == list(SEEDS)
            and report.get("bootstrap_replicates") == 1000 and report.get("bootstrap_seed") == 20260810
            and report.get("actual_fit_counts") == {"persistence": 1, "lstm": 5}
            and report.get("derived_composition_recipe_count") == 5 and report.get("native_completed_producer_count") == 11
            and report.get("shared_control_training_seed") == 17
            and report.get("independent_unit") == "source_scoped_participant"
            and report.get("seed_variability_is_participant_uncertainty") is False
            and report.get("burden_population") == "all_base_participants_including_event_free"
            and report.get("model_seed_aggregation") == "mean_of_complete_LSTM_seed_set_with_one_shared_control_fit"
            and report.get("horizon_truth") == "each_horizon_own_Ebase_and_Dbase; exact_within_horizon",
            "conditional external identity/fitting/paired-inference contract differs")
    counts = report["counts"]
    require(len(counts) == 2 and {c["horizon_minutes"] for c in counts} == set(HORIZONS), "external horizon denominators differ")
    for row in counts:
        require(row["source_id"] == COHORTS[partition], "external denominator source differs")
        for field in ("participant_count", "event_positive_participant_count", "base_event_count", "patient_days"):
            require(type(row[field]) is int and row[field] >= 0, f"invalid external denominator: {field}")
        require(row["participant_count"] > 0 and row["event_positive_participant_count"] <= row["participant_count"]
                and number(row["observation_minutes"], "external exposure") > 0, "external population/exposure differs")
    require(set(report["metrics"]) == set(EXTERNAL_METRICS), "external metric coverage differs")
    positives = [row["event_positive_participant_count"] for row in counts]
    for metric, payload in report["metrics"].items():
        sensitivity = metric == "fixed_sensitivity"
        status = payload["interval_status"]
        expected_status = ("undefined_base_sensitivity" if not all(positives) else
                           "descriptive_sparse_event_positive_participants" if min(positives) < 2 else
                           "participant_percentile_bootstrap") if sensitivity else "participant_percentile_bootstrap"
        require(status == expected_status and payload["replicates"] == (0 if status == "undefined_base_sensitivity" else 1000)
                and payload["bootstrap_seed"] == 20260810 and payload["model_seeds"] == list(SEEDS)
                and payload["horizons"] == list(HORIZONS) and payload["source_order"] == [COHORTS[partition]]
                and payload["resampling_unit"] == "source_scoped_participant"
                and payload["aggregation"] == "equal_source_then_equal_horizon_and_model_seed"
                and payload["metric"] == metric
                and payload["population"] == ("event_positive_in_at_least_one_horizon" if sensitivity else "all_base_participants")
                and payload["stratification"] == ("source_by_horizon_positivity_pattern" if sensitivity else "source"),
                "external interval population/replicates/unit differs")
        require(set(payload["contrasts"]) == set(NAMES), "external arm/contrast coverage differs")
        for name, row in payload["contrasts"].items():
            if "_minus_" in name:
                left, right = name.removesuffix("_at_54").split("_minus_")
                coefficients = {(left, 54, 1), (right, 54, -1)}
            else:
                coefficients = {(name.removesuffix("_at_54"), 54, 1)}
            actual_coefficients = [(r["arm"], r["glucose_cut"], r["coefficient"]) for r in row["coefficients"]]
            require(set(actual_coefficients) == coefficients and len(actual_coefficients) == len(coefficients), "external contrast coefficients differ")
            point, low, high = interval(row, may_be_undefined=status == "undefined_base_sensitivity")
            if status != "participant_percentile_bootstrap":
                require(low is high is None, "sparse/undefined sensitivity must not acquire a confidence interval")
            else:
                require(low is not None, "native participant interval is missing")
            require(set(row["per_horizon"]) == {"30", "60"} and set(row["per_model_seed"]) == set(map(str, SEEDS)),
                    "external horizon/seed point coverage differs")
            horizon_points = []
            for horizon in HORIZONS:
                hrow = row["per_horizon"][str(horizon)]
                hpoint, hlow, hhigh = interval(hrow, may_be_undefined=status == "undefined_base_sensitivity")
                positive = next(c["event_positive_participant_count"] for c in counts if c["horizon_minutes"] == horizon)
                if sensitivity:
                    require((hpoint is None) == (positive == 0), "external undefined horizon must not become zero")
                    if positive < 2 or status == "undefined_base_sensitivity":
                        require(hlow is hhigh is None, "sparse/undefined horizon must not acquire a sensitivity CI")
                    else:
                        require(hlow is not None, "native defined-horizon interval is missing")
                if status == "participant_percentile_bootstrap":
                    require(hlow is not None, "native per-horizon interval is missing")
                horizon_points.append(hpoint)
            seed_values = [number(v, "external seed point", undefined=point is None) for v in row["per_model_seed"].values()]
            require(all(v is None for v in seed_values) if point is None else all(v is not None for v in seed_values),
                    "undefined joint point must not be averaged over a reduced seed/horizon set")
            if point is not None:
                require(all(v is not None for v in horizon_points), "defined external joint point has undefined horizon")
                near(point, np.mean(horizon_points), "external equal-horizon point")
                near(point, np.mean(seed_values), "external equal-seed point")
        for left, right in COMPARISONS:
            for horizon in (None, *map(str, HORIZONS)):
                def estimate(name):
                    row = payload["contrasts"][name]
                    return row["estimate"] if horizon is None else row["per_horizon"][horizon]["estimate"]
                a, b, delta = estimate(f"{left}_at_54"), estimate(f"{right}_at_54"), estimate(f"{left}_minus_{right}_at_54")
                if a is not None and b is not None:
                    near(delta, a - b, "native paired contrast point")
    return {"report": report, "input_binding": binding(path)}


def source_contrasts(data, horizon):
    records = []
    for source in SOURCES:
        for member in ("persistence", "lstm"):
            for seed in SEEDS:
                comp = data["sources"]["composition", seed, "selection", horizon, source]
                reference = data["sources"][member, None if member == "persistence" else seed, "selection", horizon, source]
                records.append({"source_id": source, "horizon_minutes": horizon, "subtracted_member": member,
                                "model_seed": seed, "value": comp["fixed_sensitivity"] - reference["fixed_sensitivity"],
                                "n": comp["participant_count"], "n_positive": comp["event_positive_participant_count"],
                                "composition": comp, "member": reference})
    return records


def validate_external_pair(externals):
    require(set(externals) == set(COHORTS), "both accepted exposed external cohorts are required for Figure 8")
    require(len({e["report"]["freeze_id"] for e in externals.values()}) == 1, "external recipe freezes differ")


def make_figure(data, externals):
    validate_external_pair(externals)
    G.set_style()
    fig = plt.figure(figsize=(180 * G.MM, 182 * G.MM))
    grid = fig.add_gridspec(4, 2, height_ratios=(0.85, 1.45, 0.8, 1.03),
                            left=0.165, right=0.985, top=0.94, bottom=0.09, hspace=0.80, wspace=0.58)
    axes = {"a": fig.add_subplot(grid[0, 0]), "b": fig.add_subplot(grid[0, 1]),
            "c": fig.add_subplot(grid[1, 0]), "d": fig.add_subplot(grid[1, 1]),
            "e": fig.add_subplot(grid[2, :]), "f": fig.add_subplot(grid[3, 0]), "g": fig.add_subplot(grid[3, 1])}
    panels = {}
    for letter, horizon in zip(("a", "b"), HORIZONS, strict=True):
        ax = axes[letter]
        rows = []
        for arm in ARMS:
            group = [data["globals"][arm, seed, "selection", horizon] for seed in (SEEDS if arm != "persistence" else (None,))]
            rows.extend(group)
            if len(group) == 5:
                ax.scatter([r[BURDEN] for r in group], [100 * r[SENSITIVITY] for r in group], marker=MARKERS[arm],
                           color=COLOURS[arm], s=9, alpha=0.45)
            ax.scatter(np.mean([r[BURDEN] for r in group]), 100 * np.mean([r[SENSITIVITY] for r in group]),
                       marker=MARKERS[arm], color=COLOURS[arm], s=27, zorder=4)
        ax.set(xlabel="Achieved unmatched alarms/day", ylabel="Fixed sensitivity (%)")
        ax.set_title(f"Development, {horizon} min", loc="left", fontsize=6.5, pad=5)
        panels[letter] = {"quantity": "native selection operating points; common nominal calibration budget is not equal achieved burden", "rows": rows}
    for letter, horizon in zip(("c", "d"), HORIZONS, strict=True):
        ax = axes[letter]
        records = source_contrasts(data, horizon)
        labels = []
        for index, source in enumerate(SOURCES):
            first = next(r for r in records if r["source_id"] == source)
            labels.append(f"{source} {first['n_positive']:.0f}/{first['n']:.0f}")
            for member, offset in (("persistence", -0.15), ("lstm", 0.15)):
                group = [r for r in records if r["source_id"] == source and r["subtracted_member"] == member]
                values = [100 * r["value"] for r in group]
                ax.scatter(values, index + offset + np.linspace(-0.04, 0.04, 5), marker=MARKERS[member],
                           s=5 + math.sqrt(first["n_positive"]), color=COLOURS[member], alpha=0.4)
                ax.scatter(np.mean(values), index + offset, marker=MARKERS[member], s=12 + 2 * math.sqrt(first["n_positive"]),
                           color=COLOURS[member], clip_on=False)
        ax.axvline(0, color=G.MUTED, linewidth=0.5, linestyle=":")
        ax.set(yticks=range(11), yticklabels=labels, ylim=(10.7, -0.7), xlabel="Rank mean − member (percentage points)")
        ax.set_title(f"Development sources, {horizon} min", loc="left", fontsize=6.5, pad=5)
        panels[letter] = {"quantity": "all eleven source contrasts versus both actual members; descriptive seeds only",
                          "source_label": "n_positive/n; marker area increases with sqrt(n_positive)", "rows": records}
    ax = axes["e"]
    ordered = {(r["model_seed"], r["horizon_minutes"], r["partition"]): r for r in data["mappings"]}
    matrix = np.array([[ordered[s, h, p]["online_eligible_crossing_disagreements"] for h in HORIZONS for p in ("calibration", "selection")] for s in SEEDS])
    ax.imshow(matrix, cmap="Greys", aspect="auto", vmin=0, vmax=max(float(matrix.max()), 1), alpha=0.25)
    for i, seed in enumerate(SEEDS):
        for j, (h, role) in enumerate((h, p) for h in HORIZONS for p in ("calibration", "selection")):
            row = ordered[seed, h, role]
            text = (f"{row['online_eligible_crossing_disagreements']} | {row['step_only_emitted_alarms']}/{row['linear_only_emitted_alarms']}"
                    f" | {row['base_events_with_matched_alarm_change']} | {row['base_events_with_detection_change']}")
            ax.text(j, i, text, ha="center", va="center", fontsize=5.5)
    ax.set(yticks=range(5), yticklabels=[str(s) for s in SEEDS], xticks=range(4),
           xticklabels=("30/calibration", "30/selection", "60/calibration", "60/selection"), ylabel="LSTM seed")
    ax.set_title("Development ranks: step ECDF vs interpolation", loc="left", fontsize=6.5, pad=5)
    panels["e"] = {"quantity": "eligible crossings | step-only/interpolation-only emissions | event matched-alarm identity changes | event detection changes",
                   "threshold_rule": "saved step threshold and its calibration affine counterpart; no diagnostic refitting", "rows": data["mappings"]}
    for letter, metric in (("f", "fixed_sensitivity"), ("g", "unmatched_alarms_per_patient_day")):
        ax = axes[letter]
        rows, labels, index = [], [], 0
        scale = 100 if metric == "fixed_sensitivity" else 1
        for partition, cohort in COHORTS.items():
            report = externals[partition]["report"]
            payload = report["metrics"][metric]
            for left, right in COMPARISONS:
                name = f"{left}_minus_{right}_at_54"
                point = payload["contrasts"][name]
                value, low, high = interval(point, may_be_undefined=True)
                if low is not None:
                    ax.plot([low * scale, high * scale], [index, index], color=COLOURS[left], linewidth=0.85)
                if value is not None:
                    ax.scatter(value * scale, index, color=COLOURS[left], marker=MARKERS[left], s=20, clip_on=False)
                else:
                    ax.annotate("N/A", (0, index), fontsize=5, xytext=(3, 0), textcoords="offset points")
                labels.append(f"{'HUPA' if partition == 'stress_hupa_ucm' else 'Loop'}: {'rank' if left == 'composition' else 'LSTM'} − {'current' if right == 'persistence' else 'LSTM'}")
                rows.append({"partition": partition, "contrast": name, "interval_status": payload["interval_status"],
                             "native_paired_result": point, "denominators": report["counts"],
                             "source": externals[partition]["input_binding"]["path"]})
                index += 1
        ax.set(yticks=range(6), yticklabels=labels, ylim=(5.6, -0.6),
               xlabel="Δ sensitivity (percentage points)" if metric == "fixed_sensitivity" else "Δ unmatched alarms/day")
        ax.axvline(0, color=G.MUTED, linewidth=0.5, linestyle=":")
        ax.axhline(2.5, color="#dddddd", linewidth=0.35)
        ax.set_title("Previously exposed external cohorts", loc="left", fontsize=6.5, pad=5)
        panels[letter] = {"quantity": metric, "rows": rows,
                          "uncertainty": "native paired 95% participant-percentile CIs where defined; prescribed1000 draws jointly across horizons/seeds; native sparse/undefined status retained; no new bootstrap"}
    for letter, ax in axes.items():
        G.panel_label(ax, letter, dx=-0.20 if letter != "e" else -0.075, dy=1.12)
    fig.legend(handles=[Line2D([], [], color=COLOURS[a], marker=MARKERS[a], linestyle="none", label=LABELS[a]) for a in ARMS],
               loc="upper center", ncol=3, fontsize=5.5)
    fig.text(0.165, 0.042, "Source marks: subtracted member; labels n+/n. Mapping cells: crossings | emissions | matched identity | detection.", fontsize=5)
    fig.text(0.165, 0.021, "Nominal calibration 0.5 FA/day · Development seed points are descriptive · External intervals use participants.", fontsize=5.5)
    receipt = {"panels": panels, "width_mm": 180, "height_mm": 182, "development_coverage": data["coverage"],
               "input_bindings": data["input_bindings"] + [externals[p]["input_binding"] for p in COHORTS],
               "caption_metadata": {"composition": "untrained arithmetic mean of two frozen calibration step-ECDF ranks; not an event probability",
                   "members": "five LSTM fits and one actual seed17 shared current-CGM fit; five derived recipes",
                   "source_variation": "all eleven sources retained including Shanghai T1DM and T1D-UOM; no participant CI from seed dispersion",
                   "mapping": "small score differences and unchanged detected-event totals do not imply equal emissions or matched-alarm identity",
                   "external": "already exposed cohorts, frozen base54 recipe, no external fitting/selection; each cohort inferred separately",
                   "external_counts": {p: externals[p]["report"]["counts"] for p in COHORTS},
                   "nominal_budget": "common nominal calibration budget does not establish equal achieved selection/external burden"}}
    return fig, receipt


def render(data, externals, output_dir, acceptance_path):
    validate_external_pair(externals)
    accepted = read_json(acceptance_path)
    all_bindings = data["input_bindings"] + [externals[p]["input_binding"] for p in COHORTS]
    require(accepted.get("owner") == "primary" and accepted.get("status") in ("accepted", "accepted_for_figure_use")
            and accepted.get("figure_input_bindings") == all_bindings, "primary acceptance must bind the exact complete Figure 8 inputs")
    require(all(binding(Path(b["path"])) == b for b in all_bindings), "accepted Figure 8 input changed")
    require(output_dir.resolve().is_relative_to(Path("/private/tmp")), "candidate output must stay under /private/tmp")
    require(not output_dir.exists() or not any(output_dir.iterdir()), "refusing to overwrite candidate output")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, receipt = make_figure(data, externals)
    receipt.update(primary_acceptance=binding(acceptance_path), state="candidate_for_primary_visual_and_caption_acceptance")
    with matplotlib.rc_context({"savefig.bbox": None, "savefig.pad_inches": 0.0}):
        fig.savefig(output_dir / "fig8-corrected-candidate.png", dpi=220)
    G.save(fig, str(output_dir / "fig8-corrected-candidate.pdf"), caption_words=200)
    (output_dir / "panel-data.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("conditions", "sources", "composition-summaries", "development-acceptance", "development-audit", "mapping-dir"):
        parser.add_argument(f"--{arg}", type=Path, required=True)
    parser.add_argument("--external-hupa", type=Path)
    parser.add_argument("--external-loop", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--development-only", action="store_true")
    parser.add_argument("--acceptance", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    data = load_development(args.conditions, args.sources, args.composition_summaries, args.development_acceptance,
                            args.development_audit, args.mapping_dir)
    if args.development_only:
        require(args.validate_only, "development-only mode cannot render an incomplete external figure")
        print(json.dumps({"development": data["coverage"], "external_read": False, "rendered": False,
                          "development_input_bindings": data["input_bindings"]}))
        return
    require(args.external_hupa is not None and args.external_loop is not None, "both external reports are required")
    externals = {"stress_hupa_ucm": load_external(args.external_hupa, "stress_hupa_ucm"),
                 "external_loop": load_external(args.external_loop, "external_loop")}
    validate_external_pair(externals)
    if args.validate_only:
        print(json.dumps({"development": data["coverage"], "external_cohorts": list(externals), "rendered": False,
                          "figure_input_bindings": data["input_bindings"] + [externals[p]["input_binding"] for p in COHORTS]}))
        return
    require(args.acceptance is not None and args.output_dir is not None, "rendering requires primary acceptance and an output directory")
    render(data, externals, args.output_dir, args.acceptance)


if __name__ == "__main__":
    main()
