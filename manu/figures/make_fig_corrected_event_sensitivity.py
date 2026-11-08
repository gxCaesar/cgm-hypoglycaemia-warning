#!/usr/bin/env python3
'Make fig corrected event sensitivity.'








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
VARIANTS = ("observed_only", "strict_merge_boundary")
DEFINITIONS = ("primary", *VARIANTS)
SOURCES = ("AZT1D", "BrisT1D", "CTR3", "DCLP3", "DCLP5", "Flair", "IOBP2",
           "PEDAP", "ReplaceBG", "Shanghai T1DM", "T1D-UOM")
BASELINES = ("persistence", "linear", "summary_logistic", "xgboost")
IDENTITIES = {(a, s) for a in ("lstm", "tcn", "composition") for s in SEEDS} | {(a, None) for a in BASELINES}
PAIR_IDENTITIES = (("persistence", None), *(("lstm", s) for s in SEEDS))
SENSITIVITY = "equal_source_participant_macro_fixed_sensitivity"
BURDEN = "equal_source_participant_macro_unmatched_alarms_per_patient_day"
COUNTS = ("participant_count", "event_positive_participant_count", "base_event_count", "reachable_event_count",
          "detected_event_count", "patient_days", "eligible_decision_count", "base_evaluable_decision_count",
          "eligible_evaluable_decision_count", "emitted_alarm_count", "delivered_alarm_count",
          "unevaluable_alarm_count", "matched_alarm_count", "unmatched_alarm_count", "missed_event_count")
ADDITIVE = (*COUNTS, "observation_minutes")
INVARIANT = ("participant_count", "patient_days", "observation_minutes", "eligible_decision_count",
             "base_evaluable_decision_count", "eligible_evaluable_decision_count", "emitted_alarm_count",
             "delivered_alarm_count", "unevaluable_alarm_count")
CATALOGUE_COUNTS = ("participant_count", "normalized_reading_count", "normalized_interpolated_reading_count",
                    "normalized_lower_censored_reading_count", "primary_full_catalogue_count",
                    "variant_full_catalogue_count", "primary_base_event_count", "variant_base_event_count",
                    "variant_events_without_original_base_leadin", "shared_base_onset_count",
                    "primary_only_base_onset_count", "variant_only_base_onset_count",
                    "primary_base_events_overlapped_by_variant", "primary_base_events_split_by_variant",
                    "variant_base_events_overlapping_primary", "primary_onset_confirmations_with_interpolated_reading",
                    "primary_onset_confirmations_with_nonobserved_reading", "primary_onset_confirmations_with_censored_reading")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    require(path.is_file() and not path.is_symlink(), f"missing/unsafe aggregate: {path}")
    return json.loads(path.read_text())


def number(value, name, *, undefined=False):
    missing = value is None or (type(value) in (float, int) and not math.isfinite(value))
    if missing:
        require(undefined, f"undefined/non-finite {name}")
        return None
    require(type(value) in (float, int) and math.isfinite(value), f"invalid {name}")
    return float(value)


def near(a, b, name):
    require(math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-9), f"aggregate mismatch: {name}")


def binding(path):
    stat = path.stat()
    return {"path": str(path.resolve()), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def expected_cases(config):
    require(config.get("revision_contract") == CONTRACT and config.get("partition") == "selection",
            "configuration must retain the prospective selection contract")
    runs = config["runs"]
    require(len(runs) == 19 and {(r["arm"], r["model_seed"]) for r in runs} == IDENTITIES,
            "configuration must contain exactly nineteen original fits/compositions")
    cases = set()
    for run in runs:
        wanted = {(h, c, 0.5) for h in HORIZONS for c in ((54,) if run["arm"] == "composition" else (54, 70))}
        actual = [(c["horizon_minutes"], c["glucose_cut"], c["nominal_fapd"]) for c in run["conditions"]]
        require(set(actual) == wanted and len(actual) == len(wanted), "approved condition matrix differs")
        cases.update((run["arm"], run["model_seed"], v, h, c) for v in VARIANTS for h, c, _ in actual)
    return cases


def load_inputs(config_path, aggregate_dir, primary_conditions):
    config = read_json(config_path)
    cases = expected_cases(config)
    status_path = aggregate_dir / "status.json"
    require(read_json(status_path).get("status") == "complete", "sensitivity output is not terminal complete")
    summary_path = aggregate_dir / "event-sensitivity-summary.json"
    catalogue_path = aggregate_dir / "catalogue-differences.json"
    summary = read_json(summary_path)
    require(summary.get("revision_contract") == CONTRACT and summary.get("partition") == "selection"
            and summary.get("interpretation") == "descriptive_variant_catalogues_not_same_events"
            and summary.get("raw_acquisition_provenance_established") is False
            and summary.get("alarms_scores_policies_refractory_changed") is False
            and summary.get("base_exposure_changed") is False, "sensitivity interpretation/invariance contract differs")
    catalogue = {}
    for index, raw in enumerate(read_json(catalogue_path)["rows"]):
        key = raw["source_id"], raw["variant"], raw["horizon_minutes"]
        require(key not in catalogue and key[0] in (*SOURCES, "pooled") and key[1] in VARIANTS and key[2] in HORIZONS,
                "catalogue identity is duplicated/unknown")
        for field in CATALOGUE_COUNTS:
            require(type(raw[field]) is int and raw[field] >= 0, f"invalid catalogue count: {field}")
        require(raw["primary_base_event_count"] == raw["shared_base_onset_count"] + raw["primary_only_base_onset_count"]
                and raw["variant_base_event_count"] == raw["shared_base_onset_count"] + raw["variant_only_base_onset_count"],
                "onset count decomposition differs")
        require(raw["variant_full_catalogue_count"] == raw["variant_base_event_count"] + raw["variant_events_without_original_base_leadin"]
                and raw["primary_full_catalogue_count"] >= raw["primary_base_event_count"]
                and raw["primary_base_events_split_by_variant"] <= raw["primary_base_events_overlapped_by_variant"] <= raw["primary_base_event_count"]
                and raw["variant_base_events_overlapping_primary"] <= raw["variant_base_event_count"], "catalogue bounds differ")
        catalogue[key] = {**raw, "source": str(catalogue_path.resolve()), "json_pointer": f"/rows/{index}"}
    require(set(catalogue) == {(s, v, h) for s in (*SOURCES, "pooled") for v in VARIANTS for h in HORIZONS},
            "catalogue source/variant/horizon coverage is incomplete")
    for variant in VARIANTS:
        for horizon in HORIZONS:
            pooled = catalogue["pooled", variant, horizon]
            for field in CATALOGUE_COUNTS:
                near(pooled[field], sum(catalogue[s, variant, horizon][field] for s in SOURCES), f"catalogue source sum {field}")
    for source in (*SOURCES, "pooled"):
        for horizon in HORIZONS:
            a, b = (catalogue[source, v, horizon] for v in VARIANTS)
            for field in ("participant_count", "primary_base_event_count", "primary_full_catalogue_count",
                          "normalized_reading_count", "normalized_interpolated_reading_count", "normalized_lower_censored_reading_count"):
                require(a[field] == b[field], f"primary catalogue/input quantity changed across variants: {field}")
    rows, truth = {}, {}
    for index, raw in enumerate(summary["rows"]):
        require(raw["scope"] in ("source", "equal_source_and_pooled"), "unknown sensitivity aggregate scope")
        source = raw["source_id"] if raw["scope"] == "source" else "pooled"
        require(source in SOURCES if raw["scope"] == "source" else raw.get("source_id") in (None, "pooled"),
                "source ID contradicts aggregate scope")
        case = raw["arm"], raw["model_seed"], raw["variant"], raw["horizon_minutes"], raw["glucose_cut"]
        key = (*case, source)
        require(case in cases and source in (*SOURCES, "pooled") and key not in rows and raw["nominal_fapd"] == 0.5,
                "sensitivity case/source is unexpected or duplicated")
        for field in COUNTS:
            require(type(raw[field]) is int and raw[field] >= 0, f"invalid native count: {field}")
        require(number(raw["observation_minutes"], "observation_minutes") > 0 and raw["patient_days"] > 0,
                "base exposure is missing")
        cat = catalogue[source, case[2], case[3]]
        require(raw["base_event_count"] == cat["variant_base_event_count"]
                and raw["participant_count"] == cat["participant_count"], "variant denominator differs from catalogue")
        require(raw["emitted_alarm_count"] == raw["delivered_alarm_count"] + raw["unevaluable_alarm_count"]
                and raw["delivered_alarm_count"] == raw["matched_alarm_count"] + raw["unmatched_alarm_count"]
                and raw["detected_event_count"] == raw["matched_alarm_count"]
                and raw["missed_event_count"] == raw["base_event_count"] - raw["detected_event_count"]
                and raw["detected_event_count"] <= raw["reachable_event_count"] <= raw["base_event_count"]
                and raw["event_positive_participant_count"] <= min(raw["participant_count"], raw["base_event_count"])
                and (raw["event_positive_participant_count"] == 0) == (raw["base_event_count"] == 0),
                "native event/alarm accounting differs")
        truth_key = case[2], case[3], source
        population = raw["participant_count"], raw["event_positive_participant_count"], raw["base_event_count"]
        require(truth.setdefault(truth_key, population) == population, "variant event-positive population differs across arms/cuts/seeds")
        field = "fixed_sensitivity" if source != "pooled" else SENSITIVITY
        if source == "pooled":
            require(raw["source_count"] == 11, "source-macro denominator changed")
            undefined = raw["undefined_sensitivity_sources"]
            require(isinstance(undefined, list) and len(undefined) == len(set(undefined))
                    and set(undefined) <= set(SOURCES), "undefined source identities differ")
            missing = bool(undefined)
            burden_field = BURDEN
        else:
            missing = raw["event_positive_participant_count"] == 0
            burden_field = "participant_macro_unmatched_alarms_per_patient_day"
        sensitivity = number(raw[field], field, undefined=missing)
        require((sensitivity is None) == missing and (sensitivity is None or 0 <= sensitivity <= 1),
                "empty strata must remain undefined, never zero or silently dropped")
        require(number(raw[burden_field], burden_field) >= 0, "invalid unmatched burden")
        rows[key] = {**raw, "source": str(summary_path.resolve()), "json_pointer": f"/rows/{index}"}
    require(set(rows) == {(*case, s) for case in cases for s in (*SOURCES, "pooled")}, "incomplete 19-run sensitivity matrix")
    for case in cases:
        pooled = rows[*case, "pooled"]
        source_rows = [rows[*case, s] for s in SOURCES]
        for field in ADDITIVE:
            near(pooled[field], sum(r[field] for r in source_rows), f"sensitivity source sum {field}")
        undefined = [s for s, r in zip(SOURCES, source_rows, strict=True) if r["event_positive_participant_count"] == 0]
        require(set(pooled["undefined_sensitivity_sources"]) == set(undefined), "empty-source provenance differs")
        if not undefined:
            near(pooled[SENSITIVITY], np.mean([r["fixed_sensitivity"] for r in source_rows]), "equal-source sensitivity")
        near(pooled[BURDEN], np.mean([r["participant_macro_unmatched_alarms_per_patient_day"] for r in source_rows]), "equal-source burden")
        for source in (*SOURCES, "pooled"):
            original = rows[*case, source]
            alternate = rows[case[0], case[1], VARIANTS[1] if case[2] == VARIANTS[0] else VARIANTS[0], case[3], case[4], source]
            for field in INVARIANT:
                near(original[field], alternate[field], f"unchanged emitted sequence/exposure aggregate {field}")
    primary = {}
    with primary_conditions.open(newline="") as stream:
        for line, raw in enumerate(csv.DictReader(stream), start=2):
            identity = raw["arm"], int(raw["model_seed"]) if raw["model_seed"] else None
            if identity not in PAIR_IDENTITIES or raw["mode"] != "refitted" or int(raw["glucose_cut"]) not in (54, 70):
                continue
            key = (*identity, int(raw["horizon_minutes"]), int(raw["glucose_cut"]))
            require(key not in primary and raw["partition"] == "selection" and float(raw["nominal_fapd"]) == 0.5,
                    "primary fixed-pair condition identity differs")
            undefined = json.loads(raw["undefined_sensitivity_sources"])
            require(isinstance(undefined, list) and len(set(undefined)) == len(undefined)
                    and set(undefined) <= set(SOURCES) and int(raw["source_count"]) == 11,
                    "primary defined-source denominator differs")
            item = {**raw, "undefined_sensitivity_sources": undefined}
            for field in (*ADDITIVE, SENSITIVITY, BURDEN):
                value = float(raw[field])
                item[field] = number(value, field, undefined=field == SENSITIVITY and bool(undefined))
            require((item[SENSITIVITY] is None) == bool(undefined), "primary undefined sensitivity must retain empty sources")
            item.update(source_count=11, source=str(primary_conditions.resolve()), csv_line=line)
            primary[key] = item
    require(set(primary) == {(*i, h, c) for i in PAIR_IDENTITIES for h in HORIZONS for c in (54, 70)}, "primary fixed-pair coverage differs")
    for key, original in primary.items():
        for variant in VARIANTS:
            current = rows[key[0], key[1], variant, key[2], key[3], "pooled"]
            require(original["base_event_count"] == catalogue["pooled", variant, key[2]]["primary_base_event_count"], "primary event catalogue differs")
            for field in INVARIANT:
                near(original[field], current[field], f"primary alarm/exposure parity {field}")
    return {"catalogue": catalogue, "rows": rows, "primary": primary,
            "input_bindings": [binding(p) for p in (config_path, status_path, summary_path, catalogue_path, primary_conditions)],
            "coverage": {"runs": 19, "primary_conditions": len(cases) // 2, "variant_conditions": len(cases),
                         "source_and_pooled_rows": len(rows), "catalogue_rows": len(catalogue)}}


def pair_value(data, arm, seed, definition, horizon, cut):
    return (data["primary"][arm, seed, horizon, cut] if definition == "primary"
            else data["rows"][arm, seed, definition, horizon, cut, "pooled"])


def contrasts(data, metric, horizon):
    records = []
    for definition in DEFINITIONS:
        for cut in (54, 70):
            control = pair_value(data, "persistence", None, definition, horizon, cut)
            for seed in SEEDS:
                candidate = pair_value(data, "lstm", seed, definition, horizon, cut)
                for field in ("base_event_count", "participant_count", "event_positive_participant_count", "patient_days", "observation_minutes"):
                    near(candidate[field], control[field], f"within-definition fixed pair {field}")
                require(candidate["undefined_sensitivity_sources"] == control["undefined_sensitivity_sources"], "within-definition undefined sources differ")
                value = None if candidate[metric] is None or control[metric] is None else candidate[metric] - control[metric]
                records.append({"definition": definition, "horizon_minutes": horizon, "glucose_cut": cut,
                                "model_seed": seed, "quantity": metric, "value": value,
                                "shared_control_actual_fit_count": 1, "base_event_count": candidate["base_event_count"],
                                "n": candidate["participant_count"], "n_positive": candidate["event_positive_participant_count"],
                                "source_count": 11, "defined_source_count": 11 - len(candidate["undefined_sensitivity_sources"]),
                                "undefined_sources": candidate["undefined_sensitivity_sources"],
                                "candidate": candidate, "control": control})
    return records


def make_figure(data):
    G.set_style()
    fig, axes = plt.subplots(3, 2, figsize=(180 * G.MM, 159 * G.MM),
                             gridspec_kw={"height_ratios": [1.18, 1, 1], "width_ratios": [1.02, 0.98]})
    fig.subplots_adjust(left=0.16, right=0.985, top=0.93, bottom=0.11, hspace=0.68, wspace=0.40)
    panels = {}
    for col, horizon in enumerate(HORIZONS):
        ax = axes[0, col]
        positions, labels, raw_rows = [], [], []
        for group, variant in enumerate(VARIANTS):
            cat = data["catalogue"]["pooled", variant, horizon]
            raw_rows.append(cat)
            for offset, which in enumerate(("primary", "variant")):
                y = group * 2.6 + offset
                shared, only = cat["shared_base_onset_count"], cat[f"{which}_only_base_onset_count"]
                ax.barh(y, shared, height=0.7, color="#777777", edgecolor=G.INK, linewidth=0.35)
                ax.barh(y, only, left=shared, height=0.7, color="white" if which == "primary" else "#dddddd",
                        edgecolor=G.INK, linewidth=0.35, hatch="///" if which == "primary" else "xx")
                ax.text(shared + only, y, f"  {shared + only:,}", va="center", fontsize=5.5)
                positions.append(y)
                labels.append("Primary" if which == "primary" else "Observed only" if group == 0 else "Strict <30 min")
        ax.set_yticks(positions, labels)
        ax.set_ylim(4.25, -0.65)
        ax.set_xlim(0, max(max(c["primary_base_event_count"], c["variant_base_event_count"]) for c in raw_rows) * 1.25)
        ax.set_xlabel("Base events (source-scoped onsets)")
        letter = chr(97 + col)
        G.panel_label(ax, letter, dx=-0.18, dy=1.10)
        ax.set_title(f"Catalogue accounting, {horizon} min", loc="left", fontsize=6.5, pad=5)
        panels[letter] = {"quantity": "shared and primary-only/variant-only onset counts; no shifted-event attribution", "rows": raw_rows}
        for row, metric in ((1, SENSITIVITY), (2, BURDEN)):
            ax = axes[row, col]
            records = contrasts(data, metric, horizon)
            scale = 100 if metric == SENSITIVITY else 1
            for index, definition in enumerate(DEFINITIONS):
                for j, cut in enumerate((54, 70)):
                    group = [r for r in records if r["definition"] == definition and r["glucose_cut"] == cut]
                    values = [r["value"] for r in group]
                    centre = index + (-0.11 if j == 0 else 0.11)
                    if all(v is not None for v in values):
                        displayed = np.asarray(values) * scale
                        ax.scatter(centre + np.linspace(-0.035, 0.035, 5), displayed, s=10,
                                   facecolors=G.NOMINAL["blue"] if j == 0 else "white", edgecolors=G.NOMINAL["blue"],
                                   linewidths=0.4, marker="v", alpha=0.55, clip_on=False)
                        ax.scatter(centre, np.mean(displayed), s=28, marker="v", linewidths=0.8,
                                   facecolors=G.NOMINAL["blue"] if j == 0 else "white", edgecolors=G.NOMINAL["blue"], clip_on=False)
                    elif j == 0:
                        missing = len(group[0]["undefined_sources"])
                        ax.annotate(
                            f"Not estimable\n{missing} source without events",
                            (index, 0), fontsize=5.0, ha="center",
                            xytext=(0, 6), textcoords="offset points",
                        )
            ax.axhline(0, color=G.MUTED, linestyle=":", linewidth=0.6)
            ax.set_xticks(range(3), ("Primary", "Observed only", "Strict <30 min"), rotation=18, ha="right")
            ax.set_xlim(-0.45, 2.45)
            ax.set_ylabel("Δ fixed sensitivity (percentage points)" if row == 1 else "Δ unmatched alarms / day")
            letter = chr(97 + row * 2 + col)
            G.panel_label(ax, letter, dx=-0.18, dy=1.10)
            ax.set_title(f"LSTM − current CGM, {horizon} min", loc="left", fontsize=6.5, pad=5)
            panels[letter] = {"quantity": metric, "display_scale": scale, "rows": records,
                              "fit_means": [{"definition": definition, "glucose_cut": cut, "fit_count": 5,
                                             "mean": float(np.mean([r["value"] for r in records
                                                  if r["definition"] == definition and r["glucose_cut"] == cut]))
                                             if all(r["value"] is not None for r in records
                                                  if r["definition"] == definition and r["glucose_cut"] == cut) else None}
                                            for definition in DEFINITIONS for cut in (54, 70)],
                              "inference": "descriptive five-seed point contrasts only; no CI or reduced-source estimand"}
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor="#777777", edgecolor=G.INK, label="Shared onset"),
                        Patch(facecolor="white", edgecolor=G.INK, hatch="///", label="Primary-only onset"),
                        Patch(facecolor="#dddddd", edgecolor=G.INK, hatch="xx", label="Variant-only onset")],
               loc="upper center", ncol=3, fontsize=5.5)
    fig.legend(handles=[Line2D([], [], color=G.NOMINAL["blue"], marker="v", linestyle="none", label=f"CGM ≥{cut} mg/dL",
                               markerfacecolor=G.NOMINAL["blue"] if cut == 54 else "white") for cut in (54, 70)],
               loc="lower center", bbox_to_anchor=(0.58, 0.044), ncol=2, fontsize=5.5)
    fig.text(0.08, 0.018, "Definition-specific catalogues · Descriptive seed points and fit means · Identical emitted alarms and base exposure.", fontsize=5.5)
    denominators = []
    for definition in DEFINITIONS:
        for horizon in HORIZONS:
            for cut in (54, 70):
                row = pair_value(data, "persistence", None, definition, horizon, cut)
                denominators.append({"definition": definition, "horizon_minutes": horizon, "glucose_cut": cut,
                                     "n": row["participant_count"], "n_positive": row["event_positive_participant_count"],
                                     "base_event_count": row["base_event_count"], "source_count": 11,
                                     "defined_source_count": 11 - len(row["undefined_sensitivity_sources"]),
                                     "undefined_sources": row["undefined_sensitivity_sources"]})
    source_denominators = [{"definition": variant, "horizon_minutes": horizon, "source_id": source,
                           "n": data["rows"]["persistence", None, variant, horizon, 54, source]["participant_count"],
                           "n_positive": data["rows"]["persistence", None, variant, horizon, 54, source]["event_positive_participant_count"],
                           "base_event_count": data["rows"]["persistence", None, variant, horizon, 54, source]["base_event_count"]}
                          for variant in VARIANTS for horizon in HORIZONS for source in SOURCES]
    receipt = {"coverage": data["coverage"], "input_bindings": data["input_bindings"], "panels": panels,
               "width_mm": 180, "height_mm": 159,
               "caption_metadata": {"independent_unit": "source-scoped participant",
                   "uncertainty": "none; seed points are descriptive and are not participant confidence intervals",
                   "estimand": "each definition has its own event catalogue and event-positive population",
                   "denominators": denominators, "source_denominators": source_denominators,
                   "undefined_rule": "source-macro sensitivity remains undefined if any source has no event-positive participants; no sources are dropped",
                   "invariance_proof": "raw alarm-sequence/exposure proof belongs to the primary acceptance and producer; this adapter checks aggregate counters only",
                   "upstream_acquisition_provenance_established": False,
                   "unavailable_interpretations": ["one-to-one shifted/merged/removed-event attribution", "same-event paired confidence intervals"]}}
    return fig, receipt


def render(data, output_dir, acceptance_path):
    acceptance = read_json(acceptance_path)
    require(acceptance.get("owner") == "primary" and acceptance.get("status") in ("accepted", "accepted_for_figure_use"),
            "primary scientific receipt acceptance is required before rendering real results")
    require(acceptance.get("figure_input_bindings") == data["input_bindings"],
            "primary acceptance must bind these exact figure inputs")
    for accepted in data["input_bindings"]:
        require(binding(Path(accepted["path"])) == accepted, "accepted figure input changed after validation")
    require(output_dir.resolve().is_relative_to(Path("/private/tmp")), "candidate output must remain under /private/tmp")
    require(not output_dir.exists() or not any(output_dir.iterdir()), "refusing to overwrite candidate output")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, receipt = make_figure(data)
    receipt["primary_acceptance"] = binding(acceptance_path)
    receipt["state"] = "candidate_for_primary_visual_and_caption_acceptance"
    with matplotlib.rc_context({"savefig.bbox": None, "savefig.pad_inches": 0.0}):
        fig.savefig(output_dir / "fig7-corrected-candidate.png", dpi=220)
    G.save(fig, str(output_dir / "fig7-corrected-candidate.pdf"), caption_words=200)
    (output_dir / "panel-data.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--aggregate-dir", type=Path, required=True)
    parser.add_argument("--primary-conditions", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--acceptance", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    data = load_inputs(args.config, args.aggregate_dir, args.primary_conditions)
    if args.validate_only:
        print(json.dumps({"validated_aggregate_contract": data["coverage"], "rendered": False}))
        return
    require(args.acceptance is not None and args.output_dir is not None, "rendering requires --acceptance and --output-dir")
    render(data, args.output_dir, args.acceptance)
    print(json.dumps({"output_dir": str(args.output_dir), "coverage": data["coverage"]}))


if __name__ == "__main__":
    main()
