#!/usr/bin/env python3
"""Render candidate Figures 3/4/S2/S4 from accepted fixed-frame aggregate exports.

This adapter validates and displays existing aggregate quantities. It never reads
participant records, fits a policy, matches an alarm or resamples a participant.
Outputs are candidates and cannot be written into the canonical manuscript tree.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code/scripts/figures"))
import glycofig as G  # noqa: E402

ARMS = ("persistence", "linear", "summary_logistic", "xgboost", "lstm", "tcn")
SEEDS = (17, 23, 42, 71, 101)
CUTS = (54, 60, 70, 80, 90, 100)
HORIZONS = (30, 60)
MODES = ("refitted", "gate_only")
SOURCES = (
    "AZT1D", "BrisT1D", "CTR3", "DCLP3", "DCLP5", "Flair", "IOBP2",
    "PEDAP", "ReplaceBG", "Shanghai T1DM", "T1D-UOM",
)
LABELS = {
    "persistence": "Current CGM", "linear": "Linear",
    "summary_logistic": "Summary logistic", "xgboost": "XGBoost",
    "lstm": "LSTM", "tcn": "TCN",
}
COLOURS = {
    "persistence": G.NOMINAL["rose"], "linear": G.NEUTRAL,
    "summary_logistic": G.NOMINAL["amber"], "xgboost": G.NOMINAL["teal"],
    "lstm": G.NOMINAL["blue"], "tcn": G.NOMINAL["olive"],
}
MARKERS = dict(zip(ARMS, ("o", "s", "^", "D", "v", "P"), strict=True))
LINESTYLES = dict(zip(ARMS, ("-", "--", ":", "-.", "-", "--"), strict=True))
SENSITIVITY = "equal_source_participant_macro_fixed_sensitivity"
BURDEN = "equal_source_participant_macro_unmatched_alarms_per_patient_day"
SOURCE_SENSITIVITY = "fixed_sensitivity"
SOURCE_BURDEN = "participant_macro_unmatched_alarms_per_patient_day"
COUNT_FIELDS = (
    "base_event_count", "reachable_event_count", "detected_event_count",
    "patient_days", "eligible_decision_count", "base_evaluable_decision_count",
    "eligible_evaluable_decision_count", "emitted_alarm_count",
    "delivered_alarm_count", "unevaluable_alarm_count", "matched_alarm_count",
    "unmatched_alarm_count", "late_detection_count", "missed_event_count",
    "participant_count", "event_positive_participant_count",
    *(f"coverage_{lead}_event_count" for lead in (5, 10, 15, 20)),
)
FRACTION_FIELDS = (
    "event_pooled_sensitivity", "reachable_event_fraction",
    *(f"coverage_{lead}_event_fraction" for lead in (5, 10, 15, 20)),
)
TRUTH_FIELDS = (
    "base_event_count", "reachable_event_count", "patient_days",
    "observation_minutes", "eligible_decision_count",
    "base_evaluable_decision_count", "eligible_evaluable_decision_count",
    "participant_count", "event_positive_participant_count",
)
CONTRAST_NAMES = (
    "lstm_minus_persistence_at_54", "lstm_minus_persistence_at_70",
    "interaction_70_minus_54",
)
Key = tuple[str, int | None, int, int, str]
Row = dict[str, Any]


class InputError(ValueError):
    """An export does not satisfy the accepted plotting contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InputError(message)


def _near(observed: float, expected: float, message: str) -> None:
    _require(math.isclose(observed, expected, rel_tol=1e-9, abs_tol=1e-10), message)


def _number(value: str, field: str, location: str) -> float:
    try:
        number = float(value)
    except (ValueError, TypeError) as exc:
        raise InputError(f"{location}: invalid {field}: {value!r}") from exc
    _require(math.isfinite(number), f"{location}: non-finite {field}")
    return number


def _read_rows(path: Path, *, source: bool) -> dict:
    metric_fields = (SOURCE_SENSITIVITY, SOURCE_BURDEN) if source else (SENSITIVITY, BURDEN)
    required = {
        "arm", "model_seed", "seed_identity", "horizon_minutes", "glucose_cut",
        "mode", "partition", "nominal_fapd", "observation_minutes", "ppv",
        *COUNT_FIELDS, *FRACTION_FIELDS, *metric_fields,
    }
    if source:
        required.add("source_id")
    result = {}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        _require(required <= set(reader.fieldnames or ()),
                 f"{path.name}: missing columns {sorted(required - set(reader.fieldnames or ()))}")
        for line, raw in enumerate(reader, 2):
            if raw["arm"] == "composition":
                continue  # Explicitly outside Figures 3/4, including its missing global metrics.
            location = f"{path.name}:{line}"
            arm = raw["arm"]
            _require(arm in ARMS, f"{location}: unexpected arm {arm!r}")
            if arm in ("lstm", "tcn"):
                seed = int(_number(raw["model_seed"], "model_seed", location))
                _require(str(seed) == raw["model_seed"] and seed in SEEDS
                         and raw["seed_identity"] == str(seed),
                         f"{location}: invalid real model seed identity")
            else:
                seed = None
                _require(raw["model_seed"] == "" and raw["seed_identity"] == "shared",
                         f"{location}: shared fit must have null model_seed and shared identity")
            horizon = _number(raw["horizon_minutes"], "horizon_minutes", location)
            cut = _number(raw["glucose_cut"], "glucose_cut", location)
            _require(horizon in HORIZONS and cut in CUTS and raw["mode"] in MODES,
                     f"{location}: unexpected condition")
            _require(raw["partition"] == "selection", f"{location}: not development selection")
            _near(_number(raw["nominal_fapd"], "nominal_fapd", location), 0.5,
                  f"{location}: nominal calibration budget is not 0.5")
            key: Key = (arm, seed, int(horizon), int(cut), raw["mode"])
            full_key = key + (raw["source_id"],) if source else key
            _require(full_key not in result, f"{location}: duplicate {'source' if source else 'condition'} key")
            row: Row = dict(raw)
            row.update(arm=arm, model_seed=seed, horizon_minutes=int(horizon), glucose_cut=int(cut))
            for field in COUNT_FIELDS:
                value = _number(raw[field], field, location)
                _require(value >= 0 and value.is_integer(), f"{location}: invalid count {field}")
                row[field] = int(value)
            for field in (*FRACTION_FIELDS, *metric_fields, "observation_minutes"):
                row[field] = _number(raw[field], field, location)
            for field in (*FRACTION_FIELDS, metric_fields[0]):
                _require(0 <= row[field] <= 1, f"{location}: fraction outside [0,1]: {field}")
            _require(row[metric_fields[1]] >= 0, f"{location}: negative unmatched burden")
            _validate_counts(row, location)
            if row["delivered_alarm_count"]:
                row["ppv"] = _number(raw["ppv"], "ppv", location)
                _near(row["ppv"], row["matched_alarm_count"] / row["delivered_alarm_count"],
                      f"{location}: ppv is not matched/delivered alarms")
            else:
                _require(raw["ppv"] == "", f"{location}: zero-delivery ppv must be undefined")
                row["ppv"] = None
            result[full_key] = row
    return result


def _validate_counts(row: Row, location: str) -> None:
    _require(row["patient_days"] > 0 and row["observation_minutes"] > 0
             and row["base_evaluable_decision_count"] > 0,
             f"{location}: non-positive base exposure")
    _require(0 < row["event_positive_participant_count"] <= row["participant_count"],
             f"{location}: invalid event-positive participant exposure")
    _require(row["eligible_evaluable_decision_count"] <= row["base_evaluable_decision_count"]
             and row["eligible_evaluable_decision_count"] <= row["eligible_decision_count"],
             f"{location}: eligible evaluable count exceeds its exposure")
    _require(row["detected_event_count"] <= row["reachable_event_count"] <= row["base_event_count"],
             f"{location}: detected/reachable/base event counts disagree")
    _require(row["delivered_alarm_count"] == row["matched_alarm_count"] + row["unmatched_alarm_count"]
             and row["emitted_alarm_count"] == row["delivered_alarm_count"] + row["unevaluable_alarm_count"]
             and row["matched_alarm_count"] == row["detected_event_count"]
             and row["missed_event_count"] == row["base_event_count"] - row["detected_event_count"],
             f"{location}: alarm/event count identities disagree")
    base = row["base_event_count"]
    _require(base > 0, f"{location}: empty base event population")
    for numerator, fraction in (
        ("detected_event_count", "event_pooled_sensitivity"),
        ("reachable_event_count", "reachable_event_fraction"),
        *((f"coverage_{lead}_event_count", f"coverage_{lead}_event_fraction")
          for lead in (5, 10, 15, 20)),
    ):
        _near(row[fraction], row[numerator] / base, f"{location}: {fraction} denominator mismatch")
    coverage = [row[f"coverage_{lead}_event_count"] for lead in (5, 10, 15, 20)]
    _require(row["detected_event_count"] >= coverage[0]
             and coverage == sorted(coverage, reverse=True),
             f"{location}: timely-event coverage counts are not nested")


@dataclass
class AggregateData:
    conditions: dict[Key, Row]
    sources: dict[tuple, Row]
    inference: dict[str, Any]
    input_paths: dict[str, str]

    def rows(self, arm: str, cut: int = 54, mode: str = "refitted",
             horizon: int | None = None, source: str | None = None) -> list[Row]:
        table = self.conditions if source is None else self.sources
        return [row for key, row in table.items()
                if key[0] == arm and key[3] == cut and key[4] == mode
                and (horizon is None or key[2] == horizon)
                and (source is None or key[5] == source)]

    def mean(self, arm: str, metric: str, **selection: Any) -> float:
        values = [row[metric] for row in self.rows(arm, **selection)]
        _require(bool(values) and all(value is not None for value in values),
                 f"undefined plotted metric {arm}/{metric}/{selection}")
        return float(np.mean(values))

    def seed_means(self, arm: str, metric: str, **selection: Any) -> list[float]:
        rows = self.rows(arm, **selection)
        seeds = SEEDS if arm in ("lstm", "tcn") else (None,)
        return [float(np.mean([row[metric] for row in rows if row["model_seed"] == seed]))
                for seed in seeds]


def load_inputs(conditions: Path, sources: Path, inference: Path) -> AggregateData:
    """Validate the complete accepted six-arm product, not an available subset."""
    condition_rows = _read_rows(conditions, source=False)
    source_rows = _read_rows(sources, source=True)
    fits = [(arm, seed) for arm in ARMS
            for seed in (SEEDS if arm in ("lstm", "tcn") else (None,))]
    expected = {(arm, seed, horizon, cut, mode)
                for (arm, seed), horizon, cut, mode
                in itertools.product(fits, HORIZONS, CUTS, MODES)}
    _require(set(condition_rows) == expected,
             f"condition grid incomplete/unexpected: expected 336, got {len(condition_rows)}; "
             f"missing {len(expected - set(condition_rows))}, extra {len(set(condition_rows) - expected)}")
    expected_sources = {key + (source,) for key, source in itertools.product(expected, SOURCES)}
    _require(set(source_rows) == expected_sources,
             f"source grid incomplete/unexpected: expected 3696, got {len(source_rows)}; "
             f"missing {len(expected_sources - set(source_rows))}, extra {len(set(source_rows) - expected_sources)}")
    truth: dict[tuple, tuple] = {}
    base_exposure: dict[str, tuple] = {}
    event_positive_n: dict[str, int] = {}
    for key, condition in condition_rows.items():
        rows = [source_rows[key + (source,)] for source in SOURCES]
        for field in COUNT_FIELDS:
            _require(sum(row[field] for row in rows) == condition[field],
                     f"{key}: source sum disagrees with condition {field}")
        _near(sum(row["observation_minutes"] for row in rows), condition["observation_minutes"],
              f"{key}: source observation exposure sum mismatch")
        for global_metric, local_metric in ((SENSITIVITY, SOURCE_SENSITIVITY), (BURDEN, SOURCE_BURDEN)):
            _near(condition[global_metric], float(np.mean([row[local_metric] for row in rows])),
                  f"{key}: {global_metric} is not the equal-source mean")
        for source, row in zip(SOURCES, rows, strict=True):
            truth_key = (source, key[2], key[3])
            observed_truth = tuple(row[field] for field in TRUTH_FIELDS)
            _require(truth.setdefault(truth_key, observed_truth) == observed_truth,
                     f"{key}/{source}: shared reachability/eligibility truth differs across fits or modes")
            exposure = tuple(row[field] for field in (
                "patient_days", "observation_minutes", "base_evaluable_decision_count", "participant_count"))
            _require(base_exposure.setdefault(source, exposure) == exposure,
                     f"{source}: base exposure changed across conditions")
            n = row["event_positive_participant_count"]
            _require(event_positive_n.setdefault(source, n) == n,
                     f"{source}: event-positive participant n differs across horizons/conditions")
    infer = json.loads(inference.read_text(encoding="utf-8"))
    data = AggregateData(condition_rows, source_rows, infer, {
        "conditions": str(conditions.resolve()), "sources": str(sources.resolve()),
        "inference": str(inference.resolve()),
    })
    _validate_inference(data)
    return data


def _validate_inference(data: AggregateData) -> None:
    infer = data.inference
    _require(infer.get("primary_pair") == ["lstm", "persistence"]
             and infer.get("primary_contrast") == "interaction_70_minus_54",
             "inference: wrong fixed pair or contrast")
    shared = infer.get("shared_control", {})
    _require(shared.get("unique_fit_count") == 1 and shared.get("model_seed") is None
             and shared.get("aliases_are_independent_fits") is False
             and shared.get("bootstrap_alias_seeds") == list(SEEDS)
             and infer.get("lstm_unique_fit_count") == 5,
             "inference: shared-control/real-fit identity mismatch")
    for name, metric, declared in (
        ("sensitivity", SENSITIVITY, "fixed_sensitivity"),
        ("unmatched_alarm_burden", BURDEN, "unmatched_alarms_per_patient_day"),
    ):
        block = infer.get(name, {})
        _require(block.get("metric") == declared and block.get("model_seeds") == list(SEEDS)
                 and block.get("horizons") == list(HORIZONS)
                 and block.get("source_order") == list(SOURCES)
                 and block.get("resampling_unit") == "source_scoped_participant"
                 and block.get("aggregation") == "equal_source_then_equal_horizon_and_model_seed",
                 f"inference: {name} estimator identity mismatch")
        contrasts = block.get("contrasts", {})
        for contrast_name in CONTRAST_NAMES:
            item = contrasts.get(contrast_name, {})
            at54 = data.mean("lstm", metric) - data.mean("persistence", metric)
            at70 = data.mean("lstm", metric, cut=70) - data.mean("persistence", metric, cut=70)
            expected = at54 if contrast_name.endswith("at_54") else (
                at70 if contrast_name.endswith("at_70") else at70 - at54)
            try:
                estimate, low, high = (float(item[field]) for field in ("estimate", "ci_low", "ci_high"))
            except (KeyError, TypeError, ValueError) as exc:
                raise InputError(f"inference: invalid {name}/{contrast_name}") from exc
            _require(all(math.isfinite(value) for value in (estimate, low, high)) and low <= estimate <= high,
                     f"inference: invalid interval {name}/{contrast_name}")
            _near(estimate, expected, f"inference: {name}/{contrast_name} estimate differs from export")
            actual_coefficients = {(row["arm"], row["glucose_cut"]): row["coefficient"]
                                   for row in item.get("coefficients", [])}
            coefficients = {("lstm", 54): 1, ("persistence", 54): -1}
            if contrast_name.endswith("at_70"):
                coefficients = {("lstm", 70): 1, ("persistence", 70): -1}
            elif contrast_name == "interaction_70_minus_54":
                coefficients = {("lstm", 70): 1, ("persistence", 70): -1,
                                ("lstm", 54): -1, ("persistence", 54): 1}
            _require(actual_coefficients == coefficients,
                     f"inference: coefficient direction mismatch {name}/{contrast_name}")


def _panel(ax, letter: str, title: str) -> None:
    ax.text(-0.02, 1.14, f"{letter})", transform=ax.transAxes,
            fontsize=7, fontweight="bold", ha="left", va="bottom")
    ax.set_title(title, loc="left", pad=5, fontsize=6.5)
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.35, zorder=0)


def _handle(arm: str) -> Line2D:
    return Line2D([], [], color=COLOURS[arm], marker=MARKERS[arm],
                  linestyle=LINESTYLES[arm], label=LABELS[arm])


def _identity(row: Row, *, source: bool = False) -> dict[str, Any]:
    fields = ("arm", "model_seed", "horizon_minutes", "glucose_cut", "mode")
    return {field: row[field] for field in (*fields, *(("source_id",) if source else ()))}


def _receipt(data: AggregateData, figure: int | str, height: float) -> dict[str, Any]:
    return {
        "schema_version": "1.0", "figure": figure, "status": "candidate_only",
        "dimensions_mm": [180, height], "caption_words_ceiling": 299,
        "source_order": list(SOURCES), "input_paths": data.input_paths,
        "validated_main_condition_count": len(data.conditions),
        "validated_main_source_row_count": len(data.sources),
        "nominal_calibration_budget": 0.5,
        "nominal_budget_is_not_achieved_selection_burden": True,
        "aggregation": "Within each arm, equal horizons then equal real model seeds; classical arms are single shared fits.",
        "inference": "Existing joint source-scoped-participant bootstrap intervals only; no bootstrap is performed here.",
        "seed_variation": "Points or ranges describe model seeds, not participant confidence intervals.",
        "prohibited_operations_performed": [], "panels": {},
    }


def _bind(receipt: dict, letter: str, data: AggregateData, metrics: list[str],
          rows: list[Row], values: Any, *, source: bool = False,
          aggregation: str = "Equal horizon and real-fit mean within each arm.") -> None:
    receipt["panels"][letter] = {
        "metrics": metrics, "aggregation": aggregation,
        "table": "sources" if source else "conditions",
        "input_row_count": len(rows),
        "input_keys": [_identity(row, source=source) for row in rows],
        "display_values": values,
    }


def figure3(data: AggregateData, output_dir: Path) -> Path:
    G.set_style()
    fig = plt.figure(figsize=(180 * G.MM, 165 * G.MM))
    receipt = _receipt(data, 3, 165)
    ordered = sorted(ARMS, key=lambda arm: data.mean(arm, SENSITIVITY), reverse=True)
    axes = [fig.add_axes([left, 0.65, 0.17, 0.255]) for left in (0.12, 0.345, 0.57, 0.795)]
    titles = ("Fixed sensitivity", "Achieved alarm burden", "Sensitivity by horizon", "Pooled alarm PPV")
    metrics = (SENSITIVITY, BURDEN, SENSITIVITY, "ppv")
    base_rows = [row for row in data.conditions.values() if row["glucose_cut"] == 54 and row["mode"] == "refitted"]
    for index, (ax, title, metric) in enumerate(zip(axes, titles, metrics, strict=True)):
        _panel(ax, "abcd"[index], title)
        values = {}
        for y, arm in enumerate(ordered):
            if index == 2:
                values[arm] = {}
                for horizon, offset, face in ((30, -0.12, COLOURS[arm]), (60, 0.12, "white")):
                    mean = data.mean(arm, metric, horizon=horizon)
                    ax.plot(mean, y + offset, marker=MARKERS[arm], color=COLOURS[arm],
                            markerfacecolor=face, markeredgewidth=0.7, linestyle="none", zorder=3)
                    values[arm][str(horizon)] = mean
            else:
                mean = data.mean(arm, metric)
                seeds = data.seed_means(arm, metric)
                ax.plot(mean, y, marker=MARKERS[arm], color=COLOURS[arm], linestyle="none", zorder=3)
                if arm in ("lstm", "tcn"):
                    ax.scatter(seeds, y + np.linspace(-0.13, 0.13, len(seeds)),
                               s=7, color=COLOURS[arm], marker=".", alpha=0.8, zorder=2)
                values[arm] = {"mean": mean, "horizon_mean_per_real_fit": seeds}
        ax.set(yticks=range(6), ylim=(5.6, -0.6))
        ax.set_yticklabels([LABELS[arm] for arm in ordered] if index == 0 else [])
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        if index in (0, 2):
            ax.set(xlim=(0, 1), xticks=(0, 0.5, 1), xlabel="Sensitivity")
        elif index == 1:
            ax.axvline(0.5, color=G.MUTED, linestyle=":", linewidth=0.7)
            ax.set(xlim=(0, max(0.7, max(data.mean(arm, BURDEN) for arm in ARMS) * 1.12)),
                   xlabel="Unmatched alarms /\npatient-calendar day\n(0.5 nominal reference)")
        else:
            ax.set(xlim=(0, max(0.35, max(data.mean(arm, "ppv") for arm in ARMS) * 1.1)),
                   xlabel="Matched / delivered alarms")
        _bind(receipt, "abcd"[index], data, [metric], base_rows, values)
    axes[2].set_xlabel("Sensitivity\nFilled: 30 min; open: 60 min", fontsize=5.5)

    ax_e = fig.add_axes([0.19, 0.14, 0.38, 0.38])
    _panel(ax_e, "e", "Source-resolved fixed sensitivity")
    heat = np.array([[data.mean(arm, SOURCE_SENSITIVITY, source=source) for arm in ARMS] for source in SOURCES])
    image = ax_e.imshow(heat, cmap="cividis", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    counts = {source: data.rows("persistence", source=source)[0]["event_positive_participant_count"] for source in SOURCES}
    ax_e.set(xticks=range(6), yticks=range(11))
    ax_e.set_xticklabels([LABELS[arm] for arm in ARMS], rotation=40, ha="right", rotation_mode="anchor")
    ax_e.set_yticklabels([f"{source} (n={counts[source]})" for source in SOURCES])
    ax_e.tick_params(length=0)
    ax_e.grid(False)
    for y, x in itertools.product(range(11), range(6)):
        ax_e.text(x, y, f"{heat[y, x]:.2f}", ha="center", va="center", fontsize=5,
                  color="white" if heat[y, x] < 0.5 else G.INK)
    cax = fig.add_axes([0.588, 0.14, 0.012, 0.38])
    fig.colorbar(image, cax=cax, ticks=[0, 0.5, 1]).set_label("Sensitivity", fontsize=5.5)
    source_base = [row for row in data.sources.values() if row["glucose_cut"] == 54 and row["mode"] == "refitted"]
    _bind(receipt, "e", data, [SOURCE_SENSITIVITY, "event_positive_participant_count"],
          source_base, {"arms": list(ARMS), "sources": list(SOURCES), "sensitivity": heat.tolist(),
                        "event_positive_participant_n": counts}, source=True)

    ax_f = fig.add_axes([0.72, 0.14, 0.245, 0.38])
    _panel(ax_f, "f", "Timely-event coverage")
    lead = (5, 10, 15, 20)
    values = {}
    for arm in ARMS:
        values[arm] = [data.mean(arm, f"coverage_{minute}_event_fraction") for minute in lead]
        ax_f.plot(lead, values[arm], color=COLOURS[arm], marker=MARKERS[arm], linestyle=LINESTYLES[arm])
    ax_f.set(xticks=lead, xlim=(4, 21), ylim=(0, 1), xlabel="Required lead time (min)", ylabel="Fraction of base events")
    ax_f.legend(handles=[_handle(arm) for arm in ARMS], loc="upper right",
                bbox_to_anchor=(1.02, 1.0), handlelength=1.6, labelspacing=0.35)
    _bind(receipt, "f", data, [f"coverage_{minute}_event_fraction" for minute in lead], base_rows,
          {"required_lead_minutes": lead, "event_pooled_coverage": values},
          aggregation="Event-pooled fraction in each condition, then equal horizons and real fits within arm.")
    return _save(fig, receipt, output_dir, 3)


def _cut_curve(ax, data: AggregateData, arm: str, metric: str, *, mode: str = "refitted",
               linestyle: str | None = None, seed_range: bool = False) -> list[float]:
    values = [data.mean(arm, metric, cut=cut, mode=mode) for cut in CUTS]
    ax.plot(CUTS, values, color=COLOURS[arm], marker=MARKERS[arm],
            linestyle=linestyle or LINESTYLES[arm], markerfacecolor="white" if mode == "gate_only" else COLOURS[arm])
    if seed_range and arm in ("lstm", "tcn"):
        seeds = np.array([data.seed_means(arm, metric, cut=cut, mode=mode) for cut in CUTS])
        ax.fill_between(CUTS, seeds.min(axis=1), seeds.max(axis=1), color=COLOURS[arm], alpha=0.12, linewidth=0)
    ax.set(xticks=CUTS, xlim=(52, 102), xlabel="Eligibility cut (mg/dL)")
    ax.tick_params(axis="x", labelrotation=45, pad=1)
    return values


def _seed_range_receipt(data: AggregateData, arm: str, metric: str, mode: str = "refitted") -> dict:
    values = [data.seed_means(arm, metric, cut=cut, mode=mode) for cut in CUTS]
    return {"horizon_mean_per_real_fit": values,
            "minimum": [min(row) for row in values], "maximum": [max(row) for row in values],
            "meaning": "Descriptive model-seed range, not a participant confidence interval."}


def figure4(data: AggregateData, output_dir: Path) -> Path:
    G.set_style()
    fig = plt.figure(figsize=(180 * G.MM, 170 * G.MM))
    receipt = _receipt(data, 4, 170)
    # Preserve approximately 30/25/47 mm plotting heights while removing the
    # unused inter-row bands in the first 185 mm candidate. Text stays 5–7 pt.
    top = [fig.add_axes([left, 0.755, 0.18, 0.175]) for left in (0.085, 0.32, 0.555, 0.79)]
    refitted = [row for row in data.conditions.values() if row["mode"] == "refitted"]
    pair_rows = [row for row in refitted if row["arm"] in ("lstm", "persistence")]
    for letter, ax, title, metric in (
        ("a", top[0], "Fixed-pair sensitivity", SENSITIVITY),
        ("b", top[1], "Achieved alarm burden", BURDEN),
    ):
        _panel(ax, letter, title)
        values = {arm: _cut_curve(ax, data, arm, metric, seed_range=True) for arm in ("lstm", "persistence")}
        ax.set_ylabel("Fixed sensitivity" if letter == "a" else "Unmatched alarms /\npatient-calendar day")
        ax.set_ylim(bottom=0)
        if letter == "a":
            ax.set_ylim(0, 1)
        else:
            ax.axhline(0.5, color=G.MUTED, linestyle=":", linewidth=0.7)
        _bind(receipt, letter, data, [metric], pair_rows,
              {"cuts": CUTS, "arm_means": values,
               "lstm_seed_range": _seed_range_receipt(data, "lstm", metric)})
    top[0].legend(handles=[_handle(arm) for arm in ("lstm", "persistence")],
                  loc="lower left", labelspacing=0.2, handlelength=1.6)
    for letter, ax, title, metric in (
        ("c", top[2], "Event reachability", "reachable_event_fraction"),
        ("d", top[3], "Evaluable decisions retained", "eligible_evaluable_decision_fraction"),
    ):
        _panel(ax, letter, title)
        values = {}
        for horizon, colour, marker, linestyle in ((30, G.INK, "o", "-"), (60, G.NEUTRAL, "s", "--")):
            rows = [data.rows("persistence", cut=cut, horizon=horizon)[0] for cut in CUTS]
            values[str(horizon)] = [row[metric] if letter == "c" else
                                    row["eligible_evaluable_decision_count"] / row["base_evaluable_decision_count"]
                                    for row in rows]
            ax.plot(CUTS, values[str(horizon)], color=colour, marker=marker, linestyle=linestyle,
                    markerfacecolor="white" if horizon == 60 else colour, label=f"{horizon} min")
        ax.set(xticks=CUTS, xlim=(52, 102), ylim=(0, 1.03), xlabel="Eligibility cut (mg/dL)",
               ylabel="Fraction of base events" if letter == "c" else "Eligible evaluable /\nbase evaluable decisions")
        ax.tick_params(axis="x", labelrotation=45, pad=1)
        ax.legend(loc="lower left", labelspacing=0.2, handlelength=1.6)
        _bind(receipt, letter, data,
              [metric] if letter == "c" else ["eligible_evaluable_decision_count", "base_evaluable_decision_count"],
              refitted, {"cuts": CUTS, "horizon_values": values},
              aggregation="Pooled event or pooled evaluable-decision ratio; truth asserted identical across all fits and modes.")

    contrast_labels = ("At 54 mg/dL", "At 70 mg/dL", "Interaction (70 − 54)")
    for letter, left, name, title, xlabel in (
        ("e", 0.18, "sensitivity", "Paired sensitivity contrasts", "LSTM − current CGM: fixed sensitivity"),
        ("f", 0.665, "unmatched_alarm_burden", "Paired alarm-burden contrasts", "LSTM − current CGM: unmatched alarms / day"),
    ):
        ax = fig.add_axes([left, 0.49, 0.305, 0.15])
        _panel(ax, letter, title)
        contrasts = data.inference[name]["contrasts"]
        values = []
        for y, key in enumerate(CONTRAST_NAMES):
            item = contrasts[key]
            estimate, low, high = (item[field] for field in ("estimate", "ci_low", "ci_high"))
            ax.errorbar(estimate, y, xerr=[[estimate - low], [high - estimate]],
                        fmt="o", color=G.INK, capsize=2, elinewidth=0.8, markersize=3)
            values.append({"contrast": key, "estimate": estimate, "ci_low": low, "ci_high": high})
        ax.axvline(0, color=G.MUTED, linestyle=":", linewidth=0.7)
        ax.set(yticks=range(3), ylim=(2.6, -0.6), xlabel=xlabel)
        ax.set_yticklabels(contrast_labels)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        receipt["panels"][letter] = {
            "table": "inference", "json_pointer": f"/{name}/contrasts",
            "display_values": values, "interval": "Existing joint participant bootstrap 95% CI",
            "resampling_unit": data.inference[name]["resampling_unit"],
            "coefficients": "LSTM minus current CGM; interaction = contrast at 70 minus contrast at 54.",
        }

    source_values = {cut: [data.mean("lstm", SOURCE_SENSITIVITY, cut=cut, source=source)
                           - data.mean("persistence", SOURCE_SENSITIVITY, cut=cut, source=source)
                           for source in SOURCES] for cut in (54, 70)}
    extent = max(0.1, math.ceil(max(abs(value) for values in source_values.values() for value in values) * 10) / 10)
    source_counts = {source: data.rows("persistence", source=source)[0]["event_positive_participant_count"] for source in SOURCES}
    for letter, left, cut in (("g", 0.16, 54), ("h", 0.39, 70)):
        ax = fig.add_axes([left, 0.105, 0.14, 0.275])
        _panel(ax, letter, f"Source contrast: {cut} mg/dL")
        ax.axvline(0, color=G.MUTED, linestyle=":", linewidth=0.7)
        ax.scatter(source_values[cut], range(11), color=G.NOMINAL["blue"], marker="o", s=10)
        ax.set(xlim=(-extent, extent), xticks=[-extent, 0, extent], yticks=range(11),
               ylim=(10.6, -0.6), xlabel="LSTM − current CGM\nfixed sensitivity")
        ax.set_yticklabels([f"{source} (n={source_counts[source]})" for source in SOURCES] if letter == "g" else [])
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        rows = [row for row in data.sources.values()
                if row["glucose_cut"] == cut and row["mode"] == "refitted" and row["arm"] in ("lstm", "persistence")]
        _bind(receipt, letter, data, [SOURCE_SENSITIVITY, "event_positive_participant_count"], rows,
              {"sources": SOURCES, "lstm_minus_current_cgm": source_values[cut],
               "event_positive_participant_n": source_counts, "shared_symmetric_x_limit": extent}, source=True)
    for letter, left, arm in (("i", 0.595, "lstm"), ("j", 0.81, "persistence")):
        ax = fig.add_axes([left, 0.105, 0.16, 0.275])
        _panel(ax, letter, f"{LABELS[arm]}: policy refitting")
        values = {mode: _cut_curve(ax, data, arm, SENSITIVITY, mode=mode,
                                   linestyle="-" if mode == "refitted" else "--", seed_range=True)
                  for mode in MODES}
        ax.set(ylim=(0, 1), ylabel="Fixed sensitivity")
        ax.legend(handles=[
            Line2D([], [], color=COLOURS[arm], linestyle="-", marker=MARKERS[arm], label="Refitted"),
            Line2D([], [], color=COLOURS[arm], linestyle="--", marker=MARKERS[arm], markerfacecolor="white", label="Gate only"),
        ], loc="lower left", labelspacing=0.25, handlelength=1.5)
        rows = [row for row in data.conditions.values() if row["arm"] == arm]
        _bind(receipt, letter, data, [SENSITIVITY], rows,
              {"cuts": CUTS, "mode_means": values,
               "seed_ranges": {mode: _seed_range_receipt(data, arm, SENSITIVITY, mode) for mode in MODES}})
    return _save(fig, receipt, output_dir, 4)


def figure_s2(data: AggregateData, output_dir: Path) -> Path:
    """Comparator-complete, horizon-resolved fixed-sensitivity cut trajectories."""
    G.set_style()
    fig = plt.figure(figsize=(180 * G.MM, 170 * G.MM))
    receipt = _receipt(data, "S2", 170)
    receipt["aggregation"] = "Each horizon separately; equal real model fits within arm, or one shared classical fit."
    receipt["inference"] = "Descriptive model-seed lines only; no participant confidence intervals."
    for arm_index, arm in enumerate(ARMS):
        for horizon_index, horizon in enumerate(HORIZONS):
            panel = 2 * arm_index + horizon_index
            row_index, column = divmod(panel, 4)
            ax = fig.add_axes([0.075 + 0.235 * column, 0.76 - 0.315 * row_index, 0.185, 0.18])
            letter = "abcdefghijkl"[panel]
            _panel(ax, letter, f"{LABELS[arm]}: {horizon} min")
            values = {}
            selected = []
            for mode in MODES:
                seeds = SEEDS if arm in ("lstm", "tcn") else (None,)
                per_seed = []
                for seed in seeds:
                    seed_values = []
                    for cut in CUTS:
                        key = (arm, seed, horizon, cut, mode)
                        condition = data.conditions[key]
                        selected.append(condition)
                        seed_values.append(condition[SENSITIVITY])
                    per_seed.append(seed_values)
                    if len(seeds) > 1:
                        ax.plot(CUTS, seed_values, color=COLOURS[arm],
                                linestyle="-" if mode == "refitted" else "--",
                                linewidth=0.35, alpha=0.25, zorder=1)
                means = np.mean(per_seed, axis=0).tolist()
                ax.plot(CUTS, means, color=COLOURS[arm], marker=MARKERS[arm],
                        linestyle="-" if mode == "refitted" else "--",
                        markerfacecolor=COLOURS[arm] if mode == "refitted" else "white", zorder=2)
                values[mode] = {"mean": means, "real_model_seeds": list(seeds),
                                "per_real_fit": per_seed}
            ax.set(xlim=(52, 102), ylim=(0, 1), xticks=CUTS, yticks=(0, 0.5, 1),
                   xlabel="Eligibility cut (mg/dL)", ylabel="Fixed sensitivity" if column % 2 == 0 else "")
            ax.tick_params(axis="x", labelrotation=45, pad=1)
            _bind(receipt, letter, data, [SENSITIVITY], selected,
                  {"arm": arm, "horizon_minutes": horizon, "cuts": CUTS, "modes": values},
                  aggregation="One horizon; equal real-model-seed mean within arm. Thin lines show actual real fits.")
    fig.legend(handles=[
        Line2D([], [], color=G.INK, marker="o", linestyle="-", label="Refitted"),
        Line2D([], [], color=G.INK, marker="o", markerfacecolor="white", linestyle="--", label="Gate only"),
        Line2D([], [], color=G.INK, linewidth=0.35, alpha=0.5, label="Thin lines: model seeds"),
    ], loc="lower center", bbox_to_anchor=(0.52, 0.025), ncol=3,
        columnspacing=2.0, handlelength=2.0)
    return _save(fig, receipt, output_dir, "S2")


def _source_seed_matrix(data: AggregateData, arm: str, horizon: int) -> tuple[np.ndarray, list[Row]]:
    rows = [data.sources[(arm, seed, horizon, 54, "refitted", source)]
            for source in SOURCES for seed in SEEDS]
    matrix = np.array([row[SOURCE_SENSITIVITY] for row in rows]).reshape(len(SOURCES), len(SEEDS))
    return matrix, rows


def figure_s4(data: AggregateData, output_dir: Path) -> Path:
    """Actual source/seed cells and descriptive five-seed ranges at base 54."""
    G.set_style()
    fig = plt.figure(figsize=(180 * G.MM, 185 * G.MM))
    receipt = _receipt(data, "S4", 185)
    receipt["aggregation"] = "Heatmap cells retain each source, real model seed and horizon without averaging. Ranges are max minus min over five real seeds."
    receipt["inference"] = "Model-seed dispersion only, not participant confidence intervals."
    counts = {source: data.rows("persistence", source=source)[0]["event_positive_participant_count"]
              for source in SOURCES}
    matrices = {}
    source_rows = {}
    image = None
    for index, (arm, horizon) in enumerate(itertools.product(("lstm", "tcn"), HORIZONS)):
        row_index, column = divmod(index, 2)
        ax = fig.add_axes([0.18 + 0.48 * column, 0.70 - 0.31 * row_index, 0.28, 0.23])
        letter = "abcd"[index]
        _panel(ax, letter, f"{LABELS[arm]}: {horizon} min")
        matrix, rows = _source_seed_matrix(data, arm, horizon)
        matrices[arm, horizon] = matrix
        source_rows[arm, horizon] = rows
        image = ax.imshow(matrix, vmin=0, vmax=1, cmap="cividis", aspect="auto", interpolation="nearest")
        ax.grid(False)
        ax.set(xticks=range(5), yticks=range(11), xlabel="Model seed")
        ax.set_xticklabels(SEEDS)
        ax.set_yticklabels([f"{source} (n={counts[source]})" for source in SOURCES])
        ax.tick_params(length=0)
        for y, x in itertools.product(range(11), range(5)):
            ax.text(x, y, f"{matrix[y, x]:.2f}", ha="center", va="center", fontsize=5,
                    color="white" if matrix[y, x] < 0.5 else G.INK)
        _bind(receipt, letter, data, [SOURCE_SENSITIVITY, "event_positive_participant_count"], rows,
              {"arm": arm, "horizon_minutes": horizon, "sources": SOURCES,
               "model_seeds": SEEDS, "fixed_sensitivity": matrix.tolist(),
               "event_positive_participant_n": counts}, source=True,
              aggregation="Exact existing source × real model seed cells, base 54/refitted; no averaging.")
    colour_axis = fig.add_axes([0.955, 0.39, 0.012, 0.54])
    fig.colorbar(image, cax=colour_axis, ticks=(0, 0.5, 1))
    colour_axis.set_title("Fixed\nsensitivity", fontsize=5.5, pad=5)
    ranges = {key: matrix.max(axis=1) - matrix.min(axis=1) for key, matrix in matrices.items()}
    limit = max(0.1, math.ceil(max(float(values.max()) for values in ranges.values()) * 10) / 10)
    for index, arm in enumerate(("lstm", "tcn")):
        ax = fig.add_axes([0.18 + 0.48 * index, 0.09, 0.28, 0.21])
        letter = "ef"[index]
        _panel(ax, letter, f"{LABELS[arm]}: five-seed range")
        values = {}
        for horizon, offset, marker, face in (
            (30, -0.12, "o", COLOURS[arm]), (60, 0.12, "s", "white"),
        ):
            values[str(horizon)] = ranges[arm, horizon].tolist()
            ax.scatter(ranges[arm, horizon], np.arange(11) + offset, color=COLOURS[arm],
                       marker=marker, facecolors=face, linewidths=0.7, s=9, zorder=3)
        ax.set(xlim=(0, limit), ylim=(10.6, -0.6), yticks=range(11),
               xlabel="Maximum − minimum fixed sensitivity")
        ax.set_yticklabels([f"{source} (n={counts[source]})" for source in SOURCES])
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        rows = source_rows[arm, 30] + source_rows[arm, 60]
        _bind(receipt, letter, data, [SOURCE_SENSITIVITY], rows,
              {"arm": arm, "sources": SOURCES, "model_seeds": SEEDS,
               "range_by_horizon": values, "shared_x_limit": limit,
               "event_positive_participant_n": counts}, source=True,
              aggregation="For each source/horizon, maximum minus minimum over all five actual model seeds.")
    fig.legend(handles=[
        Line2D([], [], color=G.INK, marker="o", linestyle="none", label="30 min"),
        Line2D([], [], color=G.INK, marker="s", markerfacecolor="white", linestyle="none", label="60 min"),
    ], loc="lower left", bbox_to_anchor=(0.17, 0.014), ncol=2, columnspacing=1.2,
        handletextpad=0.4)
    fig.text(0.64, 0.026, "Model-seed dispersion, not participant CI", ha="center", fontsize=5.5)
    return _save(fig, receipt, output_dir, "S4")


def _save(fig, receipt: dict, output_dir: Path, number: int | str) -> Path:
    stem = f"fig{number}-corrected-development-candidate"
    path = output_dir / f"{stem}.pdf"
    G.assert_compliant(fig, str(path), caption_words=299)
    G.save(fig, str(path), caption_words=299)
    (output_dir / f"{stem}.data-receipt.json").write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", required=True, type=Path)
    parser.add_argument("--sources", required=True, type=Path)
    parser.add_argument("--inference", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="New/empty task-owned candidate directory outside the canonical manu tree")
    parser.add_argument("--figure", choices=("3", "4", "all", "S2", "S4", "all-supp"), default="all")
    args = parser.parse_args(argv)
    try:
        data = load_inputs(args.conditions, args.sources, args.inference)
        output = args.output_dir.resolve()
        _require(not output.is_relative_to(PROJECT_ROOT / "manu"), "candidate output cannot be written into canonical manu tree")
        _require(not output.exists() or (output.is_dir() and not any(output.iterdir())),
                 "candidate output directory must be new or empty")
        output.mkdir(parents=True, exist_ok=True)
        paths = []
        if args.figure in ("3", "all"):
            paths.append(figure3(data, output))
        if args.figure in ("4", "all"):
            paths.append(figure4(data, output))
        if args.figure in ("S2", "all-supp"):
            paths.append(figure_s2(data, output))
        if args.figure in ("S4", "all-supp"):
            paths.append(figure_s4(data, output))
    except (InputError, OSError, json.JSONDecodeError) as exc:
        parser.exit(2, f"input error: {exc}\n")
    print(json.dumps({"status": "candidate_only", "pdfs": [str(path) for path in paths],
                      "validated_conditions": 336, "validated_source_rows": 3696}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
