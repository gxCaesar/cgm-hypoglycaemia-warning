#!/usr/bin/env python3
"""Candidate-only Figures 5/6 and Supplementary Figures 1/3 from accepted aggregates.

Only aggregate reports are read. The adapter performs no fitting, calibration,
bootstrap or event matching. Five deep-model seeds are descriptive fit-level replicates;
each classical control is one actual fit. Outputs are restricted to /private/tmp.
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

ARMS = ("persistence", "linear", "summary_logistic", "xgboost", "lstm", "tcn")
SEEDS = (17, 23, 42, 71, 101)
HEADS = ("hypo_30", "hypo_60", "hyper_30", "hyper_60")
FORECAST_ARMS = ("persistence", "linear", "lstm", "tcn")
IDENTITIES = tuple((a, s) for a in ARMS for s in (SEEDS if a in ("lstm", "tcn") else (None,)))
LABELS = {"persistence": "Current CGM", "linear": "Linear", "summary_logistic": "Summary logistic",
          "xgboost": "XGBoost", "lstm": "LSTM", "tcn": "TCN"}
COLOURS = {"persistence": G.NOMINAL["rose"], "linear": G.NEUTRAL,
           "summary_logistic": G.NOMINAL["amber"], "xgboost": G.NOMINAL["teal"],
           "lstm": G.NOMINAL["blue"], "tcn": G.NOMINAL["olive"]}
MARKERS = dict(zip(ARMS, ("o", "s", "^", "D", "v", "P"), strict=True))
STYLES = dict(zip(ARMS, ("-", "--", ":", "-.", "-", "--"), strict=True))
CONTRACT = "2026-11-04-fixed-frame"
SENSITIVITY = "equal_source_participant_macro_fixed_sensitivity"
BURDEN = "equal_source_participant_macro_unmatched_alarms_per_patient_day"
PARITY_FIELDS = (SENSITIVITY, BURDEN, "ppv", "base_event_count", "participant_count",
                 "patient_days", "observation_minutes", "matched_alarm_count",
                 "unmatched_alarm_count", "unevaluable_alarm_count")
POPULATION = {"selection": (153, 24574, {30: 3616, 60: 3674}),
              "calibration": (150, 23890, {30: 3691, 60: 3739})}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    require(path.is_file() and not path.is_symlink(), f"missing/unsafe aggregate: {path}")
    return json.loads(path.read_text())


def finite(value, label, *, undefined=False):
    if value is None or (isinstance(value, (int, float)) and not math.isfinite(value)):
        require(undefined, f"undefined/non-finite {label}")
        return None
    require(type(value) in (float, int) and math.isfinite(value), f"invalid {label}")
    return float(value)


def near(a, b, label):
    require(math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-10), f"aggregate mismatch: {label}")


def local_binding(path):
    stat = path.stat()
    return {"path": str(path.resolve()), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def diagnostic_metrics(diagnostic):
    if diagnostic["metrics"] is None:
        require(diagnostic["n"] == 0, "native null probability metrics require zero retained windows")
        return {"n": 0, "prevalence": None, "auprc": None, "auroc": None, "brier": None, "ece": None}
    return diagnostic["metrics"]


def load_inputs(training_dir, acceptance_path, conditions_path, alarms_path):
    acceptance = read_json(acceptance_path)
    bindings_path = training_dir / "source-bindings.json"
    bindings = read_json(bindings_path)["files"]
    require(acceptance.get("status") == "accepted_for_figure_use" and acceptance.get("owner") == "primary"
            and (acceptance.get("reports"), acceptance.get("fixed_frame_point_checks"),
                 acceptance.get("probability_head_partition_objects"), acceptance.get("continuous_horizon_partition_objects"))
            == (14, 280, 112, 48), "primary training aggregate acceptance differs")
    require(bindings == acceptance["files"], "source bindings differ from primary acceptance")
    expected = set(IDENTITIES)
    require(len(bindings) == 14 and {(b["arm"], b["model_seed"]) for b in bindings} == expected,
            "training reports must contain all fourteen unique fits")
    reports, paths, undefined = {}, {}, []
    for binding in bindings:
        identity = binding["arm"], binding["model_seed"]
        name = f"{identity[0]}-{identity[1] if identity[1] is not None else 'shared'}.json"
        require(binding["local_name"] == name and Path(binding["path"]).is_absolute(), "report identity/path differs")
        path = training_dir / name
        report = read_json(path)
        require(path.stat().st_size == binding["size_bytes"], f"report size differs from accepted binding: {name}")
        require(report.get("revision_contract") == CONTRACT and report.get("evaluator") == "fixed_frame"
                and report.get("status") == "observed_development_revision"
                and report.get("legacy_pooled_evaluator_reportable") is False
                and set(report["partitions"]) == set(POPULATION), f"report role/contract differs: {name}")
        op = report["operational_fixed_frame"]
        require(op["evaluation_partition"] == "selection" and op["threshold_partition"] == "calibration"
                and op["glucose_cut"] == 54, "native fixed-policy boundary differs")
        for role, (people, days, events) in POPULATION.items():
            part = report["partitions"][role]
            require(set(part["probability_diagnostics"]) == set(HEADS), "four-head probability coverage differs")
            require(set(part["fixed_frame"]) == {"30", "60"}, "fixed-frame horizon coverage differs")
            for horizon in (30, 60):
                frames = part["fixed_frame"][str(horizon)]
                require(set(frames) == {"0.5", "1.0"}, "actual two-budget policy coverage differs")
                for budget, frame in frames.items():
                    require(frame["participant_count"] == people and frame["patient_days"] == days
                            and frame["base_event_count"] == events[horizon] and frame["source_count"] == 11,
                            "fixed event/exposure denominator differs")
                    require(frame == op["horizons"][str(horizon)]["partitions"][role][budget],
                            "native duplicate fixed-frame objects differ")
                    near(frame["emitted_alarm_count"], frame["matched_alarm_count"] + frame["unmatched_alarm_count"]
                         + frame["unevaluable_alarm_count"], "emitted alarm partition")
                    for lead in (5, 10, 15, 20):
                        near(frame[f"coverage_{lead}_event_fraction"],
                             frame[f"coverage_{lead}_event_count"] / frame["base_event_count"], "coverage denominator")
                    for metric in (*PARITY_FIELDS, "emitted_alarm_count", "delivered_alarm_count"):
                        finite(frame[metric], metric)
                    for metric in ("matched_only_lead_minutes_mean", "matched_only_lead_minutes_median"):
                        finite(frame[metric], metric, undefined=frame["matched_alarm_count"] == 0)
            for head, diagnostic in part["probability_diagnostics"].items():
                target = f"severe_{head}_label" if head.startswith("hypo") else f"{head}_label"
                require(diagnostic["target_column"] == target and diagnostic["mask_column"] == f"{head}_training_mask",
                        "probability target/mask differs")
                metrics, n = diagnostic_metrics(diagnostic), diagnostic["n"]
                require(type(n) is int and n >= 0 and metrics["n"] == n, "probability n differs")
                prevalence = finite(metrics["prevalence"], "prevalence", undefined=n == 0)
                require(prevalence is None or 0 <= prevalence <= 1, "invalid prevalence")
                for metric in ("auprc", "auroc", "brier", "ece"):
                    value = finite(metrics[metric], metric, undefined=n == 0 or prevalence in (0, 1))
                    require(value is None or 0 <= value <= 1, f"invalid {metric}")
                    if value is None:
                        undefined.append([*identity, role, head, metric])
            forecast = part.get("continuous_forecast", {})
            require(set(forecast) == ({"30", "60"} if identity[0] in FORECAST_ARMS else set()),
                    "forecast is missing or invented for a non-forecast arm")
            for horizon, metrics in forecast.items():
                require(metrics["n"] == part["fixed_frame"][horizon]["0.5"]["base_evaluable_decision_count"],
                        "forecast n differs from native base-evaluable windows")
                for metric in ("rmse", "mae"):
                    require(finite(metrics[metric], metric) >= 0, f"invalid {metric}")
        reports[identity], paths[identity] = report, path.resolve()
    for role in POPULATION:
        for head in HEADS:
            populations = {(r["partitions"][role]["probability_diagnostics"][head]["n"],
                            diagnostic_metrics(r["partitions"][role]["probability_diagnostics"][head])["prevalence"])
                           for r in reports.values()}
            require(len(populations) == 1, "probability n/prevalence differs across actual fits")
    with conditions_path.open(newline="") as stream:
        raw = list(csv.DictReader(stream))
    conditions = {}
    for row in raw:
        if row["arm"] == "composition" or row["mode"] != "refitted" or row["glucose_cut"] != "54":
            continue
        key = row["arm"], int(row["model_seed"]) if row["model_seed"] else None, int(row["horizon_minutes"])
        require(key not in conditions and row["partition"] == "selection" and float(row["nominal_fapd"]) == 0.5,
                "condition identity/partition/nominal budget differs")
        conditions[key] = row
    require(set(conditions) == {(*i, h) for i in IDENTITIES for h in (30, 60)}, "selection condition coverage differs")
    checks = 0
    for (arm, seed, horizon), row in conditions.items():
        frame = reports[arm, seed]["partitions"]["selection"]["fixed_frame"][str(horizon)]["0.5"]
        for field in PARITY_FIELDS:
            near(row[field], frame[field], field)
            checks += 1
    alarms = {}
    for index, row in enumerate(read_json(alarms_path)):
        if row["scope"] != "pooled_alarms" or row["glucose_cut"] != 54 or row["mode"] != "refitted":
            continue
        if row["alarm_category"] not in ("matched", "unmatched"):
            continue
        key = row["arm"], row["model_seed"], row["horizon_minutes"], row["alarm_category"]
        require(key not in alarms and row["unit"] == "mg/dL" and row["source_id"] is None,
                "global alarm summary identity differs")
        frame = reports[key[:2]]["partitions"]["selection"]["fixed_frame"][str(key[2])]["0.5"]
        require(row["alarm_count"] == frame[f"{key[3]}_alarm_count"], "global alarm count differs")
        finite(row["median"], "global alarm median", undefined=row["alarm_count"] == 0)
        alarms[key] = {**row, "source_json_pointer": f"/{index}/median"}
    require(set(alarms) == {(*i, h, c) for i in IDENTITIES for h in (30, 60) for c in ("matched", "unmatched")},
            "global matched/unmatched alarm coverage differs")
    return {"reports": reports, "paths": paths, "alarms": alarms, "alarm_source": alarms_path.resolve(), "undefined": undefined,
            "coverage": {"reports": 14, "probability_head_partition_objects": 112,
                         "continuous_horizon_partition_objects": 48, "fixed_frame_point_checks": checks},
            "bindings": bindings, "local_inputs": [local_binding(p) for p in
                [acceptance_path, bindings_path, conditions_path, alarms_path, *paths.values()]]}


def record(data, identity, quantity, value, *, horizon, n, series, pointer, **extra):
    return {"arm": identity[0], "model_seed": identity[1], "partition": "selection",
            "horizon_minutes": horizon, "quantity": quantity, "series": str(series),
            "value": finite(value, quantity, undefined=True), "n": n,
            "source": str(data["paths"][identity]), "json_pointer": pointer, **extra}


def fixed_rows(data, metric, horizon, *, budget="0.5", series=None):
    rows = []
    for identity in IDENTITIES:
        frame = data["reports"][identity]["partitions"]["selection"]["fixed_frame"][str(horizon)][budget]
        n, n_unit = frame["base_event_count"], "base_events"
        if metric.startswith("matched_only"):
            n, n_unit = frame["matched_alarm_count"], "matched_alarms"
        elif metric == BURDEN:
            n, n_unit = frame["patient_days"], "participant_calendar_days"
        elif metric == SENSITIVITY:
            n, n_unit = frame["event_positive_participant_count"], "event_positive_participants_within_11_sources"
        elif metric.endswith("_alarm_count"):
            n, n_unit = frame["participant_count"], "base_participants"
        rows.append(record(data, identity, metric, frame[metric], horizon=horizon,
                           n=n, n_unit=n_unit, series=series or budget,
                           pointer=f"/partitions/selection/fixed_frame/{horizon}/{budget}/{metric}",
                           nominal_fapd=float(budget), participant_count=frame["participant_count"],
                           patient_days=frame["patient_days"], observation_minutes=frame["observation_minutes"],
                           base_event_count=frame["base_event_count"], matched_alarm_count=frame["matched_alarm_count"]))
    return rows


def probability_rows(data, metric, head, *, series):
    rows = []
    for identity in IDENTITIES:
        diag = data["reports"][identity]["partitions"]["selection"]["probability_diagnostics"][head]
        metrics = diagnostic_metrics(diag)
        pointer = f"/partitions/selection/probability_diagnostics/{head}/metrics"
        rows.append(record(data, identity, metric, metrics[metric], horizon=int(head.split("_")[1]),
                           n=diag["n"], series=series, pointer=pointer if diag["metrics"] is None else f"{pointer}/{metric}",
                           head=head, n_unit="windows", prevalence=metrics["prevalence"],
                           score_status="temperature_scaled_probability", temperature_fit_partition="calibration"))
    return rows


def title(ax, letter, text):
    G.panel_label(ax, letter, dx=-0.13, dy=1.09)
    ax.set_title(text, loc="left", fontsize=6.5, pad=5)


def point_panel(ax, rows, series, *, horizontal=False, arms=ARMS):
    offsets = np.linspace(-0.16, 0.16, len(series)) if len(series) > 1 else [0]
    for i, arm in enumerate(arms):
        for j, name in enumerate(series):
            group = [r for r in rows if r["arm"] == arm and r["series"] == str(name)]
            require(len(group) == (5 if arm in ("lstm", "tcn") else 1), "plot silently lost an actual fit")
            values = [r["value"] for r in group if r["value"] is not None]
            centre = i + offsets[j]
            fill = COLOURS[arm] if j == 0 else "white"
            if len(group) > 1 and values:
                locations = centre + np.linspace(-0.045, 0.045, len(values))
                ax.scatter(values if horizontal else locations, locations if horizontal else values,
                           s=8, marker=MARKERS[arm], color=COLOURS[arm], alpha=0.42, linewidths=0.3, clip_on=False)
            if len(values) == len(group):
                mean = float(np.mean(values))
                ax.scatter(mean if horizontal else centre, centre if horizontal else mean,
                           s=23, marker=MARKERS[arm], facecolors=fill, edgecolors=COLOURS[arm], linewidths=0.75, zorder=4, clip_on=False)
            else:
                ax.annotate("NA", (0, centre) if horizontal else (centre, 0), fontsize=5)
    labels = [LABELS[a] for a in arms]
    if horizontal:
        ax.set_yticks(range(len(arms)), labels)
        ax.set_ylim(len(arms) - 0.5, -0.5)
    else:
        ax.set_xticks(range(len(arms)), labels, rotation=38, ha="right")
        ax.set_xlim(-0.5, len(arms) - 0.5)
        ax.set_ylim(bottom=0)
    ax.grid(axis="x" if horizontal else "y", color="#dddddd", linewidth=0.35, zorder=0)


def series_legend(ax, labels):
    handles = [Line2D([], [], marker="o", color=G.INK, linestyle="none", markersize=3,
                      markerfacecolor=G.INK if i == 0 else "white", label=label) for i, label in enumerate(labels)]
    ax.legend(handles=handles, loc="upper right", ncol=len(labels), handletextpad=0.3, columnspacing=0.8, fontsize=5)


def arm_legend(fig):
    fig.legend(handles=[Line2D([], [], color=COLOURS[a], marker=MARKERS[a], linestyle=STYLES[a], label=LABELS[a])
                        for a in ARMS], loc="upper center", ncol=6, fontsize=5.5, handlelength=1.8,
               columnspacing=1.1, bbox_to_anchor=(0.53, 0.995))


def save(fig, out, name, panels, receipt):
    for panel in panels.values():
        groups = {}
        for row in panel["rows"]:
            groups.setdefault((row["arm"], row["series"], row["quantity"]), []).append(row["value"])
        panel["plotted_fit_means"] = [{"arm": k[0], "series": k[1], "quantity": k[2], "fit_count": len(v),
                                      "mean": float(np.mean(v)) if all(x is not None for x in v) else None}
                                     for k, v in groups.items()]
    fig.canvas.draw()
    with matplotlib.rc_context({"savefig.bbox": None, "savefig.pad_inches": 0.0}):
        fig.savefig(out / f"{name}.png", dpi=220)
    receipt[name] = {"width_mm": 180, "height_mm": fig.get_figheight() / G.MM, "panels": panels,
                     "encoding": "large marks are arithmetic means of fit-level metrics; small deep-model marks are five seeds; no CIs",
                     "state": "candidate_for_primary_acceptance"}
    G.save(fig, str(out / f"{name}.pdf"), caption_words=200)


def figure5(data, out, receipt):
    fig, axs = plt.subplots(4, 2, figsize=(180 * G.MM, 178 * G.MM),
                            gridspec_kw={"height_ratios": [1.15, 1, 1, 1.05]})
    fig.subplots_adjust(left=0.12, right=0.985, top=0.93, bottom=0.09, hspace=0.80, wspace=0.42)
    arm_legend(fig)
    panels = {}
    for column, horizon in enumerate((30, 60)):
        ax = axs[0, column]
        rows = []
        for lead in (5, 10, 15, 20):
            rows.extend(fixed_rows(data, f"coverage_{lead}_event_fraction", horizon, series=str(lead)))
        for arm in ARMS:
            means = []
            for lead in (5, 10, 15, 20):
                group = [r["value"] * 100 for r in rows if r["arm"] == arm and r["series"] == str(lead)]
                means.append(float(np.mean(group)))
                if len(group) == 5:
                    ax.scatter(np.full(5, lead), group, s=5, color=COLOURS[arm], marker=MARKERS[arm], alpha=0.35)
            ax.plot((5, 10, 15, 20), means, color=COLOURS[arm], marker=MARKERS[arm], linestyle=STYLES[arm])
        ax.set(xticks=(5, 10, 15, 20), ylim=(0, 100), xlabel="Minimum lead (min)", ylabel="Base events covered (%)")
        letter = chr(97 + column)
        title(ax, letter, f"Event coverage, {horizon} min")
        panels[letter] = {"quantity": "event-pooled coverage of fixed base events", "display_scale": 100, "rows": rows}
        ax = axs[1, column]
        rows = fixed_rows(data, "matched_only_lead_minutes_mean", horizon, series="Mean")
        rows += fixed_rows(data, "matched_only_lead_minutes_median", horizon, series="Median")
        point_panel(ax, rows, ("Mean", "Median"), horizontal=True)
        ax.set_xlabel("Matched-only lead per fit (min)")
        series_legend(ax, ("Mean", "Median"))
        letter = chr(99 + column)
        title(ax, letter, f"Alarm timing, {horizon} min")
        panels[letter] = {"quantity": "mean and median within each actual fit, never pooled seed quantiles", "rows": rows}
        ax = axs[2, column]
        rows = []
        for identity in IDENTITIES:
            for category in ("matched", "unmatched"):
                alarm = data["alarms"][*identity, horizon, category]
                rows.append(record(data, identity, "global_alarm_glucose_median", alarm["median"], horizon=horizon,
                                   n=alarm["alarm_count"], n_unit=f"{category}_alarms", series=category,
                                   pointer=alarm["source_json_pointer"], source=str(data["alarm_source"]),
                                   source_kind="accepted_alarm_glucose_summary", alarm_category=category))
        point_panel(ax, rows, ("matched", "unmatched"), horizontal=True)
        ax.set_xlabel("Median CGM per fit (mg/dL)")
        series_legend(ax, ("Matched", "Unmatched"))
        letter = chr(101 + column)
        title(ax, letter, f"CGM at alarms, {horizon} min")
        panels[letter] = {"quantity": "global median within actual condition; seed means are not pooled medians/IQRs", "rows": rows}
        ax = axs[3, column]
        rows = []
        for name in ("matched", "unmatched", "unevaluable"):
            rows += fixed_rows(data, f"{name}_alarm_count", horizon, series=name)
        left = np.zeros(6)
        for name, colour, hatch in zip(("matched", "unmatched", "unevaluable"), ("#555555", "#bbbbbb", "white"), ("", "///", "xx"), strict=True):
            values = [np.mean([r["value"] for r in rows if r["arm"] == a and r["series"] == name]) for a in ARMS]
            ax.barh(range(6), values, left=left, height=0.64, color=colour, hatch=hatch, linewidth=0.35, edgecolor=G.INK, label=name.capitalize())
            left += values
        for i, arm in enumerate(ARMS):
            if arm in ("lstm", "tcn"):
                totals = [data["reports"][arm, seed]["partitions"]["selection"]["fixed_frame"][str(horizon)]["0.5"]["emitted_alarm_count"]
                          for seed in SEEDS]
                ax.scatter(totals, i + np.linspace(-0.08, 0.08, 5), s=9, marker=MARKERS[arm],
                           color=COLOURS[arm], edgecolor="white", linewidths=0.25, zorder=5)
        ax.set(yticks=range(6), yticklabels=[LABELS[a] for a in ARMS], ylim=(5.5, -0.5), xlabel="Mean emitted alarm count")
        ax.set_xlim(left=0)
        letter = chr(103 + column)
        title(ax, letter, f"Alarm accounting, {horizon} min")
        panels[letter] = {"quantity": "stacked mean counts across five deep seeds or one actual control",
                          "rows": rows + fixed_rows(data, "emitted_alarm_count", horizon, series="emitted")}
    handles, labels = axs[3, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.55, 0.034), ncol=3,
               fontsize=5, handlelength=1.0, columnspacing=1.0)
    fig.text(0.12, 0.012, "Selection · CGM ≥54 mg/dL · nominal 0.5 FA/day. Large marks: fit means; small deep-model marks: five seeds.", fontsize=5.5)
    save(fig, out, "fig5-corrected-candidate", panels, receipt)


def figure6(data, out, receipt):
    fig, axs = plt.subplots(2, 3, figsize=(180 * G.MM, 140 * G.MM), gridspec_kw={"width_ratios": [1.08, 1.08, 0.92]})
    fig.subplots_adjust(left=0.075, right=0.985, top=0.92, bottom=0.16, hspace=0.68, wspace=0.5)
    fig.legend(handles=[Line2D([], [], marker="o", color=G.INK, linestyle="none", markersize=3,
                               markerfacecolor=G.INK if i == 0 else "white", label=f"{h} min")
                        for i, h in enumerate((30, 60))], loc="upper center", ncol=2, fontsize=5.5)
    panels = {}
    for ax, metric, letter in ((axs[0, 0], "auprc", "a"), (axs[0, 1], "auroc", "b"),
                               (axs[1, 0], "brier", "d"), (axs[1, 1], "ece", "e")):
        rows = sum((probability_rows(data, metric, f"hypo_{h}", series=str(h)) for h in (30, 60)), [])
        point_panel(ax, rows, ("30", "60"))
        title(ax, letter, f"Level-2 window {'Brier' if metric == 'brier' else metric.upper()}")
        panels[letter] = {"quantity": "calibration-fitted temperature-scaled probability diagnostic evaluated on selection windows", "rows": rows}
    for row, metric, letter in ((0, "rmse", "c"), (1, "mae", "f")):
        rows = []
        for identity in IDENTITIES:
            if identity[0] not in FORECAST_ARMS:
                continue
            for horizon in (30, 60):
                metrics = data["reports"][identity]["partitions"]["selection"]["continuous_forecast"][str(horizon)]
                rows.append(record(data, identity, metric, metrics[metric], horizon=horizon, n=metrics["n"], series=str(horizon),
                                   pointer=f"/partitions/selection/continuous_forecast/{horizon}/{metric}", n_unit="windows"))
        point_panel(axs[row, 2], rows, ("30", "60"), arms=FORECAST_ARMS)
        axs[row, 2].set_ylabel("mg/dL")
        title(axs[row, 2], letter, f"Forecast {metric.upper()}")
        panels[letter] = {"quantity": "native continuous glucose forecast error; no logistic/XGBoost forecast", "rows": rows}
    n_probability = data["reports"]["persistence", None]["partitions"]["selection"]["probability_diagnostics"]["hypo_30"]["n"]
    n_forecast = data["reports"]["persistence", None]["partitions"]["selection"]["continuous_forecast"]["30"]["n"]
    fig.text(0.075, 0.045, f"Selection windows · Level-2 n={n_probability:,}; forecast n={n_forecast:,.0f}. Filled: 30 min; open: 60 min.", fontsize=5.5)
    fig.text(0.075, 0.025, "Calibration-fitted temperatures · Selection diagnostics · Large marks: fit means; small deep-model marks: five seeds.", fontsize=5.5)
    save(fig, out, "fig6-corrected-candidate", panels, receipt)


def supplement1(data, out, receipt):
    fig, axs = plt.subplots(2, 2, figsize=(180 * G.MM, 133 * G.MM))
    fig.subplots_adjust(left=0.10, right=0.985, top=0.90, bottom=0.13, hspace=0.58, wspace=0.35)
    arm_legend(fig)
    panels = {}
    for row, metric in enumerate((SENSITIVITY, BURDEN)):
        for col, horizon in enumerate((30, 60)):
            ax = axs[row, col]
            rows = sum((fixed_rows(data, metric, horizon, budget=b) for b in ("0.5", "1.0")), [])
            for arm in ARMS:
                means = []
                for budget in (0.5, 1.0):
                    group = [r["value"] for r in rows if r["arm"] == arm and r["nominal_fapd"] == budget]
                    means.append(float(np.mean(group)))
                    if len(group) == 5:
                        ax.scatter(np.full(5, budget), group, s=7, marker=MARKERS[arm], color=COLOURS[arm], alpha=0.4)
                ax.plot((0.5, 1.0), means, marker=MARKERS[arm], color=COLOURS[arm], linestyle=STYLES[arm])
            ax.set_xticks((0.5, 1.0), ("0.5", "1.0"))
            ax.set_xlim(0.44, 1.06)
            ax.set_xlabel("Nominal calibration FA/day")
            ax.set_ylabel("Fixed sensitivity" if row == 0 else "Achieved unmatched alarms/day")
            if row == 0:
                ax.set_ylim(0, 1)
            letter = chr(97 + row * 2 + col)
            title(ax, letter, f"{'Sensitivity' if row == 0 else 'Achieved burden'}, {horizon} min")
            panels[letter] = {"quantity": "selection response at the two actual frozen calibration policies, not a dense sweep", "rows": rows}
    fig.text(0.10, 0.035, "153 participants · 24,574 participant-calendar days · 3,616 / 3,674 base events at 30 / 60 min · CGM ≥54 mg/dL.", fontsize=5.5)
    save(fig, out, "supp1-corrected-candidate", panels, receipt)


def supplement3(data, out, receipt):
    fig, axs = plt.subplots(4, 2, figsize=(180 * G.MM, 182 * G.MM), gridspec_kw={"width_ratios": [1.04, 0.96]})
    fig.subplots_adjust(left=0.085, right=0.985, top=0.94, bottom=0.09, hspace=1.02, wspace=0.35)
    fig.legend(handles=[Line2D([], [], marker="o", color=G.INK, linestyle="none", markersize=3,
                               markerfacecolor=G.INK if i == 0 else "white", label=label)
                        for i, label in enumerate(("AUPRC / Brier", "AUROC / ECE"))],
               loc="upper center", ncol=2, fontsize=5.5)
    panels = {}
    for row, head in enumerate(HEADS):
        diags = [data["reports"][i]["partitions"]["selection"]["probability_diagnostics"][head] for i in IDENTITIES]
        ns, prevalences = {d["n"] for d in diags}, {diagnostic_metrics(d)["prevalence"] for d in diags}
        require(len(ns) == len(prevalences) == 1, "head n/prevalence varies across fits; a single annotation would hide it")
        n, prevalence = next(iter(ns)), next(iter(prevalences))
        head_label = ("Level-2" if head.startswith("hypo") else "Hyperglycaemia") + f", {head.split('_')[1]} min"
        for col, metrics in enumerate((("auprc", "auroc"), ("brier", "ece"))):
            ax = axs[row, col]
            rows = sum((probability_rows(data, m, head, series=m) for m in metrics), [])
            point_panel(ax, rows, metrics)
            letter = chr(97 + row * 2 + col)
            prevalence_label = "undefined" if prevalence is None else f"{prevalence:.3%}"
            title(ax, letter, head_label if col == 0 else f"n={n:,}; prevalence={prevalence_label}")
            panels[letter] = {"quantity": "temperature-scaled probability diagnostic; native scalar summaries evaluated on selection", "head": head, "rows": rows}
    fig.text(0.085, 0.018, "Calibration-fitted temperatures · Selection diagnostics · Large marks: fit means; small deep-model marks: five seeds.", fontsize=5.5)
    save(fig, out, "supp3-corrected-candidate", panels, receipt)


def build(data, output_dir):
    require(output_dir.resolve().is_relative_to(Path("/private/tmp")), "candidate outputs must remain under /private/tmp")
    require(not output_dir.exists() or not any(output_dir.iterdir()), "refusing to overwrite candidate outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    G.set_style()
    receipt = {"coverage": data["coverage"], "source_bindings": data["bindings"], "local_inputs": data["local_inputs"],
               "undefined": data["undefined"], "render_scope": "selection_only_candidates",
               "notes": ["Remote mtime bindings describe producing files; local copy mtimes are recorded separately.",
                         "Probability diagnostics use calibration-fitted temperatures and are evaluated on selection; no new fitting is performed by this figure adapter.",
                         "Producer: code/src/glycoshift/revision_training.py calibration at410-453, probability diagnostics at674-702.",
                         "No native reliability bins are present; no reliability curve or pooled seed IQR is fabricated.",
                         "Figure 5 uses eight panels to separate timing, glucose and complete alarm accounting."]}
    for renderer in (figure5, figure6, supplement1, supplement3):
        renderer(data, output_dir, receipt)
    (output_dir / "panel-data.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--training-acceptance", type=Path, required=True)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--alarm-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    data = load_inputs(args.training_dir, args.training_acceptance, args.conditions, args.alarm_summary)
    result = build(data, args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir), "coverage": result["coverage"],
                      "figures": [k for k in result if k.endswith("candidate")]}))


if __name__ == "__main__":
    main()
