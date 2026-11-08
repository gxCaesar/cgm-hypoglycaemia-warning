#!/usr/bin/env python3
"""Build four result tables from accepted fixed-frame aggregate outputs only."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from statistics import fmean
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "manu/figures"))
import make_fig_corrected_development as D  # noqa: E402

PROBABILITY_METRICS = (
    "auprc",
    "auroc",
    "brier",
    "ece",
    "calibration_intercept",
    "calibration_slope",
)
PARITY_FIELDS = (
    D.SENSITIVITY,
    D.BURDEN,
    "ppv",
    "base_event_count",
    "participant_count",
    "patient_days",
    "observation_minutes",
    "matched_alarm_count",
    "unmatched_alarm_count",
    "unevaluable_alarm_count",
)
FILENAMES = (
    "tab_budget_sweep.tex",
    "tab_cut_curve.tex",
    "tab_quality.tex",
    "tab_per_source.tex",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> Any:
    require(
        path.is_file() and not path.is_symlink(), f"missing/unsafe aggregate: {path}"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def number(value: Any, field: str, *, undefined: bool = False) -> float | None:
    if value is None:
        require(undefined, f"undefined {field}")
        return None
    require(type(value) in (int, float) and math.isfinite(value), f"invalid {field}")
    return float(value)


def near(left: Any, right: Any, field: str) -> None:
    require(
        math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-10),
        f"aggregate mismatch: {field}",
    )


def load_training(
    training_dir: Path, acceptance_path: Path, development: D.AggregateData
) -> dict:
    acceptance = read_json(acceptance_path)
    bindings = read_json(training_dir / "source-bindings.json")["files"]
    require(
        acceptance.get("status") == "accepted_for_figure_use"
        and acceptance.get("owner") == "primary"
        and (acceptance.get("reports"), acceptance.get("fixed_frame_point_checks"))
        == (14, 280),
        "primary training acceptance differs",
    )
    require(
        bindings == acceptance["files"],
        "source bindings differ from primary acceptance",
    )
    identities = {
        (arm, seed)
        for arm in D.ARMS
        for seed in (D.SEEDS if arm in ("lstm", "tcn") else (None,))
    }
    require(
        len(bindings) == 14
        and {(row["arm"], row["model_seed"]) for row in bindings} == identities,
        "training reports must cover fourteen unique fits",
    )
    reports = {}
    checks = 0
    window_populations = {}
    for binding in bindings:
        identity = binding["arm"], binding["model_seed"]
        name = (
            f"{identity[0]}-{identity[1] if identity[1] is not None else 'shared'}.json"
        )
        require(binding["local_name"] == name, "training report fit identity differs")
        path = training_dir / name
        report = read_json(path)
        require(
            path.stat().st_size == binding["size_bytes"],
            f"accepted report size differs: {name}",
        )
        require(
            report.get("revision_contract") == "2026-11-04-fixed-frame"
            and report.get("evaluator") == "fixed_frame"
            and report.get("status") == "observed_development_revision"
            and report.get("legacy_pooled_evaluator_reportable") is False,
            f"report scientific role differs: {name}",
        )
        operation = report["operational_fixed_frame"]
        require(
            operation["evaluation_partition"] == "selection"
            and operation["threshold_partition"] == "calibration"
            and operation["glucose_cut"] == 54
            and operation["nominal_fapd"] == 0.5,
            "native policy/calibration boundary differs",
        )
        selection = report["partitions"]["selection"]
        require(
            set(selection["fixed_frame"]) == {"30", "60"},
            "native horizon coverage differs",
        )
        for horizon in D.HORIZONS:
            frames = selection["fixed_frame"][str(horizon)]
            require(set(frames) == {"0.5", "1.0"}, "actual two-budget coverage differs")
            baseline = development.conditions[(*identity, horizon, 54, "refitted")]
            for budget, frame in frames.items():
                require(frame["source_count"] == 11, "native source coverage differs")
                for field in (
                    "base_event_count",
                    "participant_count",
                    "patient_days",
                    "observation_minutes",
                ):
                    near(
                        frame[field],
                        baseline[field],
                        f"native {budget}/{horizon}/{field}",
                    )
                for field in (D.SENSITIVITY, D.BURDEN):
                    value = number(frame[field], field)
                    require(
                        value >= 0 and (field != D.SENSITIVITY or value <= 1),
                        f"invalid {field}",
                    )
            for field in PARITY_FIELDS:
                near(
                    frames["0.5"][field],
                    baseline[field],
                    f"selection base54/.5 {identity}/{horizon}/{field}",
                )
                checks += 1
            diagnostic = selection["probability_diagnostics"][f"hypo_{horizon}"]
            require(
                diagnostic["target_column"] == f"severe_hypo_{horizon}_label"
                and diagnostic["mask_column"] == f"hypo_{horizon}_training_mask",
                "level-2 probability target/mask differs",
            )
            n = diagnostic["n"]
            require(
                type(n) is int and n == baseline["eligible_evaluable_decision_count"],
                "probability windows differ from base54 eligible evaluable population",
            )
            metrics = diagnostic["metrics"]
            require(
                metrics is not None and metrics["n"] == n,
                "probability metric n differs",
            )
            population = (n, number(metrics["prevalence"], "prevalence"))
            require(
                window_populations.setdefault(horizon, population) == population,
                "probability n/prevalence differs across actual fits",
            )
            for metric in PROBABILITY_METRICS:
                value = number(metrics[metric], metric, undefined=True)
                if metric in ("auprc", "auroc", "brier", "ece") and value is not None:
                    require(
                        0 <= value <= 1, f"probability metric outside [0,1]: {metric}"
                    )
        reports[identity] = report
    require(checks == 280, "native point parity coverage differs")
    return {
        "reports": reports,
        "bindings": bindings,
        "fixed_frame_point_checks": checks,
        "window_populations": window_populations,
    }


def load_composition(sources_path: Path, development: D.AggregateData) -> dict:
    """Validate all role identities but retain only the 110 selection rows."""
    with sources_path.open(newline="", encoding="utf-8") as stream:
        raw = [row for row in csv.DictReader(stream) if row["arm"] == "composition"]
    expected = {
        (role, seed, horizon, source)
        for role in ("calibration", "selection")
        for seed in D.SEEDS
        for horizon in D.HORIZONS
        for source in D.SOURCES
    }
    identities = set()
    selected = {}
    for line, row in enumerate(raw):
        require(
            row["model_seed"] in {str(seed) for seed in D.SEEDS}
            and row["seed_identity"] == row["model_seed"],
            "composition real-seed identity differs",
        )
        key = (
            row["partition"],
            int(row["model_seed"]),
            int(row["horizon_minutes"]),
            row["source_id"],
        )
        require(key not in identities, "duplicate composition source identity")
        identities.add(key)
        require(
            row["mode"] == "frozen_composition"
            and float(row["glucose_cut"]) == 54
            and float(row["nominal_fapd"]) == 0.5,
            "composition policy identity differs",
        )
        if key[0] != "selection":
            continue
        parsed = dict(row)
        for field in D.COUNT_FIELDS:
            value = float(row[field])
            require(
                math.isfinite(value) and value >= 0 and value.is_integer(),
                f"invalid composition count: {field}",
            )
            parsed[field] = int(value)
        for field in (*D.FRACTION_FIELDS, "observation_minutes", D.SOURCE_SENSITIVITY):
            parsed[field] = float(row[field])
            require(
                math.isfinite(parsed[field]), f"invalid composition metric: {field}"
            )
        D._validate_counts(parsed, f"composition source row {line}")
        require(
            0 <= parsed[D.SOURCE_SENSITIVITY] <= 1,
            "invalid composition fixed sensitivity",
        )
        baseline = development.sources[
            ("persistence", None, key[2], 54, "refitted", key[3])
        ]
        for field in D.TRUTH_FIELDS:
            near(
                parsed[field],
                baseline[field],
                f"composition selection truth: {key}/{field}",
            )
        selected[key[1:]] = parsed
    require(
        identities == expected,
        f"composition source grid incomplete/unexpected: {len(identities)}/220",
    )
    require(len(selected) == 110, "composition selection source coverage differs")
    return selected


def _mean(values: list[float | None]) -> float | None:
    # Undefined fits never disappear into a complete-case mean.
    return None if any(value is None for value in values) else fmean(values)


def _cell(value: float | None, *, decimals: int = 4, count: bool = False) -> dict:
    if value is None:
        display = "--"
    elif count:
        require(
            value >= 0 and int(value) == value,
            "table count is not a nonnegative integer",
        )
        display = f"{int(value):,}".replace(",", "{,}")
    else:
        require(math.isfinite(value), "non-finite table value")
        display = f"{value:.{decimals}f}"
        if float(display) == 0:
            display = display.lstrip("-")
    return {"value": value, "display": display, "decimals": decimals, "count": count}


def _row(
    prefix: list[str], cells: list[dict], quantities: list[str], anchors: list[dict]
) -> dict:
    require(len(cells) == len(quantities), "table cell metadata coverage differs")
    return {
        "prefix": prefix,
        "cells": cells,
        "quantities": quantities,
        "source_values": anchors,
    }


def _table(
    rows: list[dict], *, columns: str, header: list[str], caption: str, label: str
) -> str:
    return "\n".join(
        [
            "% Generated by code/scripts/build_corrected_revision_tables.py. "
            "Do not edit by hand.",
            r"\begin{table}[t]",
            r"\centering",
            r"{\footnotesize\setlength{\tabcolsep}{4pt}",
            r"\begin{tabular}{" + columns + "}",
            r"\toprule",
            *header,
            r"\midrule",
            *(
                " & ".join(row["prefix"] + [cell["display"] for cell in row["cells"]])
                + r" \\"
                for row in rows
            ),
            r"\bottomrule",
            r"\end{tabular}}",
            r"\caption{" + caption + "}",
            r"\label{" + label + "}",
            r"\end{table}",
            "",
        ]
    )


def _fits(arm: str) -> tuple:
    return D.SEEDS if arm in ("lstm", "tcn") else (None,)


def _native_values(
    training: dict, arm: str, horizon: int, budget: str, metric: str
) -> list[float]:
    return [
        training["reports"][arm, seed]["partitions"]["selection"]["fixed_frame"][
            str(horizon)
        ][budget][metric]
        for seed in _fits(arm)
    ]


def build_tables(
    development: D.AggregateData, training: dict, composition: dict
) -> tuple[dict, dict]:
    tables, evidence = {}, {}
    rows = []
    for arm in D.ARMS:
        for budget in ("0.5", "1.0"):
            values = [
                _native_values(training, arm, horizon, budget, metric)
                for horizon in D.HORIZONS
                for metric in (D.SENSITIVITY, D.BURDEN)
            ]
            rows.append(
                _row(
                    [D.LABELS[arm]],
                    [_cell(float(budget), decimals=1)]
                    + [_cell(_mean(v)) for v in values],
                    ["nominal_fapd"]
                    + [
                        f"{metric}_{horizon}"
                        for horizon in D.HORIZONS
                        for metric in (D.SENSITIVITY, D.BURDEN)
                    ],
                    [
                        {
                            "arm": arm,
                            "budget": budget,
                            "model_seeds": _fits(arm),
                            "per_fit_values": values,
                        }
                    ],
                )
            )
    evidence[FILENAMES[0]] = rows
    tables[FILENAMES[0]] = _table(
        rows,
        columns="lrrrrr",
        header=[
            r"Arm & Nominal budget & \multicolumn{2}{c}{30 min} & "
            r"\multicolumn{2}{c}{60 min} \\",
            r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
            r" & (FA/day) & Sensitivity & Burden & Sensitivity & Burden \\",
        ],
        caption=(
            "Selection performance at the two evaluated nominal calibration budgets. "
            "Sensitivity is the participant mean among event-positive participants "
            "within each source, followed by an equal-source mean across the eleven "
            "sources. "
            "Burden is unmatched alarms per participant-calendar day, averaged across "
            "participants within each source and then equally across sources. Higher "
            "sensitivity and lower burden are preferable. Thresholds are fitted on "
            "calibration data separately at each budget, and the nominal budget is "
            "distinct from the achieved selection burden. The long short-term memory "
            "network (LSTM) and temporal convolutional network (TCN) entries each "
            "average five training seeds, whereas each classical method uses one "
            "shared fit or deterministic rule. FA/day "
            "denotes false alarms per participant-calendar day."
        ),
        label="tab:budget_sweep",
    )

    rows = []
    base = {
        h: development.rows("persistence", horizon=h)[0]["base_event_count"]
        for h in D.HORIZONS
    }
    people = development.rows("persistence")[0]["participant_count"]
    days = development.rows("persistence")[0]["patient_days"]
    for cut in D.CUTS:
        for mode in D.MODES:
            values = [
                [row[metric] for row in development.rows(arm, cut=cut, mode=mode)]
                for arm in ("persistence", "lstm")
                for metric in (D.SENSITIVITY, D.BURDEN)
            ]
            reachable = [
                development.rows("persistence", cut=cut, mode=mode, horizon=h)[0][
                    "reachable_event_count"
                ]
                for h in D.HORIZONS
            ]
            rows.append(
                _row(
                    [str(cut), "Refitted" if mode == "refitted" else "Gate only"],
                    [_cell(_mean(v)) for v in values]
                    + [_cell(v, count=True) for v in reachable],
                    [
                        f"{arm}_{metric}"
                        for arm in ("persistence", "lstm")
                        for metric in (D.SENSITIVITY, D.BURDEN)
                    ]
                    + ["reachable_events_30", "reachable_events_60"],
                    [
                        {
                            "glucose_cut": cut,
                            "mode": mode,
                            "per_condition_values": values,
                        }
                    ],
                )
            )
    evidence[FILENAMES[1]] = rows
    tables[FILENAMES[1]] = _table(
        rows,
        columns="rlrrrrrr",
        header=[
            r"Cut & Policy & \multicolumn{2}{c}{Current CGM} & "
            r"\multicolumn{2}{c}{LSTM} & \multicolumn{2}{c}{Reachable events} \\",
            r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
            r"(mg/dL) & & Sensitivity & Burden & Sensitivity & Burden & "
            r"30 min & 60 min \\",
        ],
        caption=(
            "Fixed-population sensitivity, achieved alarm burden and event "
            "reachability across eligibility cuts. Sensitivity averages "
            "participant-level detected/base-event fractions across event-positive "
            "participants within each source, while burden averages unmatched alarms "
            "per participant-calendar day across all base participants within each "
            "source. Both metrics then weight sources equally before averaging "
            "equally over the "
            "30- and 60-minute horizons. The long short-term memory network (LSTM) "
            "entries average five training seeds, whereas the current-CGM rule uses "
            "one shared fit. Refitted policies fit each cut on calibration data at "
            "a nominal 0.5 false alarms per participant-calendar day, and gate-only "
            "policies retain the base-cut threshold. Reachability counts events "
            "with at least one eligible lead-in and is shared by both methods and "
            "policy modes. The fixed denominators remain "
            f"{_cell(base[30], count=True)['display']} events at 30 minutes and "
            f"{_cell(base[60], count=True)['display']} at 60 minutes, with "
            f"{_cell(people, count=True)['display']} participants and "
            f"{_cell(days, count=True)['display']} participant-calendar days at "
            "every cut. Zero-opportunity events remain in the sensitivity "
            "denominator. Higher sensitivity and lower burden are preferable."
        ),
        label="tab:cut_curve",
    )

    rows = []
    for arm in D.ARMS:
        for horizon in D.HORIZONS:
            values = [
                [
                    training["reports"][arm, seed]["partitions"]["selection"][
                        "probability_diagnostics"
                    ][f"hypo_{horizon}"]["metrics"][metric]
                    for seed in _fits(arm)
                ]
                for metric in PROBABILITY_METRICS
            ]
            rows.append(
                _row(
                    [D.LABELS[arm], str(horizon)],
                    [
                        _cell(
                            _mean(v),
                            decimals=3 if metric.startswith("calibration_") else 4,
                        )
                        for metric, v in zip(PROBABILITY_METRICS, values, strict=True)
                    ],
                    list(PROBABILITY_METRICS),
                    [
                        {
                            "arm": arm,
                            "horizon_minutes": horizon,
                            "model_seeds": _fits(arm),
                            "per_fit_values": values,
                        }
                    ],
                )
            )
    evidence[FILENAMES[2]] = rows
    counts = [training["window_populations"][h][0] for h in D.HORIZONS]
    require(
        counts[0] == counts[1],
        "probability window denominators differ between horizons",
    )
    tables[FILENAMES[2]] = _table(
        rows,
        columns="lrrrrrrr",
        header=[
            r"Arm & $h$ (min) & AUPRC & AUROC & Brier & ECE & Intercept & Slope \\",
        ],
        caption=(
            "Level 2 probability discrimination and calibration on selection windows. "
            f"Each horizon includes {counts[0]:,} eligible evaluable windows. "
            "All metrics use task probabilities after temperature scaling fitted "
            "only on calibration data. AUPRC and AUROC are areas under the "
            "precision--recall and receiver operating characteristic curves. ECE "
            "is expected calibration error. Higher AUPRC and AUROC and lower Brier "
            "score and ECE are preferable. Ideal calibration intercept and slope "
            "are 0 and 1. The intercept and slope are descriptive logistic "
            "diagnostics fitted on selection windows and are not further score "
            "transformations. The long short-term memory network (LSTM) and "
            "temporal convolutional network (TCN) each average five training seeds. "
            "Classical methods have one shared fit. These window-level metrics "
            "differ from event-warning sensitivity. An en-dash denotes a metric "
            "undefined in at least one required fit, without dropping that fit "
            "from an average."
        ).replace(f"{counts[0]:,}", f"{counts[0]:,}".replace(",", "{,}")),
        label="tab:quality",
    )

    rows = []
    for source in D.SOURCES:
        baseline = development.rows("persistence", horizon=30, source=source)[0]
        values = [
            [
                row[D.SOURCE_SENSITIVITY]
                for row in development.rows(arm, horizon=horizon, source=source)
            ]
            for arm in ("persistence", "lstm")
            for horizon in D.HORIZONS
        ]
        values += [
            [
                composition[seed, horizon, source][D.SOURCE_SENSITIVITY]
                for seed in D.SEEDS
            ]
            for horizon in D.HORIZONS
        ]
        cells = [
            _cell(baseline[field], count=True)
            for field in ("participant_count", "event_positive_participant_count")
        ]
        rows.append(
            _row(
                [source],
                cells + [_cell(_mean(v), decimals=3) for v in values],
                ["participant_count", "event_positive_participant_count"]
                + [
                    f"{arm}_sensitivity_{h}"
                    for arm in ("persistence", "lstm", "composition")
                    for h in D.HORIZONS
                ],
                [{"source_id": source, "per_fit_values": values}],
            )
        )
    evidence[FILENAMES[3]] = rows
    tables[FILENAMES[3]] = _table(
        rows,
        columns="lrrrrrrrr",
        header=[
            r"Source & $n$ & $n^{+}$ & \multicolumn{2}{c}{Current CGM} & "
            r"\multicolumn{2}{c}{LSTM} & \multicolumn{2}{c}{Composition} \\",
            r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
            r" & & & 30 min & 60 min & 30 min & 60 min & 30 min & 60 min \\",
        ],
        caption=(
            "Fixed level 2 event sensitivity by development source at the base "
            "eligibility cut of 54~mg/dL and nominal calibration budget of 0.5 "
            "false alarms per participant-calendar day. Columns $n$ and $n^{+}$ "
            "report all base participants and event-positive participants in each "
            "source, which are shared across all methods and both horizons. "
            "Each sensitivity is the mean of participant-level detected/base-event "
            "fractions among the event-positive participants, retaining events "
            "with no eligible opportunity. Higher values indicate greater sensitivity. "
            "The long "
            "short-term memory network (LSTM) and its equal-weight calibration-rank "
            "composition with current CGM each average five training seeds, whereas "
            "the "
            "current-CGM rule uses one shared fit. All values use the selection "
            "partition and frozen calibration-fitted policies."
        ),
        label="tab:per_source",
    )
    for filename, text in tables.items():
        check_table_cells(text, evidence[filename])
    return tables, evidence


def check_table_cells(text: str, rows: list[dict]) -> int:
    """Check every rendered data cell against the source-derived cell ledger."""
    body = text.split(r"\midrule", 1)[1].split(r"\bottomrule", 1)[0]
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    require(len(lines) == len(rows), "rendered table row coverage differs")
    checked = 0
    for line, row in zip(lines, rows, strict=True):
        require(line.endswith(r"\\"), "rendered table row terminator differs")
        fields = [field.strip() for field in line[:-2].split("&")]
        expected = row["prefix"] + [cell["display"] for cell in row["cells"]]
        require(fields == expected, f"rendered table cell mismatch: {row['prefix']}")
        checked += len(row["cells"]) + sum(field.isdigit() for field in row["prefix"])
    return checked


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", required=True, type=Path)
    parser.add_argument("--sources", required=True, type=Path)
    parser.add_argument("--inference", required=True, type=Path)
    parser.add_argument("--training-dir", required=True, type=Path)
    parser.add_argument("--training-acceptance", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        development = D.load_inputs(args.conditions, args.sources, args.inference)
        training = load_training(
            args.training_dir, args.training_acceptance, development
        )
        composition = load_composition(args.sources, development)
        tables, evidence = build_tables(development, training, composition)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for filename, text in tables.items():
            (args.output_dir / filename).write_text(text, encoding="utf-8")
        checks = {
            filename: check_table_cells(
                (args.output_dir / filename).read_text(), evidence[filename]
            )
            for filename in FILENAMES
        }
        receipt = {
            "status": "source_derived_table_candidates",
            "input_paths": development.input_paths,
            "training_dir": str(args.training_dir.resolve()),
            "training_acceptance": str(args.training_acceptance.resolve()),
            "native_point_parity_checks": training["fixed_frame_point_checks"],
            "validated_main_conditions": len(development.conditions),
            "validated_main_sources": len(development.sources),
            "validated_composition_selection_sources": len(composition),
            "numeric_cell_checks": checks,
            "total_numeric_cell_checks": sum(checks.values()),
            "probability_definition": (
                "Calibration-only temperature-scaled task_probability; "
                "selection intercept/slope are descriptive diagnostics."
            ),
            "tables": evidence,
        }
        (args.output_dir / "corrected-table-data-receipt.json").write_text(
            json.dumps(receipt, indent=2, allow_nan=False) + "\n"
        )
    except (ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
        parser.exit(2, f"input/table error: {exc}\n")
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "tables": list(tables),
                "numeric_cell_checks": sum(checks.values()),
                "native_point_parity_checks": training["fixed_frame_point_checks"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
