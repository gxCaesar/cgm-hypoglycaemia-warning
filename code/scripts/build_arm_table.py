'Build arm table.'

















from __future__ import annotations

import argparse
import json
from pathlib import Path

HORIZONS = ("30", "60")
TOLERANCE = 1e-12
DIAGNOSTIC = "diagnostic_only_retained_checkpoint_proxy_mismatch"

# Family by architecture, which is the only grouping recoverable from the artifacts. No
# metric JSON stores a family field, so this mapping is declared here and is the thing a
# reader should disagree with if they disagree with "seven families".
FAMILY = {
    "persistence": "cheap control",
    "constant_global": "cheap control",
    "constant_source_conditional": "cheap control",
    "label_shuffle_logistic": "cheap control",
    "linear": "cheap control",
    "logistic_regression": "cheap control",
    "xgboost": "cheap control",
    "balanced_lstm_erm": "LSTM",
    "erm_lstm": "LSTM",
    "tcn_erm": "temporal convolution",
    "erm_tcn": "temporal convolution",
    "erm_patchtst": "patch transformer",
    "erm_nomemory_glycoshift": "memory transformer",
    "dro_tau_0p5_lstm": "GroupDRO sweep",
    "dro_tau_1p0_lstm": "GroupDRO sweep",
    "dro_tau_2p0_lstm": "GroupDRO sweep",
    "tls_lstm": "tail-loss shaping",
}
UNTRAINED = {"persistence", "constant_global", "constant_source_conditional"}

# The manuscript carries no code identifiers, so each configuration gets a plain name.
# The two temporal-convolution rows and the two LSTM rows are different recipes and are
# told apart by the checkpoint column rather than by a cryptic suffix.
DISPLAY = {
    "persistence": "Current glucose (control)",
    "constant_global": "Constant score, one value",
    "constant_source_conditional": "Constant score, one per source",
    "label_shuffle_logistic": "Label-shuffled logistic",
    "linear": "Linear extrapolation",
    "logistic_regression": "Weighted logistic regression",
    "xgboost": "Gradient-boosted trees",
    "balanced_lstm_erm": "LSTM, source-balanced ERM",
    "erm_lstm": "LSTM, ERM",
    "tcn_erm": "Temporal convolution, ERM",
    "erm_tcn": "Temporal convolution, ERM",
    "erm_patchtst": "Patch transformer, ERM",
    # Renamed for the reader. The artifact id is unchanged. This architecture is a
    # comparator here, and the project name it once carried now denotes the
    # evaluation harness, so the two must not share a token in one document.
    "erm_nomemory_glycoshift": "Memory transformer, no memory",
    "dro_tau_0p5_lstm": "GroupDRO, $\\tau=0.5$",
    "dro_tau_1p0_lstm": "GroupDRO, $\\tau=1.0$",
    "dro_tau_2p0_lstm": "GroupDRO, $\\tau=2.0$",
    "tls_lstm": "Tail-loss shaping",
}


def _configuration(arm_id: str) -> str:
    return arm_id.rsplit("_seed", 1)[0]


def _read(path: Path, retained: bool) -> dict:
    payload = json.loads(path.read_text())
    if retained:
        primary = float(payload["mean_severe_event_sensitivity"])
        fapd = float(payload["mean_achieved_fapd"])
        inner = {h: payload["horizons"][h]["selection"] for h in HORIZONS}
        arm = payload["arm_id"]
        flag = payload.get("reportability")
        if flag != DIAGNOSTIC:
            raise SystemExit(f"{arm}: expected the diagnostic flag, found {flag!r}")
    else:
        primary = float(payload["primary_equal_horizon_mean_event_sensitivity"])
        fapd = float(payload["equal_horizon_mean_achieved_fapd"])
        inner = {h: payload["horizons"][h]["selection_metrics"] for h in HORIZONS}
        arm = path.parent.name
        if payload["evaluation_partition"] != "selection":
            raise SystemExit(f"{arm}: scored on {payload['evaluation_partition']}")
        if "reportability" in payload:
            raise SystemExit(f"{arm}: a current-proxy arm carries a reportability flag")
    if payload["nominal_fapd"] != 0.5:
        raise SystemExit(f"{arm}: nominal budget {payload['nominal_fapd']}")

    row = {
        "arm": arm,
        "configuration": _configuration(arm),
        "family": FAMILY.get(_configuration(arm), "UNMAPPED"),
        "display": DISPLAY.get(_configuration(arm), "UNMAPPED"),
        "seed": int(arm.rsplit("_seed", 1)[1]) if "_seed" in arm else 0,
        "trained": _configuration(arm) not in UNTRAINED,
        "checkpoint_proxy": "superseded" if retained else "current",
        "primary": primary,
        "achieved_fapd": fapd,
    }
    if "UNMAPPED" in (row["family"], row["display"]):
        raise SystemExit(
            f"{arm}: no family or name for {row['configuration']!r}")
    for h in HORIZONS:
        block = inner[h]
        if int(block["source_count"]) != 11:
            raise SystemExit(
                f"{arm} at {h}: {block['source_count']} sources, not eleven")
        row[f"sensitivity_{h}"] = float(
            block["equal_source_participant_macro_event_sensitivity"]
        )
        row[f"fapd_{h}"] = float(
            block["equal_source_participant_macro_false_alarms_per_patient_day"]
        )
        row[f"alarms_{h}"] = int(block["alarm_count"])
    rebuilt = 0.5 * (row["sensitivity_30"] + row["sensitivity_60"])
    if abs(rebuilt - primary) > TOLERANCE:
        raise SystemExit(
            f"{arm}: horizons rebuild {rebuilt:.12f} against a stored {primary:.12f}"
        )
    return row


def _seed_counts(root: Path, rows: list[dict]) -> dict[str, tuple[int, str]]:
    """Fits per configuration, each bound to the artifact that says so."""
    counts: dict[str, tuple[int, str]] = {}
    scale = root / "experiments/20260903-erm-lstm-seeds-v1/seed-scale.json"
    if scale.exists():
        counts["balanced_lstm_erm"] = (
            int(json.loads(scale.read_text())["n"]), "seed-scale.json"
        )
    tcn = root / "experiments/20260813-d0-tcn-seed-variance-v1/d0-readout.json"
    if tcn.exists():
        counts["erm_tcn"] = (int(json.loads(tcn.read_text())["n"]), "d0-readout.json")
    # anything else: the fits present in this table
    seen: dict[str, int] = {}
    for row in rows:
        seen[row["configuration"]] = seen.get(row["configuration"], 0) + 1
    for configuration, n in seen.items():
        counts.setdefault(configuration, (n, "this table"))
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--tex-out", type=Path)
    arguments = parser.parse_args(argv)
    root = arguments.root.resolve()

    rows = []
    current = root / (
        "experiments/20260817-glycoshift-v1-5-baseline-closure-v1/"
        "l20-run-results/runs"
    )
    for directory in sorted(current.iterdir()):
        path = directory / "operational_metrics.json"
        if path.exists():
            rows.append(_read(path, retained=False))
    retained = root / (
        "experiments/20260813-stage0-operational-diagnostic-v1/operational")
    for path in sorted(retained.glob("*.json")):
        if path.name == "summary.json":
            continue
        rows.append(_read(path, retained=True))

    rows.sort(key=lambda r: -r["primary"])
    seeds = _seed_counts(root, rows)
    for row in rows:
        row["seeds_available"], row["seeds_source"] = seeds[row["configuration"]]

    trained = [r for r in rows if r["trained"]]
    control = next(r for r in rows if r["configuration"] == "persistence")
    beaten = [r for r in trained if r["primary"] < control["primary"]]
    summary = {
        "arms_total": len(rows),
        "trained_fits": len(trained),
        "trained_configurations": len({r["configuration"] for r in trained}),
        "trained_families": sorted({r["family"] for r in trained}),
        "current_proxy_trained": sum(
            1 for r in trained if r["checkpoint_proxy"] == "current"
        ),
        "superseded_proxy_trained": sum(
            1 for r in trained if r["checkpoint_proxy"] == "superseded"
        ),
        "control_primary": control["primary"],
        "trained_arms_below_control": len(beaten),
        "control_leads_every_trained_arm": len(beaten) == len(trained),
        "rows": rows,
    }
    if arguments.json_out:
        arguments.json_out.write_text(json.dumps(summary, indent=1) + "\n")

    print(f"  {summary['arms_total']} arms, {summary['trained_fits']} trained fits of "
          f"{summary['trained_configurations']} configurations")
    print(f"  families among trained arms: {len(summary['trained_families'])} "
          f"{summary['trained_families']}")
    print(f"  checkpoint proxy: {summary['current_proxy_trained']} current, "
          f"{summary['superseded_proxy_trained']} superseded")
    print(f"  control leads every trained arm: "
          f"{summary['control_leads_every_trained_arm']} "
          f"({summary['trained_arms_below_control']}/{summary['trained_fits']})")
    print()
    for row in rows:
        mark = " " if row["trained"] else "*"
        print(f"  {mark}{row['arm']:34s} {row['family']:22s} "
              f"{row['checkpoint_proxy']:10s} {row['primary']:.6f} "
              f"{row['sensitivity_30']:.4f} {row['sensitivity_60']:.4f} "
              f"{row['achieved_fapd']:.4f} {row['seeds_available']}")

    if arguments.tex_out:
        arguments.tex_out.write_text(_latex(rows, control))
    return 0


def _latex(rows: list[dict], control: dict) -> str:
    out = [
        "% Generated by code/scripts/build_arm_table.py. Do not edit by hand.",
        "\\begin{table}[t]",
        "\\centering",
        "{\\footnotesize\\setlength{\\tabcolsep}{3.2pt}",
        "\\begin{tabular}{llrrrrrrr}",
        "\\toprule",
        "Arm & Checkpoint & Seed & Primary & 30 min & 60 min & FA/day & "
        "\\shortstack{Alarms\\\\(30 min)} &"
        " Fits \\\\",
        "\\midrule",
    ]
    for row in rows:
        name = row["display"] + ("" if row["trained"] else "$^{\\dagger}$")
        seed = f"{row['seed']:03d}" if row["trained"] else "--"
        out.append(
            f"{name} & {row['checkpoint_proxy']} & {seed} & "
            f"{row['primary']:.4f} & {row['sensitivity_30']:.4f} & "
            f"{row['sensitivity_60']:.4f} & {row['achieved_fapd']:.4f} & "
            f"{row['alarms_30']:,} & {row['seeds_available']} \\\\"
        )
    out += [
        "\\bottomrule",
        "\\end{tabular}}",
        "\\caption{Performance of all eighteen evaluated arms on the frozen "
        "unrestricted "
        "selection partition, ordered by the primary endpoint. The 30- and 60-minute "
        "columns report level 2 event sensitivity, averaged over event-positive "
        "participants within each source and equally over the eleven sources. Higher "
        "values are better. \\emph{Primary} is the arithmetic mean of these two "
        "columns. "
        "\\emph{FA\\slash day} is the corresponding equal-horizon mean of the achieved "
        "false-alarm rate per participant-day at a nominal budget of 0.5. "
        "\\emph{Alarms} "
        "counts refractory-collapsed triggered alarms at 30 minutes only. "
        "\\emph{Checkpoint} identifies the epoch-selection proxy for trained arms, "
        "where "
        "\\emph{current} denotes the operational budget-frontier proxy and "
        "\\emph{superseded} denotes the macro window area under the precision--recall "
        "curve proxy from earlier campaigns retained as diagnostic rescorings rather "
        "than comparator results. \\emph{Fits} counts available seeds or runs for that "
        "configuration, rather than replicates averaged in the displayed row. Rows "
        "marked "
        "$\\dagger$ are untrained rules and are not among the fifteen trained arms. "
        "Their training seed is inapplicable and shown as an en-dash. ERM, empirical "
        "risk minimisation; LSTM, long short-term memory network; GroupDRO, group "
        "distributionally robust optimisation."
        "}",
        "\\label{tab:all_arms}",
        "\\end{table}",
        "",
    ]
    return "\n".join(out)


if __name__ == "__main__":
    raise SystemExit(main())
