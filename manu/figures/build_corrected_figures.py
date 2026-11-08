#!/usr/bin/env python3
"""Recreate Figures 3-8 and Supplementary Figures 1-4 from aggregate plot inputs."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import make_fig_corrected_development as D
import make_fig_corrected_diagnostics as Q
import make_fig_corrected_event_sensitivity as E
import make_fig_corrected_composition_external as C

GROUPS = ("development", "diagnostics", "event-sensitivity", "composition-external")


def decode(value):
    if isinstance(value, dict):
        if set(value) == {"tuple_items"}:
            entries = value["tuple_items"]
            result = {tuple(key): decode(item) for key, item in entries}
            if len(result) != len(entries):
                raise ValueError("duplicate aggregate identity")
            return result
        return {key: decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode(item) for item in value]
    return value


def load(path, group):
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 1 or payload.get("figure_group") != group:
        raise ValueError("aggregate plot-input schema or group differs")
    return decode(payload["data"])


def render(group, data, output):
    if group == "development":
        data = D.AggregateData(**data, input_paths={"aggregate_input": "development.json"})
        D._validate_inference(data)
        for function in (D.figure3, D.figure4, D.figure_s2, D.figure_s4):
            function(data, output)
    elif group == "diagnostics":
        data["paths"] = {key: "diagnostics.json" for key in data["reports"]}
        data["alarm_source"] = "diagnostics.json"
        for row in data["alarms"].values():
            row["source_json_pointer"] = "aggregate_alarm_summary"
        Q.G.set_style()
        receipt = {}
        for function in (Q.figure5, Q.figure6, Q.supplement1, Q.supplement3):
            function(data, output, receipt)
        (output / "plot-values.json").write_text(json.dumps(receipt, allow_nan=False))
    else:
        data["input_bindings"] = []
        if group == "event-sensitivity":
            figure, receipt = E.make_figure(data)
            name = "fig7-corrected-candidate.pdf"
        else:
            externals = data.pop("externals")
            if data.pop("shared_external_recipe") is not True:
                raise ValueError("external plot inputs must describe the same fixed recipe")
            for external in externals.values():
                # A descriptive renderer key, not a publication or acceptance ID.
                external["report"]["freeze_id"] = "shared-base54-recipe"
                external["input_binding"] = {"path": "composition-external.json"}
            figure, receipt = C.make_figure(data, externals)
            name = "fig8-corrected-candidate.pdf"
        E.G.save(figure, str(output / name), caption_words=200)
        (output / "plot-values.json").write_text(json.dumps(receipt, allow_nan=False))
    plt.close("all")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-inputs", type=Path, required=True,
                        help="data/plot-inputs from the companion Supplementary Data")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="a new directory for rendered PDFs and plotted values")
    parser.add_argument("--group", choices=(*GROUPS, "all"), default="all")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("output directory already exists; choose a new directory")
    groups = GROUPS if args.group == "all" else (args.group,)
    loaded = [(group, load(args.plot_inputs / (group + ".json"), group)) for group in groups]
    args.output_dir.mkdir(parents=True)
    for group, data in loaded:
        output = args.output_dir / group
        output.mkdir()
        render(group, data, output)
    print("Rendered aggregate plots; no model fitting, event matching or resampling performed.")


if __name__ == "__main__":
    main()
