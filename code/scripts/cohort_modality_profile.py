'Cohort modality profile.'




























from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from glycoshift.adapters import load_source_map  # noqa: E402
from glycoshift.config import load_config, resolve_path  # noqa: E402
from glycoshift.role_ingest import _exact_raw_values, _selected_filter  # noqa: E402

GLUCOSE_COLUMN = "CGM"
PARTICIPANT_COLUMN = "id"

# The action and wearable channels the design claims to represent. Grouped because the
# figure has one row per group, and because `workout_*` is three columns describing one
# thing; a matrix with three near-identical workout rows says less than one.
CHANNELS: dict[str, tuple[str, ...]] = {
    "Glucose": ("CGM",),
    "Basal insulin": ("basal",),
    "Bolus insulin": ("bolus",),
    "Insulin, total": ("insulin",),
    "Carbohydrate": ("carbs",),
    "Meal label": ("meal_label",),
    "Heart rate": ("heartrate",),
    "Steps": ("steps",),
    "Energy expenditure": ("calories_burned",),
    "Skin temperature": ("skin_temp",),
    "Ambient temperature": ("air_temp",),
    "Electrodermal activity": ("galvanic_skin_response",),
    "Workout": ("workout_label", "workout_duration", "workout_intensity"),
}

# Categorical metadata, summarised as value counts rather than as a fraction.
CATEGORICAL = (
    "cgm_device",
    "insulin_delivery_modality",
    "insulin_delivery_algorithm",
    "treatment_group",
)

# Demographics are constant within a participant and repeated on every row, so they are
# collected per participant. Averaging them over rows would weight each person by how
# long they wore a sensor, which is a different quantity and not the one anybody means.
DEMOGRAPHIC_NUMERIC = ("age", "height", "weight", "age_of_diagnosis")
DEMOGRAPHIC_CATEGORICAL = ("gender", "ethnicity")


def _columns() -> list[str]:
    wanted = [PARTICIPANT_COLUMN, GLUCOSE_COLUMN]
    for group in CHANNELS.values():
        wanted.extend(group)
    wanted.extend(CATEGORICAL)
    wanted.extend(DEMOGRAPHIC_NUMERIC)
    wanted.extend(DEMOGRAPHIC_CATEGORICAL)
    return sorted(set(wanted))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bin-width", type=float, default=2.0)
    parser.add_argument("--batch-rows", type=int, default=1_000_000)
    arguments = parser.parse_args(argv)

    config = load_config(arguments.config)
    data = config["data"]
    development = list(data["development_sources"])
    forbidden = set(data.get("forbidden_development_sources", []))
    # resolve_path takes the whole config, not a base directory: a relative path
    # resolves against the config file's own directory, which is why protocol.yaml
    # is full of ../.. and why passing a parent directory here silently fails.
    source_map = load_source_map(resolve_path(config, data["source_map"]))

    # The project's own resolution of canonical source -> exact reviewed raw value, and
    # its own dataset filter built from those values. Not a reimplementation.
    exact = _exact_raw_values(source_map, set(development))
    raw_to_source = {raw: source for source, raw in exact.items()}
    if forbidden & set(exact):
        raise SystemExit("a forbidden source resolved into the development selection")
    selection = _selected_filter(source_map, set(exact.values()))

    lo, hi = float(data["glucose_range"]["minimum_mg_dl"]), float(
        data["glucose_range"]["maximum_mg_dl"]
    )
    edges = np.arange(lo, hi + arguments.bin_width, arguments.bin_width)

    dataset = ds.dataset(str(arguments.input), format="parquet")
    source_column = source_map.source_column

    histogram: dict[str, np.ndarray] = defaultdict(
        lambda: np.zeros(len(edges) - 1, dtype=np.int64)
    )
    rows = Counter()
    out_of_range = Counter()
    non_null: dict[str, Counter] = defaultdict(Counter)
    categories: dict[str, dict[str, Counter]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    people: dict[str, dict[str, dict]] = defaultdict(dict)
    channel_people: dict[str, dict[str, set]] = defaultdict(
        lambda: defaultdict(set)
    )
    observed_raw: set[str] = set()

    started = time.time()
    scanner = dataset.scanner(
        columns=_columns() + [source_column],
        filter=selection,
        batch_size=arguments.batch_rows,
    )
    for batch in scanner.to_batches():
        frame = batch.to_pandas()
        if frame.empty:
            continue
        observed_raw.update(frame[source_column].astype(str).unique().tolist())
        frame["_source"] = frame[source_column].astype(str).map(raw_to_source)
        for source, part in frame.groupby("_source", sort=False):
            rows[source] += len(part)
            glucose = part[GLUCOSE_COLUMN].to_numpy(dtype=float)
            finite = glucose[np.isfinite(glucose)]
            inside = finite[(finite >= lo) & (finite <= hi)]
            out_of_range[source] += int(finite.size - inside.size)
            if inside.size:
                histogram[source] += np.histogram(inside, bins=edges)[0]
            for column in _columns():
                if column in (PARTICIPANT_COLUMN,):
                    continue
                non_null[source][column] += int(part[column].notna().sum())
            for column in CATEGORICAL:
                values = part[column].dropna().astype(str)
                if not values.empty:
                    categories[source][column].update(values.value_counts().to_dict())
            for participant, block in part.groupby(PARTICIPANT_COLUMN, sort=False):
                key = str(participant)
                record = people[source].setdefault(key, {})
                for column in DEMOGRAPHIC_NUMERIC + DEMOGRAPHIC_CATEGORICAL:
                    if column in record:
                        continue
                    values = block[column].dropna()
                    if not values.empty:
                        record[column] = values.iloc[0]
                # Row-level coverage answers "how densely is this channel sampled",
                # which is not the question the figure asks. The release's row grid is
                # not the CGM grid -- glucose itself is non-null on 33% of Shanghai
                # T1DM's rows because that source contributes many insulin and meal
                # rows between readings -- so a row fraction understates a channel that
                # is present but sparse and reads as though glucose were missing.
                # Participant coverage answers "is this modality here at all".
                for name, group in CHANNELS.items():
                    if any(block[column].notna().any() for column in group):
                        channel_people[source][name].add(key)

    # A filter that matched nothing, or matched something extra, would otherwise produce
    # a confident and wrong figure. This is the assertion that makes the isolation
    # checkable after the fact rather than merely intended.
    expected_raw = set(exact.values())
    if observed_raw != expected_raw:
        raise SystemExit(
            f"selection observed {sorted(observed_raw)} against the reviewed "
            f"{sorted(expected_raw)}"
        )
    if set(rows) != set(development):
        raise SystemExit(
            f"profiled {sorted(rows)} against the configured {sorted(development)}"
        )

    payload = {
        "input": str(arguments.input),
        "source_column": source_column,
        "reviewed_values": exact,
        "glucose_range_mg_dl": [lo, hi],
        "bin_width_mg_dl": arguments.bin_width,
        "bin_edges_mg_dl": edges.tolist(),
        "elapsed_seconds": round(time.time() - started, 1),
        "channels": {name: list(group) for name, group in CHANNELS.items()},
        "sources": {},
        "total_rows": int(sum(rows.values())),
    }
    for source in development:
        counts = histogram[source]
        demographics = people[source]
        numeric = {
            column: sorted(
                float(record[column])
                for record in demographics.values()
                if column in record
            )
            for column in DEMOGRAPHIC_NUMERIC
        }
        payload["sources"][source] = {
            "rows": int(rows[source]),
            "participants": len(demographics),
            "glucose_histogram": counts.tolist(),
            "glucose_in_range": int(counts.sum()),
            "glucose_out_of_range": int(out_of_range[source]),
            "non_null": {c: int(n) for c, n in sorted(non_null[source].items())},
            "channel_non_null": {
                name: int(max(non_null[source][c] for c in group))
                for name, group in CHANNELS.items()
            },
            "channel_participants": {
                name: len(channel_people[source][name]) for name in CHANNELS
            },
            "categorical": {
                column: dict(sorted(counter.items(), key=lambda kv: -kv[1]))
                for column, counter in categories[source].items()
            },
            "demographic_numeric": {
                column: {
                    "n": len(values),
                    "median": float(np.median(values)) if values else None,
                    "p25": float(np.percentile(values, 25)) if values else None,
                    "p75": float(np.percentile(values, 75)) if values else None,
                }
                for column, values in numeric.items()
            },
            "demographic_categorical": {
                column: dict(
                    Counter(
                        str(record[column])
                        for record in demographics.values()
                        if column in record
                    ).most_common()
                )
                for column in DEMOGRAPHIC_CATEGORICAL
            },
        }

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=1) + "\n")

    print(f"profiled {len(development)} development sources in "
          f"{payload['elapsed_seconds']}s, {payload['total_rows']:,} rows")
    for source in development:
        block = payload["sources"][source]
        below70 = int(counts_below(block, edges, 70.0))
        in_range = block["glucose_in_range"]
        share = below70 / in_range if in_range else 0.0
        print(f"  {source:16s} {block['participants']:4d} people "
              f"{block['glucose_in_range']:>10,} readings   "
              f"<70 mg/dL {share:6.2%}")
    return 0


def counts_below(block: dict, edges: np.ndarray, value: float) -> int:
    counts = np.asarray(block["glucose_histogram"], dtype=np.int64)
    return int(counts[edges[1:] <= value].sum())


if __name__ == "__main__":
    raise SystemExit(main())
