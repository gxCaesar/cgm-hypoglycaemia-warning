'Cohort modality traces.'
























from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.dataset as ds

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from glycoshift.adapters import load_source_map  # noqa: E402
from glycoshift.config import load_config, resolve_path  # noqa: E402
from glycoshift.role_ingest import _exact_raw_values, _selected_filter  # noqa: E402

GLUCOSE = "CGM"
PARTICIPANT = "id"
TIME = "date"
LEVEL2_MG_DL = 54.0
# The event definition the Methods freeze: at least three consecutive readings below the
# threshold, which is the fifteen consecutive minutes the consensus sources specify.
RUN_LENGTH = 3
# Channels worth drawing under a trace. Each is a column that is either observed at an
# instant or absent; none is imputed.
ACTION = ("basal", "bolus", "carbs", "meal_label", "steps", "heartrate")
TOD_BINS = 48


def _first_event(times: np.ndarray, glucose: np.ndarray) -> int | None:
    """Index of the first reading opening a run of RUN_LENGTH below the threshold."""
    below = glucose < LEVEL2_MG_DL
    if below.size < RUN_LENGTH:
        return None
    run = 0
    for index, flag in enumerate(below):
        run = run + 1 if flag else 0
        if run == RUN_LENGTH:
            return index - RUN_LENGTH + 1
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--window-hours", type=float, default=6.0)
    parser.add_argument("--batch-rows", type=int, default=1_000_000)
    arguments = parser.parse_args(argv)

    config = load_config(arguments.config)
    data = config["data"]
    development = list(data["development_sources"])
    forbidden = set(data.get("forbidden_development_sources", []))
    source_map = load_source_map(resolve_path(config, data["source_map"]))

    exact = _exact_raw_values(source_map, set(development))
    raw_to_source = {raw: source for source, raw in exact.items()}
    if forbidden & set(exact):
        raise SystemExit("a forbidden source resolved into the development selection")
    selection = _selected_filter(source_map, set(exact.values()))
    column = source_map.source_column
    dataset = ds.dataset(str(arguments.input), format="parquet")

    # ---- pass 1: time of day, and the statistic the selection rule ranks on ----------
    started = time.time()
    tod_all: dict[str, np.ndarray] = defaultdict(
        lambda: np.zeros(TOD_BINS, dtype=np.int64))
    tod_low: dict[str, np.ndarray] = defaultdict(
        lambda: np.zeros(TOD_BINS, dtype=np.int64))
    low_count: dict[str, Counter] = defaultdict(Counter)
    observed: set[str] = set()

    scanner = dataset.scanner(
        columns=[PARTICIPANT, TIME, GLUCOSE, column],
        filter=selection, batch_size=arguments.batch_rows)
    for batch in scanner.to_batches():
        frame = batch.to_pandas()
        if frame.empty:
            continue
        observed.update(frame[column].astype(str).unique().tolist())
        frame["_source"] = frame[column].astype(str).map(raw_to_source)
        frame = frame[frame[GLUCOSE].notna() & frame[TIME].notna()]
        if frame.empty:
            continue
        slot = ((frame[TIME].dt.hour * 60 + frame[TIME].dt.minute)
                // (1440 // TOD_BINS)).to_numpy()
        low = frame[GLUCOSE].to_numpy(dtype=float) < LEVEL2_MG_DL
        for source, index in frame.groupby("_source", sort=False).indices.items():
            tod_all[source] += np.bincount(slot[index], minlength=TOD_BINS)
            if low[index].any():
                tod_low[source] += np.bincount(slot[index][low[index]],
                                               minlength=TOD_BINS)
                block = frame.iloc[index]
                counts = (block.loc[low[index], PARTICIPANT].astype(str)
                          .value_counts().to_dict())
                low_count[source].update(counts)

    expected = set(exact.values())
    if observed != expected:
        raise SystemExit(
            f"selection observed {sorted(observed)} against the reviewed "
            f"{sorted(expected)}")

    # The rule, applied. Median by count of readings below the level 2 threshold, ties
    # broken on the identifier so the choice does not depend on dictionary order.
    chosen: dict[str, str] = {}
    for source in sorted(exact):
        ranked = sorted(low_count[source].items(), key=lambda kv: (kv[1], kv[0]))
        if not ranked:
            continue
        chosen[source] = ranked[(len(ranked) - 1) // 2][0]
    if not chosen:
        raise SystemExit("no source has a participant with a reading below 54 mg/dL")
    pass1 = time.time() - started

    # ---- pass 2: the traces themselves ----------------------------------------------
    wanted = sorted({pid for pid in chosen.values()})
    scanner = dataset.scanner(
        columns=[PARTICIPANT, TIME, GLUCOSE, column, *ACTION],
        filter=selection & pc.field(PARTICIPANT).isin(wanted),
        batch_size=arguments.batch_rows)
    collected: dict[str, list] = defaultdict(list)
    for batch in scanner.to_batches():
        frame = batch.to_pandas()
        if frame.empty:
            continue
        frame["_source"] = frame[column].astype(str).map(raw_to_source)
        for source, part in frame.groupby("_source", sort=False):
            keep = part[part[PARTICIPANT].astype(str) == chosen.get(source, "")]
            if not keep.empty:
                collected[source].append(keep)

    traces = {}
    half = int(round(arguments.window_hours * 60))
    for source, blocks in collected.items():
        import pandas as pd
        frame = pd.concat(blocks).sort_values(TIME)
        frame = frame[frame[TIME].notna()]
        # The event definition counts consecutive READINGS, not consecutive rows. The
        # release interleaves rows that carry an action and no glucose, so running the
        # test over raw rows lets a single such row break a run. That silently produced
        # no trace at all for the one cohort whose rows are mostly action rows, 1,606
        # rows against 536 readings, and it looked like a cohort with no event rather
        # than like a defect.
        readings = frame[frame[GLUCOSE].notna()]
        onset = _first_event(readings[TIME].to_numpy(),
                             readings[GLUCOSE].to_numpy(dtype=float))
        if onset is None:
            continue
        centre = readings[TIME].iloc[onset]
        window = frame[(frame[TIME] >= centre - pd.Timedelta(minutes=half))
                       & (frame[TIME] <= centre + pd.Timedelta(minutes=half))]
        if window[GLUCOSE].notna().sum() < 12:
            continue
        minutes = ((window[TIME] - centre).dt.total_seconds() / 60.0).tolist()
        record = {
            "participant": chosen[source],
            "onset_hour": int(centre.hour),
            "minutes": [round(v, 2) for v in minutes],
            "glucose": [None if not np.isfinite(v) else round(float(v), 1)
                        for v in window[GLUCOSE].to_numpy(dtype=float)],
        }
        for name in ACTION:
            series = window[name]
            if name == "meal_label":
                marks = [round(m, 2) for m, v in zip(minutes, series, strict=True)
                         if isinstance(v, str) and v]
                record[name] = {"marks": marks}
            else:
                values = series.to_numpy(dtype=float, na_value=np.nan)
                observed_at = [(round(m, 2), round(float(v), 3))
                               for m, v in zip(minutes, values, strict=True)
                               if np.isfinite(v) and v != 0.0]
                record[name] = {"observed": observed_at}
        traces[source] = record

    payload = {
        "input": str(arguments.input),
        "level2_mg_dl": LEVEL2_MG_DL,
        "run_length": RUN_LENGTH,
        "window_hours": arguments.window_hours,
        "time_of_day_bins": TOD_BINS,
        "selection_rule": (
            "median participant by count of readings below the level 2 "
            "threshold, ties broken on the participant identifier. The window is "
            "centred on that participant's first level 2 event"),
        "chosen": chosen,
        "time_of_day": {
            source: {"all": tod_all[source].tolist(),
                     "below_54": tod_low[source].tolist()}
            for source in sorted(exact)
        },
        "traces": traces,
        "elapsed_seconds": round(time.time() - started, 1),
        "pass1_seconds": round(pass1, 1),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload) + "\n")
    print(f"wrote {arguments.output}  {len(traces)} traces of {len(chosen)} "
          f"chosen participants, {len(exact)} sources, "
          f"{payload['elapsed_seconds']}s")
    for source in sorted(traces):
        r = traces[source]
        acts = [n for n in ACTION
                if r[n].get("observed") or r[n].get("marks")]
        seen = sum(1 for g in r["glucose"] if g is not None)
        print(f"    {source:<16} participant {r['participant']:<10} "
              f"onset {r['onset_hour']:02d}h {seen:>4} readings  channels {acts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
