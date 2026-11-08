#!/usr/bin/env python3
"""Export aggregate population counts from prepared revision inputs.

This utility reads existing catalogues, exposure and supervision masks. It does
not reconstruct events or windows, apply glucose cuts, or fit a model. Only
source/partition/horizon aggregates are written.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PARTITIONS = ("train", "selection", "calibration")
HORIZONS = (30, 60)
KEYS = ["source_id", "participant_id"]
BATCH_ROWS = 65_536
TIMELINE_COLUMNS = [
    *KEYS,
    "partition",
    "context_end",
    "cgm_last",
    "base_evaluable",
    *[
        column
        for horizon in HORIZONS
        for column in (
            f"base_evaluable_{horizon}",
            f"hypo_{horizon}_training_mask",
            f"hypo_{horizon}_training_target",
        )
    ],
]
INPUT_COUNTS = (
    "participants",
    "base_evaluable_decisions",
    "observed_anchor_decisions",
    "legacy_interpolated_anchor_decisions",
)


def _integer(value, name: str, *, positive: bool = False) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or value < int(positive)
    ):
        raise ValueError(
            f"{name} must be a {'positive' if positive else 'nonnegative'} integer"
        )
    return int(value)


def _identifiers(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        if column not in frame or frame[column].isna().any():
            raise ValueError(f"missing {column} identity")
        if frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"empty {column} identity")
        frame[column] = frame[column].astype(str)


def _boolean(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = frame[column]
    if values.isna().any() or not pd.api.types.is_bool_dtype(values.dtype):
        raise ValueError(f"{column} must contain complete boolean values")
    return values.to_numpy(dtype=bool)


def input_counts(audit: dict) -> pd.DataFrame:
    """Validate the audit-defined source grid without disclosing private paths."""
    if audit.get("revision_contract") != "2026-11-04-fixed-frame":
        raise ValueError("input audit has the wrong revision contract")
    if audit.get("status") != "passed":
        raise ValueError("input preparation audit has not passed")
    sources = audit.get("source_order", [])
    if (
        not sources
        or any(not isinstance(s, str) or not s.strip() for s in sources)
        or len(set(sources)) != len(sources)
    ):
        raise ValueError("audit source_order must contain unique source identities")
    rows = []
    for row in audit["source_partition_counts"]:
        selected = {name: row[name] for name in ("source_id", "partition")}
        for name in INPUT_COUNTS:
            selected[name] = _integer(row[name], name, positive=name == "participants")
        if selected["base_evaluable_decisions"] > selected["observed_anchor_decisions"]:
            raise ValueError("base decisions exceed observed-anchor decisions")
        # Preparation summed this participant-level metadata alongside counts.
        if row["base_uses_maximum_horizon_completeness"] != (
            selected["participants"] * max(HORIZONS)
        ):
            raise ValueError("maximum-horizon completeness metadata is inconsistent")
        selected["base_future_completeness_minutes"] = max(HORIZONS)
        rows.append(selected)
    frame = pd.DataFrame(rows)
    _identifiers(frame, ["source_id", "partition"])
    expected = {(source, role) for source in sources for role in PARTITIONS}
    actual = list(frame[["source_id", "partition"]].itertuples(index=False, name=None))
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError(
            "audit must contain exactly one row for every source/partition"
        )
    if int(frame.participants.sum()) != _integer(
        audit["original_participants"], "original_participants", positive=True
    ):
        raise ValueError("audit participant total differs from original_participants")
    return (
        frame.set_index(["source_id", "partition"])
        .loc[[(source, role) for role in PARTITIONS for source in sources]]
        .reset_index()
    )


def _bundle_file(prepared: Path, path: Path, name: str) -> Path:
    """Require ordinary files within the selected bundle, without symlink hops."""
    try:
        relative = path.relative_to(prepared) if path.is_absolute() else path
    except ValueError as error:
        raise ValueError("prepared input path escapes bundle") from error
    if ".." in relative.parts:
        raise ValueError("prepared input path escapes bundle")
    current = prepared
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            raise ValueError("prepared inputs must not use symlinks")
    if not current.is_file():
        raise ValueError(f"{name} is missing or not a regular file")
    return current


def _check_output(output: Path) -> None:
    if output.is_symlink() or (
        output.exists() and (not output.is_dir() or any(output.iterdir()))
    ):
        raise FileExistsError("population report output must be absent or empty")


def _read_table(path: Path, columns: list[str]) -> pd.DataFrame:
    parquet = pq.ParquetFile(path)
    if not set(columns).issubset(parquet.schema_arrow.names):
        raise ValueError("endpoint table is missing required columns")
    return parquet.read(columns=columns, use_threads=False).to_pandas()


def _with_totals(rows: list[dict], sources: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    values = [c for c in frame if c not in {"source_id", "partition", "horizon"}]
    frame.insert(0, "row_type", "source")
    totals = frame.groupby(["partition", "horizon"], as_index=False)[values].sum()
    totals.insert(0, "source_id", "")
    totals.insert(0, "row_type", "role_total")
    combined = pd.concat([frame, totals], ignore_index=True)
    combined["_role"] = combined.partition.map({r: i for i, r in enumerate(PARTITIONS)})
    combined["_source"] = combined.source_id.map({s: i for i, s in enumerate(sources)})
    return (
        combined.sort_values(["_role", "horizon", "_source"], na_position="last")
        .drop(columns=["_role", "_source"])
        .reset_index(drop=True)
    )


def _endpoints(prepared: Path, counts: pd.DataFrame, sources: list[str]):
    expected = counts.set_index(["source_id", "partition"]).participants.to_dict()
    participants, rows, assigned = {}, [], {}
    for role in PARTITIONS:
        reference, event_reference = None, None
        for horizon in HORIZONS:
            directory = prepared / "endpoints" / role
            events = _read_table(
                _bundle_file(
                    prepared, directory / f"events-{horizon}.parquet", "event table"
                ),
                KEYS + ["event_id", "event_onset"],
            )
            exposure = _read_table(
                _bundle_file(
                    prepared,
                    directory / f"exposure-{horizon}.parquet",
                    "exposure table",
                ),
                KEYS + ["patient_days", "observation_minutes"],
            )
            _identifiers(events, KEYS + ["event_id"])
            _identifiers(exposure, KEYS)
            if exposure.duplicated(KEYS).any():
                raise ValueError("duplicate exposure participant identity")
            if events.duplicated(KEYS + ["event_id"]).any():
                raise ValueError("duplicate event identity within a horizon")
            onsets = pd.to_datetime(events.event_onset, utc=True, errors="coerce")
            if onsets.isna().any():
                raise ValueError("event onsets must be valid")
            events["event_onset"] = onsets
            if events.duplicated(KEYS + ["event_onset"]).any():
                raise ValueError("duplicate event onset within a participant/horizon")
            event_index = events.set_index(KEYS + ["event_id"]).event_onset
            if event_reference is not None:
                common = event_reference.index.intersection(event_index.index)
                if not event_reference.loc[common].equals(event_index.loc[common]):
                    raise ValueError(
                        "shared event identity has conflicting horizon onsets"
                    )
            event_reference = event_index
            for column in ("patient_days", "observation_minutes"):
                values = pd.to_numeric(exposure[column], errors="coerce").to_numpy(
                    float
                )
                if not np.isfinite(values).all() or (values <= 0).any():
                    raise ValueError(f"{column} must be finite and positive")
                if column == "patient_days" and (values != np.floor(values)).any():
                    raise ValueError(
                        "patient_days must contain integer calendar-day counts"
                    )
                exposure[column] = values
            keys = set(exposure[KEYS].itertuples(index=False, name=None))
            event_keys = set(events[KEYS].itertuples(index=False, name=None))
            if not event_keys.issubset(keys):
                raise ValueError("event participant is missing from base exposure")
            if {key[0] for key in keys} != set(sources):
                raise ValueError("exposure sources differ from audit source_order")
            for source in sources:
                if sum(key[0] == source for key in keys) != expected[source, role]:
                    raise ValueError("exposure participant count differs from audit")
            indexed = exposure.set_index(KEYS).sort_index()
            if reference is not None and not indexed.equals(reference):
                raise ValueError("base exposure differs between horizons")
            reference = indexed
            participants[role, horizon] = keys
            for key in keys:
                if key in assigned and assigned[key] != role:
                    raise ValueError("source-scoped participant overlaps partitions")
                assigned[key] = role
            for source in sources:
                source_exposure = exposure.loc[exposure.source_id == source]
                source_events = events.loc[events.source_id == source]
                minutes = float(source_exposure.observation_minutes.sum())
                rows.append(
                    {
                        "source_id": source,
                        "partition": role,
                        "horizon": horizon,
                        "base_events": len(source_events),
                        "event_positive_participants": len(
                            source_events[KEYS].drop_duplicates()
                        ),
                        "base_participants": len(source_exposure),
                        "calendar_days": int(source_exposure.patient_days.sum()),
                        "observation_minutes": minutes,
                        "effective_24h_days": minutes / 1440,
                    }
                )
    return _with_totals(rows, sources), participants


def _timelines(prepared: Path, counts: pd.DataFrame, sources: list[str], participants):
    manifest_path = prepared / "participant-manifest.private.json"
    if not manifest_path.exists() and not manifest_path.is_symlink():
        return None, {"status": "not_run", "reason": "participant manifest absent"}
    manifest_path = _bundle_file(prepared, manifest_path, "participant manifest")
    manifest = json.loads(manifest_path.read_text())["participants"]
    expected_counts = counts.set_index(["source_id", "partition"])
    seen, aggregate = set(), Counter()
    for record in manifest:
        identity = pd.DataFrame([{k: record[k] for k in (*KEYS, "partition")}])
        _identifiers(identity, [*KEYS, "partition"])
        source, participant, role = identity.iloc[0]
        key = source, participant
        if role not in PARTITIONS or source not in sources:
            raise ValueError("manifest has an unknown source or partition")
        if key in seen:
            raise ValueError("duplicate manifest participant identity")
        seen.add(key)
        if key not in participants[role, HORIZONS[0]]:
            raise ValueError("manifest participant differs from base exposure")
        path = _bundle_file(
            prepared, Path(record["timeline_path"]), "manifest timeline"
        )
        stored_rows = _integer(record["stored_rows"], "stored_rows", positive=True)
        parquet = pq.ParquetFile(path)
        if not set(TIMELINE_COLUMNS).issubset(parquet.schema_arrow.names):
            raise ValueError("timeline is missing required projected columns")
        if parquet.metadata.num_rows != stored_rows:
            raise ValueError("timeline stored row count differs from manifest")
        previous, read_rows = None, 0
        for batch in parquet.iter_batches(
            columns=TIMELINE_COLUMNS, batch_size=BATCH_ROWS, use_threads=False
        ):
            frame = batch.to_pandas()
            read_rows += len(frame)
            _identifiers(frame, [*KEYS, "partition"])
            if any(
                not frame[column].eq(value).all()
                for column, value in zip(
                    (*KEYS, "partition"), (source, participant, role), strict=True
                )
            ):
                raise ValueError("timeline identity differs from manifest")
            times = pd.to_datetime(frame.context_end, utc=True, errors="coerce")
            if (
                times.isna().any()
                or times.duplicated().any()
                or not times.is_monotonic_increasing
                or (previous is not None and times.iloc[0] <= previous)
            ):
                raise ValueError(
                    "timeline decision identity must be unique and chronological"
                )
            previous = times.iloc[-1]
            glucose = pd.to_numeric(frame.cgm_last, errors="coerce").to_numpy(float)
            if not np.isfinite(glucose).all():
                raise ValueError("timeline current CGM must be finite")
            base = _boolean(frame, "base_evaluable")
            if role == "train" and not base.all():
                raise ValueError("training timeline contains non-base rows")
            for horizon in HORIZONS:
                if not np.array_equal(
                    base, _boolean(frame, f"base_evaluable_{horizon}")
                ):
                    raise ValueError("timeline horizon base masks differ")
                mask = _boolean(frame, f"hypo_{horizon}_training_mask")
                if not np.array_equal(mask, base & (glucose >= 54)):
                    raise ValueError(
                        "level-2 training mask differs from base online eligibility"
                    )
                target = pd.to_numeric(
                    frame[f"hypo_{horizon}_training_target"], errors="coerce"
                ).to_numpy(float)
                if not np.isin(target[mask], (0, 1)).all():
                    raise ValueError("retained level-2 training targets must be binary")
                for name, value in (
                    ("stored_rows", len(frame)),
                    ("base_evaluable_decisions", int(base.sum())),
                    ("retained_training_windows", int(mask.sum())),
                    ("positive_training_windows", int((target[mask] == 1).sum())),
                ):
                    aggregate[source, role, horizon, name] += value
        if read_rows != stored_rows:
            raise ValueError("timeline streamed row count differs from manifest")
    if seen != set().union(*(participants[role, HORIZONS[0]] for role in PARTITIONS)):
        raise ValueError("manifest is missing base exposure participants")
    rows = []
    for role in PARTITIONS:
        for source in sources:
            for horizon in HORIZONS:
                values = {
                    name: aggregate[source, role, horizon, name]
                    for name in (
                        "stored_rows",
                        "base_evaluable_decisions",
                        "retained_training_windows",
                        "positive_training_windows",
                    )
                }
                if (
                    values["base_evaluable_decisions"]
                    != expected_counts.loc[(source, role), "base_evaluable_decisions"]
                ):
                    raise ValueError("timeline base decision count differs from audit")
                stored_expected = expected_counts.loc[
                    (source, role),
                    "base_evaluable_decisions"
                    if role == "train"
                    else "observed_anchor_decisions",
                ]
                if values["stored_rows"] != stored_expected:
                    raise ValueError(
                        "timeline storage count differs from audit role definition"
                    )
                rows.append(
                    {
                        "source_id": source,
                        "partition": role,
                        "horizon": horizon,
                        **values,
                    }
                )
    return _with_totals(rows, sources), {
        "status": "passed",
        "participants_read": len(seen),
        "batch_rows": BATCH_ROWS,
    }


def summarise_population(prepared: Path, output: Path) -> dict:
    _check_output(output)
    if prepared.is_symlink() or not prepared.is_dir():
        raise ValueError("prepared bundle must be an ordinary directory")
    prepared = prepared.resolve(strict=True)
    audit_path = _bundle_file(prepared, Path("input-audit.json"), "input audit")
    audit = json.loads(audit_path.read_text())
    counts = input_counts(audit)
    sources = audit["source_order"]
    endpoints, participants = _endpoints(prepared, counts, sources)
    windows, timeline_status = _timelines(prepared, counts, sources, participants)
    summary = {
        "status": "passed",
        "source_count": len(sources),
        "partitions": list(PARTITIONS),
        "horizons_minutes": list(HORIZONS),
        "original_participants": int(counts.participants.sum()),
        "participants_by_partition": {
            role: int(counts.loc[counts.partition == role, "participants"].sum())
            for role in PARTITIONS
        },
        "definitions": {
            "base_future_completeness_minutes": max(HORIZONS),
            "observed_anchor_decisions": (
                "Complete observed-anchor alarm timeline, including non-base tails."
            ),
            "interpolated_anchors_excluded": (
                "legacy_interpolated_anchor_decisions from original retained windows."
            ),
            "events": (
                "Independent prepared base catalogues at each horizon; no glucose cut."
            ),
            "exposure": (
                "All base participants, including those with no events; "
                "calendar days and actual observation minutes."
            ),
            "training_windows": (
                "Actual hypo masks/targets before training subsampling; "
                "positive windows are not distinct events."
            ),
            "role_totals": "Pooled sums across sources, not source-macro averages.",
        },
        "endpoint_role_totals": endpoints.loc[endpoints.row_type == "role_total"]
        .drop(columns=["row_type", "source_id"])
        .to_dict(orient="records"),
        "timeline_counts": timeline_status,
        "rows": {
            "input_counts": len(counts),
            "endpoint_counts": len(endpoints),
            "training_window_counts": 0 if windows is None else len(windows),
        },
    }
    files = {
        "input-counts.csv": counts,
        "endpoint-counts.csv": endpoints,
        **({"training-window-counts.csv": windows} if windows is not None else {}),
    }
    _check_output(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in files.items():
        frame.to_csv(output / name, index=False)
    (output / "population-summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarise_population(args.prepared, args.output)
    print(json.dumps({"status": result["status"], "rows": result["rows"]}))


if __name__ == "__main__":
    main()
