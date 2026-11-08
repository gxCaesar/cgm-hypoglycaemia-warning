"""Prepare prospective replay shards without resplitting retained participants.

This is a remote, private analysis utility. Participant-level outputs are not a
public release and must remain with the approved normalized inputs.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from .config import load_config
from .revision_inputs import build_participant_replay

KEYS = ["source_id", "participant_id"]
CUTS = (54, 60, 70, 80, 90, 100)


def _json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")


def _fragment_frame(fragment, columns=None) -> pd.DataFrame:
    physical = set(fragment.physical_schema.names)
    selected = None if columns is None else [c for c in columns if c in physical]
    frame = fragment.to_table(columns=selected, use_threads=False).to_pandas()
    for name, value in ds.get_partition_keys(fragment.partition_expression).items():
        if name in frame and not frame[name].astype(str).eq(str(value)).all():
            raise ValueError(f"hive/file identity mismatch: {name}")
        frame[name] = value
    return frame


def _index_fragments(path: Path, sources: set[str], *, windows: bool) -> dict:
    dataset = ds.dataset(path, format="parquet", partitioning="hive")
    if "source_id" not in dataset.schema.names:
        raise ValueError("dataset has no source identity")
    mapping = defaultdict(list)
    partitions = {}
    for fragment in dataset.get_fragments(
        filter=ds.field("source_id").isin(sorted(sources))
    ):
        columns = KEYS + (["partition"] if windows else [])
        identifiers = _fragment_frame(fragment, columns).drop_duplicates()
        if len(identifiers) != 1:
            raise ValueError(
                "replay preparation requires participant-specific input shards"
            )
        row = identifiers.iloc[0]
        key = tuple(str(row[c]) for c in KEYS)
        if key[0] not in sources:
            raise ValueError("source filter failed before record ingestion")
        if windows:
            partition = str(row["partition"])
            if partition not in {"train", "selection", "calibration"}:
                raise ValueError("legacy shard contains a non-development partition")
            if key in partitions and partitions[key] != partition:
                raise ValueError("legacy participant overlaps partitions")
            partitions[key] = partition
        mapping[key].append(fragment)
    return {key: (fragments, partitions.get(key)) for key, fragments in mapping.items()}


def _concat(fragments) -> pd.DataFrame:
    return pd.concat([_fragment_frame(f) for f in fragments], ignore_index=True)


def aggregate_denominators(rows) -> pd.DataFrame:
    return (
        pd.DataFrame(rows)
        .groupby(
            ["source_id", "partition", "horizon", "cut"],
            as_index=False,
            observed=True,
        )
        .sum()
    )


def _validate_legacy_binding(
    old: pd.DataFrame, normalized: pd.DataFrame, replay
) -> dict:
    old = old.copy()
    old["context_end"] = pd.to_datetime(old["context_end"], utc=True)
    if old.context_end.duplicated().any():
        raise ValueError("duplicate legacy decision key")
    normalized = normalized.sort_values("timestamp").reset_index(drop=True)
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], utc=True)
    at = normalized.set_index("timestamp").reindex(old.context_end)
    if at.cgm_mg_dl.isna().any():
        raise ValueError("legacy decision missing from retained normalized input")
    if not np.allclose(at.cgm_mg_dl, old.cgm_last, rtol=0, atol=1e-5):
        raise ValueError("legacy/normalized anchor glucose mismatch")
    keep = (at.cgm_observed & ~at.cgm_interpolated & ~at.cgm_lower_censored).to_numpy()
    expected = old.loc[keep].set_index("context_end").sort_index()
    actual = (
        replay.timeline.loc[replay.timeline.base_evaluable]
        .set_index("context_end")
        .sort_index()
    )
    if not expected.index.equals(actual.index):
        raise ValueError(
            "observed legacy/replay base keys differ beyond "
            "interpolated-anchor exclusion: "
            f"legacy_only={len(expected.index.difference(actual.index))}, "
            f"replay_only={len(actual.index.difference(expected.index))}"
        )
    for horizon in (30, 60):
        for kind in ("hypo", "severe_hypo", "hyper"):
            column = f"{kind}_{horizon}_label"
            if not np.array_equal(
                expected[column].to_numpy(), actual[column].to_numpy()
            ):
                raise ValueError(f"legacy/replay outcome mismatch: {column}")
    for column in ("cgm_mean", "cgm_std", "cgm_slope", "cgm_slope_30"):
        if not np.allclose(expected[column], actual[column], rtol=1e-5, atol=1e-4):
            raise ValueError(f"legacy/replay input feature mismatch: {column}")
    return {
        "legacy_decisions": len(old),
        "legacy_interpolated_anchor_decisions": int(at.cgm_interpolated.sum()),
        "legacy_nonobserved_anchor_decisions": int((~at.cgm_observed).sum()),
        "legacy_observed_retained_decisions": int(keep.sum()),
        "legacy_key_outcome_feature_binding_passed": True,
    }


def prepare_replays(
    normalized_path: Path, legacy_windows: Path, output: Path, config: dict
) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite replay output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    sources = set(map(str, config["data"]["development_sources"]))
    if sources.intersection(map(str, config["data"]["forbidden_development_sources"])):
        raise ValueError("development and forbidden sources overlap")
    old_index = _index_fragments(legacy_windows, sources, windows=True)
    normalized_index = _index_fragments(normalized_path, sources, windows=False)
    if not old_index or not set(old_index).issubset(normalized_index):
        raise ValueError(
            "retained normalized inputs do not cover original participants"
        )
    if {key[0] for key in old_index} != sources:
        raise ValueError("original participant shards do not cover development sources")
    aggregate = defaultdict(lambda: defaultdict(int))
    denominator_rows, manifests, event_tables, exposure_tables = (
        [],
        [],
        defaultdict(list),
        defaultdict(list),
    )
    try:
        for ordinal, key in enumerate(sorted(old_index)):
            old_fragments, partition = old_index[key]
            norm_fragments, _ = normalized_index[key]
            old, normalized = _concat(old_fragments), _concat(norm_fragments)
            replay = build_participant_replay(normalized, config, partition=partition)
            binding = _validate_legacy_binding(old, normalized, replay)
            if replay.exposure[30].empty:
                raise ValueError(
                    "an original participant has no observed-anchor base exposure"
                )
            summary = aggregate[(key[0], partition)]
            summary["participants"] += 1
            for column, value in {**replay.audit, **binding}.items():
                if isinstance(value, int) and not isinstance(value, bool):
                    summary[column] += value
            timeline = replay.timeline
            # Evaluation tails affect refractory state. Training has no outcome
            # supervision there and therefore stores base rows only.
            stored = (
                timeline.loc[timeline.base_evaluable]
                if partition == "train"
                else timeline
            )
            shard = (
                output
                / "timelines"
                / f"partition={partition}"
                / f"part-{ordinal:05d}.parquet"
            )
            shard.parent.mkdir(parents=True, exist_ok=True)
            stored.to_parquet(shard, index=False)
            manifests.append(
                {
                    "source_id": key[0],
                    "participant_id": key[1],
                    "partition": partition,
                    "timeline_path": str(shard),
                    "stored_rows": len(stored),
                    "normalized_paths": [str(f.path) for f in norm_fragments],
                    "legacy_paths": [str(f.path) for f in old_fragments],
                }
            )
            for horizon in (30, 60):
                event_tables[(partition, horizon)].append(replay.events[horizon])
                exposure_tables[(partition, horizon)].append(replay.exposure[horizon])
                base_times = timeline.loc[timeline.base_evaluable]
                base_onsets = replay.events[horizon].event_onset
                for cut in CUTS:
                    eligible = base_times.loc[base_times.cgm_last >= cut]
                    legacy_cut = old.loc[old.cgm_last >= cut]
                    times = (
                        pd.to_datetime(eligible.context_end, utc=True)
                        .astype("int64")
                        .to_numpy()
                    )
                    onsets = (
                        pd.to_datetime(base_onsets, utc=True).astype("int64").to_numpy()
                    )
                    reach = np.searchsorted(times, onsets) > np.searchsorted(
                        times, onsets - horizon * 60_000_000_000
                    )
                    denominator_rows.append(
                        {
                            "source_id": key[0],
                            "partition": partition,
                            "horizon": horizon,
                            "cut": cut,
                            "base_participants": 1,
                            "eligible_participants": int(len(eligible) > 0),
                            "base_days": int(
                                replay.exposure[horizon].patient_days.sum()
                            ),
                            "base_observation_minutes": int(
                                replay.exposure[horizon].observation_minutes.sum()
                            ),
                            "eligible_days": int(eligible.patient_day.nunique()),
                            "base_decisions": len(base_times),
                            "eligible_decisions": len(eligible),
                            "base_events": len(base_onsets),
                            "reachable_events": int(reach.sum()),
                            "legacy_filtered_days": int(
                                legacy_cut.patient_day.nunique()
                            ),
                            "legacy_filtered_label_events": int(
                                legacy_cut[f"severe_hypo_{horizon}_event_id"].nunique()
                            ),
                        }
                    )
            if (ordinal + 1) % 25 == 0:
                print(
                    json.dumps(
                        {"processed_participants": ordinal + 1, "total": len(old_index)}
                    ),
                    flush=True,
                )
        endpoint_paths = {}
        for (partition, horizon), tables in event_tables.items():
            directory = output / "endpoints" / partition
            directory.mkdir(parents=True, exist_ok=True)
            events_path, exposure_path = (
                directory / f"events-{horizon}.parquet",
                directory / f"exposure-{horizon}.parquet",
            )
            pd.concat(tables, ignore_index=True).to_parquet(events_path, index=False)
            pd.concat(
                exposure_tables[(partition, horizon)], ignore_index=True
            ).to_parquet(exposure_path, index=False)
            endpoint_paths[f"{partition}/{horizon}"] = {
                "events": str(events_path),
                "exposure": str(exposure_path),
            }
        aggregate_denominators(denominator_rows).to_csv(
            output / "denominator-audit.csv", index=False
        )
        summary = {
            "revision_contract": "2026-11-04-fixed-frame",
            "status": "passed",
            "normalized_path": str(normalized_path),
            "legacy_windows": str(legacy_windows),
            "source_order": sorted(sources),
            "original_participants": len(old_index),
            "normalized_participants": len(normalized_index),
            "original_excluded_participants_not_added": len(
                set(normalized_index) - set(old_index)
            ),
            "source_partition_counts": [
                {"source_id": key[0], "partition": key[1], **values}
                for key, values in sorted(aggregate.items())
            ],
            "endpoint_paths": endpoint_paths,
            "action_arrival_timestamp_available": False,
            "action_timing_conclusion": (
                "No verified arrival-time contract; main comparison uses CGM, "
                "observation mask and clock only."
            ),
        }
        _json(output / "participant-manifest.private.json", {"participants": manifests})
        _json(output / "input-audit.json", summary)
        return summary
    except Exception as error:
        _json(
            output / "preparation-failure.json",
            {
                "status": "failed",
                "error": str(error),
                "completed_participants": len(manifests),
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--normalized", type=Path, required=True)
    parser.add_argument("--legacy-windows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    prepare_replays(
        args.normalized, args.legacy_windows, args.output, load_config(args.config)
    )


if __name__ == "__main__":
    main()
