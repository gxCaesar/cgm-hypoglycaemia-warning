"""Run matched revision models from accepted, partition-preserving replay shards."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from .config import load_config, resolve_path
from .io import _validate_real_data_provenance, load_frame
from .revision_inputs import ReplayInputs
from .sequences import REVISION_CONTRACT


def revision_config(original: dict, *, seed: int, output: Path) -> dict:
    config = copy.deepcopy(original)
    if seed not in (17, 23, 42, 71, 101):
        raise ValueError("seed is outside the prospective revision contract")
    config["training"].update(
        seed=seed,
        revision_contract=REVISION_CONTRACT,
        event_objective="severe_retarget",
    )
    config["model"]["input_feature_mode"] = "cgm_mask_clock"
    config["output"]["run_dir"] = str(output.resolve())
    return config


def load_replay_bundle(prepared: Path, config: dict, *, sequences: bool = True):
    prepared = prepared.resolve()
    audit = json.loads((prepared / "input-audit.json").read_text())
    if (
        audit.get("status") != "passed"
        or audit.get("revision_contract") != REVISION_CONTRACT
    ):
        raise ValueError("replay inputs lack a passing revision preparation audit")
    sources = tuple(sorted(map(str, config["data"]["development_sources"])))
    if tuple(audit.get("source_order", ())) != sources:
        raise ValueError("replay source order differs from effective configuration")
    if not config["project"].get("synthetic", False):
        _validate_real_data_provenance(config)
        for setting, field in (
            ("input_path", "normalized_path"),
            ("training_window_shards", "legacy_windows"),
        ):
            configured = resolve_path(config, config["data"].get(setting))
            if (
                configured is None
                or configured.resolve() != Path(audit[field]).resolve()
            ):
                raise ValueError(f"replay {field} differs from retained input binding")
    manifest = json.loads((prepared / "participant-manifest.private.json").read_text())[
        "participants"
    ]
    if len(manifest) != audit["original_participants"]:
        raise ValueError("replay manifest count differs from audit")
    keys, frames = set(), {p: [] for p in ("train", "selection", "calibration")}
    for entry in manifest:
        key = (str(entry["source_id"]), str(entry["participant_id"]))
        partition = entry["partition"]
        if key in keys or key[0] not in sources or partition not in frames:
            raise ValueError("replay participant identity or partition is invalid")
        keys.add(key)
        path = Path(entry["timeline_path"])
        if not path.is_relative_to(prepared) or path.is_symlink() or not path.is_file():
            raise ValueError("replay timeline path escapes its accepted run")
        table = pq.ParquetFile(path).read().to_pandas()
        if len(table) != entry["stored_rows"]:
            raise ValueError("replay timeline row count differs from manifest")
        if not (
            table.source_id.astype(str).eq(key[0])
            & table.participant_id.astype(str).eq(key[1])
            & table.partition.eq(partition)
        ).all():
            raise ValueError("replay table identity differs from manifest")
        if partition == "train" and not table.base_evaluable.all():
            raise ValueError("training replay contains unevaluable tails")
        frames[partition].append(table)
    replays = {}
    for partition, tables in frames.items():
        if not tables:
            raise ValueError(f"empty revision partition: {partition}")
        timeline = pd.concat(tables, ignore_index=True)
        if set(timeline.source_id.astype(str)) != set(sources):
            raise ValueError(f"revision partition lacks a source: {partition}")
        events, exposure = {}, {}
        for horizon in (30, 60):
            for category, target in (("events", events), ("exposure", exposure)):
                path = (
                    prepared / "endpoints" / partition / f"{category}-{horizon}.parquet"
                )
                expected = Path(
                    audit["endpoint_paths"][f"{partition}/{horizon}"][category]
                )
                if path != expected or path.is_symlink():
                    raise ValueError("endpoint path differs from preparation binding")
                target[horizon] = pq.ParquetFile(path).read().to_pandas()
        replays[partition] = ReplayInputs(timeline, events, exposure, audit)
    windows = pd.concat([replays[p].timeline for p in frames], ignore_index=True)
    frame = (
        load_frame(config, sources=set(sources), participant_keys=keys)
        if sequences
        else None
    )
    return (frame, windows, audit), replays


def run_revision_deep(config: dict, prepared: Path, *, architecture: str) -> Path:
    from .deep_training import run_deep_training
    from .revision_training import RevisionTrainingContext

    if architecture not in {"lstm", "tcn"}:
        raise ValueError("revision deep arm must be lstm or tcn")
    data, replays = load_replay_bundle(prepared, config)
    sources = tuple(sorted(map(str, config["data"]["development_sources"])))
    context = RevisionTrainingContext(
        replays,
        source_order=sources,
        output_dir=Path(config["output"]["run_dir"]) / "fixed-frame",
    )
    return run_deep_training(
        config,
        architecture=architecture,
        prepared_data=data,
        revision_checkpoint_proxy=context.checkpoint_proxy,
        revision_calibrator=context.calibrate,
        revision_reporter=context.report,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--architecture", choices=("lstm", "tcn"), required=True)
    args = parser.parse_args()
    config = revision_config(
        load_config(args.config), seed=args.seed, output=args.output
    )
    run_revision_deep(config, args.prepared, architecture=args.architecture)


if __name__ == "__main__":
    main()
