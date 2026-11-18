"""Run matched revision models from accepted, partition-preserving replay shards."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from .config import load_config, resolve_path
from .io import _validate_real_data_provenance, load_frame
from .revision_crossfit_prepare import (
    INNER_ROLES,
    _bound_file,
    validated_crossfit_metadata,
)
from .revision_inputs import ReplayInputs, validate_confirmation_masks
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
    if audit.get("ascertainment_contract") != config["protocol"].get(
        "ascertainment_contract"
    ):
        raise ValueError("replay ascertainment contract differs from configuration")
    if (
        audit.get("status") != "passed"
        or audit.get("revision_contract") != REVISION_CONTRACT
    ):
        raise ValueError("replay inputs lack a passing revision preparation audit")
    if audit.get("timeline_storage", "original_partitions") != "original_partitions":
        raise ValueError("full-online shared pools require the cross-fitting loader")
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
        validate_confirmation_masks(
            table, config["protocol"].get("ascertainment_contract")
        )
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


def _participant_filter(keys: set[tuple[str, str]]):
    predicate = None
    for source in sorted({source for source, _ in keys}):
        identifiers = sorted(participant for s, participant in keys if s == source)
        term = (ds.field("source_id") == source) & ds.field("participant_id").isin(
            identifiers
        )
        predicate = term if predicate is None else predicate | term
    if predicate is None:
        raise ValueError("cross-fitting role has no participant identities")
    return predicate


def _endpoint_counts(table: pd.DataFrame) -> dict:
    """Count only real source-scoped identities, excluding unused categories."""
    return table.groupby(
        ["source_id", "participant_id"], observed=True
    ).size().to_dict()


def _crossfit_endpoints(
    shared: Path, shared_audit: dict, manifest: list[dict], keys: set, *, role: str
) -> tuple[dict, dict]:
    """Push each admitted role's identity filter into Arrow before value reads."""
    events, exposure = {}, {}
    admitted = [
        entry for entry in manifest
        if (entry["source_id"], entry["participant_id"]) in keys
    ]
    for horizon in (30, 60):
        for category, target in (("events", events), ("exposure", exposure)):
            tables = []
            for original_partition in INNER_ROLES:
                expected = (
                    shared / "endpoints" / original_partition
                    / f"{category}-{horizon}.parquet"
                )
                path = _bound_file(
                    shared_audit["endpoint_paths"][f"{original_partition}/{horizon}"][
                        category
                    ],
                    shared,
                    expected=expected,
                )
                original_keys = {
                    (entry["source_id"], entry["participant_id"])
                    for entry in manifest if entry["partition"] == original_partition
                }
                identity = pq.ParquetFile(path).read(
                    columns=["source_id", "participant_id"]
                ).to_pandas()
                identity_keys = set(
                    identity.astype(str).itertuples(index=False, name=None)
                )
                if not identity_keys.issubset(original_keys):
                    raise ValueError(
                        "cross-fitting endpoint identity differs from original binding"
                    )
                selected_keys = keys.intersection(original_keys)
                if selected_keys:
                    table = ds.dataset(path, format="parquet").to_table(
                        filter=_participant_filter(selected_keys), use_threads=False
                    ).to_pandas()
                    actual_keys = set(
                        table[["source_id", "participant_id"]].astype(str).itertuples(
                            index=False, name=None
                        )
                    )
                    if not actual_keys.issubset(selected_keys):
                        raise ValueError("cross-fitting endpoint role filter failed")
                    tables.append(table)
            if not tables:
                raise ValueError(f"cross-fitting endpoint has no source tables: {role}")
            combined = pd.concat(tables, ignore_index=True)
            counts = _endpoint_counts(combined)
            for entry in admitted:
                key = (entry["source_id"], entry["participant_id"])
                if counts.get(key, 0) != entry["endpoint_rows"][str(horizon)][category]:
                    raise ValueError(
                        "cross-fitting endpoint count differs from preparation binding"
                    )
            target[horizon] = combined
    return events, exposure


def _load_crossfit_bundle(
    prepared: Path,
    config: dict,
    *,
    outer_fold: int,
    roles: tuple[str, ...],
    sequences: bool,
):
    audit, shared_audit, manifest, metadata = validated_crossfit_metadata(
        prepared, config, outer_fold=outer_fold
    )
    selected = metadata.loc[metadata.role.isin(roles)]
    role_by_key = {
        (row.source_id, row.participant_id): row.role
        for row in selected.itertuples(index=False)
    }
    # Role authorization above precedes every timeline/endpoint value read and
    # the normalized load_frame participant predicate below.
    selected_manifest = [
        entry for entry in manifest
        if (str(entry["source_id"]), str(entry["participant_id"])) in role_by_key
    ]
    frames = {role: [] for role in roles}
    for entry in selected_manifest:
        key = (str(entry["source_id"]), str(entry["participant_id"]))
        role = role_by_key[key]
        table = pq.ParquetFile(entry["timeline_path"]).read().to_pandas()
        validate_confirmation_masks(
            table, config["protocol"].get("ascertainment_contract")
        )
        if len(table) != entry["stored_rows"]:
            raise ValueError(
                "cross-fitting timeline row count differs from preparation binding"
            )
        if not (
            table.source_id.astype(str).eq(key[0])
            & table.participant_id.astype(str).eq(key[1])
            & table.partition.eq(entry["partition"])
        ).all():
            raise ValueError("cross-fitting timeline identity differs from binding")
        table = table.copy()
        table["original_partition"] = entry["partition"]
        table["partition"] = role
        table["outer_fold"] = outer_fold
        if role == "train":
            table = table.loc[table.base_evaluable].copy()
        frames[role].append(table)
    bundle_audit = {
        **shared_audit,
        "crossfit_contract": audit["crossfit_contract"],
        "outer_fold": int(outer_fold),
        "crossfit_prepared_dir": str(Path(prepared).resolve()),
        "bundle_roles": list(roles),
        "loaded_participants": len(role_by_key),
        "timeline_storage": "fold_role_view",
    }
    replays = {}
    for role, tables in frames.items():
        if not tables:
            raise ValueError(f"cross-fitting role is empty: {role}")
        timeline = pd.concat(tables, ignore_index=True)
        keys = {key for key, assigned in role_by_key.items() if assigned == role}
        events, exposure = _crossfit_endpoints(
            Path(audit["shared_prepared"]), shared_audit, manifest, keys, role=role
        )
        replays[role] = ReplayInputs(timeline, events, exposure, bundle_audit)
    windows = pd.concat([replays[role].timeline for role in roles], ignore_index=True)
    frame = (
        load_frame(
            config,
            sources=set(audit["source_order"]),
            participant_keys=set(role_by_key),
        )
        if sequences else None
    )
    if frame is not None:
        loaded_keys = set(
            frame[["source_id", "participant_id"]].astype(str).itertuples(
                index=False, name=None
            )
        )
        if loaded_keys != set(role_by_key):
            raise ValueError("cross-fitting normalized values differ from role binding")
    return (frame, windows, bundle_audit), replays


def load_crossfit_inner_bundle(
    prepared: Path, config: dict, *, outer_fold: int, sequences: bool = True
):
    """Load only the inner train/selection/calibration roles for one outer fold."""
    return _load_crossfit_bundle(
        prepared, config, outer_fold=outer_fold, roles=INNER_ROLES, sequences=sequences
    )


def load_crossfit_outer_bundle(
    prepared: Path, config: dict, *, outer_fold: int, sequences: bool = True
):
    """Independent inference entrypoint; never used by the fitting runner."""
    data, replays = _load_crossfit_bundle(
        prepared, config, outer_fold=outer_fold,
        roles=("outer_evaluation",), sequences=sequences,
    )
    frame, _, audit = data
    audit["role"] = "outer_evaluation"
    return frame, replays["outer_evaluation"], audit


def run_revision_deep(
    config: dict, prepared: Path, *, architecture: str, outer_fold: int | None = None
) -> Path:
    from .deep_training import run_deep_training
    from .revision_training import RevisionTrainingContext

    if architecture not in {"lstm", "tcn"}:
        raise ValueError("revision deep arm must be lstm or tcn")
    data, replays = (
        load_replay_bundle(prepared, config)
        if outer_fold is None
        else load_crossfit_inner_bundle(prepared, config, outer_fold=outer_fold)
    )
    sources = tuple(sorted(map(str, config["data"]["development_sources"])))
    context = RevisionTrainingContext(
        replays,
        source_order=sources,
        output_dir=Path(config["output"]["run_dir"]) / "fixed-frame",
        sensitivity_source_policy=(
            "all_sources" if outer_fold is None else "supported_sources"
        ),
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
    parser.add_argument("--outer-fold", type=int, choices=range(5))
    args = parser.parse_args()
    config = revision_config(
        load_config(args.config), seed=args.seed, output=args.output
    )
    run_revision_deep(
        config, args.prepared,
        architecture=args.architecture, outer_fold=args.outer_fold,
    )


if __name__ == "__main__":
    main()
