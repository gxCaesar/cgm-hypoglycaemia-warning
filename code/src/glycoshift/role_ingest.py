from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from .adapters import (
    METABONET_ALIASES,
    SourceMap,
    load_source_map,
    normalize_hupa,
    normalize_metabonet,
    sha256_file,
)
from .config import load_config, resolve_path
from .io import write_json
from .schema import ACTION_FIELDS

METABONET_ROLES = {"development", "external_loop"}
HUPA_MEMBER_PATTERN = re.compile(
    r"^HUPA-UCM Diabetes Dataset/Preprocessed/(HUPA\d{4}P)\.csv$"
)


def _prepare_output(output_dir: str | Path, manifest_path: str | Path) -> Path:
    output = Path(output_dir).expanduser().resolve()
    manifest = Path(manifest_path).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty dataset: {output}")
    if manifest.exists():
        raise FileExistsError(f"refusing to overwrite manifest: {manifest}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _exact_raw_values(
    source_map: SourceMap,
    canonical_sources: set[str],
) -> dict[str, str]:
    selected: dict[str, str] = {}
    for source in sorted(canonical_sources):
        rules = [rule for rule in source_map.rules if rule.source_id == source]
        if len(rules) != 1 or rules[0].value is None:
            raise ValueError(
                "role ingestion requires one exact reviewed value for "
                f"source {source!r}"
            )
        selected[source] = str(rules[0].value)
    raw_values = list(selected.values())
    if len(raw_values) != len(set(raw_values)):
        raise ValueError("reviewed role source values must be unique")
    return selected


def _resolve_columns(schema_names: set[str]) -> dict[str, str | None]:
    resolved: dict[str, str | None] = {}
    for canonical, aliases in METABONET_ALIASES.items():
        resolved[canonical] = next(
            (candidate for candidate in aliases if candidate in schema_names),
            None,
        )
    missing = [
        name
        for name in ("participant_id", "timestamp", "cgm_mg_dl")
        if resolved[name] is None
    ]
    if missing:
        raise ValueError(f"MetaboNet release is missing core columns: {missing}")
    return resolved


def _selected_filter(source_map: SourceMap, raw_values: set[str]) -> ds.Expression:
    return ds.field(source_map.source_column).isin(sorted(raw_values))


def _source_preflight(
    dataset: ds.Dataset,
    *,
    source_map: SourceMap,
    raw_values: set[str],
    resolved: dict[str, str | None],
    batch_size: int,
    glucose_minimum_mg_dl: float,
    glucose_maximum_mg_dl: float,
) -> tuple[dict[str, dict[str, bool]], dict[str, Any]]:
    availability: dict[str, dict[str, bool]] = defaultdict(
        lambda: {field: False for field in ACTION_FIELDS}
    )
    action_columns = {
        field: resolved[field]
        for field in ACTION_FIELDS
        if resolved[field] is not None
    }
    cgm_column = str(resolved["cgm_mg_dl"])
    columns = [
        source_map.source_column,
        cgm_column,
        *dict.fromkeys(action_columns.values()),
    ]
    audit: dict[str, Any] = {
        "minimum_mg_dl": float(glucose_minimum_mg_dl),
        "maximum_mg_dl": float(glucose_maximum_mg_dl),
        "nonmissing_rows": 0,
        "out_of_range_rows": 0,
        "below_minimum_rows": 0,
        "above_maximum_rows": 0,
        "by_source": {},
        "observed_out_of_range_values_mg_dl": [],
    }
    observed_invalid_values: set[float] = set()
    for batch in dataset.to_batches(
        columns=columns,
        filter=_selected_filter(source_map, raw_values),
        batch_size=batch_size,
    ):
        frame = batch.to_pandas()
        for raw_source, group in frame.groupby(
            source_map.source_column,
            sort=False,
        ):
            canonical = source_map.map_value(raw_source)
            glucose = pd.to_numeric(group[cgm_column], errors="coerce")
            below = glucose.notna() & glucose.lt(glucose_minimum_mg_dl)
            above = glucose.notna() & glucose.gt(glucose_maximum_mg_dl)
            invalid = below | above
            audit["nonmissing_rows"] += int(glucose.notna().sum())
            audit["out_of_range_rows"] += int(invalid.sum())
            audit["below_minimum_rows"] += int(below.sum())
            audit["above_maximum_rows"] += int(above.sum())
            if invalid.any():
                audit["by_source"][canonical] = int(
                    audit["by_source"].get(canonical, 0)
                ) + int(invalid.sum())
                observed_invalid_values.update(
                    map(float, glucose.loc[invalid].unique())
                )
            for field, column in action_columns.items():
                availability[canonical][field] = (
                    availability[canonical][field]
                    or bool(
                        pd.to_numeric(group[str(column)], errors="coerce")
                        .notna()
                        .any()
                    )
                )
    audit["by_source"] = dict(sorted(audit["by_source"].items()))
    audit["observed_out_of_range_values_mg_dl"] = sorted(
        observed_invalid_values
    )
    return dict(availability), audit


def _approve_glucose_range_audit(
    audit: dict[str, Any],
    *,
    input_file: Path,
    policy: str,
    allowed_sources: set[str],
    maximum_rows: int,
    approved_values_mg_dl: tuple[float, ...],
    expected_input_sha256: str | None,
) -> str | None:
    out_of_range = int(audit["out_of_range_rows"])
    audit["policy"] = policy
    verified_hash: str | None = None
    if policy in {"exclude_noninterpolable", "left_censor"}:
        if expected_input_sha256 is None:
            raise ValueError(
                "approved glucose-range exception requires a release SHA-256"
            )
        expected_hash = expected_input_sha256.lower()
        verified_hash = sha256_file(input_file)
        if verified_hash != expected_hash:
            raise ValueError(
                "MetaboNet release SHA-256 does not match the approved "
                "glucose-range amendment"
            )
    if out_of_range == 0:
        audit["action"] = "none"
        audit["release_sha256_verified"] = verified_hash is not None
        return verified_hash
    if policy not in {"exclude_noninterpolable", "left_censor"}:
        raise ValueError(
            f"cgm_mg_dl contains {out_of_range} values outside "
            f"{audit['minimum_mg_dl']:g}–{audit['maximum_mg_dl']:g}"
        )
    if int(audit["above_maximum_rows"]):
        raise ValueError(
            "approved glucose-range exception covers lower-bound records only"
        )
    observed_sources = set(map(str, audit["by_source"]))
    unexpected_sources = sorted(observed_sources.difference(allowed_sources))
    if unexpected_sources:
        raise ValueError(
            "out-of-range glucose occurred in unapproved sources: "
            f"{unexpected_sources}"
        )
    if out_of_range > maximum_rows:
        raise ValueError(
            "out-of-range glucose count exceeds approved maximum: "
            f"{out_of_range} > {maximum_rows}"
        )
    observed_values = [
        float(value)
        for value in audit["observed_out_of_range_values_mg_dl"]
    ]
    unapproved_values = [
        value
        for value in observed_values
        if not any(
            abs(value - approved) <= 1e-9
            for approved in approved_values_mg_dl
        )
    ]
    if unapproved_values:
        raise ValueError(
            "out-of-range glucose contains unapproved values: "
            f"{unapproved_values}"
        )
    audit["action"] = (
        "excluded_without_interpolation"
        if policy == "exclude_noninterpolable"
        else "left_censored"
    )
    audit["release_sha256_verified"] = True
    return verified_hash


def _iter_raw_participants(
    dataset: ds.Dataset,
    *,
    source_map: SourceMap,
    raw_values: set[str],
    resolved: dict[str, str | None],
    batch_size: int,
) -> Iterator[pd.DataFrame]:
    columns = [
        source_map.source_column,
        *(
            column
            for column in resolved.values()
            if column is not None
        ),
    ]
    columns = list(dict.fromkeys(columns))
    participant_column = str(resolved["participant_id"])
    carry = pd.DataFrame()
    completed: set[str] = set()
    for batch in dataset.to_batches(
        columns=columns,
        filter=_selected_filter(source_map, raw_values),
        batch_size=batch_size,
    ):
        frame = batch.to_pandas()
        if frame.empty:
            continue
        combined = pd.concat((carry, frame), ignore_index=True)
        combined["_participant_key"] = (
            combined[source_map.source_column].astype(str)
            + "\x1f"
            + combined[participant_column].astype(str)
        )
        last_key = str(combined["_participant_key"].iloc[-1])
        ready = combined.loc[combined["_participant_key"] != last_key]
        carry = combined.loc[
            combined["_participant_key"] == last_key
        ].drop(columns="_participant_key")
        for key, participant in ready.groupby("_participant_key", sort=False):
            string_key = str(key)
            if string_key in completed:
                raise ValueError(
                    "raw MetaboNet release is not participant-contiguous; "
                    f"repeated key {string_key!r}"
                )
            completed.add(string_key)
            yield participant.drop(
                columns="_participant_key"
            ).reset_index(drop=True)
    if not carry.empty:
        final_key = (
            str(carry[source_map.source_column].iloc[0])
            + "\x1f"
            + str(carry[participant_column].iloc[0])
        )
        if final_key in completed:
            raise ValueError(
                "raw MetaboNet release repeats the final participant key"
            )
        yield carry.reset_index(drop=True)


def _write_participant(frame: pd.DataFrame, output: Path) -> tuple[str, str]:
    sources = frame["source_id"].astype(str).unique()
    participants = frame["participant_id"].astype(str).unique()
    if len(sources) != 1 or len(participants) != 1:
        raise ValueError("each normalized shard must contain one participant")
    source = str(sources[0])
    participant = str(participants[0])
    if Path(source).name != source:
        raise ValueError(f"unsafe source identifier: {source!r}")
    digest = hashlib.sha256(
        f"{source}\x1f{participant}".encode()
    ).hexdigest()[:20]
    partition_dir = output / f"source_id={source}"
    partition_dir.mkdir(parents=True, exist_ok=True)
    destination = partition_dir / f"participant-{digest}.parquet"
    if destination.exists():
        raise FileExistsError(f"duplicate participant shard: {destination}")
    table = pa.Table.from_pandas(
        frame.drop(columns="source_id"),
        preserve_index=False,
    )
    pq.write_table(table, destination, compression="zstd")
    return source, participant


def _finalize_manifest(
    *,
    input_path: Path,
    manifest_path: str | Path,
    release_id: str,
    role: str,
    origin: str,
    source_map_name: str | None,
    columns: list[str],
    source_rows: dict[str, int],
    source_participants: dict[str, set[str]],
    input_sha256: str | None = None,
    quality_audit: dict[str, Any] | None = None,
    duplicate_participant_exclusion: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = {
        "release_id": str(release_id),
        "input_file": input_path.name,
        "sha256": input_sha256 or sha256_file(input_path),
        "source_map": source_map_name,
        "role": role,
        "origin": origin,
        "columns": columns,
        "sources": {
            source: {
                "participants": len(source_participants[source]),
                "rows": int(rows),
            }
            for source, rows in sorted(source_rows.items())
        },
    }
    if quality_audit is not None:
        manifest["glucose_range_audit"] = quality_audit
    # Its own field, not folded into the glucose-range audit. Two amendments with
    # different approvals and different evidence do not belong in one record.
    if duplicate_participant_exclusion is not None:
        manifest["duplicate_participant_exclusion"] = duplicate_participant_exclusion
    write_json(manifest, manifest_path)
    return manifest


def ingest_metabonet_role(
    input_path: str | Path,
    output_dir: str | Path,
    manifest_path: str | Path,
    *,
    source_map: SourceMap,
    canonical_sources: set[str],
    role: str,
    release_id: str,
    batch_size: int = 250_000,
    source_map_name: str | None = None,
    glucose_range_policy: str = "fail",
    glucose_minimum_mg_dl: float = 20.0,
    glucose_maximum_mg_dl: float = 600.0,
    lower_censor_proxy_mg_dl: float = 39.0,
    allowed_invalid_glucose_sources: set[str] | None = None,
    maximum_invalid_glucose_rows: int = 0,
    approved_invalid_glucose_values_mg_dl: tuple[float, ...] = (),
    expected_input_sha256: str | None = None,
    expected_release_id: str | None = None,
    excluded_participants: frozenset[tuple[str, str]] = frozenset(),
    expected_excluded_rows: int | None = None,
) -> dict[str, Any]:
    if role not in METABONET_ROLES:
        raise ValueError(f"unsupported MetaboNet role: {role}")
    if not canonical_sources:
        raise ValueError("role ingestion requires at least one source")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if (
        expected_release_id is not None
        and str(release_id) != str(expected_release_id)
    ):
        raise ValueError(
            "MetaboNet release ID does not match the approved "
            "glucose-range amendment"
        )
    input_file = Path(input_path).expanduser().resolve()
    if not input_file.is_file():
        raise FileNotFoundError(f"MetaboNet release not found: {input_file}")
    output_candidate = Path(output_dir).expanduser().resolve()
    manifest_candidate = Path(manifest_path).expanduser().resolve()
    if output_candidate.exists() and any(output_candidate.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty dataset: {output_candidate}"
        )
    if manifest_candidate.exists():
        raise FileExistsError(
            f"refusing to overwrite manifest: {manifest_candidate}"
        )
    selected = _exact_raw_values(source_map, canonical_sources)
    dataset = ds.dataset(input_file, format="parquet")
    schema_names = set(dataset.schema.names)
    if source_map.source_column not in schema_names:
        raise ValueError(
            "reviewed source column is absent from MetaboNet release: "
            f"{source_map.source_column}"
        )
    resolved = _resolve_columns(schema_names)
    raw_values = set(selected.values())
    print(
        json.dumps(
            {
                "event": "ingest_stage",
                "stage": "source_preflight_started",
                "selected_sources": len(canonical_sources),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    availability, glucose_range_audit = _source_preflight(
        dataset,
        source_map=source_map,
        raw_values=raw_values,
        resolved=resolved,
        batch_size=batch_size,
        glucose_minimum_mg_dl=glucose_minimum_mg_dl,
        glucose_maximum_mg_dl=glucose_maximum_mg_dl,
    )
    verified_input_sha256 = _approve_glucose_range_audit(
        glucose_range_audit,
        input_file=input_file,
        policy=glucose_range_policy,
        allowed_sources=allowed_invalid_glucose_sources or set(),
        maximum_rows=maximum_invalid_glucose_rows,
        approved_values_mg_dl=approved_invalid_glucose_values_mg_dl,
        expected_input_sha256=expected_input_sha256,
    )
    print(
        json.dumps(
            {
                "event": "ingest_stage",
                "stage": "source_preflight_completed",
                "glucose_range_audit": glucose_range_audit,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    output = _prepare_output(output_dir, manifest_path)
    print(
        json.dumps(
            {
                "event": "ingest_stage",
                "stage": "participant_normalization_started",
            },
            sort_keys=True,
        ),
        flush=True,
    )
    source_rows: dict[str, int] = defaultdict(int)
    source_participants: dict[str, set[str]] = defaultdict(set)
    columns: list[str] | None = None
    participants_processed = 0
    # DATA_QUALITY_AMENDMENT_LOOP_DUPLICATE_PARTICIPANTS.md. Two Loop identifiers carry
    # the same ninety-two CGM days, and the endpoint is an equal-source PARTICIPANT
    # macro, so one record enrolled twice would be weighted twice. The exclusion is
    # bound to exactly which participants and exactly how many rows, the same way the
    # T1D-UOM lower-range exception is bound, so a different release cannot inherit it.
    excluded_rows = 0
    excluded_seen: set[tuple[str, str]] = set()
    for raw_participant in _iter_raw_participants(
        dataset,
        source_map=source_map,
        raw_values=raw_values,
        resolved=resolved,
        batch_size=batch_size,
    ):
        normalized = normalize_metabonet(
            raw_participant,
            source_map=source_map,
            action_availability=availability,
            glucose_range_policy=glucose_range_policy,
            glucose_minimum_mg_dl=glucose_minimum_mg_dl,
            glucose_maximum_mg_dl=glucose_maximum_mg_dl,
            lower_censor_proxy_mg_dl=lower_censor_proxy_mg_dl,
        )
        observed_sources = set(normalized["source_id"].astype(str))
        if not observed_sources.issubset(canonical_sources):
            raise ValueError(
                "role ingestion produced an unselected source: "
                f"{sorted(observed_sources.difference(canonical_sources))}"
            )
        identity = (
            str(normalized["source_id"].iloc[0]),
            str(normalized["participant_id"].iloc[0]),
        )
        if identity in excluded_participants:
            excluded_rows += len(normalized)
            excluded_seen.add(identity)
            continue
        source, participant = _write_participant(normalized, output)
        source_rows[source] += len(normalized)
        source_participants[source].add(participant)
        columns = columns or list(normalized.columns)
        participants_processed += 1
        if participants_processed == 1 or participants_processed % 25 == 0:
            print(
                json.dumps(
                    {
                        "event": "ingest_progress",
                        "participants_processed": participants_processed,
                        "rows_written": sum(source_rows.values()),
                        "sources_started": len(source_rows),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    # The approval names exactly which participants and exactly how many of their rows.
    # A release where either differs has not been reviewed, so it does not inherit this.
    if excluded_participants:
        if excluded_seen != set(excluded_participants):
            raise ValueError(
                "approved duplicate-participant exclusion does not match the data: "
                f"approved={sorted(excluded_participants)} "
                f"present={sorted(excluded_seen)}"
            )
        if (
            expected_excluded_rows is not None
            and excluded_rows != expected_excluded_rows
        ):
            raise ValueError(
                "approved duplicate-participant exclusion covers "
                f"{expected_excluded_rows} rows, this release has {excluded_rows}"
            )
    elif expected_excluded_rows:
        raise ValueError(
            "an excluded row count was approved with no participants to exclude"
        )
    missing_sources = sorted(canonical_sources.difference(source_rows))
    if missing_sources:
        raise ValueError(
            f"selected MetaboNet sources have no rows: {missing_sources}"
        )
    exclusion_record = None
    if excluded_participants:
        exclusion_record = {
            "participants": sorted(
                f"{source}/{participant}" for source, participant in excluded_seen
            ),
            "rows_excluded": int(excluded_rows),
            "amendment": "DATA_QUALITY_AMENDMENT_LOOP_DUPLICATE_PARTICIPANTS.md",
        }
    return _finalize_manifest(
        input_path=input_file,
        manifest_path=manifest_path,
        release_id=release_id,
        role=role,
        origin="metabonet-public",
        source_map_name=source_map_name,
        columns=columns or [],
        source_rows=dict(source_rows),
        source_participants=dict(source_participants),
        input_sha256=verified_input_sha256,
        quality_audit=glucose_range_audit,
        duplicate_participant_exclusion=exclusion_record,
    )


def ingest_hupa_archive(
    input_path: str | Path,
    output_dir: str | Path,
    manifest_path: str | Path,
    *,
    release_id: str,
    expected_participant_count: int = 25,
) -> dict[str, Any]:
    input_file = Path(input_path).expanduser().resolve()
    if not input_file.is_file():
        raise FileNotFoundError(f"HUPA archive not found: {input_file}")
    output = _prepare_output(output_dir, manifest_path)
    source_rows: dict[str, int] = defaultdict(int)
    source_participants: dict[str, set[str]] = defaultdict(set)
    columns: list[str] | None = None
    with ZipFile(input_file) as archive:
        members: list[tuple[str, str]] = []
        for name in archive.namelist():
            match = HUPA_MEMBER_PATTERN.fullmatch(name)
            if match is not None:
                members.append((name, match.group(1)))
        if len(members) != expected_participant_count:
            raise ValueError(
                "unexpected HUPA preprocessed participant count: "
                f"{len(members)} != {expected_participant_count}"
            )
        participant_ids = [participant for _, participant in members]
        if len(participant_ids) != len(set(participant_ids)):
            raise ValueError("duplicate HUPA preprocessed participant files")
        for member, participant in sorted(members):
            with archive.open(member) as handle:
                raw = pd.read_csv(handle, sep=";")
            normalized = normalize_hupa(
                raw,
                participant_id=participant,
            )
            source, normalized_participant = _write_participant(
                normalized,
                output,
            )
            source_rows[source] += len(normalized)
            source_participants[source].add(normalized_participant)
            columns = columns or list(normalized.columns)
    return _finalize_manifest(
        input_path=input_file,
        manifest_path=manifest_path,
        release_id=release_id,
        role="stress_hupa_ucm",
        origin="mendeley-direct",
        source_map_name=None,
        columns=columns or [],
        source_rows=dict(source_rows),
        source_participants=dict(source_participants),
    )


def _role_destination(
    config: dict[str, Any],
    role: str,
) -> tuple[Path, Path, set[str]]:
    if role == "development":
        output = resolve_path(config, config["data"].get("input_path"))
        manifest = resolve_path(config, config["data"].get("source_manifest"))
        sources = set(map(str, config["data"]["development_sources"]))
    else:
        specification = config["data"]["evaluation_inputs"][role]
        output = resolve_path(config, specification.get("input_path"))
        manifest = resolve_path(config, specification.get("source_manifest"))
        sources = {str(specification["expected_source"])}
    if output is None or manifest is None:
        raise ValueError(f"{role} output and manifest paths are required")
    return output, manifest, sources


def _duplicate_exclusion(
    config: dict[str, Any],
    role: str,
    release_id: str,
    range_policy: dict[str, Any],
) -> dict[str, Any]:
    """The approved duplicate-participant exclusion, or nothing.

    It applies to one role of one release. A different role, a different release
    identifier or a different release hash gets an empty exclusion rather than an
    inherited approval, and the ingestion then fails on the duplicate-day gate exactly
    as it did before this amendment existed.
    """
    block = config.get("data", {}).get("duplicate_participant_exclusion")
    if not block:
        return {}
    if str(block.get("role")) != str(role):
        return {}
    if str(block.get("approved_release_id")) != str(release_id):
        raise ValueError(
            "duplicate-participant exclusion is approved for release "
            f"{block.get('approved_release_id')!r}, not {release_id!r}"
        )
    if str(block.get("approved_release_sha256")) != str(
        range_policy.get("approved_release_sha256")
    ):
        raise ValueError(
            "duplicate-participant exclusion names a different release hash from "
            "the glucose-range amendment"
        )
    source = str(block["source"])
    return {
        "excluded_participants": frozenset(
            (source, str(item)) for item in block["participants"]
        ),
        "expected_excluded_rows": int(block["expected_excluded_rows"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize one frozen CGM dataset role at a time"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    metabonet = subparsers.add_parser("metabonet")
    metabonet.add_argument("--config", required=True)
    metabonet.add_argument("--input", required=True)
    metabonet.add_argument(
        "--role",
        choices=tuple(sorted(METABONET_ROLES)),
        required=True,
    )
    metabonet.add_argument("--release-id", required=True)
    metabonet.add_argument("--batch-size", type=int, default=250_000)
    hupa = subparsers.add_parser("hupa")
    hupa.add_argument("--config", required=True)
    hupa.add_argument("--input", required=True)
    hupa.add_argument(
        "--release-id",
        default="doi:10.17632/3hbcscwz44.1",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "metabonet":
        output, manifest, sources = _role_destination(config, args.role)
        source_map_path = resolve_path(
            config,
            config["data"].get("source_map"),
        )
        if source_map_path is None or not source_map_path.is_file():
            raise FileNotFoundError("reviewed MetaboNet source map is absent")
        range_policy = config["data"].get("glucose_range", {})
        payload = ingest_metabonet_role(
            args.input,
            output,
            manifest,
            source_map=load_source_map(source_map_path),
            canonical_sources=sources,
            role=args.role,
            release_id=args.release_id,
            batch_size=args.batch_size,
            source_map_name=source_map_path.name,
            glucose_range_policy=str(range_policy.get("policy", "fail")),
            glucose_minimum_mg_dl=float(
                range_policy.get("minimum_mg_dl", 20)
            ),
            glucose_maximum_mg_dl=float(
                range_policy.get("maximum_mg_dl", 600)
            ),
            lower_censor_proxy_mg_dl=float(
                range_policy.get("lower_censor_proxy_mg_dl", 39)
            ),
            allowed_invalid_glucose_sources=set(
                map(str, range_policy.get("allowed_sources", []))
            ),
            maximum_invalid_glucose_rows=int(
                range_policy.get("maximum_rows", 0)
            ),
            approved_invalid_glucose_values_mg_dl=tuple(
                map(
                    float,
                    range_policy.get(
                        "approved_values_mg_dl",
                        [],
                    ),
                )
            ),
            expected_input_sha256=range_policy.get(
                "approved_release_sha256"
            ),
            expected_release_id=range_policy.get("approved_release_id"),
            **_duplicate_exclusion(
                config, args.role, args.release_id, range_policy
            ),
        )
    else:
        output, manifest, _ = _role_destination(
            config,
            "stress_hupa_ucm",
        )
        payload = ingest_hupa_archive(
            args.input,
            output,
            manifest,
            release_id=args.release_id,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
