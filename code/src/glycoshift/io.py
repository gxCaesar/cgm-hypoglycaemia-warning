from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .config import resolve_path
from .schema import validate_frame
from .synthetic import make_synthetic_cgm


def _read_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)
    if not isinstance(manifest, dict):
        raise ValueError("source manifest must be a YAML mapping")
    return manifest


def _validate_manifest(
    manifest_path: Path,
    *,
    expected_sources: set[str],
    role: str,
    required_origin: str,
    forbidden_sources: set[str] | None = None,
) -> dict[str, Any]:
    if manifest_path is None or not manifest_path.is_file():
        raise FileNotFoundError(
            f"real-data role {role} requires a reviewed local manifest"
        )
    manifest = _read_manifest(manifest_path)
    required = {
        "release_id",
        "input_file",
        "sha256",
        "source_map",
        "sources",
        "role",
        "origin",
    }
    missing = sorted(required.difference(manifest))
    if missing:
        raise ValueError(f"source manifest is missing fields: {missing}")
    sha256 = str(manifest["sha256"]).lower()
    invalid_hash_character = any(
        character not in "0123456789abcdef"
        for character in sha256
    )
    if len(sha256) != 64 or invalid_hash_character:
        raise ValueError("source manifest sha256 must be a 64-character hex digest")
    if not isinstance(manifest["sources"], dict) or not manifest["sources"]:
        raise ValueError("source manifest sources must be a non-empty mapping")
    observed_sources = set(map(str, manifest["sources"]))
    contamination = sorted(
        observed_sources.intersection(forbidden_sources or set())
    )
    if contamination:
        raise ValueError(
            f"{role} manifest contains held-out sources: {contamination}"
        )
    if observed_sources != expected_sources:
        raise ValueError(
            f"{role} manifest source mismatch: expected "
            f"{sorted(expected_sources)}, observed {sorted(observed_sources)}"
        )
    if str(manifest["role"]) != role:
        raise ValueError(
            f"source manifest role mismatch: {manifest['role']!r} != {role!r}"
        )
    if str(manifest["origin"]) != required_origin:
        raise ValueError(
            "source manifest origin mismatch: "
            f"{manifest['origin']!r} != {required_origin!r}"
        )
    return manifest


def _validate_real_data_provenance(config: dict[str, Any]) -> None:
    manifest_path = resolve_path(config, config["data"].get("source_manifest"))
    source_map_path = resolve_path(config, config["data"].get("source_map"))
    if manifest_path is None:
        raise FileNotFoundError(
            "real-data runs require data.source_manifest"
        )
    if source_map_path is None or not source_map_path.is_file():
        raise FileNotFoundError(
            "real-data runs require data.source_map pointing to a reviewed "
            "local source mapping"
        )
    development_sources = set(
        map(str, config["data"]["development_sources"])
    )
    manifest = _validate_manifest(
        manifest_path,
        expected_sources=development_sources,
        role="development",
        required_origin="metabonet-public",
        forbidden_sources=set(
            map(
                str,
                config["data"].get(
                    "forbidden_development_sources",
                    [],
                ),
            )
        ),
    )
    forbidden = set(
        map(str, config["data"].get("forbidden_development_sources", []))
    )
    contamination = sorted(forbidden.intersection(manifest["sources"]))
    if contamination:
        raise ValueError(
            f"development manifest contains held-out sources: {contamination}"
        )
    if str(manifest["source_map"]) != source_map_path.name:
        raise ValueError(
            "development manifest does not reference the configured source map"
        )
    range_config = config["data"].get("glucose_range")
    range_audit = manifest.get("glucose_range_audit")
    if not isinstance(range_config, dict) or not isinstance(range_audit, dict):
        raise ValueError(
            "development provenance requires a glucose-range audit"
        )
    expected_policy = str(range_config.get("policy", "fail"))
    if str(range_audit.get("policy")) != expected_policy:
        raise ValueError(
            "development glucose-range policy conflicts with the manifest"
        )
    if (
        float(range_audit.get("minimum_mg_dl", float("nan")))
        != float(range_config.get("minimum_mg_dl", 20))
        or float(range_audit.get("maximum_mg_dl", float("nan")))
        != float(range_config.get("maximum_mg_dl", 600))
    ):
        raise ValueError(
            "development glucose-range bounds conflict with the manifest"
        )
    out_of_range = int(range_audit.get("out_of_range_rows", -1))
    if out_of_range < 0 or out_of_range > int(
        range_config.get("maximum_rows", 0)
    ):
        raise ValueError(
            "development glucose-range count exceeds the approved manifest"
        )
    if str(manifest["release_id"]) != str(
        range_config.get("approved_release_id", "")
    ):
        raise ValueError(
            "development release ID conflicts with the approved "
            "glucose-range amendment"
        )
    if str(manifest["sha256"]).lower() != str(
        range_config.get("approved_release_sha256", "")
    ).lower():
        raise ValueError(
            "development release hash conflicts with the approved "
            "glucose-range amendment"
        )
    if (
        expected_policy != "fail"
        and not bool(range_audit.get("release_sha256_verified", False))
    ):
        raise ValueError(
            "development glucose-range policy lacks hash verification"
        )
    if out_of_range:
        approved_sources = set(
            map(str, range_config.get("allowed_sources", []))
        )
        observed_sources = set(
            map(str, range_audit.get("by_source", {}))
        )
        if not observed_sources.issubset(approved_sources):
            raise ValueError(
                "development glucose-range audit contains unapproved sources"
            )


def load_frame(
    config: dict[str, Any],
    *,
    sources: set[str] | None = None,
    participant_keys: set[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    normalized_keys: set[tuple[str, str]] | None = None
    if participant_keys is not None:
        normalized_keys = {
            (str(source), str(participant))
            for source, participant in participant_keys
        }
        if not normalized_keys:
            raise ValueError("participant_keys must be non-empty when supplied")
        if any(
            not source or not participant
            for source, participant in normalized_keys
        ):
            raise ValueError("participant_keys must contain non-empty identifiers")
        if sources is not None:
            unexpected_key_sources = sorted(
                {source for source, _ in normalized_keys}.difference(
                    set(map(str, sources))
                )
            )
            if unexpected_key_sources:
                raise ValueError(
                    "participant_keys contain sources outside the requested scope: "
                    f"{unexpected_key_sources}"
                )
    if bool(config["project"].get("synthetic", False)):
        frame = make_synthetic_cgm(config, seed=int(config["split"]["seed"]))
        if sources is not None:
            frame = frame.loc[
                frame["source_id"].astype(str).isin(sources)
            ].copy()
        if normalized_keys is not None:
            row_keys = pd.MultiIndex.from_frame(
                frame[["source_id", "participant_id"]].astype(str)
            )
            frame = frame.loc[
                row_keys.isin(pd.MultiIndex.from_tuples(sorted(normalized_keys)))
            ].copy()
    else:
        _validate_real_data_provenance(config)
        development_sources = set(
            map(str, config["data"]["development_sources"])
        )
        requested_sources = (
            development_sources
            if sources is None
            else set(map(str, sources))
        )
        nondevelopment = sorted(
            requested_sources.difference(development_sources)
        )
        if nondevelopment:
            raise ValueError(
                "development input cannot load held-out sources: "
                f"{nondevelopment}"
            )
        input_path = resolve_path(config, config["data"].get("input_path"))
        if input_path is None or not input_path.exists():
            raise FileNotFoundError(
                "normalized CGM input is absent; follow data/README.md and "
                "RUN_INSTRUCTIONS.md"
            )
        if normalized_keys is None:
            filters = (
                [("source_id", "in", sorted(requested_sources))]
                if requested_sources
                else None
            )
        else:
            participants_by_source = {
                source: sorted(
                    participant
                    for current_source, participant in normalized_keys
                    if current_source == source
                )
                for source in sorted({source for source, _ in normalized_keys})
            }
            # Nested filter lists are OR-ed by the Parquet engine, while each
            # source/participant clause is AND-ed. This preserves source-scoped
            # identities when raw participant IDs repeat across datasets.
            filters = [
                [
                    ("source_id", "==", source),
                    ("participant_id", "in", participants),
                ]
                for source, participants in participants_by_source.items()
            ]
        frame = pd.read_parquet(input_path, filters=filters)
    validate_frame(frame)
    if normalized_keys is not None:
        observed_keys = set(
            zip(  # noqa: B905
                frame["source_id"].astype(str),
                frame["participant_id"].astype(str),
            )
        )
        if observed_keys != normalized_keys:
            raise ValueError(
                "loaded participant keys differ from the requested source-scoped scope"
            )
    return frame


def _stress_partition(source: str) -> str:
    return f"stress_{source.lower().replace('-', '_').replace(' ', '_')}"


def load_evaluation_frame(
    config: dict[str, Any],
    partition: str,
) -> pd.DataFrame:
    if bool(config["project"].get("synthetic", False)):
        raise ValueError(
            "synthetic evaluation uses the in-memory synthetic fixture"
        )
    specifications = config["data"].get("evaluation_inputs")
    if not isinstance(specifications, dict) or partition not in specifications:
        raise ValueError(f"evaluation input is not configured: {partition}")
    specification = specifications[partition]
    expected_source = str(specification["expected_source"])
    if partition == "external_loop":
        configured_source = str(config["data"]["external_source"])
    else:
        stress_sources = set(map(str, config["data"].get("stress_sources", [])))
        matches = {
            source
            for source in stress_sources
            if _stress_partition(source) == partition
        }
        if len(matches) != 1:
            raise ValueError(
                f"partition is not a configured stress source: {partition}"
            )
        configured_source = matches.pop()
    if expected_source != configured_source:
        raise ValueError(
            "evaluation expected_source conflicts with frozen protocol: "
            f"{expected_source!r} != {configured_source!r}"
        )
    input_path = resolve_path(config, specification.get("input_path"))
    manifest_path = resolve_path(
        config,
        specification.get("source_manifest"),
    )
    development_path = resolve_path(config, config["data"].get("input_path"))
    if input_path is None or not input_path.exists():
        raise FileNotFoundError(
            f"normalized evaluation input is absent: {partition}"
        )
    if development_path is not None and input_path == development_path:
        raise ValueError(
            "evaluation input must be physically isolated from development"
        )
    if manifest_path is None:
        raise FileNotFoundError(
            f"evaluation manifest is not configured: {partition}"
        )
    _validate_manifest(
        manifest_path,
        expected_sources={expected_source},
        role=partition,
        required_origin=str(specification["required_origin"]),
    )
    frame = pd.read_parquet(
        input_path,
        filters=[("source_id", "==", expected_source)],
    )
    observed_sources = set(frame["source_id"].astype(str).unique())
    if observed_sources != {expected_source}:
        raise ValueError(
            f"evaluation data source mismatch: {sorted(observed_sources)}"
        )
    validate_frame(frame)
    return frame


def write_json(payload: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
