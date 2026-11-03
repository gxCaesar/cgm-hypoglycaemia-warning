"""Finalize runtime provenance for a post-preparation C2 training config.

The preparation config cannot contain hashes of artifacts that do not exist
yet.  Consequently its runtime receipt must never be reused by the later C2
training config.  This module binds that final, fully pinned config to the same
input bytes and to the immutable preparation summary without self-referential
YAML hashes.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .c2_prepare import (
    PREPARATION_METHOD,
    PREPARATION_PROTOCOL_ID,
    PREPARATION_SCHEMA_VERSION,
    validate_participant_shard_manifest,
)
from .c2_training import load_c2_reference_vectors
from .config import load_config, resolve_path
from .event_reference_authority import (
    read_production_event_authority,
    validate_production_event_authority_against_shards,
)
from .mero_reference import read_canonical_json
from .mero_weighted_reference import read_weighted_reference_artifact
from .runtime_provenance import (
    RUNTIME_PROVENANCE_BINDING_ORDER,
    build_runtime_provenance_receipt,
    read_runtime_provenance_receipt,
    runtime_provenance_pins,
    verify_runtime_provenance_receipt_against_paths,
    write_runtime_provenance_receipt,
)
from .shard_manifest import read_window_manifest


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    return value


def _exact_fields(
    value: Mapping[str, Any],
    *,
    fields: set[str],
    field: str,
) -> None:
    missing = sorted(fields.difference(value))
    extra = sorted(map(repr, set(value).difference(fields)))
    if missing or extra:
        raise ValueError(f"{field} fields differ: missing={missing}, extra={extra}")


def _sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _required_path(config: dict[str, Any], field: str) -> Path:
    value = _mapping(config.get("training"), field="training").get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"training.{field} is required")
    path = resolve_path(config, value)
    if path is None:
        raise ValueError(f"training.{field} could not be resolved")
    return path.resolve()


def _runtime_paths(config: dict[str, Any]) -> dict[str, Path]:
    return {
        argument: _required_path(config, field)
        for argument, field in (
            ("resolved_config_path", "c2_resolved_config_path"),
            ("input_manifest_path", "c2_input_manifest_path"),
            ("split_manifest_path", "c2_split_manifest_path"),
            ("preprocessing_contract_path", "c2_preprocessing_contract_path"),
            ("loss_score_contract_path", "c2_loss_score_contract_path"),
            (
                "participant_shard_manifest_path",
                "c2_participant_shard_manifest_path",
            ),
            ("participant_shard_root", "c2_participant_shard_root"),
        )
    }


def _assert_config_authority(
    config: dict[str, Any],
    *,
    resolved_config_path: Path,
) -> None:
    loaded_path = Path(str(config.get("_config_path", ""))).resolve()
    if loaded_path != resolved_config_path:
        raise ValueError("final C2 config is not its resolved config authority")
    reloaded = load_config(resolved_config_path)
    current_payload = {
        key: value for key, value in config.items() if key != "_config_path"
    }
    reloaded_payload = {
        key: value for key, value in reloaded.items() if key != "_config_path"
    }
    if current_payload != reloaded_payload:
        raise ValueError("in-memory final C2 config differs from live YAML")


def _validate_preparation_summary(path: str | Path) -> dict[str, Any]:
    summary = read_canonical_json(path)
    _exact_fields(
        summary,
        fields={
            "schema_version",
            "protocol_id",
            "method",
            "scope",
            "paths",
            "runtime_pins",
            "artifacts",
            "post_preparation_training_pins",
            "receipt_roles",
            "reference_receipt",
            "artifact_sha256",
        },
        field="preparation summary",
    )
    if summary.get("schema_version") != PREPARATION_SCHEMA_VERSION:
        raise ValueError("preparation summary schema_version is incompatible")
    if summary.get("protocol_id") != PREPARATION_PROTOCOL_ID:
        raise ValueError("preparation summary protocol_id is incompatible")
    if summary.get("method") != PREPARATION_METHOD:
        raise ValueError("preparation summary method is incompatible")
    scope = _mapping(summary.get("scope"), field="preparation summary scope")
    expected_scope = {
        "fit_partition": "train",
        "held_partition": "train",
        "selection_rows_loaded": False,
        "calibration_rows_loaded": False,
        "external_rows_loaded": False,
    }
    if dict(scope) != expected_scope:
        raise ValueError("preparation summary is not train-only")
    runtime_pins = _mapping(
        summary.get("runtime_pins"),
        field="preparation summary runtime_pins",
    )
    if set(runtime_pins) != set(RUNTIME_PROVENANCE_BINDING_ORDER):
        raise ValueError("preparation summary runtime pins are incomplete")
    for name, value in runtime_pins.items():
        _sha256(value, field=f"preparation summary runtime_pins.{name}")
    paths = _mapping(summary.get("paths"), field="preparation summary paths")
    _exact_fields(
        paths,
        fields={
            "runtime_provenance",
            "base_reference",
            "event_authority",
            "event_reference",
            "event_execution_root",
        },
        field="preparation summary paths",
    )
    for name, value in paths.items():
        if not isinstance(value, str) or not value or not Path(value).is_absolute():
            raise ValueError(f"preparation summary paths.{name} must be absolute")
    artifacts = _mapping(
        summary.get("artifacts"), field="preparation summary artifacts"
    )
    _exact_fields(
        artifacts,
        fields={
            "runtime_provenance_artifact_sha256",
            "base_reference_artifact_sha256",
            "event_authority_artifact_sha256",
            "event_reference_artifact_sha256",
        },
        field="preparation summary artifacts",
    )
    for name, value in artifacts.items():
        _sha256(value, field=f"preparation summary artifacts.{name}")
    post_pins = _mapping(
        summary.get("post_preparation_training_pins"),
        field="preparation summary post-training pins",
    )
    _exact_fields(
        post_pins,
        fields={
            "c2_base_reference_artifact_sha256",
            "c2_base_loss_contract_sha256",
            "c2_event_authority_artifact_sha256",
            "c2_event_reference_artifact_sha256",
        },
        field="preparation summary post-training pins",
    )
    for name, value in post_pins.items():
        _sha256(value, field=f"preparation summary post-training pins.{name}")
    for artifact_name, pin_name in (
        ("base_reference_artifact_sha256", "c2_base_reference_artifact_sha256"),
        ("event_authority_artifact_sha256", "c2_event_authority_artifact_sha256"),
        ("event_reference_artifact_sha256", "c2_event_reference_artifact_sha256"),
    ):
        if artifacts[artifact_name] != post_pins[pin_name]:
            raise ValueError(
                f"preparation summary artifact and training pin differ: {artifact_name}"
            )
    if post_pins["c2_base_loss_contract_sha256"] != runtime_pins[
        "loss_score_contract"
    ]:
        raise ValueError("preparation summary base loss contract differs from runtime")
    roles = _mapping(
        summary.get("receipt_roles"), field="preparation summary receipt_roles"
    )
    expected_roles = {
        "preparation_runtime_receipt": (
            "binds_preparation_config_only_not_valid_for_c2_training"
        ),
        "required_next_step": (
            "build_fully_pinned_training_config_then_run_glycoshift.c2_finalize"
        ),
    }
    if dict(roles) != expected_roles:
        raise ValueError("preparation summary receipt roles are incompatible")
    reference_receipt = _mapping(
        summary.get("reference_receipt"),
        field="preparation summary reference_receipt",
    )
    if not reference_receipt:
        raise ValueError("preparation summary reference_receipt must be non-empty")
    preparation_receipt = read_runtime_provenance_receipt(
        paths["runtime_provenance"]
    )
    if runtime_provenance_pins(preparation_receipt) != dict(runtime_pins):
        raise ValueError("preparation summary runtime pins differ from receipt")
    if (
        preparation_receipt["artifact_sha256"]
        != artifacts["runtime_provenance_artifact_sha256"]
    ):
        raise ValueError("preparation summary runtime artifact differs from receipt")
    return summary


def _expected_runtime_pins(
    config: dict[str, Any],
    *,
    preparation_pins: Mapping[str, Any],
) -> dict[str, str]:
    supplied = _mapping(
        _mapping(config.get("training"), field="training").get(
            "c2_runtime_expected_pins"
        ),
        field="training.c2_runtime_expected_pins",
    )
    expected_names = set(RUNTIME_PROVENANCE_BINDING_ORDER).difference(
        {"resolved_config"}
    )
    if set(supplied) != expected_names:
        raise ValueError(
            "final C2 config must pin every non-self runtime binding and must "
            "not predeclare resolved_config"
        )
    output = {
        name: _sha256(value, field=f"training.c2_runtime_expected_pins.{name}")
        for name, value in supplied.items()
    }
    for name in sorted(expected_names):
        if output[name] != preparation_pins[name]:
            raise ValueError(
                f"final C2 runtime pin differs from preparation: {name}"
            )
    return output


def _validate_artifact_bindings(
    config: dict[str, Any],
    *,
    summary: Mapping[str, Any],
    runtime_paths: Mapping[str, Path],
    runtime_pins: Mapping[str, str],
) -> None:
    training = _mapping(config.get("training"), field="training")
    summary_paths = _mapping(summary.get("paths"), field="preparation summary paths")
    post_pins = _mapping(
        summary.get("post_preparation_training_pins"),
        field="preparation summary post-training pins",
    )
    path_bindings = (
        (
            "c2_base_reference_path",
            "base_reference",
            "c2_base_reference_artifact_sha256",
        ),
        (
            "c2_event_authority_path",
            "event_authority",
            "c2_event_authority_artifact_sha256",
        ),
        (
            "c2_event_reference_path",
            "event_reference",
            "c2_event_reference_artifact_sha256",
        ),
    )
    resolved_paths: dict[str, Path] = {}
    configured_pins: dict[str, str] = {}
    for path_field, summary_path_field, pin_field in path_bindings:
        path = _required_path(config, path_field)
        if path != Path(str(summary_paths[summary_path_field])).resolve():
            raise ValueError(f"training.{path_field} differs from preparation summary")
        expected = _sha256(post_pins.get(pin_field), field=f"summary.{pin_field}")
        configured = _sha256(training.get(pin_field), field=f"training.{pin_field}")
        if configured != expected:
            raise ValueError(f"C2 artifact binding mismatch: {pin_field}")
        resolved_paths[summary_path_field] = path
        configured_pins[pin_field] = configured
    loss_pin = _sha256(
        post_pins.get("c2_base_loss_contract_sha256"),
        field="summary.c2_base_loss_contract_sha256",
    )
    if (
        _sha256(
            training.get("c2_base_loss_contract_sha256"),
            field="training.c2_base_loss_contract_sha256",
        )
        != loss_pin
    ):
        raise ValueError("C2 base loss-contract pin differs from preparation")
    if loss_pin != runtime_pins["loss_score_contract"]:
        raise ValueError("C2 base loss contract differs from live runtime contract")

    base = read_weighted_reference_artifact(resolved_paths["base_reference"])
    if base["artifact_sha256"] != configured_pins[
        "c2_base_reference_artifact_sha256"
    ]:
        raise ValueError("C2 weighted reference schema or artifact pin mismatch")
    manifest = read_window_manifest(runtime_paths["participant_shard_manifest_path"])
    authority = validate_production_event_authority_against_shards(
        read_production_event_authority(resolved_paths["event_authority"]),
        parent_base_reference_artifact=base,
        shard_root=runtime_paths["participant_shard_root"],
        participant_shard_manifest=manifest,
    )
    if authority["artifact_sha256"] != configured_pins[
        "c2_event_authority_artifact_sha256"
    ]:
        raise ValueError("C2 event authority schema or artifact pin mismatch")
    authority_bindings = _mapping(
        authority.get("bindings"), field="event authority bindings"
    )
    if (
        authority_bindings["split_manifest_sha256"]
        != runtime_pins["split_manifest"]
        or authority_bindings["preprocessing_config_sha256"]
        != runtime_pins["preprocessing_contract"]
        or authority_bindings["window_manifest_sha256"]
        != runtime_pins["participant_shard_root"]
    ):
        raise ValueError("C2 event authority disagrees with live runtime bindings")
    reference = load_c2_reference_vectors(
        base_reference_path=resolved_paths["base_reference"],
        event_reference_path=resolved_paths["event_reference"],
        expected_sha256={
            "base_artifact_sha256": base["artifact_sha256"],
            "event_artifact_sha256": configured_pins[
                "c2_event_reference_artifact_sha256"
            ],
            "event_authority_sha256": authority["artifact_sha256"],
            "split_manifest_sha256": runtime_pins["split_manifest"],
            "window_manifest_sha256": runtime_pins["participant_shard_root"],
            "preprocessing_config_sha256": runtime_pins[
                "preprocessing_contract"
            ],
            "base_input_manifest_sha256": runtime_pins["input_manifest"],
            "base_loss_contract_sha256": loss_pin,
            "event_loss_score_contract_sha256": runtime_pins[
                "loss_score_contract"
            ],
        },
    )
    if reference.receipt() != dict(
        _mapping(
            summary.get("reference_receipt"),
            field="preparation summary reference_receipt",
        )
    ):
        raise ValueError("C2 reference receipt differs from preparation summary")


def finalize_c2_training_runtime(
    config: dict[str, Any],
    *,
    preparation_summary_path: str | Path,
) -> dict[str, Any]:
    """Write the final config-specific runtime receipt and reverify live bytes."""

    expected_summary_path = _required_path(config, "c2_preparation_summary_path")
    if Path(preparation_summary_path).expanduser().resolve() != expected_summary_path:
        raise ValueError("final C2 preparation summary path differs from config")
    summary = _validate_preparation_summary(preparation_summary_path)
    configured_summary_sha256 = _sha256(
        _mapping(config.get("training"), field="training").get(
            "c2_preparation_summary_artifact_sha256"
        ),
        field="training.c2_preparation_summary_artifact_sha256",
    )
    if configured_summary_sha256 != summary["artifact_sha256"]:
        raise ValueError("final C2 preparation summary artifact pin mismatch")
    runtime_paths = _runtime_paths(config)
    _assert_config_authority(
        config,
        resolved_config_path=runtime_paths["resolved_config_path"],
    )
    preparation_pins = _mapping(
        summary["runtime_pins"],
        field="preparation summary runtime_pins",
    )
    expected_pins = _expected_runtime_pins(
        config,
        preparation_pins=preparation_pins,
    )
    receipt = build_runtime_provenance_receipt(
        **runtime_paths,
        participant_shard_partition="train",
        expected_pins=expected_pins,
    )
    runtime_pins = runtime_provenance_pins(receipt)
    train_manifest = validate_participant_shard_manifest(
        participant_shard_manifest_path=runtime_paths[
            "participant_shard_manifest_path"
        ],
        participant_shard_root=runtime_paths["participant_shard_root"],
        split_manifest_path=runtime_paths["split_manifest_path"],
        expected_manifest_file_sha256=runtime_pins[
            "participant_shard_manifest"
        ],
    )
    if train_manifest["sha256"] != runtime_pins["participant_shard_root"]:
        raise ValueError("C2 train shard ledger differs from runtime provenance")
    _validate_artifact_bindings(
        config,
        summary=summary,
        runtime_paths=runtime_paths,
        runtime_pins=runtime_pins,
    )
    verified_receipt = verify_runtime_provenance_receipt_against_paths(
        receipt,
        **runtime_paths,
        participant_shard_partition="train",
        expected_pins=expected_pins,
    )
    receipt_path = _required_path(config, "c2_runtime_provenance_path")
    preparation_receipt_path = Path(
        str(_mapping(summary["paths"], field="summary paths")["runtime_provenance"])
    ).resolve()
    if receipt_path == preparation_receipt_path:
        raise ValueError(
            "final C2 runtime receipt must differ from the preparation receipt"
        )
    write_runtime_provenance_receipt(verified_receipt, receipt_path)
    return verified_receipt


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finalize a fully pinned C2 training runtime receipt."
    )
    parser.add_argument("--config", required=True, help="final C2 YAML config")
    parser.add_argument(
        "--preparation-summary",
        required=True,
        help="canonical summary emitted by glycoshift.c2_prepare",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    receipt = finalize_c2_training_runtime(
        load_config(args.config),
        preparation_summary_path=args.preparation_summary,
    )
    print(
        json.dumps(
            {
                "runtime_provenance_artifact_sha256": receipt["artifact_sha256"],
                "runtime_pins": runtime_provenance_pins(receipt),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
