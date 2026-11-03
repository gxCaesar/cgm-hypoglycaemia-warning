"""Fail-closed, content-addressed runtime provenance receipts.

The runtime binder deliberately hashes the bytes it is about to use.  YAML
digest pins can be supplied as a redundant assertion, but are never accepted
as evidence in place of re-reading regular local files and the explicitly
bound live shard scope.
"""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Mapping, Sequence
from numbers import Integral
from pathlib import Path, PurePosixPath
from typing import Any

from .mero_reference import (
    bind_artifact_sha256,
    canonical_json_bytes,
    read_canonical_json,
    verify_artifact_payload,
    write_immutable_json,
)
from .shard_manifest import (
    WINDOW_MANIFEST_SERIALIZATION,
    read_window_manifest,
    validate_live_window_partition,
)

RUNTIME_PROVENANCE_SCHEMA_VERSION = "1.1"
RUNTIME_PROVENANCE_PROTOCOL_ID = "glycoshift-sota-method-v1.3"
RUNTIME_PROVENANCE_METHOD = "runtime-content-addressed-provenance-v1.1"
RUNTIME_PROVENANCE_BINDING_ORDER = (
    "resolved_config",
    "input_manifest",
    "split_manifest",
    "preprocessing_contract",
    "loss_score_contract",
    "participant_shard_manifest",
    "participant_shard_root",
)

_FILE_KIND = "regular_file"
_DIRECTORY_KIND = "directory_tree"
_PARTITION_KIND = "manifest_partition_tree"
_DIRECTORY_MANIFEST_SCHEMA = "runtime-provenance-directory-tree-v1"
_HASH_CHUNK_BYTES = 8 * 1024 * 1024


def _validate_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _validate_non_negative_integer(value: Any, *, field: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError(f"{field} must be an integer")
    resolved = int(value)
    if resolved < 0:
        raise ValueError(f"{field} must be non-negative")
    return resolved


def _require_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    return value


def _require_sequence(value: Any, *, field: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{field} must be a sequence")
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


def _absolute_path(value: str | Path, *, field: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise TypeError(f"{field} must be a path string or Path")
    raw = Path(value).expanduser()
    if not raw.is_absolute():
        raw = Path.cwd() / raw
    return Path(os.path.abspath(os.fspath(raw)))


def _path_components(path: Path) -> tuple[Path, ...]:
    if not path.is_absolute():
        raise ValueError("provenance paths must be absolute after resolution")
    current = Path(path.anchor)
    components = [current]
    for part in path.parts[1:]:
        current = current / part
        components.append(current)
    return tuple(components)


def _existing_path_without_symlinks(value: str | Path, *, field: str) -> Path:
    """Resolve lexical ``.``/``..`` spelling without following any symlink."""

    path = _absolute_path(value, field=field)
    for component in _path_components(path):
        try:
            metadata = component.lstat()
        except FileNotFoundError as error:
            raise FileNotFoundError(f"{field} is absent: {path}") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"{field} must not contain a symlink: {component}")
    return path


def _ensure_safe_output_parent(path: Path, *, field: str) -> Path:
    """Create missing output directories while refusing symlink traversal."""

    parent = path.parent
    components = _path_components(parent)
    for component in components:
        try:
            metadata = component.lstat()
        except FileNotFoundError:
            component.mkdir()
            metadata = component.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"{field} must not contain a symlink: {component}")
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"{field} parent is not a directory: {component}")
    try:
        output_metadata = path.lstat()
    except FileNotFoundError:
        return path
    if stat.S_ISLNK(output_metadata.st_mode):
        raise ValueError(f"{field} must not be a symlink")
    if not stat.S_ISREG(output_metadata.st_mode):
        raise ValueError(f"{field} must be a regular file when it already exists")
    return path


def _regular_file_sha256(path: Path, *, field: str) -> tuple[str, int]:
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{field} must be a regular non-symlink file")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    after = path.lstat()
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(getattr(before, name) != getattr(after, name) for name in stable_fields):
        raise ValueError(f"{field} changed while deriving runtime provenance")
    return digest.hexdigest(), int(after.st_size)


def _regular_file_binding(value: str | Path, *, field: str) -> dict[str, Any]:
    path = _existing_path_without_symlinks(value, field=field)
    digest, byte_count = _regular_file_sha256(path, field=field)
    return {
        "kind": _FILE_KIND,
        "resolved_path": str(path),
        "bytes": byte_count,
        "sha256": digest,
    }


def _safe_relative_path(path: Path, *, root: Path, field: str) -> str:
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError(f"{field} escapes participant_shard_root") from error
    pure = PurePosixPath(relative)
    if (
        not relative
        or "\\" in relative
        or pure.is_absolute()
        or any(part in {"", ".", ".."} for part in pure.parts)
        or pure.as_posix() != relative
    ):
        raise ValueError(f"{field} is not a safe normalized relative path")
    return relative


def _directory_manifest_sha256(entries: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "schema": _DIRECTORY_MANIFEST_SCHEMA,
                "entries": list(entries),
            }
        )
    ).hexdigest()


def _directory_binding(value: str | Path, *, field: str) -> dict[str, Any]:
    root = _existing_path_without_symlinks(value, field=field)
    root_metadata = root.lstat()
    if not stat.S_ISDIR(root_metadata.st_mode):
        raise ValueError(f"{field} must be a directory")

    entries: list[dict[str, Any]] = []

    def visit(directory: Path) -> None:
        before = directory.lstat()
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
            raise ValueError(f"{field} contains a non-directory traversal node")
        with os.scandir(directory) as scan:
            children = sorted(scan, key=lambda entry: os.fsencode(entry.name))
        for child in children:
            child_path = Path(child.path)
            metadata = child.stat(follow_symlinks=False)
            relative = _safe_relative_path(child_path, root=root, field=field)
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError(f"{field} contains a symlink: {relative}")
            if stat.S_ISDIR(metadata.st_mode):
                visit(child_path)
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError(f"{field} contains a non-regular file: {relative}")
            digest, byte_count = _regular_file_sha256(child_path, field=field)
            entries.append(
                {
                    "relative_path": relative,
                    "bytes": byte_count,
                    "sha256": digest,
                }
            )
        after = directory.lstat()
        stable_fields = ("st_dev", "st_ino", "st_mtime_ns")
        if any(getattr(before, name) != getattr(after, name) for name in stable_fields):
            raise ValueError(f"{field} changed while deriving runtime provenance")

    visit(root)
    if not entries:
        raise ValueError(f"{field} must contain at least one regular file")
    if entries != sorted(entries, key=lambda row: os.fsencode(row["relative_path"])):
        raise AssertionError("directory traversal did not produce lexical order")
    return {
        "kind": _DIRECTORY_KIND,
        "resolved_path": str(root),
        "sha256": _directory_manifest_sha256(entries),
        "file_count": len(entries),
        "bytes": sum(int(entry["bytes"]) for entry in entries),
        "entries": entries,
    }


def _partition_binding(
    value: str | Path,
    *,
    manifest_path: str | Path,
    partition: str,
    field: str,
) -> dict[str, Any]:
    if partition != "train":
        raise ValueError("participant_shard_partition must be train")
    root = _existing_path_without_symlinks(value, field=field)
    ledger = validate_live_window_partition(
        root,
        read_window_manifest(manifest_path),
        partition=partition,
    )
    return {
        "kind": _PARTITION_KIND,
        "resolved_path": str(root),
        "partition": partition,
        "serialization": ledger["serialization"],
        "sha256": ledger["sha256"],
        "file_count": ledger["files"],
        "bytes": ledger["bytes"],
        "rows": ledger["rows"],
        "entries": [dict(record) for record in ledger["records"]],
    }


def _validate_resolved_path(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty absolute path")
    path = Path(value)
    if not path.is_absolute() or os.path.abspath(value) != value:
        raise ValueError(f"{field} must use normalized absolute spelling")
    return value


def _validate_relative_path(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{field} must be a non-empty POSIX relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != value
    ):
        raise ValueError(f"{field} must use normalized safe POSIX spelling")
    return value


def _validate_file_binding(value: Any, *, field: str) -> dict[str, Any]:
    row = _require_mapping(value, field=field)
    _exact_fields(
        row,
        fields={"kind", "resolved_path", "bytes", "sha256"},
        field=field,
    )
    if row["kind"] != _FILE_KIND:
        raise ValueError(f"{field}.kind must be {_FILE_KIND}")
    return {
        "kind": _FILE_KIND,
        "resolved_path": _validate_resolved_path(
            row["resolved_path"],
            field=f"{field}.resolved_path",
        ),
        "bytes": _validate_non_negative_integer(row["bytes"], field=f"{field}.bytes"),
        "sha256": _validate_sha256(row["sha256"], field=f"{field}.sha256"),
    }


def _validate_directory_binding(value: Any, *, field: str) -> dict[str, Any]:
    row = _require_mapping(value, field=field)
    _exact_fields(
        row,
        fields={
            "kind",
            "resolved_path",
            "sha256",
            "file_count",
            "bytes",
            "entries",
        },
        field=field,
    )
    if row["kind"] != _DIRECTORY_KIND:
        raise ValueError(f"{field}.kind must be {_DIRECTORY_KIND}")
    entries: list[dict[str, Any]] = []
    for index, supplied in enumerate(
        _require_sequence(row["entries"], field=f"{field}.entries")
    ):
        entry = _require_mapping(supplied, field=f"{field}.entries[{index}]")
        _exact_fields(
            entry,
            fields={"relative_path", "bytes", "sha256"},
            field=f"{field}.entries[{index}]",
        )
        entries.append(
            {
                "relative_path": _validate_relative_path(
                    entry["relative_path"],
                    field=f"{field}.entries[{index}].relative_path",
                ),
                "bytes": _validate_non_negative_integer(
                    entry["bytes"],
                    field=f"{field}.entries[{index}].bytes",
                ),
                "sha256": _validate_sha256(
                    entry["sha256"],
                    field=f"{field}.entries[{index}].sha256",
                ),
            }
        )
    if not entries:
        raise ValueError(f"{field}.entries must be non-empty")
    if len({entry["relative_path"] for entry in entries}) != len(entries):
        raise ValueError(f"{field}.entries must not repeat relative paths")
    ordered = sorted(entries, key=lambda entry: os.fsencode(entry["relative_path"]))
    if entries != ordered:
        raise ValueError(f"{field}.entries must use lexical path order")
    file_count = _validate_non_negative_integer(
        row["file_count"],
        field=f"{field}.file_count",
    )
    byte_count = _validate_non_negative_integer(row["bytes"], field=f"{field}.bytes")
    if file_count != len(entries) or byte_count != sum(
        int(entry["bytes"]) for entry in entries
    ):
        raise ValueError(f"{field} aggregate counts disagree with entries")
    normalized = {
        "kind": _DIRECTORY_KIND,
        "resolved_path": _validate_resolved_path(
            row["resolved_path"],
            field=f"{field}.resolved_path",
        ),
        "sha256": _validate_sha256(row["sha256"], field=f"{field}.sha256"),
        "file_count": file_count,
        "bytes": byte_count,
        "entries": entries,
    }
    if normalized["sha256"] != _directory_manifest_sha256(entries):
        raise ValueError(f"{field}.sha256 disagrees with the directory entries")
    return normalized


def _validate_partition_binding(value: Any, *, field: str) -> dict[str, Any]:
    row = _require_mapping(value, field=field)
    _exact_fields(
        row,
        fields={
            "kind",
            "resolved_path",
            "partition",
            "serialization",
            "sha256",
            "file_count",
            "bytes",
            "rows",
            "entries",
        },
        field=field,
    )
    if row["kind"] != _PARTITION_KIND:
        raise ValueError(f"{field}.kind must be {_PARTITION_KIND}")
    if row["partition"] != "train":
        raise ValueError(f"{field}.partition must be train")
    if row["serialization"] != WINDOW_MANIFEST_SERIALIZATION:
        raise ValueError(f"{field}.serialization is incompatible")
    entries: list[dict[str, Any]] = []
    for index, supplied in enumerate(
        _require_sequence(row["entries"], field=f"{field}.entries")
    ):
        entry = _require_mapping(supplied, field=f"{field}.entries[{index}]")
        _exact_fields(
            entry,
            fields={"relative_path", "bytes", "rows", "sha256"},
            field=f"{field}.entries[{index}]",
        )
        relative = _validate_relative_path(
            entry["relative_path"],
            field=f"{field}.entries[{index}].relative_path",
        )
        if not relative.startswith("partition=train/"):
            raise ValueError(f"{field}.entries must be train-only")
        entries.append(
            {
                "relative_path": relative,
                "bytes": _validate_non_negative_integer(
                    entry["bytes"], field=f"{field}.entries[{index}].bytes"
                ),
                "rows": _validate_non_negative_integer(
                    entry["rows"], field=f"{field}.entries[{index}].rows"
                ),
                "sha256": _validate_sha256(
                    entry["sha256"], field=f"{field}.entries[{index}].sha256"
                ),
            }
        )
    if not entries or any(entry["bytes"] < 1 or entry["rows"] < 1 for entry in entries):
        raise ValueError(f"{field}.entries must contain positive train records")
    if len({entry["relative_path"] for entry in entries}) != len(entries):
        raise ValueError(f"{field}.entries repeat a relative path")
    if entries != sorted(entries, key=lambda entry: entry["relative_path"].encode()):
        raise ValueError(f"{field}.entries must use lexical path order")
    file_count = _validate_non_negative_integer(
        row["file_count"], field=f"{field}.file_count"
    )
    byte_count = _validate_non_negative_integer(row["bytes"], field=f"{field}.bytes")
    row_count = _validate_non_negative_integer(row["rows"], field=f"{field}.rows")
    if (
        file_count != len(entries)
        or byte_count != sum(entry["bytes"] for entry in entries)
        or row_count != sum(entry["rows"] for entry in entries)
    ):
        raise ValueError(f"{field} aggregate counts disagree with entries")
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(
            (
                f"{entry['relative_path']}\t{entry['bytes']}\t{entry['rows']}\t"
                f"{entry['sha256']}\n"
            ).encode()
        )
    normalized = {
        "kind": _PARTITION_KIND,
        "resolved_path": _validate_resolved_path(
            row["resolved_path"], field=f"{field}.resolved_path"
        ),
        "partition": "train",
        "serialization": WINDOW_MANIFEST_SERIALIZATION,
        "sha256": _validate_sha256(row["sha256"], field=f"{field}.sha256"),
        "file_count": file_count,
        "bytes": byte_count,
        "rows": row_count,
        "entries": entries,
    }
    if normalized["sha256"] != digest.hexdigest():
        raise ValueError(f"{field}.sha256 disagrees with partition entries")
    return normalized


def _normalize_expected_pins(
    expected_pins: Mapping[str, str] | None,
) -> dict[str, str]:
    if expected_pins is None:
        return {}
    pins = _require_mapping(expected_pins, field="expected_pins")
    if any(not isinstance(name, str) for name in pins):
        raise TypeError("expected_pins keys must be strings")
    unknown = sorted(set(pins).difference(RUNTIME_PROVENANCE_BINDING_ORDER))
    if unknown:
        raise ValueError(f"expected_pins contain unknown bindings: {unknown}")
    return {
        name: _validate_sha256(value, field=f"expected_pins.{name}")
        for name, value in pins.items()
    }


def _assert_expected_pins(
    bindings: Mapping[str, Mapping[str, Any]],
    expected_pins: Mapping[str, str] | None,
) -> None:
    pins = _normalize_expected_pins(expected_pins)
    for name, expected in pins.items():
        if bindings[name]["sha256"] != expected:
            raise ValueError(f"runtime provenance pin mismatch: {name}")


def _payload(bindings: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": RUNTIME_PROVENANCE_SCHEMA_VERSION,
        "protocol_id": RUNTIME_PROVENANCE_PROTOCOL_ID,
        "method": RUNTIME_PROVENANCE_METHOD,
        "binding_order": list(RUNTIME_PROVENANCE_BINDING_ORDER),
        "bindings": {
            name: dict(bindings[name]) for name in RUNTIME_PROVENANCE_BINDING_ORDER
        },
    }


def build_runtime_provenance_receipt(
    *,
    resolved_config_path: str | Path,
    input_manifest_path: str | Path,
    split_manifest_path: str | Path,
    preprocessing_contract_path: str | Path,
    loss_score_contract_path: str | Path,
    participant_shard_manifest_path: str | Path,
    participant_shard_root: str | Path,
    participant_shard_partition: str | None = None,
    expected_pins: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Derive a self-hashed receipt directly from safe local input bytes.

    ``expected_pins`` is a mapping from a name in
    :data:`RUNTIME_PROVENANCE_BINDING_ORDER` to its expected digest.  It is an
    assertion over the newly observed bytes, never a substitute for hashing.
    """

    expected = _normalize_expected_pins(expected_pins)
    bindings: dict[str, Mapping[str, Any]] = {
        "resolved_config": _regular_file_binding(
            resolved_config_path,
            field="resolved_config_path",
        ),
        "input_manifest": _regular_file_binding(
            input_manifest_path,
            field="input_manifest_path",
        ),
        "split_manifest": _regular_file_binding(
            split_manifest_path,
            field="split_manifest_path",
        ),
        "preprocessing_contract": _regular_file_binding(
            preprocessing_contract_path,
            field="preprocessing_contract_path",
        ),
        "loss_score_contract": _regular_file_binding(
            loss_score_contract_path,
            field="loss_score_contract_path",
        ),
        "participant_shard_manifest": _regular_file_binding(
            participant_shard_manifest_path,
            field="participant_shard_manifest_path",
        ),
    }
    for name in RUNTIME_PROVENANCE_BINDING_ORDER[:-1]:
        if name in expected and bindings[name]["sha256"] != expected[name]:
            raise ValueError(f"runtime provenance pin mismatch: {name}")
    bindings["participant_shard_root"] = (
        _directory_binding(
            participant_shard_root,
            field="participant_shard_root",
        )
        if participant_shard_partition is None
        else _partition_binding(
            participant_shard_root,
            manifest_path=participant_shard_manifest_path,
            partition=participant_shard_partition,
            field="participant_shard_root",
        )
    )
    _assert_expected_pins(bindings, expected)
    return bind_artifact_sha256(_payload(bindings))


def validate_runtime_provenance_receipt(
    payload: Mapping[str, Any],
    *,
    expected_pins: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Validate receipt schema, self-hash, and all rebuilt directory derivations."""

    if not isinstance(payload, Mapping):
        raise TypeError("runtime provenance receipt must be a mapping")
    verify_artifact_payload(payload)
    _exact_fields(
        payload,
        fields={
            "schema_version",
            "protocol_id",
            "method",
            "binding_order",
            "bindings",
            "artifact_sha256",
        },
        field="runtime provenance receipt",
    )
    if payload["schema_version"] != RUNTIME_PROVENANCE_SCHEMA_VERSION:
        raise ValueError("unsupported runtime provenance schema_version")
    if payload["protocol_id"] != RUNTIME_PROVENANCE_PROTOCOL_ID:
        raise ValueError("runtime provenance protocol_id is incompatible")
    if payload["method"] != RUNTIME_PROVENANCE_METHOD:
        raise ValueError("runtime provenance method is incompatible")
    if (
        tuple(_require_sequence(payload["binding_order"], field="binding_order"))
        != RUNTIME_PROVENANCE_BINDING_ORDER
    ):
        raise ValueError("runtime provenance binding_order is incompatible")
    supplied_bindings = _require_mapping(payload["bindings"], field="bindings")
    _exact_fields(
        supplied_bindings,
        fields=set(RUNTIME_PROVENANCE_BINDING_ORDER),
        field="runtime provenance bindings",
    )
    bindings: dict[str, Mapping[str, Any]] = {}
    for name in RUNTIME_PROVENANCE_BINDING_ORDER:
        if name == "participant_shard_root":
            binding = _require_mapping(
                supplied_bindings[name], field=f"bindings.{name}"
            )
            if binding.get("kind") == _PARTITION_KIND:
                bindings[name] = _validate_partition_binding(
                    binding, field=f"bindings.{name}"
                )
            else:
                bindings[name] = _validate_directory_binding(
                    binding,
                    field=f"bindings.{name}",
                )
        else:
            bindings[name] = _validate_file_binding(
                supplied_bindings[name],
                field=f"bindings.{name}",
            )
    rebuilt = bind_artifact_sha256(_payload(bindings))
    if rebuilt != dict(payload):
        raise ValueError("runtime provenance receipt contains derived-field drift")
    _assert_expected_pins(bindings, expected_pins)
    return rebuilt


def runtime_provenance_pins(payload: Mapping[str, Any]) -> dict[str, str]:
    """Return the observed digest pins after validating the complete receipt."""

    receipt = validate_runtime_provenance_receipt(payload)
    return {
        name: str(receipt["bindings"][name]["sha256"])
        for name in RUNTIME_PROVENANCE_BINDING_ORDER
    }


def verify_runtime_provenance_receipt_against_paths(
    payload: Mapping[str, Any],
    *,
    resolved_config_path: str | Path,
    input_manifest_path: str | Path,
    split_manifest_path: str | Path,
    preprocessing_contract_path: str | Path,
    loss_score_contract_path: str | Path,
    participant_shard_manifest_path: str | Path,
    participant_shard_root: str | Path,
    participant_shard_partition: str | None = None,
    expected_pins: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Reject a receipt unless the supplied paths still reproduce it exactly."""

    receipt = validate_runtime_provenance_receipt(
        payload,
        expected_pins=expected_pins,
    )
    observed = build_runtime_provenance_receipt(
        resolved_config_path=resolved_config_path,
        input_manifest_path=input_manifest_path,
        split_manifest_path=split_manifest_path,
        preprocessing_contract_path=preprocessing_contract_path,
        loss_score_contract_path=loss_score_contract_path,
        participant_shard_manifest_path=participant_shard_manifest_path,
        participant_shard_root=participant_shard_root,
        participant_shard_partition=participant_shard_partition,
        expected_pins=expected_pins,
    )
    if observed != receipt:
        raise ValueError("runtime provenance receipt does not match live inputs")
    return receipt


def read_runtime_provenance_receipt(
    path: str | Path,
    *,
    expected_pins: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Read only canonical bytes from a regular non-symlink receipt file."""

    receipt_path = _existing_path_without_symlinks(
        path,
        field="runtime_provenance_receipt",
    )
    _regular_file_sha256(receipt_path, field="runtime_provenance_receipt")
    return validate_runtime_provenance_receipt(
        read_canonical_json(receipt_path),
        expected_pins=expected_pins,
    )


def write_runtime_provenance_receipt(
    payload: Mapping[str, Any],
    path: str | Path,
) -> str:
    """Write a canonical receipt once, rejecting symlink output paths."""

    output = _ensure_safe_output_parent(
        _absolute_path(path, field="runtime_provenance_receipt"),
        field="runtime_provenance_receipt",
    )
    return write_immutable_json(validate_runtime_provenance_receipt(payload), output)
