"""Pure manifest validation and train-only live shard verification.

The frozen window manifest commits every development partition.  Train-derived
state must bind that immutable manifest without opening selection or
calibration shards, so this module separates two operations deliberately:

* validate/subset the supplied manifest entirely in memory; and
* verify exactly one live partition by scanning only its partition directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import pyarrow.parquet as pq

WINDOW_MANIFEST_SERIALIZATION = (
    "relative_path<TAB>bytes<TAB>rows<TAB>sha256<LF>"
)
WINDOW_PARTITIONS = ("train", "selection", "calibration")
_HASH_CHUNK_BYTES = 8 * 1024 * 1024


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    return value


def _sequence(value: Any, *, field: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{field} must be a sequence")
    return value


def _sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _positive_integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _safe_relative_path(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{field} must be a non-empty POSIX relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != value
        or len(path.parts) < 2
        or path.parts[0] not in {
            f"partition={partition}" for partition in WINDOW_PARTITIONS
        }
        or path.suffix != ".parquet"
    ):
        raise ValueError(f"{field} is not a supported normalized shard path")
    return value


def _manifest_digest(records: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(
            (
                f"{record['relative_path']}\t{record['bytes']}\t"
                f"{record['rows']}\t{record['sha256']}\n"
            ).encode()
        )
    return digest.hexdigest()


def _manifest_payload(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    normalized = [dict(record) for record in records]
    return {
        "serialization": WINDOW_MANIFEST_SERIALIZATION,
        "sha256": _manifest_digest(normalized),
        "files": len(normalized),
        "bytes": sum(int(record["bytes"]) for record in normalized),
        "rows": sum(int(record["rows"]) for record in normalized),
        "records": normalized,
    }


def validate_window_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate all manifest derivations without touching any shard path."""

    manifest = _mapping(payload, field="window manifest")
    expected_fields = {"serialization", "sha256", "files", "bytes", "rows", "records"}
    if set(manifest) != expected_fields:
        raise ValueError("window manifest fields are incompatible")
    if manifest["serialization"] != WINDOW_MANIFEST_SERIALIZATION:
        raise ValueError("window manifest serialization is incompatible")
    records: list[dict[str, Any]] = []
    for index, supplied in enumerate(
        _sequence(manifest["records"], field="window manifest records")
    ):
        row = _mapping(supplied, field=f"window manifest record {index}")
        if set(row) != {"relative_path", "bytes", "rows", "sha256"}:
            raise ValueError("window manifest record fields are incompatible")
        records.append(
            {
                "relative_path": _safe_relative_path(
                    row["relative_path"],
                    field=f"window manifest record {index}.relative_path",
                ),
                "bytes": _positive_integer(
                    row["bytes"], field=f"window manifest record {index}.bytes"
                ),
                "rows": _positive_integer(
                    row["rows"], field=f"window manifest record {index}.rows"
                ),
                "sha256": _sha256(
                    row["sha256"], field=f"window manifest record {index}.sha256"
                ),
            }
        )
    if not records:
        raise ValueError("window manifest records must be non-empty")
    paths = [record["relative_path"] for record in records]
    if len(set(paths)) != len(paths):
        raise ValueError("window manifest repeats a relative path")
    if paths != sorted(paths, key=lambda value: value.encode("utf-8")):
        raise ValueError("window manifest records must use lexical path order")
    rebuilt = _manifest_payload(records)
    if rebuilt != dict(manifest):
        raise ValueError("window manifest aggregate or digest drift")
    return rebuilt


def read_window_manifest(path: str | Path) -> dict[str, Any]:
    resolved = Path(os.path.abspath(os.fspath(Path(path).expanduser())))
    metadata = resolved.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError("window manifest must be a regular non-symlink file")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("window manifest must be valid UTF-8 JSON") from error
    return validate_window_manifest(payload)


def partition_window_manifest(
    manifest: Mapping[str, Any],
    *,
    partition: str,
) -> dict[str, Any]:
    """Derive a deterministic partition ledger from a validated full ledger."""

    if partition not in WINDOW_PARTITIONS:
        raise ValueError(f"unsupported window partition: {partition}")
    full = validate_window_manifest(manifest)
    prefix = f"partition={partition}/"
    records = [
        record
        for record in full["records"]
        if record["relative_path"].startswith(prefix)
    ]
    if not records:
        raise ValueError(f"window manifest has no {partition} records")
    return _manifest_payload(records)


def validate_window_manifest_against_split(
    manifest: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the supplied full ledger to frozen split counts and digest."""

    full = validate_window_manifest(manifest)
    split = _mapping(split_manifest, field="split manifest")
    binding = _mapping(split.get("artifact_binding"), field="split artifact_binding")
    aggregate = _mapping(
        binding.get("window_shard_manifest"),
        field="split window_shard_manifest",
    )
    expected_digest = _sha256(
        binding.get("window_shard_manifest_sha256"),
        field="split window_shard_manifest_sha256",
    )
    if split.get("split_hash") != expected_digest or full["sha256"] != expected_digest:
        raise ValueError("window manifest digest disagrees with frozen split")
    for field in ("files", "bytes", "rows"):
        if aggregate.get(field) != full[field]:
            raise ValueError(f"window manifest {field} disagrees with frozen split")
    partitions = _mapping(split.get("partitions"), field="split partitions")
    for partition in WINDOW_PARTITIONS:
        ledger = partition_window_manifest(full, partition=partition)
        expected = _mapping(partitions.get(partition), field=f"split {partition}")
        if (
            expected.get("participant_files") != ledger["files"]
            or expected.get("windows") != ledger["rows"]
        ):
            raise ValueError(
                f"window manifest {partition} counts disagree with frozen split"
            )
    return full


def _regular_file_sha256(path: Path, *, field: str) -> tuple[str, int]:
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{field} must be a regular non-symlink file")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    after = path.lstat()
    for name in ("st_dev", "st_ino", "st_size", "st_mtime_ns"):
        if getattr(before, name) != getattr(after, name):
            raise ValueError(f"{field} changed while hashing")
    return digest.hexdigest(), int(after.st_size)


def validate_live_window_partition(
    shard_root: str | Path,
    manifest: Mapping[str, Any],
    *,
    partition: str,
) -> dict[str, Any]:
    """Verify one live partition without listing or opening sibling partitions."""

    ledger = partition_window_manifest(manifest, partition=partition)
    root = Path(os.path.abspath(os.fspath(Path(shard_root).expanduser())))
    root_metadata = root.lstat()
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise ValueError("participant shard root must be a directory without symlinks")
    partition_root = root / f"partition={partition}"
    partition_metadata = partition_root.lstat()
    if stat.S_ISLNK(partition_metadata.st_mode) or not stat.S_ISDIR(
        partition_metadata.st_mode
    ):
        raise ValueError(f"{partition} shard partition must be a directory")

    observed_paths: list[str] = []

    def visit(directory: Path) -> None:
        before = directory.lstat()
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
            raise ValueError(f"{partition} partition contains unsafe traversal")
        with os.scandir(directory) as scan:
            children = sorted(scan, key=lambda entry: os.fsencode(entry.name))
        for child in children:
            path = Path(child.path)
            metadata = child.stat(follow_symlinks=False)
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError(f"{partition} partition contains a symlink")
            if stat.S_ISDIR(metadata.st_mode):
                visit(path)
                continue
            if not stat.S_ISREG(metadata.st_mode) or path.suffix != ".parquet":
                raise ValueError(f"{partition} partition contains a non-Parquet file")
            observed_paths.append(path.relative_to(root).as_posix())
        after = directory.lstat()
        for name in ("st_dev", "st_ino", "st_mtime_ns"):
            if getattr(before, name) != getattr(after, name):
                raise ValueError(f"{partition} partition changed while scanning")

    visit(partition_root)
    expected_paths = [record["relative_path"] for record in ledger["records"]]
    if observed_paths != expected_paths:
        raise ValueError(f"live {partition} shard inventory disagrees with manifest")
    for record in ledger["records"]:
        path = root / Path(record["relative_path"])
        digest, byte_count = _regular_file_sha256(
            path,
            field=f"{partition} shard {record['relative_path']}",
        )
        if digest != record["sha256"] or byte_count != record["bytes"]:
            raise ValueError(f"live {partition} shard bytes disagree with manifest")
        if int(pq.ParquetFile(path).metadata.num_rows) != record["rows"]:
            raise ValueError(f"live {partition} shard rows disagree with manifest")
    return ledger
