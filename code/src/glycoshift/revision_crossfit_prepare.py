"""Prepare one complete replay pool and metadata references for five outer folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import load_config, resolve_path
from .io import _validate_real_data_provenance
from .revision_crossfit import CROSSFIT_SEED, OUTER_FOLDS, build_crossfit_assignments
from .revision_prepare import KEYS, _index_fragments, _json, prepare_replays
from .sequences import REVISION_CONTRACT

CROSSFIT_CONTRACT = "2026-11-09-participant-crossfit"
INNER_ROLES = ("train", "selection", "calibration")


def _bound_file(value: str | Path, root: Path, *, expected: Path | None = None) -> Path:
    path = Path(value)
    if (
        not path.is_absolute()
        or not path.is_relative_to(root)
        or path.resolve() != path
        or not path.is_file()
        or (expected is not None and path != expected)
    ):
        raise ValueError("cross-fitting path escapes or differs from its binding")
    return path


def _original_metadata(index: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [(source, participant, index[(source, participant)][1])
         for source, participant in sorted(index)],
        columns=[*KEYS, "original_partition"],
    )


def _population_plan(metadata: pd.DataFrame, config: dict):
    sources = tuple(sorted(map(str, config["data"]["development_sources"])))
    if set(sources).intersection(config["data"]["forbidden_development_sources"]):
        raise ValueError("development and forbidden sources overlap")
    if not config["project"].get("synthetic", False) and len(metadata) != 1020:
        raise ValueError(
            "cross-fitting requires all 1020 original development identities"
        )
    return build_crossfit_assignments(
        metadata, expected_participants=metadata, development_sources=sources
    )


def prepare_crossfit_replays(
    normalized_path: Path, legacy_windows: Path, output: Path, config: dict
) -> dict:
    """Write full timelines once; the five fold manifests contain only metadata.

    The original participant/window inventory defines eligibility and original
    roles. Originally excluded normalized participants remain excluded. Assignment
    is fixed from identity-only reads before replay values are prepared.
    """
    normalized_path, legacy_windows, output = (
        Path(path).resolve() for path in (normalized_path, legacy_windows, output)
    )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite cross-fitting output: {output}")
    sources = set(map(str, config["data"]["development_sources"]))
    original = _original_metadata(
        _index_fragments(legacy_windows, sources, windows=True)
    )
    plan = _population_plan(original, config)
    output.mkdir(parents=True, exist_ok=True)
    try:
        shared = output / "shared"
        shared_audit = prepare_replays(
            normalized_path, legacy_windows, shared, config, retain_all_tails=True
        )
        folds = {}
        for fold in range(OUTER_FOLDS):
            directory = output / "folds" / f"fold-{fold}"
            directory.mkdir(parents=True)
            roles = plan.role_metadata(fold).merge(
                original, on=KEYS, how="left", validate="one_to_one"
            )
            path = directory / "participant-roles.private.json"
            _json(path, {"participants": roles.to_dict("records")})
            folds[str(fold)] = str(path)
        audit = {
            "status": "passed",
            "crossfit_contract": CROSSFIT_CONTRACT,
            "revision_contract": REVISION_CONTRACT,
            "ascertainment_contract": shared_audit.get("ascertainment_contract"),
            "split_seed": CROSSFIT_SEED,
            "outer_folds": OUTER_FOLDS,
            "source_order": list(plan.source_order),
            "original_participants": len(original),
            "shared_prepared": str(shared),
            "shared_manifest": str(shared / "participant-manifest.private.json"),
            "fold_manifests": folds,
            "timeline_copies_per_participant": 1,
        }
        _json(output / "crossfit-audit.json", audit)
        return audit
    except Exception as error:
        _json(
            output / "crossfit-failure.json",
            {"status": "failed", "error": str(error)},
        )
        raise


def validated_crossfit_metadata(prepared: Path, config: dict, *, outer_fold: int):
    """Validate every manifest and original identity binding before reading values."""
    prepared = Path(prepared).resolve()
    audit_path = _bound_file(prepared / "crossfit-audit.json", prepared)
    audit = json.loads(audit_path.read_text())
    sources = tuple(sorted(map(str, config["data"]["development_sources"])))
    if (
        audit.get("status") != "passed"
        or audit.get("crossfit_contract") != CROSSFIT_CONTRACT
        or audit.get("revision_contract") != REVISION_CONTRACT
        or audit.get("split_seed") != CROSSFIT_SEED
        or audit.get("outer_folds") != OUTER_FOLDS
        or tuple(audit.get("source_order", ())) != sources
        or audit.get("ascertainment_contract")
        != config["protocol"].get("ascertainment_contract")
    ):
        raise ValueError("cross-fitting preparation contract differs from config")
    shared = prepared / "shared"
    if audit.get("shared_prepared") != str(shared):
        raise ValueError("cross-fitting shared pool differs from its binding")
    shared_audit_path = _bound_file(shared / "input-audit.json", shared)
    shared_audit = json.loads(shared_audit_path.read_text())
    if (
        shared_audit.get("status") != "passed"
        or shared_audit.get("revision_contract") != REVISION_CONTRACT
        or shared_audit.get("timeline_storage") != "full_online"
        or tuple(shared_audit.get("source_order", ())) != sources
        or shared_audit.get("ascertainment_contract") != audit["ascertainment_contract"]
    ):
        raise ValueError("cross-fitting requires an accepted full-online shared pool")
    if not config["project"].get("synthetic", False):
        _validate_real_data_provenance(config)
        for setting, field in (
            ("input_path", "normalized_path"),
            ("training_window_shards", "legacy_windows"),
        ):
            configured = resolve_path(config, config["data"].get(setting))
            if (
                configured is None
                or configured.resolve() != Path(shared_audit[field]).resolve()
            ):
                raise ValueError(
                    f"cross-fitting {field} differs from retained input binding"
                )
    old_index = _index_fragments(
        Path(shared_audit["legacy_windows"]), set(sources), windows=True
    )
    normalized_index = _index_fragments(
        Path(shared_audit["normalized_path"]), set(sources), windows=False
    )
    original = _original_metadata(old_index)
    plan = _population_plan(original, config)
    expected_roles = plan.role_metadata(outer_fold).merge(
        original, on=KEYS, how="left", validate="one_to_one"
    )
    if (
        len(original) != audit.get("original_participants")
        or len(original) != shared_audit.get("original_participants")
    ):
        raise ValueError("cross-fitting participant count differs from binding")
    manifest_path = _bound_file(
        audit["shared_manifest"], shared,
        expected=shared / "participant-manifest.private.json",
    )
    manifest = json.loads(manifest_path.read_text())["participants"]
    if len(manifest) != len(original):
        raise ValueError("cross-fitting shared manifest count differs from binding")
    expected_partitions = {
        tuple(row[:2]): row[2]
        for row in original.itertuples(index=False, name=None)
    }
    expected_paths = {
        key: (
            shared / "timelines" / f"partition={partition}"
            / f"part-{index:05d}.parquet"
        )
        for index, (key, partition) in enumerate(expected_partitions.items())
    }
    seen, paths = set(), set()
    for entry in manifest:
        key = (str(entry["source_id"]), str(entry["participant_id"]))
        if (
            key in seen
            or key not in expected_partitions
            or entry["partition"] != expected_partitions[key]
            or key not in normalized_index
        ):
            raise ValueError("cross-fitting shared identity differs from binding")
        seen.add(key)
        path = _bound_file(
            entry["timeline_path"], shared / "timelines", expected=expected_paths[key]
        )
        if path in paths:
            raise ValueError("cross-fitting participants share a timeline path")
        paths.add(path)
        for field, index in (
            ("legacy_paths", old_index), ("normalized_paths", normalized_index)
        ):
            fragment_paths = sorted(str(fragment.path) for fragment in index[key][0])
            if sorted(entry[field]) != fragment_paths:
                raise ValueError(f"cross-fitting {field} differs from original binding")
    if seen != set(expected_partitions):
        raise ValueError("cross-fitting shared manifest omits original identities")
    if set(audit["fold_manifests"]) != {str(fold) for fold in range(OUTER_FOLDS)}:
        raise ValueError("cross-fitting fold manifest inventory is incomplete")
    fold_path = _bound_file(
        audit["fold_manifests"][str(outer_fold)], prepared,
        expected=(
            prepared / "folds" / f"fold-{outer_fold}"
            / "participant-roles.private.json"
        ),
    )
    roles = pd.DataFrame(json.loads(fold_path.read_text())["participants"])
    try:
        pd.testing.assert_frame_equal(
            roles.sort_values(KEYS, ignore_index=True)[expected_roles.columns],
            expected_roles.sort_values(KEYS, ignore_index=True),
            check_dtype=False,
        )
    except (AssertionError, KeyError) as error:
        raise ValueError(
            "cross-fitting roles differ from frozen identity assignment"
        ) from error
    for role in (*INNER_ROLES, "outer_evaluation"):
        selected = roles.loc[roles.role.eq(role)]
        plan.guard_role_metadata(selected, outer_fold=outer_fold, role=role)
        if role in ("train", "calibration"):
            plan.guard_fit_metadata(selected, outer_fold=outer_fold, role=role)
    return audit, shared_audit, manifest, roles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--normalized", type=Path, required=True)
    parser.add_argument("--legacy-windows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    prepare_crossfit_replays(
        args.normalized, args.legacy_windows, args.output, load_config(args.config)
    )


if __name__ == "__main__":
    main()
