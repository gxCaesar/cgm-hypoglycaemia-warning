"""Private, explicit record inventory and verified-pair bindings for inference.

The group manifest binds ``record_manifest`` and ``pair_manifest`` by canonical
path, size and mtime. All three JSON files carry schema_version, freeze_id,
partition and source_id. The inventory also binds the accepted normalized_inputs
and lists every record as {source_id, participant_id} in ``records``. The pair
file repeats its inventory binding, records status=verified_partial_overlap and
exactly two ``members``. The group file supplies complete ``groups`` rows with
source_id, participant_id, group_id. Only the verified pair may share a group.
These files are private input authorities, never inferred from glucose or names.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

from .fixed_frame_evaluation import FixedFrameEvaluation, FixedFrameKey, _aggregate
from .revision_external import _bound_file

_IDS = ["source_id", "participant_id"]


def validated_horizon_counts(value: dict | None, normalized_participants: int):
    if value is None:
        return None
    fields = {"evaluated_participants", "zero_base_exposure_participants"}
    if not isinstance(value, dict) or set(value) != {"30", "60"} or any(
        not isinstance(row, dict) or set(row) != fields
        or any(
            isinstance(count, bool) or not isinstance(count, int) or count < 0
            for count in row.values()
        ) or sum(row.values()) != normalized_participants
        for row in value.values()
    ):
        raise ValueError("horizon participant counts do not cover normalized records")
    return value


def _records(rows: list[dict]) -> set[tuple[str, str]]:
    if not isinstance(rows, list) or not rows:
        raise ValueError("record inventory must contain complete record identities")
    keys = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != set(_IDS) or any(
            not isinstance(row[column], str) or not row[column].strip()
            for column in _IDS
        ):
            raise ValueError("record inventory requires strict string identity fields")
        keys.append((row["source_id"], row["participant_id"]))
    if len(set(keys)) != len(keys):
        raise ValueError("record inventory has duplicate identities")
    return set(keys)


@dataclass(frozen=True)
class ExternalRecordGroups:
    participant_groups: dict[tuple[str, str], str]
    pair: frozenset[tuple[str, str]]
    bindings: tuple[dict, ...]

    def restrict(self, keys: set[tuple[str, str]]) -> dict[tuple[str, str], str]:
        if not keys.issubset(self.participant_groups):
            raise ValueError(
                "evaluation record is outside the complete record inventory"
            )
        return {key: self.participant_groups[key] for key in keys}

    def recheck(self) -> None:
        for binding in self.bindings:
            _bound_file(binding)

    def aggregate_counts(self) -> dict:
        return {
            "full_record_count": len(self.participant_groups),
            "full_group_count": len(set(self.participant_groups.values())),
            "verified_pair_record_count": len(self.pair),
            "record_point_weights": "original_distinct_record_weights",
        }


def load_external_record_groups(
    path: Path, *, freeze_id: str, partition: str, source: dict,
    normalized_participants: int,
) -> ExternalRecordGroups:
    """Fail closed on complete inventory, accepted source and exact pair binding."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("group manifest must be a regular non-symlink file")
    stat = path.stat()
    own_binding = {
        "path": str(path.resolve()), "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    groups = json.loads(path.read_text())
    if not isinstance(groups, dict) or not {
        "record_manifest", "pair_manifest", "groups"
    }.issubset(groups) or not isinstance(groups["groups"], list):
        raise ValueError("group manifest requires inventory, pair and complete groups")
    inventory_binding = groups["record_manifest"]
    pair_binding = groups["pair_manifest"]
    inventory = json.loads(_bound_file(inventory_binding).read_text())
    pair = json.loads(_bound_file(pair_binding).read_text())
    if (
        not isinstance(inventory, dict) or "records" not in inventory
        or not isinstance(pair, dict) or "members" not in pair
    ):
        raise ValueError("record and pair manifests lack their identity rows")
    identity = {
        "schema_version": "1.0", "freeze_id": freeze_id,
        "partition": partition, "source_id": source["source_id"],
    }
    if (
        partition != "external_loop"
        or source["source_id"] != "Loop Observational Study"
    ):
        raise ValueError("the verified-pair correction is restricted to exposed Loop")
    if any(
        any(payload.get(name) != value for name, value in identity.items())
        for payload in (groups, inventory, pair)
    ):
        raise ValueError(
            "group/inventory/pair manifest freeze or source identity differs"
        )
    if inventory.get("normalized_inputs") != source["normalized_inputs"]:
        raise ValueError(
            "record inventory is not bound to the accepted normalized inputs"
        )
    records = _records(inventory["records"])
    if len(records) != normalized_participants or any(
        source_id != source["source_id"] for source_id, _ in records
    ):
        raise ValueError(
            "complete record inventory differs from native source population"
        )
    members = _records(pair["members"])
    if (
        pair.get("status") != "verified_partial_overlap"
        or pair.get("record_manifest") != inventory_binding
        or len(members) != 2 or not members.issubset(records)
    ):
        raise ValueError(
            "verified pair is not exactly bound to the full record inventory"
        )
    mapping = {}
    for row in groups["groups"]:
        if not isinstance(row, dict) or set(row) != {*_IDS, "group_id"}:
            raise ValueError("group rows require exact record and group identity")
        key = next(iter(_records([{column: row[column] for column in _IDS}])))
        group_id = row["group_id"]
        if key in mapping or not isinstance(group_id, str) or not group_id.strip():
            raise ValueError("group rows have duplicate records or invalid group IDs")
        mapping[key] = group_id
    if set(mapping) != records:
        raise ValueError("group mapping does not exactly cover the record inventory")
    member_groups = {mapping[key] for key in members}
    if len(member_groups) != 1 or {
        key for key, group_id in mapping.items() if group_id in member_groups
    } != members or len(set(mapping.values())) != len(records) - 1:
        raise ValueError("only the two verified pair records may share a group")
    result = ExternalRecordGroups(
        mapping, frozenset(members), (own_binding, inventory_binding, pair_binding)
    )
    result.recheck()
    return result


def exclude_pair_records(
    evaluations: Mapping[FixedFrameKey, FixedFrameEvaluation],
    pair: frozenset[tuple[str, str]],
) -> dict[FixedFrameKey, FixedFrameEvaluation]:
    """Filter already-evaluated records; never refit or merge an alarm history."""
    result = {}
    for key, evaluation in evaluations.items():
        def keep(frame: pd.DataFrame) -> pd.DataFrame:
            if not set(_IDS).issubset(frame):
                return frame.copy()
            return frame.loc[[
                record not in pair for record in frame[_IDS].itertuples(
                    index=False, name=None
                )
            ]].copy()

        table = keep(evaluation.participants)
        sources = pd.DataFrame([
            {"source_id": source, **_aggregate(table.loc[table.source_id.eq(source)])}
            for source in evaluation.spec.source_order
        ])
        undefined = sources.loc[
            sources.event_positive_participant_count.eq(0), "source_id"
        ].tolist()
        summary = _aggregate(table)
        summary.update({
            "source_count": len(evaluation.spec.source_order),
            "sensitivity_defined": not undefined,
            "undefined_sensitivity_sources": undefined,
            "participant_pooled_sensitivity": summary.pop("fixed_sensitivity"),
            "equal_source_participant_macro_fixed_sensitivity": (
                float("nan") if undefined else float(sources.fixed_sensitivity.mean())
            ),
            "equal_source_participant_macro_unmatched_alarms_per_patient_day": float(
                sources.participant_macro_unmatched_alarms_per_patient_day.mean()
            ),
            "equal_source_participant_macro_delivered_alarms_per_patient_day": float(
                sources.participant_macro_delivered_alarms_per_patient_day.mean()
            ),
        })
        result[key] = replace(
            evaluation, participants=table, sources=sources,
            events=keep(evaluation.events),
            alarms=keep(evaluation.alarms), summary=summary,
        )
    return result


def evaluation_keys(
    evaluations: Mapping[FixedFrameKey, FixedFrameEvaluation],
) -> set[tuple[str, str]]:
    return set().union(*(
        set(value.participants[_IDS].itertuples(index=False, name=None))
        for value in evaluations.values()
    ))


def horizon_group_counts(
    table: pd.DataFrame, mapping: Mapping[tuple[str, str], str],
) -> tuple[int, int]:
    keys = list(table[_IDS].itertuples(index=False, name=None))
    return len({mapping[key] for key in keys}), len({
        mapping[key] for key, positive in zip(
            keys, table.base_event_count.to_numpy() > 0, strict=True
        ) if positive
    })
