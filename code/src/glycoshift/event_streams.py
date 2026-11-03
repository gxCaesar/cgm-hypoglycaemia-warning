from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np
import pandas as pd
import torch

EVENT_HORIZONS = (30, 60)
EVENT_PARTITION = "train"


def _sha256(value: str, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _identifier(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    if any(character in value for character in ("\x00", "\r", "\n", "|")):
        raise ValueError(f"{field} contains a forbidden delimiter")
    return value


def _source_order(values: Sequence[str]) -> tuple[str, ...]:
    sources = tuple(_identifier(value, field="source_id") for value in values)
    if not sources or len(set(sources)) != len(sources):
        raise ValueError("source_order must be non-empty and unique")
    if sources != tuple(sorted(sources)):
        raise ValueError("source_order must use lexical order")
    return sources


def _horizons(values: Sequence[int]) -> tuple[int, ...]:
    horizons = tuple(values)
    if horizons != EVENT_HORIZONS:
        raise ValueError(f"event horizons must be exactly {EVENT_HORIZONS}")
    return horizons


def _timestamp_text(value: Any, *, field: str) -> str:
    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(timestamp):
        raise ValueError(f"{field} contains an invalid timestamp")
    return pd.Timestamp(timestamp).isoformat()


def _optional_timestamp_text(value: Any, *, field: str) -> str | None:
    if pd.isna(value):
        return None
    return _timestamp_text(value, field=field)


def _canonical_stream_rows(
    windows: pd.DataFrame,
    *,
    horizons: tuple[int, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in windows.to_dict(orient="records"):
        payload: dict[str, Any] = {
            "source_id": _identifier(row["source_id"], field="source_id"),
            "participant_id": _identifier(
                row["participant_id"],
                field="participant_id",
            ),
            "partition": _identifier(row["partition"], field="partition"),
            "context_start": _timestamp_text(
                row["context_start"],
                field="context_start",
            ),
            "context_end": _timestamp_text(
                row["context_end"],
                field="context_end",
            ),
            "patient_day": _timestamp_text(
                row["patient_day"],
                field="patient_day",
            ),
            "participant_end_index": int(row["participant_end_index"]),
            "severe_hypo_currently_active": bool(
                row["severe_hypo_currently_active"]
            ),
        }
        for horizon in horizons:
            prefix = f"severe_hypo_{horizon}"
            label = row[f"{prefix}_label"]
            payload[f"{prefix}_label"] = None if pd.isna(label) else int(label)
            event_id = row[f"{prefix}_event_id"]
            payload[f"{prefix}_event_id"] = (
                None
                if pd.isna(event_id)
                else _identifier(str(event_id), field=f"{prefix}_event_id")
            )
            payload[f"{prefix}_event_onset"] = _optional_timestamp_text(
                row[f"{prefix}_event_onset"],
                field=f"{prefix}_event_onset",
            )
        rows.append(payload)
    return rows


def participant_stream_content_sha256(
    windows: pd.DataFrame,
    *,
    horizons: Sequence[int] = EVENT_HORIZONS,
) -> str:
    """Hash every decision row used to prove participant-stream completeness."""

    resolved_horizons = _horizons(horizons)
    required = {
        "source_id",
        "participant_id",
        "partition",
        "context_start",
        "context_end",
        "patient_day",
        "participant_end_index",
        "severe_hypo_currently_active",
    }
    for horizon in resolved_horizons:
        required.update(
            {
                f"severe_hypo_{horizon}_label",
                f"severe_hypo_{horizon}_event_id",
                f"severe_hypo_{horizon}_event_onset",
            }
        )
    missing = required.difference(windows.columns)
    if missing:
        raise ValueError(f"participant stream is missing columns: {sorted(missing)}")
    if windows.empty:
        raise ValueError("participant stream cannot be empty")
    payload = _canonical_stream_rows(windows, horizons=resolved_horizons)
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ParticipantEventStream:
    """One hash-bound, complete, eligible participant decision trajectory."""

    source_id: str
    participant_id: str
    partition: str
    window_shard_sha256: str
    content_sha256: str
    eligible_row_indices: tuple[int, ...]
    decision_minutes: torch.Tensor
    patient_days: int
    event_onsets: tuple[tuple[int, torch.Tensor], ...]

    @property
    def decision_count(self) -> int:
        return len(self.eligible_row_indices)

    def event_onset_minutes(self, horizon: int) -> torch.Tensor:
        for current, values in self.event_onsets:
            if current == horizon:
                return values
        raise KeyError(f"unknown event horizon: {horizon}")

    def event_count(self, horizon: int) -> int:
        return int(self.event_onset_minutes(horizon).numel())


def _validate_one_identity(windows: pd.DataFrame, column: str) -> str:
    values = windows[column]
    if values.isna().any():
        raise ValueError(f"participant stream {column} contains missing values")
    resolved = tuple(dict.fromkeys(values.astype(str)))
    if len(resolved) != 1:
        raise ValueError(f"participant stream must contain one {column}")
    return _identifier(resolved[0], field=column)


def build_participant_event_stream(
    windows: pd.DataFrame,
    *,
    expected_content_sha256: str,
    window_shard_sha256: str,
    horizons: Sequence[int] = EVENT_HORIZONS,
    expected_partition: str = EVENT_PARTITION,
) -> ParticipantEventStream:
    """Validate and materialize one complete train-only participant stream."""

    resolved_horizons = _horizons(horizons)
    expected_digest = _sha256(
        expected_content_sha256,
        field="expected_content_sha256",
    )
    shard_digest = _sha256(window_shard_sha256, field="window_shard_sha256")
    observed_digest = participant_stream_content_sha256(
        windows,
        horizons=resolved_horizons,
    )
    if observed_digest != expected_digest:
        raise ValueError("participant stream content hash does not match authority")

    source = _validate_one_identity(windows, "source_id")
    participant = _validate_one_identity(windows, "participant_id")
    partition = _validate_one_identity(windows, "partition")
    if partition != expected_partition or expected_partition != EVENT_PARTITION:
        raise ValueError("event participant stream must use the train partition")

    ordered = windows.reset_index(drop=True).copy()
    context_end = pd.to_datetime(ordered["context_end"], utc=True, errors="coerce")
    context_start = pd.to_datetime(
        ordered["context_start"],
        utc=True,
        errors="coerce",
    )
    if context_end.isna().any() or context_start.isna().any():
        raise ValueError("participant stream contains invalid context timestamps")
    if len(context_end) > 1 and not context_end.diff().iloc[1:].gt(
        pd.Timedelta(0)
    ).all():
        raise ValueError("participant decision times must be strictly increasing")
    if not context_start.lt(context_end).all():
        raise ValueError("participant context_start must precede context_end")

    end_indices = pd.to_numeric(
        ordered["participant_end_index"],
        errors="coerce",
    )
    if end_indices.isna().any() or not np.equal(
        end_indices.to_numpy(dtype=float),
        np.floor(end_indices.to_numpy(dtype=float)),
    ).all():
        raise ValueError("participant_end_index must contain integers")
    if len(end_indices) > 1 and bool(
        (np.diff(end_indices.to_numpy(dtype=np.int64)) <= 0).any()
    ):
        raise ValueError("participant_end_index must be strictly increasing")

    active_values = ordered["severe_hypo_currently_active"]
    if active_values.isna().any():
        raise ValueError("current severe-event status contains missing values")
    active = active_values.astype(bool).to_numpy()
    eligible_indices = tuple(np.flatnonzero(~active).tolist())
    if not eligible_indices:
        raise ValueError("participant stream has no eligible decisions")
    eligible = ordered.iloc[list(eligible_indices)].copy()
    eligible_times = context_end.iloc[list(eligible_indices)].reset_index(drop=True)
    patient_days = pd.to_datetime(
        eligible["patient_day"],
        errors="coerce",
    )
    if patient_days.isna().any():
        raise ValueError("participant stream contains invalid patient days")
    day_count = int(patient_days.nunique())
    if day_count <= 0:
        raise ValueError("participant stream has zero patient-days")

    origin = eligible_times.iloc[0]
    decision_minutes = torch.tensor(
        (eligible_times - origin).dt.total_seconds().to_numpy(dtype=float) / 60.0,
        dtype=torch.float64,
    )
    event_onsets: list[tuple[int, torch.Tensor]] = []
    for horizon in resolved_horizons:
        prefix = f"severe_hypo_{horizon}"
        labels = pd.to_numeric(eligible[f"{prefix}_label"], errors="coerce")
        if labels.isna().any() or not labels.isin((0, 1)).all():
            raise ValueError(f"{prefix} labels must be complete and binary")
        event_ids = eligible[f"{prefix}_event_id"]
        onset_values = eligible[f"{prefix}_event_onset"]
        has_event = event_ids.notna()
        has_onset = onset_values.notna()
        if not has_event.eq(labels.astype(bool)).all():
            raise ValueError(f"{prefix} labels and event IDs disagree")
        if not has_event.eq(has_onset).all():
            raise ValueError(f"{prefix} event IDs and onsets disagree")

        positives = eligible.loc[has_event].copy()
        if positives.empty:
            event_tensor = torch.empty(0, dtype=torch.float64)
        else:
            positives[f"{prefix}_event_onset"] = pd.to_datetime(
                positives[f"{prefix}_event_onset"],
                utc=True,
                errors="coerce",
            )
            if positives[f"{prefix}_event_onset"].isna().any():
                raise ValueError(f"{prefix} contains an invalid event onset")
            # Event ID columns are categorical whenever the frame descends from
            # window shards, and a grouped aggregation over a categorical
            # reindexes to every declared category, so absent IDs arrive with
            # `nunique() == 0` and this check rejects clean data. That is what
            # stopped the 2026-08-17 baseline campaign from the equivalent line
            # in `operational_metrics`; this path had not yet run on real data.
            mapping = positives.groupby(
                f"{prefix}_event_id", sort=False, observed=True
            )[f"{prefix}_event_onset"].nunique()
            if not mapping.eq(1).all():
                raise ValueError(f"{prefix} maps one event ID to multiple onsets")
            event_table = (
                positives[[f"{prefix}_event_id", f"{prefix}_event_onset"]]
                .drop_duplicates()
                .sort_values(f"{prefix}_event_onset")
            )
            if event_table[f"{prefix}_event_onset"].duplicated().any():
                raise ValueError(f"{prefix} maps multiple events to one onset")

            positive_times = pd.to_datetime(
                positives["context_end"],
                utc=True,
            )
            onset_series = positives[f"{prefix}_event_onset"]
            if not onset_series.gt(positive_times).all() or not onset_series.le(
                positive_times + pd.Timedelta(minutes=horizon)
            ).all():
                raise ValueError(f"{prefix} event lies outside its warning horizon")
            event_tensor = torch.tensor(
                (
                    event_table[f"{prefix}_event_onset"] - origin
                ).dt.total_seconds().to_numpy(dtype=float)
                / 60.0,
                dtype=torch.float64,
            )
            if len(event_tensor) > 1 and not bool(
                (event_tensor[1:] > event_tensor[:-1]).all()
            ):
                raise ValueError(f"{prefix} event onsets must be strictly increasing")
        event_onsets.append((horizon, event_tensor))

    return ParticipantEventStream(
        source_id=source,
        participant_id=participant,
        partition=partition,
        window_shard_sha256=shard_digest,
        content_sha256=observed_digest,
        eligible_row_indices=eligible_indices,
        decision_minutes=decision_minutes,
        patient_days=day_count,
        event_onsets=tuple(event_onsets),
    )


@dataclass(frozen=True)
class ParticipantEventPack:
    epoch: int
    update_index: int
    miss_participants: tuple[
        tuple[int, tuple[tuple[str, str], ...]],
        ...,
    ]
    burden_participants: tuple[tuple[str, str], ...]

    def miss_participants_for(
        self,
        horizon: int,
    ) -> tuple[tuple[str, str], ...]:
        for current, participants in self.miss_participants:
            if current == horizon:
                return participants
        raise KeyError(f"unknown event horizon: {horizon}")


def _participant_pool(
    values: Mapping[str, Sequence[str]],
    *,
    source_order: tuple[str, ...],
    field: str,
) -> dict[str, tuple[str, ...]]:
    if set(values) != set(source_order):
        raise ValueError(f"{field} sources must exactly match source_order")
    output: dict[str, tuple[str, ...]] = {}
    for source in source_order:
        participants = tuple(
            _identifier(value, field="participant_id") for value in values[source]
        )
        if not participants or len(set(participants)) != len(participants):
            raise ValueError(f"{field}[{source!r}] must be non-empty and unique")
        output[source] = participants
    return output


def _cycle_permutations(
    values: tuple[str, ...],
    *,
    length: int,
    seed: int,
    epoch: int,
    source: str,
    role: str,
) -> tuple[str, ...]:
    output: list[str] = []
    cycle = 0
    while len(output) < length:
        message = f"{seed}|{epoch}|{source}|{role}|{cycle}".encode()
        derived = int.from_bytes(hashlib.sha256(message).digest()[:8], "big")
        permutation = np.random.default_rng(derived).permutation(len(values))
        output.extend(values[int(index)] for index in permutation)
        cycle += 1
    return tuple(output[:length])


class AllSourceParticipantPackSampler:
    """Deterministic dual-queue participant packs for C2 event estimation."""

    def __init__(
        self,
        *,
        source_order: Sequence[str],
        all_participants: Mapping[str, Sequence[str]],
        event_positive_participants: Mapping[
            int,
            Mapping[str, Sequence[str]],
        ],
        updates_per_epoch: int,
        seed: int,
    ) -> None:
        self.source_order = _source_order(source_order)
        self.all_participants = _participant_pool(
            all_participants,
            source_order=self.source_order,
            field="all_participants",
        )
        if set(event_positive_participants) != set(EVENT_HORIZONS):
            raise ValueError(
                f"event-positive participant pools must cover {EVENT_HORIZONS}"
            )
        self.event_positive_participants = {
            horizon: _participant_pool(
                event_positive_participants[horizon],
                source_order=self.source_order,
                field=f"event_positive_participants[{horizon}]",
            )
            for horizon in EVENT_HORIZONS
        }
        for horizon in EVENT_HORIZONS:
            for source in self.source_order:
                if not set(
                    self.event_positive_participants[horizon][source]
                ).issubset(self.all_participants[source]):
                    raise ValueError(
                        "event-positive participants must belong to the "
                        "all-participant pool"
                    )
        if not isinstance(updates_per_epoch, Integral) or isinstance(
            updates_per_epoch,
            bool,
        ):
            raise TypeError("updates_per_epoch must be an integer")
        if int(updates_per_epoch) <= 0:
            raise ValueError("updates_per_epoch must be positive")
        if not isinstance(seed, Integral) or isinstance(seed, bool) or int(seed) < 0:
            raise ValueError("seed must be a non-negative integer")
        self.updates_per_epoch = int(updates_per_epoch)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if not isinstance(epoch, Integral) or isinstance(epoch, bool):
            raise TypeError("epoch must be an integer")
        if int(epoch) < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.updates_per_epoch

    def __iter__(self) -> Iterator[ParticipantEventPack]:
        misses = {
            horizon: {
                source: _cycle_permutations(
                    self.event_positive_participants[horizon][source],
                    length=self.updates_per_epoch,
                    seed=self.seed,
                    epoch=self.epoch,
                    source=source,
                    role=f"miss-{horizon}",
                )
                for source in self.source_order
            }
            for horizon in EVENT_HORIZONS
        }
        burdens = {
            source: _cycle_permutations(
                self.all_participants[source],
                length=self.updates_per_epoch,
                seed=self.seed,
                epoch=self.epoch,
                source=source,
                role="burden",
            )
            for source in self.source_order
        }
        for update_index in range(self.updates_per_epoch):
            yield ParticipantEventPack(
                epoch=self.epoch,
                update_index=update_index,
                miss_participants=tuple(
                    (
                        horizon,
                        tuple(
                            (source, misses[horizon][source][update_index])
                            for source in self.source_order
                        ),
                    )
                    for horizon in EVENT_HORIZONS
                ),
                burden_participants=tuple(
                    (source, burdens[source][update_index])
                    for source in self.source_order
                ),
            )
