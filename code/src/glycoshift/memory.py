from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional


def validate_past_only(
    query_start: pd.Timestamp,
    retrieved_end_times: list[pd.Timestamp],
) -> None:
    query = pd.Timestamp(query_start)
    invalid = [
        pd.Timestamp(end_time)
        for end_time in retrieved_end_times
        if pd.Timestamp(end_time) >= query
    ]
    if invalid:
        raise ValueError(
            "memory contains current or future records: "
            f"{[value.isoformat() for value in invalid[:3]]}"
        )


def past_candidates(
    windows: pd.DataFrame,
    *,
    source_id: str,
    participant_id: str,
    query_start: pd.Timestamp,
    k: int,
) -> pd.DataFrame:
    candidates = windows.loc[
        (windows["source_id"] == source_id)
        & (windows["participant_id"] == participant_id)
        & (pd.to_datetime(windows["context_end"], utc=True) < pd.Timestamp(query_start))
    ].sort_values("context_end", ascending=False)
    selected = candidates.head(k).copy()
    validate_past_only(
        pd.Timestamp(query_start),
        list(pd.to_datetime(selected["context_end"], utc=True)),
    )
    return selected


def memory_anchor_indices(
    windows: pd.DataFrame,
    *,
    stride_minutes: int,
) -> np.ndarray:
    if stride_minutes <= 0:
        raise ValueError("memory stride must be positive")
    required = {
        "source_id",
        "participant_id",
        "context_end",
    }
    missing = sorted(required.difference(windows.columns))
    if missing:
        raise ValueError(f"memory windows are missing columns: {missing}")
    if windows.empty:
        return np.empty(0, dtype=np.int64)
    context_end = pd.to_datetime(
        windows["context_end"],
        utc=True,
        errors="raise",
    )
    if context_end.isna().any():
        raise ValueError("memory context_end contains missing timestamps")
    stride_ns = int(stride_minutes) * 60 * 1_000_000_000
    keys = pd.DataFrame(
        {
            "source_id": windows["source_id"].astype(str).to_numpy(),
            "participant_id": windows["participant_id"].astype(str).to_numpy(),
            "bucket": context_end.astype("int64").to_numpy() // stride_ns,
        }
    )
    anchors = ~keys.duplicated(
        subset=["source_id", "participant_id", "bucket"],
        keep="first",
    )
    return np.flatnonzero(anchors.to_numpy()).astype(np.int64, copy=False)


@dataclass(frozen=True)
class MemoryRecord:
    dataset_index: int
    source_id: str
    participant_id: str
    context_start: pd.Timestamp
    context_end: pd.Timestamp
    embedding: torch.Tensor


class StrictPastMemoryBank:
    def __init__(self, *, k: int, stride_minutes: int = 30) -> None:
        if k <= 0:
            raise ValueError("memory k must be positive")
        if stride_minutes <= 0:
            raise ValueError("memory stride must be positive")
        self.k = int(k)
        self.stride_minutes = int(stride_minutes)
        self._records: dict[tuple[str, str], list[MemoryRecord]] = {}

    def build(
        self,
        *,
        embeddings: torch.Tensor,
        metadata: list[dict[str, object]],
    ) -> None:
        if embeddings.ndim != 2 or len(embeddings) != len(metadata):
            raise ValueError("memory embeddings and metadata do not align")
        records: dict[tuple[str, str], list[MemoryRecord]] = {}
        seen_buckets: set[tuple[str, str, pd.Timestamp]] = set()
        for dataset_index, (embedding, item) in enumerate(
            zip(embeddings, metadata)  # noqa: B905
        ):
            source_id = str(item["source_id"])
            participant_id = str(item["participant_id"])
            context_start = pd.Timestamp(item["context_start"])
            context_end = pd.Timestamp(item["context_end"])
            bucket = context_end.floor(f"{self.stride_minutes}min")
            bucket_key = (source_id, participant_id, bucket)
            if bucket_key in seen_buckets:
                continue
            seen_buckets.add(bucket_key)
            key = (source_id, participant_id)
            records.setdefault(key, []).append(
                MemoryRecord(
                    dataset_index=dataset_index,
                    source_id=source_id,
                    participant_id=participant_id,
                    context_start=context_start,
                    context_end=context_end,
                    embedding=embedding.detach().cpu().float(),
                )
            )
        for key in records:
            records[key].sort(key=lambda record: record.context_end)
        self._records = records

    def retrieve(
        self,
        *,
        query_embeddings: torch.Tensor,
        query_metadata: list[dict[str, object]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if query_embeddings.ndim != 2 or len(query_embeddings) != len(query_metadata):
            raise ValueError("memory queries and metadata do not align")
        batch_size, embedding_dim = query_embeddings.shape
        output = torch.zeros(
            batch_size,
            self.k,
            embedding_dim,
            device=query_embeddings.device,
            dtype=query_embeddings.dtype,
        )
        padding_mask = torch.ones(
            batch_size,
            self.k,
            device=query_embeddings.device,
            dtype=torch.bool,
        )
        for batch_index, item in enumerate(query_metadata):
            key = (str(item["source_id"]), str(item["participant_id"]))
            query_start = pd.Timestamp(item["context_start"])
            candidates = [
                record
                for record in self._records.get(key, [])
                if record.context_end < query_start
            ]
            validate_past_only(
                query_start,
                [record.context_end for record in candidates],
            )
            if not candidates:
                continue
            candidate_embeddings = torch.stack(
                [record.embedding for record in candidates]
            ).to(query_embeddings.device, dtype=query_embeddings.dtype)
            similarities = functional.cosine_similarity(
                candidate_embeddings,
                query_embeddings[batch_index].unsqueeze(0),
                dim=-1,
            )
            count = min(self.k, len(candidates))
            selected = torch.topk(similarities, k=count).indices
            output[batch_index, :count] = candidate_embeddings[selected]
            padding_mask[batch_index, :count] = False
        return output, padding_mask


@dataclass(frozen=True)
class _ParticipantMemorySlice:
    start: int
    stop: int


class ContiguousStrictPastMemoryBank:
    def __init__(
        self,
        *,
        k: int,
        stride_minutes: int = 30,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        transfer_chunk_rows: int = 100_000,
    ) -> None:
        if k <= 0:
            raise ValueError("memory k must be positive")
        if stride_minutes <= 0:
            raise ValueError("memory stride must be positive")
        if transfer_chunk_rows <= 0:
            raise ValueError("memory transfer chunk must be positive")
        self.k = int(k)
        self.stride_minutes = int(stride_minutes)
        self.device = torch.device(device) if device is not None else None
        self.dtype = dtype
        self.transfer_chunk_rows = int(transfer_chunk_rows)
        self._embeddings: torch.Tensor | None = None
        self._normalized_embeddings: torch.Tensor | None = None
        self._end_times_ns: torch.Tensor | None = None
        self._slices: dict[tuple[str, str], _ParticipantMemorySlice] = {}

    @property
    def record_count(self) -> int:
        return 0 if self._embeddings is None else len(self._embeddings)

    @property
    def storage_bytes(self) -> int:
        tensors = (
            self._embeddings,
            self._normalized_embeddings,
            self._end_times_ns,
        )
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in tensors
            if tensor is not None
        )

    def build(
        self,
        *,
        embeddings: torch.Tensor,
        metadata: list[dict[str, object]],
    ) -> None:
        if embeddings.ndim != 2 or len(embeddings) != len(metadata):
            raise ValueError("memory embeddings and metadata do not align")
        windows = pd.DataFrame(metadata)
        self.build_from_windows(
            embeddings=embeddings,
            windows=windows,
        )

    def build_from_windows(
        self,
        *,
        embeddings: torch.Tensor,
        windows: pd.DataFrame,
    ) -> None:
        if embeddings.ndim != 2:
            raise ValueError("memory embeddings must be a matrix")
        required = {
            "source_id",
            "participant_id",
            "context_end",
        }
        missing = sorted(required.difference(windows.columns))
        if missing:
            raise ValueError(f"memory windows are missing columns: {missing}")
        anchor_indices = memory_anchor_indices(
            windows,
            stride_minutes=self.stride_minutes,
        )
        if len(embeddings) == len(windows):
            embedding_rows = anchor_indices
        elif len(embeddings) == len(anchor_indices):
            embedding_rows = np.arange(len(anchor_indices), dtype=np.int64)
        else:
            raise ValueError(
                "memory embeddings must align with all windows or anchor rows"
            )
        if len(anchor_indices) == 0:
            raise ValueError("memory bank requires at least one anchor row")

        plan = windows.iloc[anchor_indices][
            ["source_id", "participant_id", "context_end"]
        ].copy()
        plan["_source_key"] = plan["source_id"].astype(str)
        plan["_participant_key"] = plan["participant_id"].astype(str)
        plan["_end_ns"] = pd.to_datetime(
            plan["context_end"],
            utc=True,
            errors="raise",
        ).astype("int64")
        plan["_embedding_row"] = embedding_rows
        plan["_anchor_order"] = np.arange(len(plan), dtype=np.int64)
        plan = plan.sort_values(
            [
                "_source_key",
                "_participant_key",
                "_end_ns",
                "_anchor_order",
            ],
            kind="mergesort",
        ).reset_index(drop=True)

        target_device = self.device or embeddings.device
        target_dtype = self.dtype or embeddings.dtype
        embedding_dim = int(embeddings.shape[1])
        record_count = len(plan)
        stored = torch.empty(
            (record_count, embedding_dim),
            device=target_device,
            dtype=target_dtype,
        )
        ordered_rows = plan["_embedding_row"].to_numpy(dtype=np.int64)
        for start in range(0, record_count, self.transfer_chunk_rows):
            stop = min(record_count, start + self.transfer_chunk_rows)
            source_rows = torch.from_numpy(ordered_rows[start:stop]).to(
                embeddings.device
            )
            stored[start:stop].copy_(
                embeddings.index_select(0, source_rows).to(
                    device=target_device,
                    dtype=target_dtype,
                )
            )
        normalized = torch.empty_like(stored)
        for start in range(0, record_count, self.transfer_chunk_rows):
            stop = min(record_count, start + self.transfer_chunk_rows)
            normalized[start:stop] = functional.normalize(
                stored[start:stop],
                p=2,
                dim=-1,
                eps=1e-8,
            )
        end_times_ns = torch.from_numpy(
            plan["_end_ns"].to_numpy(dtype=np.int64)
        ).to(target_device)

        slices: dict[tuple[str, str], _ParticipantMemorySlice] = {}
        offset = 0
        for key, participant in plan.groupby(
            ["_source_key", "_participant_key"],
            sort=False,
            observed=False,
        ):
            stop = offset + len(participant)
            slices[(str(key[0]), str(key[1]))] = _ParticipantMemorySlice(
                start=offset,
                stop=stop,
            )
            offset = stop
        if offset != record_count:
            raise AssertionError("participant memory slices do not cover records")

        self.device = stored.device
        self.dtype = target_dtype
        self._embeddings = stored
        self._normalized_embeddings = normalized
        self._end_times_ns = end_times_ns
        self._slices = slices

    def retrieve(
        self,
        *,
        query_embeddings: torch.Tensor,
        query_metadata: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if query_embeddings.ndim != 2 or len(query_embeddings) != len(
            query_metadata
        ):
            raise ValueError("memory queries and metadata do not align")
        if (
            self._embeddings is None
            or self._normalized_embeddings is None
            or self._end_times_ns is None
            or self.device is None
            or self.dtype is None
        ):
            raise RuntimeError("memory bank has not been built")
        if query_embeddings.device != self.device:
            raise ValueError(
                "memory queries must be on the contiguous bank device"
            )

        batch_size, embedding_dim = query_embeddings.shape
        if embedding_dim != self._embeddings.shape[1]:
            raise ValueError("memory query embedding width does not align")
        output = torch.zeros(
            batch_size,
            self.k,
            embedding_dim,
            device=query_embeddings.device,
            dtype=query_embeddings.dtype,
        )
        padding_mask = torch.ones(
            batch_size,
            self.k,
            device=query_embeddings.device,
            dtype=torch.bool,
        )
        grouped_queries: dict[
            tuple[str, str],
            list[tuple[int, int]],
        ] = {}
        for batch_index, item in enumerate(query_metadata):
            key = (str(item["source_id"]), str(item["participant_id"]))
            query_start_ns = pd.Timestamp(item["context_start"]).value
            grouped_queries.setdefault(key, []).append(
                (batch_index, query_start_ns)
            )

        for key, query_records in grouped_queries.items():
            participant_slice = self._slices.get(key)
            if participant_slice is None:
                continue
            start = participant_slice.start
            stop = participant_slice.stop
            batch_indices = torch.tensor(
                [record[0] for record in query_records],
                device=self.device,
                dtype=torch.long,
            )
            query_times = torch.tensor(
                [record[1] for record in query_records],
                device=self.device,
                dtype=torch.long,
            )
            participant_embeddings = self._embeddings[start:stop]
            if participant_embeddings.dtype == query_embeddings.dtype:
                participant_normalized = self._normalized_embeddings[
                    start:stop
                ]
            else:
                participant_embeddings = participant_embeddings.to(
                    query_embeddings.dtype
                )
                participant_normalized = functional.normalize(
                    participant_embeddings,
                    p=2,
                    dim=-1,
                    eps=1e-8,
                )
            query = query_embeddings.index_select(0, batch_indices)
            with torch.autocast(
                device_type=query_embeddings.device.type,
                enabled=False,
            ):
                # Both sides are pinned to float32 here, for two reasons that
                # happen to have the same fix.
                #
                # The equivalence this bank was adopted on is a FLOAT32 cosine.
                # Disabling autocast does not convert a tensor that is already
                # bfloat16, so a caller inside `autocast(bfloat16)` was getting
                # a bfloat16 cosine instead, and the precision the adoption
                # decision rested on was not being delivered.
                #
                # It also removes a dtype mismatch that killed C3 on its first
                # retrieval epoch, `BFloat16 != float` at the matmul below. That
                # crash was NOT reproduced on CPU across three bank/query dtype
                # combinations, so this is not offered as a located root cause.
                # What is established is narrower and testable: after this, the
                # two operands cannot disagree whatever the caller's dtype, and
                # the float32-to-float32 path every completed arm ran is bitwise
                # unchanged, which a test asserts.
                query_normalized = functional.normalize(
                    query.to(torch.float32),
                    p=2,
                    dim=-1,
                    eps=1e-8,
                )
                similarities = torch.matmul(
                    query_normalized,
                    participant_normalized.to(torch.float32).transpose(0, 1),
                )
                valid = self._end_times_ns[start:stop].unsqueeze(0) < (
                    query_times.unsqueeze(1)
                )
                similarities = similarities.masked_fill(
                    ~valid,
                    -torch.inf,
                )
                selected_count = min(self.k, stop - start)
                _, selected = torch.topk(
                    similarities,
                    k=selected_count,
                    dim=1,
                )
            selected_valid = torch.gather(valid, 1, selected)
            selected_embeddings = participant_embeddings[selected]
            selected_embeddings = torch.where(
                selected_valid.unsqueeze(-1),
                selected_embeddings,
                torch.zeros_like(selected_embeddings),
            )
            output[batch_indices, :selected_count] = selected_embeddings
            padding_mask[batch_indices, :selected_count] = ~selected_valid
        return output, padding_mask
