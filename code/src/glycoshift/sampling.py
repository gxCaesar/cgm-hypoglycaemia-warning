from __future__ import annotations

import math
from collections.abc import Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Sampler


class SourceBalancedSampler(Sampler[int]):
    """Sample sources uniformly and records uniformly within each source.

    This is distributionally equivalent to assigning every record weight
    ``1 / source_count`` in ``WeightedRandomSampler(replacement=True)``. It
    avoids ``torch.multinomial`` and its 2**24 category limit by drawing in
    bounded chunks.
    """

    def __init__(
        self,
        source_indices: Sequence[int] | np.ndarray,
        *,
        num_sources: int,
        num_samples: int | None = None,
        generator: torch.Generator | None = None,
        chunk_size: int = 65_536,
    ) -> None:
        values = np.asarray(source_indices, dtype=np.int64)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("source_indices must be a nonempty vector")
        if num_sources <= 0:
            raise ValueError("num_sources must be positive")
        if values.min() < 0 or values.max() >= num_sources:
            raise ValueError("source index is outside [0, num_sources)")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        resolved_samples = len(values) if num_samples is None else int(num_samples)
        if resolved_samples <= 0:
            raise ValueError("num_samples must be positive")

        pools = tuple(
            torch.from_numpy(
                np.flatnonzero(values == source_index).astype(
                    np.int64,
                    copy=False,
                )
            )
            for source_index in range(num_sources)
        )
        if any(len(pool) == 0 for pool in pools):
            raise ValueError("every source must contain at least one record")

        self.pools = pools
        self.num_samples = resolved_samples
        self.generator = generator
        self.chunk_size = int(chunk_size)

    def __iter__(self) -> Iterator[int]:
        remaining = self.num_samples
        while remaining:
            draw_count = min(self.chunk_size, remaining)
            source_draws = torch.randint(
                high=len(self.pools),
                size=(draw_count,),
                dtype=torch.int64,
                generator=self.generator,
            )
            record_draws = torch.empty(draw_count, dtype=torch.int64)
            for source_index, pool in enumerate(self.pools):
                positions = torch.nonzero(
                    source_draws == source_index,
                    as_tuple=False,
                ).flatten()
                if positions.numel() == 0:
                    continue
                offsets = torch.randint(
                    high=len(pool),
                    size=(positions.numel(),),
                    dtype=torch.int64,
                    generator=self.generator,
                )
                record_draws[positions] = pool[offsets]
            yield from record_draws.tolist()
            remaining -= draw_count

    def __len__(self) -> int:
        return self.num_samples


def all_source_allocation(
    *,
    batch_size: int,
    source_count: int,
    extras_cursor: int,
) -> tuple[tuple[int, ...], int]:
    """Return the frozen base-plus-rotating-extras source allocation."""

    if any(
        not isinstance(value, (int, np.integer)) or isinstance(value, bool)
        for value in (batch_size, source_count, extras_cursor)
    ):
        raise TypeError("allocation values must be integers")
    if source_count <= 0:
        raise ValueError("source_count must be positive")
    if batch_size < source_count:
        raise ValueError("an all-source batch must contain every source")
    if not 0 <= extras_cursor < source_count:
        raise ValueError("extras_cursor is outside the source order")
    base, remainder = divmod(batch_size, source_count)
    counts = [base] * source_count
    for offset in range(remainder):
        counts[(extras_cursor + offset) % source_count] += 1
    next_cursor = (extras_cursor + remainder) % source_count
    return tuple(counts), next_cursor


def effective_update_sizes(
    *,
    epoch_draws: int,
    physical_batch_size: int,
    effective_batch_size: int,
) -> tuple[int, ...]:
    """Return record counts for every protocol-v1.1 effective update."""

    if any(
        not isinstance(value, (int, np.integer)) or isinstance(value, bool)
        for value in (epoch_draws, physical_batch_size, effective_batch_size)
    ):
        raise TypeError("draw and batch sizes must be integers")
    if epoch_draws <= 0 or physical_batch_size <= 0 or effective_batch_size <= 0:
        raise ValueError("draw and batch sizes must be positive")
    if effective_batch_size % physical_batch_size:
        raise ValueError("effective_batch_size must be a physical-batch multiple")
    full_batches, final_size = divmod(epoch_draws, physical_batch_size)
    physical_sizes = [physical_batch_size] * full_batches
    if final_size:
        physical_sizes.append(final_size)
    accumulation_steps = effective_batch_size // physical_batch_size
    return tuple(
        sum(physical_sizes[start : start + accumulation_steps])
        for start in range(0, len(physical_sizes), accumulation_steps)
    )


class AllSourceBatchSampler(Sampler[list[int]]):
    """Yield deterministic replacement batches containing every source.

    Integer source indices must already follow the supplied lexical source
    order. A fresh CPU generator is seeded with ``seed + epoch`` for every
    iterator. Per-source draws and the final batch permutation consume that
    one generator in the protocol's fixed order.
    """

    def __init__(
        self,
        source_indices: Sequence[int] | np.ndarray,
        *,
        source_order: Sequence[str],
        epoch_draws: int,
        batch_size: int,
        seed: int = 42,
        epoch: int = 0,
    ) -> None:
        raw_values = np.asarray(source_indices)
        if raw_values.ndim != 1 or raw_values.size == 0:
            raise ValueError("source_indices must be a nonempty vector")
        if not np.issubdtype(raw_values.dtype, np.integer) or np.issubdtype(
            raw_values.dtype,
            np.bool_,
        ):
            raise TypeError("source_indices must contain integers")
        values = raw_values.astype(np.int64, copy=False)
        if any(not isinstance(source, str) for source in source_order):
            raise TypeError("source_order entries must be strings")
        sources = tuple(source_order)
        if not sources or len(set(sources)) != len(sources):
            raise ValueError("source_order must be non-empty and unique")
        if sources != tuple(sorted(sources)):
            raise ValueError("source_order must use lexical order")
        if any(
            not isinstance(value, (int, np.integer)) or isinstance(value, bool)
            for value in (epoch_draws, batch_size, seed, epoch)
        ):
            raise TypeError("draw, batch, seed, and epoch values must be integers")
        if epoch_draws <= 0:
            raise ValueError("epoch_draws must be positive")
        if batch_size < len(sources):
            raise ValueError("batch_size must be large enough for every source")
        if seed < 0 or epoch < 0:
            raise ValueError("seed and epoch must be non-negative")
        if values.min() < 0 or values.max() >= len(sources):
            raise ValueError("source index is outside the frozen source order")

        pools = tuple(
            torch.from_numpy(
                np.flatnonzero(values == source_index).astype(np.int64, copy=False)
            )
            for source_index in range(len(sources))
        )
        if any(len(pool) == 0 for pool in pools):
            raise ValueError("every source must contain at least one record")
        final_size = epoch_draws % batch_size
        if final_size and final_size < len(sources):
            raise ValueError("the final physical batch would omit a source")

        self.pools = pools
        self.source_order = sources
        self.epoch_draws = int(epoch_draws)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = int(epoch)

    def set_epoch(self, epoch: int) -> None:
        if not isinstance(epoch, (int, np.integer)) or isinstance(epoch, bool):
            raise TypeError("epoch must be an integer")
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        epoch_seed = self.seed + self.epoch
        if epoch_seed >= 2**63:
            raise ValueError("epoch seed exceeds the signed 63-bit contract")
        generator = torch.Generator(device="cpu").manual_seed(epoch_seed)
        remaining = self.epoch_draws
        extras_cursor = 0
        while remaining:
            current_size = min(self.batch_size, remaining)
            allocations, extras_cursor = all_source_allocation(
                batch_size=current_size,
                source_count=len(self.pools),
                extras_cursor=extras_cursor,
            )
            chunks: list[torch.Tensor] = []
            for pool, draw_count in zip(self.pools, allocations, strict=True):
                offsets = torch.randint(
                    high=len(pool),
                    size=(draw_count,),
                    dtype=torch.int64,
                    generator=generator,
                )
                chunks.append(pool[offsets])
            batch = torch.cat(chunks)
            permutation = torch.randperm(current_size, generator=generator)
            yield batch[permutation].tolist()
            remaining -= current_size

    def __len__(self) -> int:
        return math.ceil(self.epoch_draws / self.batch_size)
