from __future__ import annotations

import numpy as np
import pytest

from glycoshift.sampling import (
    AllSourceBatchSampler,
    all_source_allocation,
    effective_update_sizes,
)

SOURCE_ORDER = tuple(f"source-{index:02d}" for index in range(11))


def test_all_source_allocation_rotates_extras_exactly() -> None:
    first, cursor = all_source_allocation(
        batch_size=256,
        source_count=11,
        extras_cursor=0,
    )
    assert first[:3] == (24, 24, 24)
    assert first[3:] == (23,) * 8
    assert cursor == 3

    second, cursor = all_source_allocation(
        batch_size=256,
        source_count=11,
        extras_cursor=cursor,
    )
    assert second[3:6] == (24, 24, 24)
    assert sum(second) == 256
    assert cursor == 6

    final, cursor = all_source_allocation(
        batch_size=245,
        source_count=11,
        extras_cursor=0,
    )
    assert final[:3] == (23, 23, 23)
    assert final[3:] == (22,) * 8
    assert cursor == 3


def _small_protocol_sampler(*, epoch: int = 0) -> AllSourceBatchSampler:
    source_indices = np.repeat(np.arange(11, dtype=np.int64), 3)
    return AllSourceBatchSampler(
        source_indices,
        source_order=SOURCE_ORDER,
        epoch_draws=5 * 256 + 245,
        batch_size=256,
        seed=42,
        epoch=epoch,
    )


def test_all_source_sampler_is_deterministic_and_covers_every_source() -> None:
    source_indices = np.repeat(np.arange(11, dtype=np.int64), 3)
    first = list(_small_protocol_sampler())
    second = list(_small_protocol_sampler())
    assert first == second
    assert len(first) == 6
    assert [len(batch) for batch in first] == [256] * 5 + [245]

    expected_extra_starts = [0, 3, 6, 9, 1, 4]
    for batch, extra_start in zip(first, expected_extra_starts, strict=True):
        counts = np.bincount(source_indices[np.asarray(batch)], minlength=11)
        assert (counts > 0).all()
        base, remainder = divmod(len(batch), 11)
        expected = np.full(11, base)
        for offset in range(remainder):
            expected[(extra_start + offset) % 11] += 1
        np.testing.assert_array_equal(counts, expected)

    assert list(_small_protocol_sampler(epoch=1)) != first


def test_protocol_effective_update_sizes_end_in_exactly_501_records() -> None:
    sizes = effective_update_sizes(
        epoch_draws=27_526_645,
        physical_batch_size=256,
        effective_batch_size=1024,
    )
    assert len(sizes) == 26_882
    assert set(sizes[:-1]) == {1024}
    assert sizes[-1] == 501
    assert sum(sizes) == 27_526_645


def test_all_source_sampler_rejects_a_source_omitting_tail() -> None:
    with pytest.raises(ValueError, match="final physical batch"):
        AllSourceBatchSampler(
            np.arange(11, dtype=np.int64),
            source_order=SOURCE_ORDER,
            epoch_draws=258,
            batch_size=256,
        )
