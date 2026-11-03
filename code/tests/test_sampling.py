import numpy as np
import pandas as pd
import pytest
import torch

from glycoshift.deep_training import _loader
from glycoshift.sampling import SourceBalancedSampler


class _StubWindowDataset:
    """Minimal stand-in exposing only what ``_loader`` reads from a dataset."""

    def __init__(self, *, records: int, sources: int) -> None:
        assignment = np.arange(records, dtype=np.int64) % sources
        self.windows = pd.DataFrame(
            {"source_id": [f"src_{index:02d}" for index in assignment]}
        )
        self.source_to_index = {
            f"src_{index:02d}": index for index in range(sources)
        }

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> int:
        return index


def _training_loader(*, sampler_seed: int, records: int = 70_000):
    # More than one 65,536-record sampler chunk, so multiple chunk draws occur.
    return _loader(
        _StubWindowDataset(records=records, sources=11),
        batch_size=256,
        num_workers=0,
        source_balanced=True,
        sampler_seed=sampler_seed,
    )


def test_source_balanced_sampler_matches_weighted_probability_oracle() -> None:
    source_indices = np.asarray([0, 0, 1, 1, 1, 1, 2], dtype=np.int64)
    counts = np.bincount(source_indices, minlength=3)
    weighted = np.asarray(
        [1.0 / counts[index] for index in source_indices],
        dtype=np.float64,
    )
    weighted /= weighted.sum()
    hierarchical = np.asarray(
        [1.0 / (3 * counts[index]) for index in source_indices],
        dtype=np.float64,
    )

    np.testing.assert_allclose(hierarchical, weighted, rtol=0, atol=0)


def test_source_balanced_sampler_is_deterministic_and_balanced() -> None:
    source_indices = np.asarray(
        [0, 0, 1, 1, 1, 1, 2],
        dtype=np.int64,
    )
    first_generator = torch.Generator().manual_seed(20260727)
    second_generator = torch.Generator().manual_seed(20260727)
    first = list(
        SourceBalancedSampler(
            source_indices,
            num_sources=3,
            num_samples=120_000,
            generator=first_generator,
            chunk_size=4096,
        )
    )
    second = list(
        SourceBalancedSampler(
            source_indices,
            num_sources=3,
            num_samples=120_000,
            generator=second_generator,
            chunk_size=4096,
        )
    )

    assert first == second
    sampled_sources = source_indices[np.asarray(first)]
    source_fractions = np.bincount(sampled_sources, minlength=3) / len(first)
    np.testing.assert_allclose(
        source_fractions,
        np.full(3, 1 / 3),
        atol=0.01,
    )


def test_training_loader_sampler_does_not_consume_global_rng() -> None:
    """This decoupling is what makes ``num_workers`` a throughput-only knob.

    DataLoader eagerly prefetches ``2 * num_workers`` batches of indices, so a
    sampler drawing from the global RNG crosses its 65,536-record chunk
    boundaries at a different phase for every worker count, and the training
    loop's feature-group dropout -- which draws from the same global RNG --
    then sees a different stream. Measured 2026-08-13 on the pre-fix code: the
    index and dropout streams both diverged between 0, 2, 4 and 8 workers.
    Asserting on the global RNG state catches that at its source; asserting on
    the first few batches of a small dataset does not, because a run shorter
    than one chunk draws all of its sampler randomness up front.
    """
    loader = _training_loader(sampler_seed=17)

    before = torch.random.get_rng_state()
    drawn = list(loader.sampler)
    after = torch.random.get_rng_state()

    assert torch.equal(before, after)
    assert len(drawn) == 70_000


def test_training_loader_sampler_stream_depends_only_on_its_seed() -> None:
    first = list(_training_loader(sampler_seed=17).sampler)
    repeat = list(_training_loader(sampler_seed=17).sampler)
    other = list(_training_loader(sampler_seed=23).sampler)

    assert first == repeat
    assert first != other


def test_training_loader_requires_an_explicit_sampler_seed() -> None:
    with pytest.raises(ValueError, match="sampler seed"):
        _loader(
            _StubWindowDataset(records=512, sources=11),
            batch_size=32,
            num_workers=0,
            source_balanced=True,
        )


def test_source_balanced_sampler_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="nonempty vector"):
        SourceBalancedSampler([], num_sources=1)
    with pytest.raises(ValueError, match="every source"):
        SourceBalancedSampler([0, 0], num_sources=2)
    with pytest.raises(ValueError, match="outside"):
        SourceBalancedSampler([0, 2], num_sources=2)
