"""The fast sample assembly must be bit-identical to the one it replaces.

`FastWindowSequenceDataset` exists only to remove Python overhead from
`__getitem__`; it is not allowed to change a single value. These tests compare
it against the original implementation directly -- not against a second
reimplementation of the expected output -- so that any drift in either shows up
as a failure rather than as two implementations agreeing on the wrong thing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from glycoshift.config import load_config
from glycoshift.fast_sequences import (
    FastWindowSequenceDataset,
    _last_observed_index,
    _time_since_from_last_observed,
)
from glycoshift.sequences import (
    SequenceScaler,
    WindowSequenceDataset,
    _time_since_observation,
)
from glycoshift.synthetic import make_synthetic_cgm
from glycoshift.windows import build_windows

CONFIG = Path(__file__).parents[1] / "configs" / "smoke.yaml"


def _cohort(*, participants: int = 3, days: int = 2, context_hours: float = 0.5):
    config = load_config(CONFIG)
    config["data"]["participants_per_source"] = participants
    config["data"]["days_per_participant"] = days
    config["protocol"]["context_hours"] = context_hours
    frame = make_synthetic_cgm(config)
    blocks = []
    for _key, group in frame.groupby(["source_id", "participant_id"], sort=True):
        block = group.sort_values("timestamp").reset_index(drop=True)
        blocks.append(build_windows(block, config))
    windows = pd.concat(blocks, ignore_index=True)
    keys = set(
        zip(  # noqa: B905
            frame["source_id"].astype(str), frame["participant_id"].astype(str)
        )
    )
    scaler = SequenceScaler.fit(frame, keys)
    samples = int(
        context_hours * 60 / int(config["protocol"]["sample_minutes"])
    )
    return frame, windows, scaler, samples


def _pair(frame, windows, scaler, samples):
    slow = WindowSequenceDataset(
        frame, windows, scaler=scaler, context_samples=samples
    )
    fast = FastWindowSequenceDataset(
        frame, windows, scaler=scaler, context_samples=samples
    )
    return slow, fast


def test_time_since_identity_matches_the_original_loop() -> None:
    """The vectorised reconstruction equals the element-wise loop exactly.

    This is the only channel that is not pointwise in time: the original seeds
    its counter with the window width, so the value depends on the window a
    position is read in. The cases below are the ones that could diverge --
    nothing observed, everything observed, an observation only before the
    window, and one only at the final position.
    """

    rng = np.random.default_rng(20260817)
    width = 12
    cases = {
        "none_observed": np.zeros(64, dtype=bool),
        "all_observed": np.ones(64, dtype=bool),
        "sparse": rng.random(64) < 0.15,
        "dense": rng.random(64) < 0.85,
        "only_before_window": np.array(
            [i == 3 for i in range(64)], dtype=bool
        ),
        "only_at_window_end": np.array(
            [i == 40 for i in range(64)], dtype=bool
        ),
    }
    for name, observed in cases.items():
        last = _last_observed_index(observed)
        for start in range(0, len(observed) - width + 1):
            expected = _time_since_observation(observed[start : start + width])
            actual = _time_since_from_last_observed(last, start, width)
            assert actual.dtype == expected.dtype, name
            np.testing.assert_array_equal(actual, expected, err_msg=f"{name}@{start}")


def test_every_sample_is_bit_identical() -> None:
    frame, windows, scaler, samples = _cohort()
    slow, fast = _pair(frame, windows, scaler, samples)
    assert len(slow) == len(fast) and len(slow) > 100

    for index in range(len(slow)):
        expected = slow[index]
        actual = fast[index]
        assert set(expected) == set(actual), index
        for key, value in expected.items():
            other = actual[key]
            assert other.dtype == value.dtype, (index, key)
            assert other.shape == value.shape, (index, key)
            assert torch.equal(other, value), (index, key)


def test_bit_identical_with_action_gaps_and_missing_values() -> None:
    """Gaps are where a pointwise precompute would be most likely to diverge."""

    frame, windows, scaler, samples = _cohort(participants=2, days=2)
    frame = frame.copy()
    rng = np.random.default_rng(11)
    for field in ("insulin_units", "carbs_g", "activity_value"):
        if f"{field}_observed" not in frame:
            continue
        drop = rng.random(len(frame)) < 0.4
        frame.loc[drop, f"{field}_observed"] = False
        frame.loc[drop, field] = np.nan
    scaler = SequenceScaler.fit(
        frame,
        set(
            zip(  # noqa: B905
                frame["source_id"].astype(str),
                frame["participant_id"].astype(str),
            )
        ),
    )
    slow, fast = _pair(frame, windows, scaler, samples)
    for index in range(0, len(slow), 3):
        expected, actual = slow[index], fast[index]
        for key, value in expected.items():
            assert torch.equal(actual[key], value), (index, key)


def test_fast_dataset_refuses_lower_censored_context_like_the_original() -> None:
    """A refusal is part of the contract, so it must survive the rewrite."""

    frame, windows, scaler, samples = _cohort(participants=1, days=2)
    frame = frame.copy()
    # Spread the flag so it certainly intersects some window's context; marking
    # a prefix of the concatenated frame does not, because the earliest windows
    # of the first participant need not begin at its first reading.
    rng = np.random.default_rng(7)
    frame["cgm_lower_censored"] = rng.random(len(frame)) < 0.2
    slow, fast = _pair(frame, windows, scaler, samples)

    refused_slow = refused_fast = 0
    for index in range(min(len(slow), 40)):
        try:
            slow[index]
        except ValueError:
            refused_slow += 1
        try:
            fast[index]
        except ValueError:
            refused_fast += 1
    assert refused_slow == refused_fast
    assert refused_slow > 0, "fixture failed to exercise the refusal"


def test_fast_path_is_materially_faster() -> None:
    """Not a benchmark gate -- a floor that catches an accidental regression."""

    import time

    frame, windows, scaler, samples = _cohort(participants=3, days=2)
    slow, fast = _pair(frame, windows, scaler, samples)
    indices = list(range(min(len(slow), 300)))

    start = time.perf_counter()
    for index in indices:
        slow[index]
    slow_seconds = time.perf_counter() - start

    start = time.perf_counter()
    for index in indices:
        fast[index]
    fast_seconds = time.perf_counter() - start

    assert fast_seconds < slow_seconds, (slow_seconds, fast_seconds)
    print(
        f"\nper-sample: slow {slow_seconds / len(indices) * 1e6:.1f} us, "
        f"fast {fast_seconds / len(indices) * 1e6:.1f} us, "
        f"speedup {slow_seconds / max(fast_seconds, 1e-9):.1f}x"
    )


def test_assembly_selection_is_explicit_in_both_directions() -> None:
    """The array-backed path is the default, and the original stays selectable.

    The default was flipped only after the real-shard audit reported
    `bit_identical: true` over 20,000 calibration windows with every maximum
    absolute difference zero. What must not regress is the ability to pin the
    original explicitly: that is the lever for suspecting this path later, and a
    default is not a replacement for the reference implementation.
    """

    from glycoshift.deep_training import _window_dataset_class
    from glycoshift.sequences import WindowSequenceDataset

    assert _window_dataset_class({}) is FastWindowSequenceDataset
    assert _window_dataset_class({"training": {}}) is FastWindowSequenceDataset
    assert (
        _window_dataset_class({"training": {"fast_sample_assembly": True}})
        is FastWindowSequenceDataset
    )
    assert (
        _window_dataset_class({"training": {"fast_sample_assembly": False}})
        is WindowSequenceDataset
    ), "the reference implementation must remain reachable by configuration"


def test_participant_blocks_are_built_once_per_store_and_scaling() -> None:
    """Four datasets over one store must not hold four copies of the features.

    A training run builds train, selection, calibration and stream-adapter
    datasets over a single participant store. Per-instance precompute would keep
    four copies of the same 46.6M-row feature matrix, and the per-lane RAM
    ceiling that gates a launch is derived from that peak. The cache must also be
    keyed by the scaling, or a differently fitted scaler would silently reuse
    another's features.
    """

    from glycoshift.fast_sequences import _shared_participant_blocks
    from glycoshift.sequences import ParticipantSequenceStore

    frame, windows, scaler, samples = _cohort(participants=2, days=2)
    store = ParticipantSequenceStore.from_frame(frame)

    first = FastWindowSequenceDataset(
        frame, windows, scaler=scaler, context_samples=samples,
        participant_store=store, copy_windows=False,
    )
    second = FastWindowSequenceDataset(
        frame, windows, scaler=scaler, context_samples=samples,
        participant_store=store, copy_windows=False,
    )
    assert len(first._blocks) == len(second._blocks) > 0
    for left, right in zip(first._blocks, second._blocks):  # noqa: B905
        assert left is right, "blocks were rebuilt instead of shared"

    # A different scaling must not reuse the cached features.
    rescaled = SequenceScaler.fit(
        frame.assign(cgm_mg_dl=frame["cgm_mg_dl"] * 2.0),
        set(
            zip(  # noqa: B905
                frame["source_id"].astype(str),
                frame["participant_id"].astype(str),
            )
        ),
    )
    keys = sorted(store.participants)
    other = _shared_participant_blocks(store, rescaled, keys)
    assert other[0] is not first._blocks[0], "cache ignored the scaling"

    # And the values still match the original implementation.
    slow = WindowSequenceDataset(
        frame, windows, scaler=scaler, context_samples=samples,
        participant_store=store, copy_windows=False,
    )
    for index in range(0, len(slow), 5):
        for key, value in slow[index].items():
            assert torch.equal(second[index][key], value), (index, key)
