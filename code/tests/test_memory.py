import pandas as pd
import pytest
import torch

from glycoshift.memory import (
    ContiguousStrictPastMemoryBank,
    StrictPastMemoryBank,
    memory_anchor_indices,
    past_candidates,
    validate_past_only,
)


def test_memory_rejects_current_or_future_records() -> None:
    query = pd.Timestamp("2026-01-02T00:00:00Z")
    validate_past_only(query, [query - pd.Timedelta(minutes=5)])
    with pytest.raises(ValueError, match="future"):
        validate_past_only(query, [query])


def test_past_candidates_are_strictly_earlier() -> None:
    query = pd.Timestamp("2026-01-02T00:00:00Z")
    frame = pd.DataFrame(
        {
            "source_id": ["s", "s", "s"],
            "participant_id": ["p", "p", "p"],
            "context_end": [
                query - pd.Timedelta(hours=2),
                query - pd.Timedelta(hours=1),
                query,
            ],
        }
    )
    selected = past_candidates(
        frame,
        source_id="s",
        participant_id="p",
        query_start=query,
        k=8,
    )
    assert len(selected) == 2
    assert (pd.to_datetime(selected["context_end"], utc=True) < query).all()


def test_tensor_memory_bank_never_retrieves_overlapping_context() -> None:
    bank = StrictPastMemoryBank(k=2, stride_minutes=30)
    metadata = [
        {
            "source_id": "s",
            "participant_id": "p",
            "context_start": pd.Timestamp("2026-01-01T00:00:00Z"),
            "context_end": pd.Timestamp("2026-01-01T01:00:00Z"),
        },
        {
            "source_id": "s",
            "participant_id": "p",
            "context_start": pd.Timestamp("2026-01-01T02:00:00Z"),
            "context_end": pd.Timestamp("2026-01-01T03:00:00Z"),
        },
    ]
    bank.build(embeddings=torch.eye(2), metadata=metadata)
    memory, padding = bank.retrieve(
        query_embeddings=torch.tensor([[1.0, 0.0]]),
        query_metadata=[
            {
                "source_id": "s",
                "participant_id": "p",
                "context_start": pd.Timestamp("2026-01-01T02:00:00Z"),
                "context_end": pd.Timestamp("2026-01-01T03:00:00Z"),
            }
        ],
    )
    assert not padding[0, 0]
    assert padding[0, 1]
    torch.testing.assert_close(memory[0, 0], torch.tensor([1.0, 0.0]))


def test_memory_anchor_indices_keep_first_row_in_each_bucket() -> None:
    windows = pd.DataFrame(
        {
            "source_id": ["s", "s", "s", "s", "other"],
            "participant_id": ["p", "p", "p", "p", "p"],
            "context_end": pd.to_datetime(
                [
                    "2026-01-01T00:05:00Z",
                    "2026-01-01T00:25:00Z",
                    "2026-01-01T00:30:00Z",
                    "2026-01-01T00:35:00Z",
                    "2026-01-01T00:25:00Z",
                ],
                utc=True,
            ),
        }
    )
    assert memory_anchor_indices(
        windows,
        stride_minutes=30,
    ).tolist() == [0, 2, 4]


def test_contiguous_memory_matches_reference_oracle() -> None:
    torch.manual_seed(31)
    starts = pd.to_datetime(
        [
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:05:00Z",
            "2026-01-01T00:30:00Z",
            "2026-01-01T01:00:00Z",
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:30:00Z",
        ],
        utc=True,
    )
    ends = starts + pd.Timedelta(minutes=30)
    metadata = [
        {
            "source_id": source,
            "participant_id": participant,
            "context_start": start,
            "context_end": end,
        }
        for source, participant, start, end in zip(  # noqa: B905
            ["s", "s", "s", "s", "s", "s"],
            ["p", "p", "p", "p", "q", "q"],
            starts,
            ends,
        )
    ]
    embeddings = torch.randn(len(metadata), 16)
    oracle = StrictPastMemoryBank(k=3, stride_minutes=30)
    oracle.build(embeddings=embeddings, metadata=metadata)
    contiguous = ContiguousStrictPastMemoryBank(
        k=3,
        stride_minutes=30,
    )
    contiguous.build(embeddings=embeddings, metadata=metadata)

    query_metadata = [
        {
            "source_id": "s",
            "participant_id": "p",
            "context_start": pd.Timestamp("2026-01-01T00:30:00Z"),
        },
        {
            "source_id": "s",
            "participant_id": "q",
            "context_start": pd.Timestamp("2026-01-01T01:30:00Z"),
        },
        {
            "source_id": "s",
            "participant_id": "p",
            "context_start": pd.Timestamp("2026-01-01T02:00:00Z"),
        },
        {
            "source_id": "missing",
            "participant_id": "p",
            "context_start": pd.Timestamp("2026-01-01T02:00:00Z"),
        },
    ]
    queries = torch.randn(len(query_metadata), embeddings.shape[1])
    expected_memory, expected_padding = oracle.retrieve(
        query_embeddings=queries,
        query_metadata=query_metadata,
    )
    observed_memory, observed_padding = contiguous.retrieve(
        query_embeddings=queries,
        query_metadata=query_metadata,
    )

    assert torch.equal(observed_padding, expected_padding)
    torch.testing.assert_close(
        observed_memory,
        expected_memory,
        atol=1e-6,
        rtol=1e-6,
    )
    assert contiguous.record_count == sum(
        len(records) for records in oracle._records.values()
    )

    anchors = memory_anchor_indices(
        pd.DataFrame(metadata),
        stride_minutes=30,
    )
    anchor_only = ContiguousStrictPastMemoryBank(
        k=3,
        stride_minutes=30,
    )
    anchor_only.build_from_windows(
        embeddings=embeddings[anchors],
        windows=pd.DataFrame(metadata).iloc[anchors].reset_index(drop=True),
    )
    anchor_memory, anchor_padding = anchor_only.retrieve(
        query_embeddings=queries,
        query_metadata=query_metadata,
    )
    assert torch.equal(anchor_padding, expected_padding)
    torch.testing.assert_close(
        anchor_memory,
        expected_memory,
        atol=1e-6,
        rtol=1e-6,
    )


def test_contiguous_memory_preserves_tied_cosine_topk_order() -> None:
    metadata = [
        {
            "source_id": "s",
            "participant_id": "p",
            "context_start": pd.Timestamp("2026-01-01T00:00:00Z"),
            "context_end": pd.Timestamp("2026-01-01T00:30:00Z"),
        },
        {
            "source_id": "s",
            "participant_id": "p",
            "context_start": pd.Timestamp("2026-01-01T00:30:00Z"),
            "context_end": pd.Timestamp("2026-01-01T01:00:00Z"),
        },
    ]
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    query_embeddings = torch.tensor([[1.0, 1.0]])
    query_metadata = [
        {
            "source_id": "s",
            "participant_id": "p",
            "context_start": pd.Timestamp("2026-01-01T02:00:00Z"),
        }
    ]
    oracle = StrictPastMemoryBank(k=2, stride_minutes=30)
    oracle.build(embeddings=embeddings, metadata=metadata)
    contiguous = ContiguousStrictPastMemoryBank(
        k=2,
        stride_minutes=30,
    )
    contiguous.build(embeddings=embeddings, metadata=metadata)

    expected_memory, expected_padding = oracle.retrieve(
        query_embeddings=query_embeddings,
        query_metadata=query_metadata,
    )
    observed_memory, observed_padding = contiguous.retrieve(
        query_embeddings=query_embeddings,
        query_metadata=query_metadata,
    )

    assert torch.equal(observed_padding, expected_padding)
    torch.testing.assert_close(observed_memory, expected_memory)


def test_bank_is_identical_whether_fed_all_windows_or_anchor_rows() -> None:
    """Feeding only the anchor embeddings must build the same bank.

    This is a property of the BANK BUILDER only, and it is deliberately not a
    licence to feed it anchor-only embeddings in production.

    An anchor-only encoder was evaluated and rejected in cde7460: on real
    shards it moved representations by up to 0.00452 (Shanghai T1DM) and
    0.00399 (DCLP5) against a predeclared 5e-5 gate, because a per-sample
    transformer under bfloat16 autocast is NOT batch-composition invariant --
    different batch shapes select different cuBLAS kernels and reduction
    orders. Full-window encoding is preserved and
    test_memory_bank_preserves_full_window_encoding guards it.

    What this test shows is narrower and still worth having: given identical
    embedding values, the two accepted input shapes build the same bank, so
    the builder itself introduces no asymmetry.
    """

    rows = 60
    windows = pd.DataFrame(
        {
            "source_id": ["s"] * rows,
            "participant_id": (["p"] * (rows // 2)) + (["q"] * (rows // 2)),
            "context_end": pd.to_datetime(
                [f"2026-01-01T{h:02d}:{m:02d}:00Z"
                 for h in range(rows // 12)
                 for m in (0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55)],
                utc=True,
            ),
        }
    )
    generator = torch.Generator().manual_seed(20260801)
    full = torch.randn(rows, 16, generator=generator)

    anchors = memory_anchor_indices(windows, stride_minutes=30)
    assert 0 < len(anchors) < rows, "the stride must actually reduce the rows"

    from_all = ContiguousStrictPastMemoryBank(k=3, stride_minutes=30)
    from_all.build_from_windows(embeddings=full, windows=windows)

    from_anchors = ContiguousStrictPastMemoryBank(k=3, stride_minutes=30)
    from_anchors.build_from_windows(embeddings=full[anchors], windows=windows)

    assert from_all.record_count == from_anchors.record_count == len(anchors)
    assert from_all.storage_bytes == from_anchors.storage_bytes
    for name in ("_embeddings", "_normalized_embeddings", "_end_times_ns"):
        left = getattr(from_all, name)
        right = getattr(from_anchors, name)
        assert left is not None and right is not None
        assert torch.equal(left, right), f"{name} differs between the two paths"
    assert from_all._slices.keys() == from_anchors._slices.keys()


def test_anchor_stride_keeps_one_window_in_six() -> None:
    """Quantifies how much of each encode pass the bank discards.

    Five-minute windows with a thirty-minute stride keep one row in six, so
    five sixths of every full-window encode is thrown away. That waste is
    real, but it cannot be removed by encoding only the anchors: see
    test_bank_is_identical_whether_fed_all_windows_or_anchor_rows for why.
    """

    minutes = list(range(0, 60 * 6, 5))
    windows = pd.DataFrame(
        {
            "source_id": ["s"] * len(minutes),
            "participant_id": ["p"] * len(minutes),
            "context_end": pd.to_datetime(
                [
                    pd.Timestamp("2026-01-01T00:00:00Z")
                    + pd.Timedelta(minutes=m)
                    for m in minutes
                ],
                utc=True,
            ),
        }
    )
    anchors = memory_anchor_indices(windows, stride_minutes=30)
    assert len(anchors) * 6 == len(windows)
