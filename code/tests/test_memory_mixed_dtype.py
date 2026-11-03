"""Retrieval must survive a query whose dtype differs from the bank's.

C3 gave the LSTM baseline the retrieval pathway for the first time and died on
epoch 3, the first epoch past `warmup_memory_epochs` and therefore the first time
retrieval actually ran:

    RuntimeError: expected mat1 and mat2 to have the same dtype,
                  but got: c10::BFloat16 != float

The bank is built by encoding the training set outside autocast, so it holds
float32, while the training loop's representation is produced inside
`torch.autocast(bfloat16)`. The transformer path never hit this because autocast
returns float32 from `layer_norm`; an LSTM returns bfloat16.
"""

from __future__ import annotations

import pandas as pd
import pytest
import torch

from glycoshift.memory import ContiguousStrictPastMemoryBank


def _bank(dtype: torch.dtype) -> tuple[ContiguousStrictPastMemoryBank, list[dict]]:
    starts = pd.to_datetime(
        [
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:05:00Z",
            "2026-01-01T00:30:00Z",
            "2026-01-01T01:00:00Z",
        ],
        utc=True,
    )
    ends = starts + pd.Timedelta(minutes=30)
    windows = pd.DataFrame(
        {
            "source_id": ["s"] * 4,
            "participant_id": ["p"] * 4,
            "context_start": starts,
            "context_end": ends,
        }
    )
    torch.manual_seed(7)
    embeddings = torch.randn(4, 16, dtype=dtype)
    bank = ContiguousStrictPastMemoryBank(k=2, stride_minutes=30, dtype=dtype)
    bank.build_from_windows(embeddings=embeddings, windows=windows)
    query_metadata = [
        {
            "source_id": "s",
            "participant_id": "p",
            "context_start": starts[3],
            "context_end": ends[3],
        }
    ]
    return bank, query_metadata


@pytest.mark.parametrize(
    ("bank_dtype", "query_dtype"),
    [
        (torch.float32, torch.float32),
        (torch.float32, torch.bfloat16),
        (torch.bfloat16, torch.float32),
    ],
)
def test_retrieval_survives_a_dtype_mismatch(
    bank_dtype: torch.dtype, query_dtype: torch.dtype
) -> None:
    bank, query_metadata = _bank(bank_dtype)
    torch.manual_seed(11)
    query = torch.randn(1, 16, dtype=query_dtype)
    memory, padding = bank.retrieve(
        query_embeddings=query, query_metadata=query_metadata
    )
    assert torch.isfinite(memory.float()).all()
    assert padding.shape[0] == 1


def test_a_float32_query_is_bitwise_what_it_was() -> None:
    """The fix must not move any existing arm.

    Every completed arm ran the float32-to-float32 path. If normalising the
    dtype changes that path at all, the fix is not inert and the equivalence
    proven for the contiguous bank no longer holds.
    """

    bank, query_metadata = _bank(torch.float32)
    torch.manual_seed(11)
    query = torch.randn(1, 16, dtype=torch.float32)
    first, _ = bank.retrieve(query_embeddings=query, query_metadata=query_metadata)
    second, _ = bank.retrieve(query_embeddings=query, query_metadata=query_metadata)
    assert torch.equal(first, second)
    assert first.dtype == torch.float32
