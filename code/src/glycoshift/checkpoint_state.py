"""Deterministic tensor-state identity for selected CF5 checkpoints."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

import torch

from .mero_reference import canonical_json_bytes
from .mero_weighted_reference import CHECKPOINT_STATE_HASH_METHOD


def checkpoint_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Hash tensor names, dtype, shape, and exact contiguous CPU bytes."""

    if not isinstance(state, Mapping) or not state:
        raise ValueError("checkpoint state must be a non-empty tensor mapping")
    if any(not isinstance(name, str) or not name for name in state):
        raise TypeError("checkpoint state names must be non-empty strings")
    digest = hashlib.sha256()
    header = canonical_json_bytes({"method": CHECKPOINT_STATE_HASH_METHOD})
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    for name in sorted(state, key=str.encode):
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError("checkpoint state values must be tensors")
        tensor = value.detach().cpu().contiguous()
        metadata = canonical_json_bytes(
            {
                "name": name,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
            }
        )
        raw = tensor.view(torch.uint8).numpy().tobytes(order="C")
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()
