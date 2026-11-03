from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from .sequences import REGRESSION_HORIZONS, SEQUENCE_FEATURES, TASKS


def persistence_forecast(
    context_glucose: np.ndarray,
    *,
    horizon_samples: int,
) -> np.ndarray:
    values = np.asarray(context_glucose, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("persistence forecast requires a non-empty 1D context")
    return np.repeat(values[-1], horizon_samples)


def linear_extrapolation_forecast(
    context_glucose: np.ndarray,
    *,
    horizon_samples: int,
    fit_samples: int = 6,
) -> np.ndarray:
    values = np.asarray(context_glucose, dtype=float)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("linear extrapolation requires at least two samples")
    fit = values[-min(fit_samples, len(values)) :]
    slope, intercept = np.polyfit(np.arange(len(fit)), fit, 1)
    future = np.arange(len(fit), len(fit) + horizon_samples)
    return intercept + slope * future


@dataclass(frozen=True)
class TemporalBaselineConfig:
    d_model: int
    memory_k: int = 0


class ParticipantMemoryFusion(nn.Module):
    """Post-encoder participant-memory fusion for a temporal baseline.

    This mirrors ``GlycoShift._fuse_memory`` rather than refactoring it.
    Refactoring that class would reorder its parameter construction and change
    its initialisation RNG stream, silently invalidating every GlycoShift number
    already recorded, so the retained implementation stays untouched and
    ``test_baseline_memory.py`` asserts the two agree from matched weights.
    """

    def __init__(
        self,
        *,
        d_model: int,
        memory_k: int,
        n_heads: int,
        dropout: float,
        retrieval: bool = True,
    ) -> None:
        super().__init__()
        if memory_k <= 0:
            raise ValueError("memory_k must be positive to fuse memory")
        # retrieval=False is the parameter-matched control: identical modules,
        # identical parameter count and identical compute, differing only in
        # whether participant-specific memory is allowed to reach the fusion.
        # A win over it is attributable to retrieval; a win over a plain
        # baseline is attributable only to added capacity.
        self.retrieval = bool(retrieval)
        self.population_memory = nn.Parameter(torch.zeros(memory_k, d_model))
        nn.init.normal_(self.population_memory, std=0.02)
        self.memory_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.memory_gate = nn.Linear(d_model * 2, d_model)
        self.instrument = False
        self.last_gate: torch.Tensor | None = None
        self.last_attention_weights: torch.Tensor | None = None

    def forward(
        self,
        representation: torch.Tensor,
        memory_embeddings: torch.Tensor | None,
        memory_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size = representation.shape[0]
        if not self.retrieval:
            memory_embeddings = None
            memory_padding_mask = None
        if memory_embeddings is None:
            memory_embeddings = self.population_memory.unsqueeze(0).expand(
                batch_size,
                -1,
                -1,
            )
            memory_padding_mask = None
        elif memory_padding_mask is not None:
            population = self.population_memory.unsqueeze(0).expand(
                batch_size,
                -1,
                -1,
            )
            memory_embeddings = torch.where(
                memory_padding_mask.unsqueeze(-1),
                population,
                memory_embeddings,
            )
            memory_padding_mask = None
        query = representation.unsqueeze(1)
        # need_weights=False matches GlycoShift and keeps the fast attention
        # path; requesting weights here would change the kernel and perturb the
        # low bits of a value that feeds training.
        attended, _ = self.memory_attention(
            query,
            memory_embeddings,
            memory_embeddings,
            key_padding_mask=memory_padding_mask,
            need_weights=False,
        )
        attended = attended.squeeze(1)
        gate = torch.sigmoid(
            self.memory_gate(torch.cat((representation, attended), dim=-1))
        )
        self.last_gate = gate.detach()
        if self.instrument:
            with torch.no_grad():
                _, weights = self.memory_attention(
                    query,
                    memory_embeddings,
                    memory_embeddings,
                    key_padding_mask=memory_padding_mask,
                    need_weights=True,
                )
            self.last_attention_weights = weights.detach()
        return representation + gate * attended


class _TemporalHeads(nn.Module):
    supports_memory = False

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.config = TemporalBaselineConfig(d_model=d_model)
        self.classification_heads = nn.ModuleDict(
            {task: nn.Linear(d_model, 1) for task in TASKS}
        )
        self.regression_heads = nn.ModuleDict(
            {
                str(horizon): nn.Linear(d_model, 1)
                for horizon in REGRESSION_HORIZONS
            }
        )

    def attach_memory(
        self,
        *,
        memory_k: int,
        n_heads: int = 4,
        dropout: float = 0.1,
        retrieval: bool = True,
    ) -> None:
        """Add participant memory after the model is otherwise constructed.

        Attaching last is what keeps every shared module bitwise identical to
        the same baseline built without memory: the memory parameters are drawn
        from the RNG only after the encoder and heads already have their
        weights. A baseline built with ``memory_k = 0`` never reaches this
        method, so its initialisation stream is untouched.
        """
        self.memory = ParticipantMemoryFusion(
            d_model=self.config.d_model,
            memory_k=memory_k,
            n_heads=n_heads,
            dropout=dropout,
            retrieval=retrieval,
        )
        self.config = TemporalBaselineConfig(
            d_model=self.config.d_model,
            memory_k=memory_k,
        )
        self.supports_memory = True

    def forward_from_representation(
        self,
        representation: torch.Tensor,
        *,
        memory_embeddings: torch.Tensor | None = None,
        memory_padding_mask: torch.Tensor | None = None,
        use_memory: bool = True,
        **_: Any,
    ) -> dict[str, Any]:
        memory = getattr(self, "memory", None)
        fused = (
            memory(representation, memory_embeddings, memory_padding_mask)
            if memory is not None and use_memory
            else representation
        )
        return {
            "logits": {
                task: head(fused).squeeze(-1)
                for task, head in self.classification_heads.items()
            },
            "regression": {
                horizon: head(fused).squeeze(-1)
                for horizon, head in self.regression_heads.items()
            },
            "representation": representation,
            "fused_representation": fused,
        }

    def forward(self, sequence: torch.Tensor, **kwargs: Any) -> dict[str, Any]:
        representation = self.encode_sequence(sequence)[:, -1]
        return self.forward_from_representation(representation, **kwargs)


class LSTMBaseline(_TemporalHeads):
    def __init__(self, *, d_model: int = 128, layers: int = 2) -> None:
        super().__init__(d_model)
        self.input_projection = nn.Linear(len(SEQUENCE_FEATURES), d_model)
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=layers,
            dropout=0.1 if layers > 1 else 0.0,
            batch_first=True,
        )

    def encode_sequence(self, sequence: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.lstm(self.input_projection(sequence))
        return encoded


class _CausalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        kernel_size = 3
        self.left_padding = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        padded = functional_pad_left(values, self.left_padding)
        convolved = self.conv(padded)
        output = convolved.transpose(1, 2)
        output = torch.nn.functional.gelu(self.norm(output))
        output = self.dropout(output).transpose(1, 2)
        return values + output


def functional_pad_left(values: torch.Tensor, amount: int) -> torch.Tensor:
    return torch.nn.functional.pad(values, (amount, 0))


class TCNBaseline(_TemporalHeads):
    def __init__(self, *, d_model: int = 128, layers: int = 5) -> None:
        super().__init__(d_model)
        self.input_projection = nn.Conv1d(
            len(SEQUENCE_FEATURES),
            d_model,
            kernel_size=1,
        )
        self.blocks = nn.ModuleList(
            [_CausalBlock(d_model, 2**index, 0.1) for index in range(layers)]
        )
        self.output_norm = nn.LayerNorm(d_model)

    def encode_sequence(self, sequence: torch.Tensor) -> torch.Tensor:
        encoded = self.input_projection(sequence.transpose(1, 2))
        for block in self.blocks:
            encoded = block(encoded)
        return self.output_norm(encoded.transpose(1, 2))


class PatchTSTBaseline(_TemporalHeads):
    def __init__(
        self,
        *,
        d_model: int = 128,
        patch_size: int = 6,
        layers: int = 4,
    ) -> None:
        super().__init__(d_model)
        self.patch_size = int(patch_size)
        self.patch_projection = nn.Linear(
            len(SEQUENCE_FEATURES) * self.patch_size,
            d_model,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=layers,
            norm=nn.LayerNorm(d_model),
        )

    def encode_sequence(self, sequence: torch.Tensor) -> torch.Tensor:
        remainder = sequence.shape[1] % self.patch_size
        if remainder:
            sequence = sequence[:, remainder:]
        batch, steps, features = sequence.shape
        patches = sequence.reshape(
            batch,
            steps // self.patch_size,
            features * self.patch_size,
        )
        return self.encoder(self.patch_projection(patches))


def build_temporal_baseline(
    name: str,
    *,
    memory_k: int = 0,
    retrieval: bool = True,
) -> nn.Module:
    if name == "lstm":
        model: nn.Module = LSTMBaseline()
    elif name == "tcn":
        model = TCNBaseline()
    elif name == "patchtst":
        model = PatchTSTBaseline()
    else:
        raise ValueError(f"unknown temporal baseline: {name}")
    if memory_k > 0:
        model.attach_memory(memory_k=memory_k, retrieval=retrieval)
    return model
