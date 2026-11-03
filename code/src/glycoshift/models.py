from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .sequences import REGRESSION_HORIZONS, SEQUENCE_FEATURES, TASKS


@dataclass(frozen=True)
class GlycoShiftModelConfig:
    input_dim: int
    context_samples: int
    d_model: int
    n_heads: int
    n_layers: int
    dim_feedforward: int
    dropout: float
    memory_k: int

    @classmethod
    def variant(
        cls,
        name: str,
        *,
        context_samples: int,
        memory_k: int = 8,
        dropout: float = 0.1,
    ) -> GlycoShiftModelConfig:
        if name == "tiny":
            dimensions = {
                "d_model": 128,
                "n_heads": 4,
                "n_layers": 4,
                "dim_feedforward": 512,
            }
        elif name == "base":
            dimensions = {
                "d_model": 512,
                "n_heads": 8,
                "n_layers": 10,
                "dim_feedforward": 2048,
            }
        else:
            raise ValueError(f"unknown GlycoShift variant: {name}")
        return cls(
            input_dim=len(SEQUENCE_FEATURES),
            context_samples=context_samples,
            dropout=dropout,
            memory_k=memory_k,
            **dimensions,
        )


class GlycoShift(nn.Module):
    supports_memory = True

    def __init__(self, config: GlycoShiftModelConfig) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Sequential(
            nn.Linear(config.input_dim, config.d_model),
            nn.LayerNorm(config.d_model),
        )
        self.position_embedding = nn.Parameter(
            torch.zeros(1, config.context_samples, config.d_model)
        )
        nn.init.normal_(self.position_embedding, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.n_layers,
            norm=nn.LayerNorm(config.d_model),
        )
        self.population_memory = nn.Parameter(
            torch.zeros(config.memory_k, config.d_model)
        )
        nn.init.normal_(self.population_memory, std=0.02)
        self.memory_attention = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.memory_gate = nn.Linear(config.d_model * 2, config.d_model)
        self.classification_heads = nn.ModuleDict(
            {task: nn.Linear(config.d_model, 1) for task in TASKS}
        )
        self.regression_heads = nn.ModuleDict(
            {
                str(horizon): nn.Linear(config.d_model, 1)
                for horizon in REGRESSION_HORIZONS
            }
        )

    def encode_sequence(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 3:
            raise ValueError("sequence must have shape [batch, time, feature]")
        if sequence.shape[1] > self.config.context_samples:
            raise ValueError("sequence exceeds configured context length")
        projected = self.input_projection(sequence)
        projected = projected + self.position_embedding[:, : sequence.shape[1]]
        causal_mask = torch.triu(
            torch.ones(
                sequence.shape[1],
                sequence.shape[1],
                device=sequence.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        return self.encoder(projected, mask=causal_mask)

    def _fuse_memory(
        self,
        representation: torch.Tensor,
        memory_embeddings: torch.Tensor | None,
        memory_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size = representation.shape[0]
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
        return representation + gate * attended

    def forward(
        self,
        sequence: torch.Tensor,
        *,
        memory_embeddings: torch.Tensor | None = None,
        memory_padding_mask: torch.Tensor | None = None,
        use_memory: bool = True,
    ) -> dict[str, Any]:
        states = self.encode_sequence(sequence)
        representation = states[:, -1]
        return self.forward_from_representation(
            representation,
            memory_embeddings=memory_embeddings,
            memory_padding_mask=memory_padding_mask,
            use_memory=use_memory,
        )

    def forward_from_representation(
        self,
        representation: torch.Tensor,
        *,
        memory_embeddings: torch.Tensor | None = None,
        memory_padding_mask: torch.Tensor | None = None,
        use_memory: bool = True,
    ) -> dict[str, Any]:
        fused = (
            self._fuse_memory(
                representation,
                memory_embeddings,
                memory_padding_mask,
            )
            if use_memory
            else representation
        )
        logits = {
            task: head(fused).squeeze(-1)
            for task, head in self.classification_heads.items()
        }
        regression = {
            horizon: head(fused).squeeze(-1)
            for horizon, head in self.regression_heads.items()
        }
        return {
            "logits": logits,
            "regression": regression,
            "representation": representation,
            "fused_representation": fused,
        }


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
