"""GRU encoders with explicit batch-first contracts."""

from __future__ import annotations

import torch
from torch import nn


class GRUEncoder(nn.Module):
    """Embedding plus nn.GRU returning outputs and final hidden state."""

    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        if min(vocab_size, embed_size, hidden_size, num_layers) <= 0:
            raise ValueError("model dimensions must be positive")
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(
            embed_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

    @property
    def output_size(self) -> int:
        return self.gru.hidden_size * (2 if self.gru.bidirectional else 1)

    def forward(
        self,
        tokens: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch, steps]")
        embeddings = self.embedding(tokens.long())
        return self.gru(embeddings, state)

