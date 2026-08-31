"""RNN encoders with explicit batch-first contracts."""

from __future__ import annotations

import torch
from torch import nn


class RNNEncoder(nn.Module):
    """Embedding plus nn.RNN returning outputs and final hidden state."""

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
        self.rnn = nn.RNN(
            embed_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

    @property
    def output_size(self) -> int:
        return self.rnn.hidden_size * (2 if self.rnn.bidirectional else 1)

    def forward(
        self,
        tokens: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch, steps]")
        embeddings = self.embedding(tokens.long())
        return self.rnn(embeddings, state)


class BidirectionalRNNEncoder(RNNEncoder):
    """RNN encoder that exposes forward and backward hidden states."""

    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        hidden_size: int,
        num_layers: int = 1,
    ) -> None:
        super().__init__(
            vocab_size=vocab_size,
            embed_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
        )
