"""Minimal recurrent language-model training components."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class RNNLanguageModel(nn.Module):
    """Batch-first token language model based on a vanilla RNN."""

    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        hidden_size: int,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        if min(vocab_size, embed_size, hidden_size, num_layers) <= 0:
            raise ValueError("model dimensions must be positive")
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(
            embed_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.projection = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        tokens: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch, steps]")
        hidden_states, state = self.rnn(self.embedding(tokens.long()), state)
        return self.projection(hidden_states), state


def language_model_loss(
    model: RNNLanguageModel,
    tokens: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Compute next-token cross entropy for a batch."""
    logits, _ = model(tokens)
    if logits.shape[:2] != targets.shape:
        raise ValueError("targets must match the model batch and step dimensions")
    return F.cross_entropy(logits.reshape(-1, model.vocab_size), targets.reshape(-1).long())


def train_language_model(
    model: RNNLanguageModel,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    num_epochs: int = 1,
    lr: float = 0.01,
) -> list[float]:
    """Train on pre-built batches and return mean loss per epoch."""
    if num_epochs <= 0 or lr <= 0:
        raise ValueError("num_epochs and lr must be positive")
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    losses: list[float] = []
    for _ in range(num_epochs):
        epoch_losses: list[float] = []
        for tokens, targets in batches:
            optimizer.zero_grad(set_to_none=True)
            loss = language_model_loss(model, tokens, targets)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        if not epoch_losses:
            raise ValueError("batches must contain at least one item")
        losses.append(sum(epoch_losses) / len(epoch_losses))
    return losses

