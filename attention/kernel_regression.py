"""Nadaraya-Watson kernel regression as a compact attention example."""

from __future__ import annotations

import torch


def attention_weights(queries: torch.Tensor, keys: torch.Tensor, bandwidth: float = 0.5) -> torch.Tensor:
    """Return Gaussian-kernel attention weights for each query."""
    distances = queries.reshape(-1, 1) - keys.reshape(1, -1)
    scores = -(distances**2) / (2 * bandwidth**2)
    return torch.softmax(scores, dim=1)


def predict(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    bandwidth: float = 0.5,
) -> torch.Tensor:
    """Predict values as the attention-weighted sum of training values."""
    return attention_weights(queries, keys, bandwidth) @ values

