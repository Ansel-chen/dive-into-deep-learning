"""Attention-weighted kernel regression from first principles."""

from __future__ import annotations

import torch


def gaussian_attention_weights(
    queries: torch.Tensor,
    keys: torch.Tensor,
    bandwidth: float = 1.0,
) -> torch.Tensor:
    """Return Gaussian attention weights with shape [queries, keys]."""
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")
    queries = torch.as_tensor(queries)
    keys = torch.as_tensor(keys, device=queries.device, dtype=queries.dtype)
    queries = queries.flatten()
    keys = keys.flatten()
    scores = -0.5 * ((queries[:, None] - keys[None, :]) / bandwidth) ** 2
    return torch.softmax(scores, dim=-1)


def nadaraya_watson(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    bandwidth: float = 1.0,
    return_weights: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Predict values by a Gaussian attention-weighted average."""
    weights = gaussian_attention_weights(queries, keys, bandwidth)
    values = torch.as_tensor(values, device=weights.device, dtype=weights.dtype)
    if values.shape[0] != weights.shape[1]:
        raise ValueError("values must have one row per key")
    original_rank = values.ndim
    if values.ndim == 1:
        values = values.unsqueeze(-1)
    elif values.ndim != 2:
        raise ValueError("values must have shape [keys] or [keys, features]")
    predictions = weights @ values
    if original_rank == 1:
        predictions = predictions.squeeze(-1)
    return (predictions, weights) if return_weights else predictions

