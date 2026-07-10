"""Numerically stable softmax regression helpers."""

from __future__ import annotations

import torch


def softmax(logits: torch.Tensor) -> torch.Tensor:
    """Convert a 2-D logits tensor into row-wise probabilities."""
    shifted = logits - logits.max(dim=1, keepdim=True).values
    exponentials = torch.exp(shifted)
    return exponentials / exponentials.sum(dim=1, keepdim=True)


def cross_entropy(probabilities: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return per-example negative log likelihood for integer labels."""
    selected = probabilities[torch.arange(len(probabilities)), labels]
    return -torch.log(selected.clamp_min(torch.finfo(probabilities.dtype).tiny))

