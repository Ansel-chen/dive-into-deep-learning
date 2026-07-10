"""Linear regression utilities implemented with PyTorch tensors."""

from __future__ import annotations

import torch


def synthetic_data(weights: torch.Tensor, bias: float, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a noisy linear dataset ``y = Xw + b + noise``."""
    features = torch.normal(0, 1, (count, len(weights)))
    labels = features @ weights + bias
    labels += torch.normal(0, 0.01, labels.shape)
    return features, labels.reshape((-1, 1))


def squared_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return one-half squared error for each sample."""
    return (predictions - targets.reshape(predictions.shape)) ** 2 / 2

