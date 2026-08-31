"""A minimal perceptron with an explicit update rule."""

from __future__ import annotations

import torch


def train_perceptron(
    features: torch.Tensor,
    labels: torch.Tensor,
    num_epochs: int = 20,
    lr: float = 1.0,
) -> dict[str, object]:
    """Train on labels in {-1, +1} and return the update trace."""
    if features.ndim != 2:
        raise ValueError("features must have shape [batch, features]")
    if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
        raise ValueError("labels must have shape [batch]")
    unique_labels = set(labels.detach().cpu().tolist())
    if not unique_labels.issubset({-1, 1, -1.0, 1.0}):
        raise ValueError("perceptron labels must be -1 or +1")
    if num_epochs <= 0 or lr <= 0:
        raise ValueError("num_epochs and lr must be positive")

    weights = torch.zeros(features.shape[1], dtype=features.dtype, device=features.device)
    bias = torch.zeros((), dtype=features.dtype, device=features.device)
    mistakes_per_epoch: list[int] = []

    for _ in range(num_epochs):
        mistakes = 0
        for feature, label in zip(features, labels, strict=True):
            margin = label * (feature @ weights + bias)
            if margin <= 0:
                weights = weights + lr * label * feature
                bias = bias + lr * label
                mistakes += 1
        mistakes_per_epoch.append(mistakes)
        if mistakes == 0:
            break

    return {
        "weights": weights.detach(),
        "bias": bias.detach(),
        "mistakes": mistakes_per_epoch,
    }


def predict(features: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Predict labels in {-1, +1}."""
    scores = features @ weights + bias
    return torch.where(scores >= 0, 1.0, -1.0)

