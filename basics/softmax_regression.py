"""Numerically stable softmax-regression components."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from torch.nn import functional as F


def stable_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Compute softmax after subtracting the maximum logit."""
    if logits.ndim < 1:
        raise ValueError("logits must have at least one dimension")
    shifted = logits - logits.max(dim=dim, keepdim=True).values
    return shifted.exp() / shifted.exp().sum(dim=dim, keepdim=True)


def cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute stable multiclass cross entropy from unnormalized logits."""
    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, classes]")
    if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
        raise ValueError("targets must have shape [batch]")
    return F.cross_entropy(logits, targets.long(), reduction=reduction)


def predict(logits: torch.Tensor) -> torch.Tensor:
    """Return the most likely class index for each row."""
    return logits.argmax(dim=-1)


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Return exact-match accuracy as a Python float."""
    if logits.shape[0] != targets.shape[0]:
        raise ValueError("logits and targets must have the same batch size")
    return float((predict(logits) == targets).float().mean())


def train_softmax_classifier(
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    num_epochs: int = 20,
    batch_size: int = 32,
    lr: float = 0.1,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> dict[str, object]:
    """Train a linear multiclass classifier on an in-memory dataset."""
    if features.ndim != 2:
        raise ValueError("features must have shape [batch, features]")
    if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
        raise ValueError("labels must have shape [batch]")
    if num_classes <= 1:
        raise ValueError("num_classes must be greater than one")

    device = torch.device(device)
    torch.manual_seed(seed)
    model = nn.Linear(features.shape[1], num_classes).to(device)
    features = features.to(device)
    labels = labels.to(device).long()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    losses: list[float] = []

    for _ in range(num_epochs):
        permutation = torch.randperm(features.shape[0], device=device)
        for start in range(0, features.shape[0], batch_size):
            indices = permutation[start : start + batch_size]
            loss = cross_entropy(model(features[indices]), labels[indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        losses.append(float(cross_entropy(model(features), labels).detach()))

    return {"model": model, "losses": losses, "accuracy": accuracy(model(features), labels)}

