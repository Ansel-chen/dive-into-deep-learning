"""Small, testable linear-regression building blocks."""

from __future__ import annotations

from collections.abc import Iterator

import torch


def synthetic_data(
    weights: torch.Tensor,
    bias: float | torch.Tensor,
    num_examples: int,
    noise_std: float = 0.01,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate y = Xw + b + noise with an explicit tensor contract."""
    if num_examples <= 0:
        raise ValueError("num_examples must be positive")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")

    weights = torch.as_tensor(weights, dtype=torch.get_default_dtype()).reshape(-1)
    features = torch.randn(
        num_examples,
        weights.numel(),
        generator=generator,
        dtype=weights.dtype,
    )
    labels = features @ weights.reshape(-1, 1) + torch.as_tensor(
        bias,
        dtype=weights.dtype,
    )
    if noise_std:
        labels = labels + noise_std * torch.randn(
            labels.shape,
            generator=generator,
            dtype=labels.dtype,
        )
    return features, labels


def linreg(
    features: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Apply a linear model to a batch of feature rows."""
    return features @ weights + bias


def squared_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Return elementwise half squared error."""
    return (predictions - targets.reshape_as(predictions)) ** 2 / 2


def data_iter(
    batch_size: int,
    features: torch.Tensor,
    labels: torch.Tensor,
    shuffle: bool = True,
    generator: torch.Generator | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield mini-batches without hiding data movement or shuffling."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if len(features) != len(labels):
        raise ValueError("features and labels must have the same length")

    if shuffle:
        if generator is None:
            indices = torch.randperm(len(features), device=features.device)
        else:
            indices = torch.randperm(
                len(features),
                generator=generator,
                device=features.device,
            )
    else:
        indices = torch.arange(len(features), device=features.device)

    for start in range(0, len(features), batch_size):
        batch_indices = indices[start : start + batch_size]
        yield features[batch_indices], labels[batch_indices]


def train_linear_regression(
    num_examples: int = 1_000,
    num_epochs: int = 20,
    batch_size: int = 10,
    lr: float = 0.03,
    noise_std: float = 0.01,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor | list[float]]:
    """Train a linear model with manual SGD and return inspectable results."""
    if num_epochs <= 0:
        raise ValueError("num_epochs must be positive")
    if lr <= 0:
        raise ValueError("lr must be positive")

    device = torch.device(device)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    true_weights = torch.tensor([2.0, -3.4], device=device)
    features, labels = synthetic_data(
        true_weights.cpu(),
        4.2,
        num_examples,
        noise_std=noise_std,
        generator=generator,
    )
    features = features.to(device)
    labels = labels.to(device)

    weights = torch.zeros(
        (true_weights.numel(), 1),
        device=device,
        requires_grad=True,
    )
    bias = torch.zeros(1, device=device, requires_grad=True)
    losses: list[float] = []

    for _ in range(num_epochs):
        batch_generator = torch.Generator(device="cpu").manual_seed(seed)
        for batch_features, batch_labels in data_iter(
            batch_size,
            features,
            labels,
            shuffle=True,
            generator=batch_generator,
        ):
            predictions = linreg(batch_features, weights, bias)
            loss = squared_loss(predictions, batch_labels).mean()
            loss.backward()
            with torch.no_grad():
                weights -= lr * weights.grad
                bias -= lr * bias.grad
            weights.grad.zero_()
            bias.grad.zero_()

        with torch.no_grad():
            epoch_loss = squared_loss(
                linreg(features, weights, bias),
                labels,
            ).mean()
        losses.append(float(epoch_loss))

    return {
        "weights": weights.detach().reshape(-1),
        "bias": bias.detach().squeeze(),
        "losses": losses,
        "features": features.detach(),
        "labels": labels.detach(),
    }


if __name__ == "__main__":
    result = train_linear_regression()
    print("learned weights:", result["weights"])
    print("learned bias:", result["bias"])
    print("final loss:", result["losses"][-1])

