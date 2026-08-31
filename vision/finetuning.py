"""Explicit, network-free transfer-learning exercises."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SmallImageClassifier(nn.Module):
    """A compact backbone plus a replaceable classifier head."""

    def __init__(self, num_classes: int = 2, in_channels: int = 3) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4:
            raise ValueError("inputs must have shape [batch, channels, height, width]")
        return self.classifier(self.backbone(inputs).flatten(1))


def replace_classifier(model: nn.Module, num_classes: int) -> nn.Module:
    """Replace a common classifier/fc head and return the same model."""
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    if hasattr(model, "classifier"):
        classifier = getattr(model, "classifier")
        if isinstance(classifier, nn.Linear):
            setattr(model, "classifier", nn.Linear(classifier.in_features, num_classes))
            return model
        if isinstance(classifier, nn.Sequential):
            for index in range(len(classifier) - 1, -1, -1):
                if isinstance(classifier[index], nn.Linear):
                    classifier[index] = nn.Linear(
                        classifier[index].in_features,
                        num_classes,
                    )
                    return model
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    raise TypeError("model does not expose a replaceable classifier or fc head")


def freeze_backbone(model: nn.Module) -> None:
    """Freeze every parameter except a classifier or fc head."""
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("classifier") or name.startswith("fc")
    if not any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("model has no trainable classifier or fc parameters")


def build_finetuning_model(
    num_classes: int,
    in_channels: int = 3,
    backbone: str = "small",
    pretrained: bool = False,
) -> SmallImageClassifier:
    """Build a local model; pretrained=True is intentionally unsupported."""
    if backbone != "small":
        raise ValueError("only the network-free 'small' backbone is supported")
    if pretrained:
        raise ValueError("pretrained weights require an explicit external-data experiment")
    return SmallImageClassifier(num_classes=num_classes, in_channels=in_channels)


def training_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Run one supervised optimization step and return detached loss."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = F.cross_entropy(logits, labels.long())
    loss.backward()
    optimizer.step()
    return loss.detach()

