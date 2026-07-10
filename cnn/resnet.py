"""A compact ResNet-18 implementation for image classification practice."""

from __future__ import annotations

import torch
from torch import nn


class Residual(nn.Module):
    """Two-layer residual block with an optional projection shortcut."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        use_projection: bool = False,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.projection = (
            nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
            if use_projection
            else None
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.projection(inputs) if self.projection is not None else inputs
        outputs = self.relu(self.bn1(self.conv1(inputs)))
        outputs = self.bn2(self.conv2(outputs))
        return self.relu(outputs + residual)


def _stage(in_channels: int, out_channels: int, blocks: int, *, first: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    for index in range(blocks):
        if index == 0 and not first:
            layers.append(Residual(in_channels, out_channels, use_projection=True, stride=2))
        else:
            layers.append(Residual(out_channels, out_channels))
    return nn.Sequential(*layers)


def resnet18(num_classes: int = 10, in_channels: int = 1) -> nn.Sequential:
    """Build a ResNet-18 style classifier with adaptive global pooling."""
    return nn.Sequential(
        nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(3, stride=2, padding=1),
        _stage(64, 64, 2, first=True),
        _stage(64, 128, 2),
        _stage(128, 256, 2),
        _stage(256, 512, 2),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(512, num_classes),
    )

