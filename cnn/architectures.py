"""Compact CNN architectures used to inspect structural ideas."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class LeNet(nn.Module):
    """LeNet-style classifier with adaptive spatial pooling."""

    def __init__(self, num_classes: int = 10, in_channels: int = 1) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


class AlexNet(nn.Module):
    """A compact AlexNet-style network suitable for small smoke tests."""

    def __init__(self, num_classes: int = 10, in_channels: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs).flatten(1))


def vgg_block(
    num_convs: int,
    in_channels: int,
    out_channels: int,
) -> nn.Sequential:
    """Build one VGG block with repeated 3x3 convolutions."""
    layers: list[nn.Module] = []
    for index in range(num_convs):
        layers.extend(
            [
                nn.Conv2d(
                    in_channels if index == 0 else out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.ReLU(),
            ]
        )
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


class VGG(nn.Module):
    """A small VGG-style classifier."""

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 3,
        architecture: Sequence[tuple[int, int]] = ((1, 32), (1, 64), (2, 128)),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_channels = in_channels
        for num_convs, out_channels in architecture:
            layers.append(vgg_block(num_convs, current_channels, out_channels))
            current_channels = out_channels
        layers.append(nn.AdaptiveAvgPool2d(1))
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(current_channels, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs).flatten(1))


def nin_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    padding: int,
) -> nn.Sequential:
    """Build a Network-in-Network block."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(),
    )


class NiN(nn.Module):
    """A compact Network-in-Network classifier."""

    def __init__(self, num_classes: int = 10, in_channels: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nin_block(in_channels, 32, 5, 1, 2),
            nn.MaxPool2d(2, 2),
            nin_block(32, 64, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nin_block(64, 128, 3, 1, 1),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Conv2d(128, num_classes, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs)).flatten(1)


class Inception(nn.Module):
    """Four-branch Inception block."""

    def __init__(
        self,
        in_channels: int,
        c1: int,
        c2: tuple[int, int],
        c3: tuple[int, int],
        c4: int,
    ) -> None:
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=1),
            nn.ReLU(),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, c2[0], kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, c3[0], kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, c4, kernel_size=1),
            nn.ReLU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (
                self.branch1(inputs),
                self.branch2(inputs),
                self.branch3(inputs),
                self.branch4(inputs),
            ),
            dim=1,
        )


class GoogleNet(nn.Module):
    """A compact GoogLeNet-style classifier."""

    def __init__(self, num_classes: int = 10, in_channels: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            Inception(32, 16, (16, 24), (8, 16), 8),
            nn.MaxPool2d(2, 2),
            Inception(64, 24, (16, 32), (8, 16), 16),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(88, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs).flatten(1))


GoogLeNet = GoogleNet


class Residual(nn.Module):
    """Residual block with optional projection shortcut."""

    def __init__(
        self,
        input_channels: int,
        num_channels: int,
        use_1x1_conv: bool = False,
        strides: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            input_channels,
            num_channels,
            kernel_size=3,
            padding=1,
            stride=strides,
        )
        self.conv2 = nn.Conv2d(
            num_channels,
            num_channels,
            kernel_size=3,
            padding=1,
        )
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.conv3 = (
            nn.Conv2d(
                input_channels,
                num_channels,
                kernel_size=1,
                stride=strides,
            )
            if use_1x1_conv or input_channels != num_channels or strides != 1
            else None
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs if self.conv3 is None else self.conv3(inputs)
        output = self.relu(self.bn1(self.conv1(inputs)))
        output = self.bn2(self.conv2(output))
        return self.relu(output + residual)


def _resnet_stage(
    input_channels: int,
    output_channels: int,
    blocks: int,
    first_stride: int,
) -> nn.Sequential:
    layers: list[nn.Module] = [
        Residual(
            input_channels,
            output_channels,
            use_1x1_conv=True,
            strides=first_stride,
        )
    ]
    layers.extend(
        Residual(output_channels, output_channels)
        for _ in range(blocks - 1)
    )
    return nn.Sequential(*layers)


class ResNet(nn.Module):
    """A configurable, compact ResNet-18-style classifier."""

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 3,
        width: int = 16,
        blocks: Sequence[int] = (2, 2, 2, 2),
    ) -> None:
        super().__init__()
        if len(blocks) != 4 or any(block < 1 for block in blocks):
            raise ValueError("blocks must contain four positive stage sizes")
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        channels = [width, width * 2, width * 4, width * 8]
        stages: list[nn.Module] = []
        current_channels = width
        for index, (stage_channels, block_count) in enumerate(zip(channels, blocks, strict=True)):
            stages.append(
                _resnet_stage(
                    current_channels,
                    stage_channels,
                    block_count,
                    first_stride=1 if index == 0 else 2,
                )
            )
            current_channels = stage_channels
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(current_channels, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.stages(self.stem(inputs))
        return self.classifier(self.pool(output).flatten(1))


def resnet18(
    num_classes: int = 10,
    in_channels: int = 3,
    width: int = 16,
) -> ResNet:
    """Return the default compact ResNet-18-style model."""
    return ResNet(
        num_classes=num_classes,
        in_channels=in_channels,
        width=width,
        blocks=(2, 2, 2, 2),
    )

