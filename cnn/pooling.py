"""Pooling primitives with explicit shape and mode contracts."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


def _pair(value: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    if len(value) != 2:
        raise ValueError("pool size and stride must contain two values")
    return int(value[0]), int(value[1])


def pool2d(
    inputs: torch.Tensor,
    pool_size: int | tuple[int, int],
    mode: str = "max",
    stride: int | tuple[int, int] | None = None,
) -> torch.Tensor:
    """Apply valid max or average pooling to a rank-2 tensor."""
    if inputs.ndim != 2:
        raise ValueError("pool2d expects a rank-2 tensor")
    kernel_height, kernel_width = _pair(pool_size)
    stride_height, stride_width = _pair(stride) if stride is not None else (1, 1)
    if min(kernel_height, kernel_width, stride_height, stride_width) <= 0:
        raise ValueError("pool size and stride must be positive")
    if mode not in {"max", "avg"}:
        raise ValueError("mode must be 'max' or 'avg'")

    height, width = inputs.shape
    output_height = (height - kernel_height) // stride_height + 1
    output_width = (width - kernel_width) // stride_width + 1
    if output_height <= 0 or output_width <= 0:
        raise ValueError("pool window must fit inside inputs")

    output = torch.empty(
        (output_height, output_width),
        dtype=inputs.dtype,
        device=inputs.device,
    )
    reduce = torch.max if mode == "max" else torch.mean
    for row in range(output_height):
        for column in range(output_width):
            window = inputs[
                row * stride_height : row * stride_height + kernel_height,
                column * stride_width : column * stride_width + kernel_width,
            ]
            output[row, column] = reduce(window)
    return output


class Pooling2d(nn.Module):
    """NCHW wrapper around PyTorch pooling operators."""

    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        mode: str = "max",
        stride: int | tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        if mode == "max":
            self.pool = nn.MaxPool2d(kernel_size, stride=stride)
        elif mode == "avg":
            self.pool = nn.AvgPool2d(kernel_size, stride=stride)
        else:
            raise ValueError("mode must be 'max' or 'avg'")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4:
            raise ValueError("Pooling2d expects an NCHW tensor")
        return self.pool(inputs)


def functional_pool2d(
    inputs: torch.Tensor,
    kernel_size: int | tuple[int, int],
    mode: str = "max",
    stride: int | tuple[int, int] | None = None,
) -> torch.Tensor:
    """Functional NCHW pooling helper useful in small experiments."""
    if mode == "max":
        return F.max_pool2d(inputs, kernel_size, stride=stride)
    if mode == "avg":
        return F.avg_pool2d(inputs, kernel_size, stride=stride)
    raise ValueError("mode must be 'max' or 'avg'")
