"""Teaching-oriented two-dimensional cross-correlation."""

from __future__ import annotations

import torch


def corr2d(inputs: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Compute valid cross-correlation for two rank-2 tensors."""
    if inputs.ndim != 2 or kernel.ndim != 2:
        raise ValueError("corr2d expects rank-2 input and kernel tensors")
    height, width = inputs.shape
    kernel_height, kernel_width = kernel.shape
    if kernel_height > height or kernel_width > width:
        raise ValueError("kernel must fit inside inputs")

    output = torch.empty(
        (height - kernel_height + 1, width - kernel_width + 1),
        dtype=torch.promote_types(inputs.dtype, kernel.dtype),
        device=inputs.device,
    )
    for row in range(output.shape[0]):
        for column in range(output.shape[1]):
            window = inputs[
                row : row + kernel_height,
                column : column + kernel_width,
            ]
            output[row, column] = (window * kernel).sum()
    return output


cross_correlation2d = corr2d

