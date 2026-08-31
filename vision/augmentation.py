"""Network-free image augmentation primitives."""

from __future__ import annotations

import torch
from torch import nn


def _check_images(images: torch.Tensor) -> None:
    if images.ndim not in {3, 4}:
        raise ValueError("images must have shape [C,H,W] or [N,C,H,W]")


def horizontal_flip(images: torch.Tensor) -> torch.Tensor:
    """Flip a CHW or NCHW batch along its width dimension."""
    _check_images(images)
    return images.flip(-1)


def random_horizontal_flip(
    images: torch.Tensor,
    probability: float = 0.5,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Flip each image independently without modifying the input tensor."""
    _check_images(images)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between zero and one")
    if images.ndim == 3:
        if probability == 1.0:
            return horizontal_flip(images)
        if probability == 0.0:
            return images.clone()
        random_value = torch.rand((), generator=generator, device=images.device)
        return horizontal_flip(images) if random_value < probability else images.clone()

    batch_size = images.shape[0]
    if generator is None:
        choices = torch.rand(batch_size, device=images.device) < probability
    else:
        choices = torch.rand(batch_size, generator=generator) < probability
        choices = choices.to(images.device)
    output = images.clone()
    output[choices] = horizontal_flip(images[choices])
    return output


def add_gaussian_noise(
    images: torch.Tensor,
    noise_std: float = 0.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Add zero-mean Gaussian noise while preserving the image contract."""
    _check_images(images)
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")
    if noise_std == 0:
        return images.clone()
    noise = torch.randn(
        images.shape,
        dtype=images.dtype,
        device=images.device,
        generator=generator,
    )
    return images + noise_std * noise


class AugmentationPipeline(nn.Module):
    """A small, deterministic-under-seed augmentation pipeline."""

    def __init__(
        self,
        flip_probability: float = 0.5,
        noise_std: float = 0.0,
    ) -> None:
        super().__init__()
        self.flip_probability = flip_probability
        self.noise_std = noise_std

    def forward(
        self,
        images: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        output = random_horizontal_flip(
            images,
            probability=self.flip_probability,
            generator=generator,
        )
        return add_gaussian_noise(
            output,
            noise_std=self.noise_std,
            generator=generator,
        )


def make_augmentation_pipeline(
    flip_probability: float = 0.5,
    noise_std: float = 0.0,
) -> AugmentationPipeline:
    """Build a reusable augmentation module without downloading assets."""
    return AugmentationPipeline(flip_probability, noise_std)

