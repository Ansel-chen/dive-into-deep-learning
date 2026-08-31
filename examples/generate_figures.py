"""Generate small, deterministic figures for the project README."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attention.kernel_regression import nadaraya_watson
from basics.linear_regression import train_linear_regression
from cnn.convolution import corr2d
from vision.augmentation import make_augmentation_pipeline


def _prepare_output_dir(output_dir: str | Path) -> Path:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def generate_linear_regression_figure(output_dir: str | Path) -> Path:
    output_dir = _prepare_output_dir(output_dir)
    result = train_linear_regression(
        num_examples=128,
        num_epochs=30,
        batch_size=16,
        lr=0.03,
        noise_std=0.05,
        seed=11,
    )
    predictions = (
        result["features"] @ result["weights"].reshape(-1, 1)
        + result["bias"]
    )
    labels = result["labels"]

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(
        labels.numpy().ravel(),
        predictions.numpy().ravel(),
        alpha=0.65,
        edgecolors="none",
    )
    minimum = min(float(labels.min()), float(predictions.min()))
    maximum = max(float(labels.max()), float(predictions.max()))
    axes[0].plot([minimum, maximum], [minimum, maximum], "k--", linewidth=1)
    axes[0].set_title("Linear regression: prediction vs target")
    axes[0].set_xlabel("target")
    axes[0].set_ylabel("prediction")

    axes[1].plot(result["losses"], color="#d94841", linewidth=2)
    axes[1].set_title("Training loss")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("half squared error")
    figure.tight_layout()
    path = output_dir / "linear-regression.png"
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return path


def generate_convolution_figure(output_dir: str | Path) -> Path:
    output_dir = _prepare_output_dir(output_dir)
    image = torch.zeros(16, 16)
    image[:, 7:9] = 1.0
    kernel = torch.tensor([[1.0, -1.0], [1.0, -1.0]])
    response = corr2d(image, kernel)

    figure, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    panels = [
        (image, "input image"),
        (kernel, "2x2 kernel"),
        (response, "cross-correlation"),
    ]
    for axis, (panel, title) in zip(axes, panels, strict=True):
        axis.imshow(panel.numpy(), cmap="coolwarm", aspect="equal")
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle("Local windows create edge responses", y=1.02)
    figure.tight_layout()
    path = output_dir / "convolution-response.png"
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return path


def _synthetic_image() -> torch.Tensor:
    image = torch.zeros(3, 32, 32)
    image[0, 6:26, 8:14] = 1.0
    image[1, 10:22, 12:26] = 0.8
    image[2, 8:18, 18:28] = 0.6
    return image


def generate_augmentation_figure(output_dir: str | Path) -> Path:
    output_dir = _prepare_output_dir(output_dir)
    image = _synthetic_image()
    flipped = make_augmentation_pipeline(
        flip_probability=1.0,
        noise_std=0.0,
    )(image)
    noisy = make_augmentation_pipeline(
        flip_probability=0.0,
        noise_std=0.05,
    )(image, generator=torch.Generator().manual_seed(3))

    figure, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    for axis, panel, title in zip(
        axes,
        (image, flipped, noisy),
        ("original", "horizontal flip", "additive noise"),
        strict=True,
    ):
        axis.imshow(panel.clamp(0, 1).permute(1, 2, 0).numpy())
        axis.set_title(title)
        axis.axis("off")
    figure.suptitle("Augmentation changes observations, not label semantics", y=1.02)
    figure.tight_layout()
    path = output_dir / "augmentation-comparison.png"
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return path


def generate_attention_figure(output_dir: str | Path) -> Path:
    output_dir = _prepare_output_dir(output_dir)
    keys = torch.linspace(-2.0, 2.0, 17)
    values = torch.sin(keys * 1.5)
    queries = torch.linspace(-2.0, 2.0, 40)
    predictions, weights = nadaraya_watson(
        queries,
        keys,
        values,
        bandwidth=0.38,
        return_weights=True,
    )

    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    image = axes[0].imshow(
        weights.numpy(),
        aspect="auto",
        origin="lower",
        extent=[float(keys.min()), float(keys.max()), float(queries.min()), float(queries.max())],
        cmap="viridis",
    )
    axes[0].set_title("Gaussian attention weights")
    axes[0].set_xlabel("key position")
    axes[0].set_ylabel("query position")
    figure.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)

    axes[1].plot(keys.numpy(), values.numpy(), "o", label="key values")
    axes[1].plot(queries.numpy(), predictions.numpy(), label="weighted prediction")
    axes[1].set_title("Nadaraya–Watson prediction")
    axes[1].set_xlabel("position")
    axes[1].legend(frameon=False)
    figure.tight_layout()
    path = output_dir / "attention-weights.png"
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return path


def generate_all_figures(output_dir: str | Path = "assets/figures") -> list[Path]:
    """Generate all README figures and return their paths."""
    return [
        generate_linear_regression_figure(output_dir),
        generate_convolution_figure(output_dir),
        generate_augmentation_figure(output_dir),
        generate_attention_figure(output_dir),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("assets/figures"),
    )
    args = parser.parse_args()
    for path in generate_all_figures(args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
