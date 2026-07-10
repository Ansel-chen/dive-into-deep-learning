"""Reusable image augmentation pipelines."""

from torchvision import transforms


def build_train_transform() -> transforms.Compose:
    """Return a modest training transform suitable for small RGB datasets."""
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
            transforms.ToTensor(),
        ]
    )

