import torch

from vision.augmentation import horizontal_flip, make_augmentation_pipeline
from vision.finetuning import (
    build_finetuning_model,
    freeze_backbone,
    training_step,
)


def test_horizontal_flip_preserves_shape_and_reverses_width():
    images = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)
    flipped = horizontal_flip(images)
    assert flipped.shape == images.shape
    assert torch.equal(flipped, images.flip(-1))


def test_augmentation_pipeline_is_deterministic_when_probability_is_one():
    images = torch.rand(2, 3, 8, 8)
    pipeline = make_augmentation_pipeline(
        flip_probability=1.0,
        noise_std=0.0,
    )
    augmented = pipeline(images)
    assert torch.allclose(augmented, images.flip(-1))


def test_finetuning_model_replaces_head_and_trains_one_batch():
    model = build_finetuning_model(num_classes=4, in_channels=3)
    output = model(torch.randn(4, 3, 32, 32))
    assert output.shape == (4, 4)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss = training_step(
        model,
        optimizer,
        torch.randn(4, 3, 32, 32),
        torch.tensor([0, 1, 2, 3]),
    )
    assert torch.isfinite(loss)


def test_freeze_backbone_leaves_classifier_trainable():
    model = build_finetuning_model(num_classes=3)
    freeze_backbone(model)
    backbone_params = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("classifier")
    ]
    classifier_params = [
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith("classifier")
    ]
    assert backbone_params and classifier_params
    assert all(not parameter.requires_grad for parameter in backbone_params)
    assert all(parameter.requires_grad for parameter in classifier_params)

