import pytest
import torch

from cnn.architectures import (
    AlexNet,
    GoogleNet,
    LeNet,
    NiN,
    ResNet,
    VGG,
    Residual,
    resnet18,
)
from cnn.convolution import corr2d
from cnn.pooling import pool2d


def test_corr2d_shape_and_values():
    image = torch.arange(1.0, 17.0).reshape(4, 4)
    kernel = torch.tensor([[1.0, 0.0], [0.0, -1.0]])
    output = corr2d(image, kernel)
    assert output.shape == (3, 3)
    assert torch.equal(output, torch.full((3, 3), -5.0))


def test_pool2d_supports_max_and_average_modes():
    image = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    assert torch.equal(
        pool2d(image, (2, 2), mode="max"),
        torch.tensor([[5.0, 6.0], [8.0, 9.0]]),
    )
    assert torch.allclose(
        pool2d(image, (2, 2), mode="avg"),
        torch.tensor([[3.0, 4.0], [6.0, 7.0]]),
    )


def test_residual_handles_identity_and_channel_change():
    identity_block = Residual(4, 4)
    changed_block = Residual(4, 8, use_1x1_conv=True, strides=2)
    x = torch.randn(2, 4, 16, 16)
    assert identity_block(x).shape == x.shape
    assert changed_block(x).shape == (2, 8, 8, 8)


@pytest.mark.parametrize(
    ("model", "channels", "size"),
    [
        (LeNet(num_classes=5), 1, 28),
        (AlexNet(num_classes=5, in_channels=3), 3, 32),
        (VGG(num_classes=5, in_channels=3), 3, 32),
        (NiN(num_classes=5, in_channels=3), 3, 32),
        (GoogleNet(num_classes=5, in_channels=3), 3, 32),
        (ResNet(num_classes=5, in_channels=3, width=8), 3, 32),
    ],
)
def test_classic_architectures_have_stable_forward_shape(model, channels, size):
    output = model(torch.randn(2, channels, size, size))
    assert output.shape == (2, 5)
    assert torch.isfinite(output).all()


def test_resnet18_factory_is_callable():
    model = resnet18(num_classes=3, in_channels=3, width=8).eval()
    output = model(torch.randn(1, 3, 32, 32))
    assert output.shape == (1, 3)


def test_gpu_smoke_when_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    model = resnet18(num_classes=3, width=8).cuda().eval()
    output = model(torch.randn(1, 3, 32, 32, device="cuda"))
    assert output.is_cuda
    assert output.shape == (1, 3)
