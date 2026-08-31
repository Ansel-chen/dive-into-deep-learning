import torch

from basics.linear_regression import (
    synthetic_data,
    train_linear_regression,
)
from basics.perceptron import train_perceptron
from basics.softmax_regression import cross_entropy, stable_softmax


def test_synthetic_data_contract():
    features, labels = synthetic_data(
        torch.tensor([2.0, -3.4]),
        4.2,
        num_examples=32,
        noise_std=0.0,
    )
    assert features.shape == (32, 2)
    assert labels.shape == (32, 1)


def test_linear_regression_learns_known_parameters():
    result = train_linear_regression(
        num_examples=96,
        num_epochs=30,
        batch_size=16,
        lr=0.03,
        noise_std=0.0,
        seed=7,
    )
    assert result["losses"][0] > result["losses"][-1]
    assert torch.allclose(
        result["weights"].flatten(),
        torch.tensor([2.0, -3.4]),
        atol=0.08,
    )
    assert abs(float(result["bias"]) - 4.2) < 0.08


def test_stable_softmax_handles_large_logits():
    logits = torch.tensor([[10000.0, 9999.0, -10000.0]])
    probabilities = stable_softmax(logits)
    assert torch.isfinite(probabilities).all()
    assert torch.allclose(
        probabilities.sum(dim=1),
        torch.ones(1),
        atol=1e-6,
    )


def test_cross_entropy_is_finite_and_prefers_correct_class():
    logits = torch.tensor([[10000.0, -10000.0], [-10000.0, 10000.0]])
    targets = torch.tensor([0, 1])
    loss = cross_entropy(logits, targets)
    assert torch.isfinite(loss)
    assert float(loss) < 1e-5


def test_perceptron_converges_on_separable_data():
    features = torch.tensor(
        [[2.0, 1.0], [1.5, 2.0], [-1.0, -2.0], [-2.0, -1.0]]
    )
    labels = torch.tensor([1.0, 1.0, -1.0, -1.0])
    result = train_perceptron(features, labels, num_epochs=20)
    predictions = torch.where(
        features @ result["weights"] + result["bias"] >= 0,
        1.0,
        -1.0,
    )
    assert torch.equal(predictions, labels)
    assert result["mistakes"][-1] == 0

