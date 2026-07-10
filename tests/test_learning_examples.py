import unittest

import torch

from basics.linear_regression import squared_loss, synthetic_data
from basics.softmax_regression import cross_entropy, softmax
from cnn.resnet import Residual, resnet18
from vision.augmentation import build_train_transform
from sequence.time_machine import sequential_batches
from attention.kernel_regression import attention_weights, predict


class LinearRegressionTests(unittest.TestCase):
    def test_synthetic_data_shapes_and_zero_loss(self):
        features, labels = synthetic_data(torch.tensor([2.0, -3.4]), 4.2, 16)
        self.assertEqual(features.shape, (16, 2))
        self.assertEqual(labels.shape, (16, 1))
        self.assertTrue(torch.all(squared_loss(labels, labels) == 0))


class SoftmaxRegressionTests(unittest.TestCase):
    def test_probabilities_sum_to_one(self):
        probabilities = softmax(torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]))
        torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(2))

    def test_cross_entropy_selects_target_class(self):
        probabilities = torch.tensor([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]])
        labels = torch.tensor([1, 0])
        torch.testing.assert_close(cross_entropy(probabilities, labels), -torch.log(torch.tensor([0.7, 0.8])))


class ResNetTests(unittest.TestCase):
    def test_projection_residual_changes_shape(self):
        block = Residual(3, 8, use_projection=True, stride=2)
        self.assertEqual(block(torch.randn(2, 3, 32, 32)).shape, (2, 8, 16, 16))

    def test_resnet18_forward_shape(self):
        model = resnet18(num_classes=10, in_channels=1)
        self.assertEqual(model(torch.randn(2, 1, 96, 96)).shape, (2, 10))


class VisionTests(unittest.TestCase):
    def test_training_transform_contains_augmentation_and_tensor_conversion(self):
        names = [type(item).__name__ for item in build_train_transform().transforms]
        self.assertIn("RandomHorizontalFlip", names)
        self.assertIn("ToTensor", names)


class SequenceTests(unittest.TestCase):
    def test_sequential_batches_shift_targets_by_one(self):
        corpus = list(range(20))
        features, targets = next(sequential_batches(corpus, batch_size=2, num_steps=4, offset=0))
        torch.testing.assert_close(targets, features + 1)


class AttentionTests(unittest.TestCase):
    def test_attention_weights_are_normalized(self):
        queries = torch.tensor([0.0, 1.0])
        keys = torch.tensor([0.0, 1.0, 2.0])
        weights = attention_weights(queries, keys, bandwidth=0.5)
        self.assertEqual(weights.shape, (2, 3))
        torch.testing.assert_close(weights.sum(dim=1), torch.ones(2))

    def test_prediction_matches_query_count(self):
        queries = torch.tensor([0.0, 1.0])
        keys = torch.tensor([0.0, 1.0, 2.0])
        values = torch.tensor([1.0, 3.0, 5.0])
        self.assertEqual(predict(queries, keys, values).shape, (2,))


if __name__ == "__main__":
    unittest.main()
