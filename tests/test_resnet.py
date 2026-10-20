"""ResNet18 preprocessing and frozen-backbone invariants."""

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pv_fault.models.resnet import (
    IMAGENET_MEAN, IMAGENET_STD, WEIGHT_SHA256,
    ResNet18Thermal, preprocess_thermal,
)


class ResNet18ContractTests(unittest.TestCase):
    def test_grayscale_adapter_shape_and_normalization(self):
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        result = preprocess_thermal(torch.zeros(2, 1, 40, 24), mean, std)
        self.assertEqual(tuple(result.shape), (2, 3, 80, 48))
        for channel in range(3):
            expected = -IMAGENET_MEAN[channel] / IMAGENET_STD[channel]
            self.assertAlmostEqual(result[0, channel, 0, 0].item(), expected, delta=1e-6)

    def test_wrong_input_shape_is_rejected(self):
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        with self.assertRaises(ValueError):
            preprocess_thermal(torch.zeros(2, 3, 40, 24), mean, std)

    def test_only_classifier_trains_and_batchnorm_stays_frozen(self):
        weights = ROOT / "data/pretrained/resnet18-f37072fd.pth"
        if not weights.is_file():
            self.skipTest("Pretrained weights are not downloaded locally")
        model = ResNet18Thermal(12, weights, WEIGHT_SHA256)
        trainable = [name for name, p in model.named_parameters() if p.requires_grad]
        self.assertEqual(trainable, ["backbone.fc.weight", "backbone.fc.bias"])
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), 6156)
        model.train()
        self.assertFalse(model.backbone.bn1.training)
        self.assertTrue(model.backbone.fc.training)
        self.assertEqual(tuple(model(torch.zeros(2, 1, 40, 24)).shape), (2, 12))


if __name__ == "__main__":
    unittest.main()
