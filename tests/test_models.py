"""Checks for the controlled CNN width comparison."""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pv_fault.models.baseline import SmallCNN, WideCNN


class CNNWidthTests(unittest.TestCase):
    def test_original_model_keeps_its_shape_and_parameter_count(self):
        model = SmallCNN(num_classes=12)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 38412)
        self.assertEqual(tuple(model(torch.zeros(4, 1, 40, 24)).shape), (4, 12))

    def test_wide_model_changes_only_width(self):
        model = WideCNN(num_classes=12)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 338316)
        self.assertEqual(tuple(model(torch.zeros(4, 1, 40, 24)).shape), (4, 12))
        self.assertEqual(
            [type(layer) for layer in model.features],
            [type(layer) for layer in SmallCNN().features],
        )


if __name__ == "__main__":
    unittest.main()
