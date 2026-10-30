import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import check_validation_robustness as robustness


class RobustnessTests(unittest.TestCase):
    def test_preserves_shape_range_and_input(self):
        image = torch.linspace(0, 1, 40 * 24).reshape(1, 40, 24)
        original = image.clone()
        for condition in robustness.CONDITIONS:
            result = robustness.disturb(image, condition)
            self.assertEqual(result.shape, image.shape)
            self.assertTrue(torch.isfinite(result).all())
            self.assertGreaterEqual(result.min().item(), 0)
            self.assertLessEqual(result.max().item(), 1)
            torch.testing.assert_close(image, original)
        torch.testing.assert_close(robustness.disturb(image, 'original'), original)

    def test_shift_directions_and_padding(self):
        image = torch.zeros(1, 40, 24);image[0, 20, 12] = 1
        for condition, y, x in [('shift_left', 20, 10), ('shift_right', 20, 14),
                                ('shift_up', 18, 12), ('shift_down', 22, 12)]:
            moved = robustness.disturb(image, condition)
            self.assertEqual(moved[0, y, x].item(), 1)
            self.assertEqual(moved.sum().item(), 1)
        for condition in robustness.CONDITIONS:
            torch.testing.assert_close(robustness.disturb(torch.ones_like(image), condition), torch.ones_like(image))

    def test_lost_resolution_and_blur_suppress_fine_detail(self):
        image = (torch.arange(24) % 2).float().repeat(1, 40, 1)
        for condition in ('blur', 'half_resolution'):
            self.assertLess(robustness.disturb(image, condition).var().item(), image.var().item())

    def test_validation_only_and_invalid_inputs(self):
        for split in ('train', 'test'):
            with self.assertRaises(ValueError):
                robustness.DisturbedValidation(SimpleNamespace(split=split), 'original')
        with self.assertRaises(ValueError):
            robustness.disturb(torch.zeros(1, 40, 24), 'unknown')
        with self.assertRaises(ValueError):
            robustness.disturb(torch.zeros(1, 20, 12), 'blur')
        with self.assertRaises(ValueError):
            robustness.disturb(torch.full((1, 40, 24), float('nan')), 'original')
