import unittest

import torch
from torch import nn
from pv_fault.gradcam import grad_cam


class GradCamTests(unittest.TestCase):
    def setUp(self):
        self.model = nn.Sequential(nn.Conv2d(1, 1, 1, bias=False), nn.ReLU(),
                                   nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(1, 2, bias=False)).eval()
        with torch.no_grad():
            self.model[0].weight.fill_(1)
            self.model[4].weight.copy_(torch.tensor([[2.], [-2.]]))
        self.model.requires_grad_(False)
        self.images = torch.tensor([[[[0., 1.], [2., 4.]]], [[[0., 1.], [2., 4.]]]])

    def test_matches_analytic_map_with_frozen_weights(self):
        before = {k: v.clone() for k, v in self.model.state_dict().items()}
        with torch.no_grad():
            logits, maps = grad_cam(self.model, self.images, self.model[1], [0, 1])
        torch.testing.assert_close(maps[0, 0], self.images[0, 0] / 4)
        self.assertEqual(maps[1].count_nonzero().item(), 0)
        torch.testing.assert_close(logits, self.model(self.images))
        for key, value in self.model.state_dict().items():
            torch.testing.assert_close(value, before[key])
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in self.model.parameters()))
        self.assertFalse(self.images.requires_grad)
        self.assertFalse(self.model[1]._forward_hooks)

    def test_invalid_targets_remove_hook(self):
        for targets in ([0], [0, 2], [0., 1.]):
            with self.assertRaises(ValueError):
                grad_cam(self.model, self.images, self.model[1], targets)
            self.assertFalse(self.model[1]._forward_hooks)

    def test_rejects_training_mode(self):
        with self.assertRaises(ValueError):
            grad_cam(self.model.train(), self.images, self.model[1], [0, 1])
