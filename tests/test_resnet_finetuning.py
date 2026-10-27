"""Late-stage adaptation preserves the frozen early network and BatchNorm."""
import sys
import unittest
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pv_fault.finetuning import configure_resnet_finetuning
from pv_fault.models.resnet import ResNet18Thermal, enable_layer4_finetuning


class LateStageTests(unittest.TestCase):
    def make_model(self):
        weights = ROOT / "data/pretrained/resnet18-f37072fd.pth"
        if not weights.is_file():
            self.skipTest("Pretrained weights are not downloaded locally")
        return ResNet18Thermal(12, weights)

    def test_layer4_matches_existing_trainable_flags(self):
        model = self.make_model()
        enable_layer4_finetuning(model)
        expected = {name: p.requires_grad for name, p in model.named_parameters()}
        configure_resnet_finetuning(model)
        self.assertEqual(expected, {name: p.requires_grad for name, p in model.named_parameters()})

    def test_layer3_and_layer4_receive_gradients_with_earlier_layers_frozen(self):
        model = self.make_model()
        configure_resnet_finetuning(model, ("layer3", "layer4"))
        model.train()
        self.assertFalse(any(m.training for m in model.modules() if isinstance(m, nn.BatchNorm2d)))
        self.assertFalse(any(p.requires_grad for m in model.modules()
                             if isinstance(m, nn.BatchNorm2d) for p in m.parameters()))
        nn.CrossEntropyLoss()(model(torch.rand(2, 1, 40, 24)), torch.tensor([0, 7])).backward()
        self.assertIsNone(model.backbone.layer2[0].conv1.weight.grad)
        self.assertTrue(torch.isfinite(model.backbone.layer3[0].conv1.weight.grad).all())
        self.assertTrue(torch.isfinite(model.backbone.layer4[0].conv1.weight.grad).all())
        self.assertTrue(torch.isfinite(model.backbone.fc.weight.grad).all())

    def test_rejects_unsupported_stage_selection(self):
        with self.assertRaises(ValueError):
            configure_resnet_finetuning(None, ("layer1",))


if __name__ == "__main__":
    unittest.main()
