"""ResNet18 preprocessing and frozen-backbone invariants."""

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from train_baseline import make_scheduler

from pv_fault.finetuning import configure_resnet_finetuning
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


    def test_layer4_updates_leave_earlier_weights_and_batchnorm_frozen(self):
        weights = ROOT / "data/pretrained/resnet18-f37072fd.pth"
        if not weights.is_file():
            self.skipTest("Pretrained weights are not downloaded locally")
        model = ResNet18Thermal(12, weights, WEIGHT_SHA256)
        configure_resnet_finetuning(model)
        model.train()
        self.assertFalse(any(module.training for module in model.modules()
                             if isinstance(module, torch.nn.BatchNorm2d)))
        self.assertFalse(any(parameter.requires_grad for module in model.modules()
                             if isinstance(module, torch.nn.BatchNorm2d)
                             for parameter in module.parameters()))
        self.assertTrue(all(name.startswith(("backbone.layer4.", "backbone.fc."))
                            for name, parameter in model.named_parameters() if parameter.requires_grad))
        self.assertTrue(model.backbone.layer4[0].conv1.weight.requires_grad)
        self.assertTrue(model.backbone.fc.weight.requires_grad)
        early_weight = model.backbone.conv1.weight.detach().clone()
        late_weight = model.backbone.layer4[0].conv1.weight.detach().clone()
        head_weight = model.backbone.fc.weight.detach().clone()
        bn_mean = model.backbone.layer4[0].bn1.running_mean.clone()
        optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-5)
        loss = torch.nn.CrossEntropyLoss()(model(torch.rand(2, 1, 40, 24)), torch.tensor([0, 7]))
        loss.backward()
        self.assertIsNone(model.backbone.conv1.weight.grad)
        self.assertTrue(torch.isfinite(model.backbone.layer4[0].conv1.weight.grad).all())
        optimizer.step()
        self.assertTrue(torch.equal(early_weight, model.backbone.conv1.weight))
        self.assertTrue(torch.equal(bn_mean, model.backbone.layer4[0].bn1.running_mean))
        self.assertFalse(torch.equal(late_weight, model.backbone.layer4[0].conv1.weight))
        self.assertFalse(torch.equal(head_weight, model.backbone.fc.weight))


    def test_scheduler_reduces_both_finetuning_rates_after_a_plateau(self):
        layer_parameter = torch.nn.Parameter(torch.ones(1))
        head_parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.Adam([
            {"params": [layer_parameter], "lr": 3e-5},
            {"params": [head_parameter], "lr": 3e-4},
        ])
        scheduler = make_scheduler(optimizer, {
            "scheduler": "plateau",
            "scheduler_settings": {"factor": 0.5, "patience": 1, "min_lr": 1e-7},
        })
        scheduler.step(0.5)
        scheduler.step(0.49)
        self.assertEqual([group["lr"] for group in optimizer.param_groups], [3e-5, 3e-4])
        scheduler.step(0.49)
        self.assertEqual([group["lr"] for group in optimizer.param_groups], [1.5e-5, 1.5e-4])
        scheduler.step(0.6)
        self.assertEqual([group["lr"] for group in optimizer.param_groups], [1.5e-5, 1.5e-4])


if __name__ == "__main__":
    unittest.main()
