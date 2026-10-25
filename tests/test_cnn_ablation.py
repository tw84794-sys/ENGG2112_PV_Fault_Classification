"""Preserve CNN controls and keep validation out of BatchNorm state updates."""
import io
import sys
import unittest
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from pv_fault.models.baseline import SmallCNN, WideCNN
from pv_fault.train import train_one_batch, evaluate_one_batch
from train_baseline import load_plan, make_scheduler


class CNNAblationTests(unittest.TestCase):
    def test_batchnorm_preserves_initial_convolution_and_classifier_weights(self):
        for model_class in (SmallCNN, WideCNN):
            torch.manual_seed(2112)
            control = model_class()
            torch.manual_seed(2112)
            variant = model_class(batch_norm=True)
            original = [m for m in control.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
            modified = [m for m in variant.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
            for left, right in zip(original, modified):
                self.assertTrue(torch.equal(left.weight, right.weight))
                self.assertTrue(torch.equal(left.bias, right.bias))

    def test_validation_leaves_batchnorm_statistics_unchanged_and_checkpoint_reloads(self):
        for model_class in (SmallCNN, WideCNN):
            model = model_class(batch_norm=True)
            images = torch.rand(4, 1, 40, 24)
            labels = torch.tensor([0, 1, 2, 7])
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            train_one_batch(model, images, labels, optimizer, criterion)
            before = {name: value.clone() for name, value in model.named_buffers()}
            evaluate_one_batch(model, images, labels, criterion)
            for name, value in model.named_buffers():
                self.assertTrue(torch.equal(before[name], value))
            with torch.inference_mode():
                expected = model(images)
            checkpoint = io.BytesIO()
            torch.save(model.state_dict(), checkpoint)
            checkpoint.seek(0)
            restored = model_class(batch_norm=True).eval()
            restored.load_state_dict(torch.load(checkpoint, weights_only=True))
            with torch.inference_mode():
                self.assertTrue(torch.equal(expected, restored(images)))

    def test_plateau_scheduler_tracks_maximum_score_and_preserves_control(self):
        model = SmallCNN()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        self.assertIsNone(make_scheduler(optimizer, {"scheduler": "none"}))
        plan = load_plan(ROOT / "configs/experiment-plan-smallcnn-scheduler.json")
        scheduler = make_scheduler(optimizer, plan)
        for score in (0.2, 0.3, 0.4, 0.4):
            scheduler.step(score)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.001)
        scheduler.step(0.4)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.0003)
        scheduler.step(0.5)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.0003)


if __name__ == "__main__":
    unittest.main()
