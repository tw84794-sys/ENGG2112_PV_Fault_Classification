"""Configurable CNN variants keep logits, gradients and checkpoint contracts."""
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
from train_baseline import make_optimizer


class CNNSearchTests(unittest.TestCase):
    def test_variants_have_finite_gradients_and_roundtrip(self):
        for model_class in (SmallCNN, WideCNN):
            for activation, extra in (("gelu", False), ("relu", True), ("gelu", True)):
                kwargs = dict(batch_norm=True, activation=activation, extra_conv=extra)
                model = model_class(**kwargs)
                images = torch.rand(4, 1, 40, 24)
                logits = model(images)
                self.assertEqual(tuple(logits.shape), (4, 12))
                nn.CrossEntropyLoss()(logits, torch.tensor([0, 1, 7, 11])).backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
                self.assertEqual(sum(isinstance(m, nn.Conv2d) for m in model.modules()), 3 if extra else 2)
                model.eval()
                with torch.inference_mode():
                    expected = model(images)
                buffer = io.BytesIO()
                torch.save(model.state_dict(), buffer)
                buffer.seek(0)
                restored = model_class(**kwargs).eval()
                restored.load_state_dict(torch.load(buffer, weights_only=True))
                with torch.inference_mode():
                    self.assertTrue(torch.equal(expected, restored(images)))

    def test_optimizers_respect_explicit_settings(self):
        model = SmallCNN()
        for name, expected in (("adam", torch.optim.Adam), ("adamw", torch.optim.AdamW), ("sgd", torch.optim.SGD)):
            optimizer = make_optimizer(model, {"optimizer": name, "learning_rate": 0.01, "weight_decay": 0.0001, "momentum": 0.9})
            self.assertIsInstance(optimizer, expected)
            self.assertEqual(optimizer.param_groups[0]["lr"], 0.01)
            self.assertEqual(optimizer.param_groups[0]["weight_decay"], 0.0001)
            if name == "sgd":
                self.assertEqual(optimizer.param_groups[0]["momentum"], 0.9)


if __name__ == "__main__":
    unittest.main()
