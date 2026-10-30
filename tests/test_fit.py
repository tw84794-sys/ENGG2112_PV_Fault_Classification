"""The shared loop saves, selects, stops and reloads a tiny deterministic trial."""
import json
from pathlib import Path
import tempfile
import time
import unittest
import torch
from torch import nn
from torch.utils.data import DataLoader
from test_pipeline import TinyDataset
from pv_fault.evaluate import collect_validation_predictions
from pv_fault.train import fit


class FitTests(unittest.TestCase):
    def test_single_validation_pass_matches_loss_and_saved_predictions(self):
        class CountingModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0
            def forward(self, images):
                self.calls += 1
                return torch.zeros(len(images), 12)
        model = CountingModel()
        records, stats = collect_validation_predictions(
            model, DataLoader(TinyDataset(), batch_size=1), "cpu", criterion=nn.CrossEntropyLoss())
        self.assertEqual(model.calls, 2)
        self.assertEqual(stats["count"], 2)
        self.assertAlmostEqual(stats["loss"], float(torch.log(torch.tensor(12.))))
        self.assertEqual(len(records), 2)

    def test_best_checkpoint_roundtrip_and_early_stop(self):
        torch.manual_seed(2112)
        model = nn.Sequential(nn.Flatten(), nn.Linear(960, 12))
        train = DataLoader(TinyDataset("train"), batch_size=2)
        validation = DataLoader(TinyDataset(), batch_size=2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0)
        plan = {"max_epochs": 5, "early_stopping_patience": 1, "max_elapsed_seconds": 60}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            summary = fit(model, train, validation, optimizer, nn.CrossEntropyLoss(),
                          nn.CrossEntropyLoss(), plan, root, {"model": "tiny"}, time.monotonic())
            self.assertEqual(summary["best_epoch"], 1)
            self.assertEqual(summary["epochs_completed"], 2)
            self.assertEqual(summary["stop_reason"], "early_stopping")
            self.assertTrue(summary["checkpoint_predictions_verified"])
            saved = json.loads((root / "best_validation_predictions.json").read_text())
            self.assertEqual(saved, collect_validation_predictions(model, validation, "cpu"))
            timed = fit(model, train, validation, optimizer, nn.CrossEntropyLoss(),
                        nn.CrossEntropyLoss(), plan, root, {}, time.monotonic() - 61)
            self.assertIsNone(timed["best_epoch"])
            self.assertEqual(timed["stop_reason"], "time_limit_before_epoch")
