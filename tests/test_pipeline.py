"""Checks for training and evaluation pipeline behaviour."""

import sys
import unittest
import torch

from pathlib import Path
from torch.utils.data import DataLoader, Dataset
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pv_fault.data import CLASS_NAMES, CLASS_TO_INDEX
from pv_fault.evaluate import classification_metrics, collect_validation_predictions


class MetricsTests(unittest.TestCase):
    def test_known_predictions(self):
        result = classification_metrics(
            [0, 0, 1, 1, 2, 2],
            [0, 1, 1, 2, 1, 2],
            class_names=("FaultA", "No-Anomaly", "FaultB"),
        )

        self.assertEqual(
            result["confusion_matrix"],
            [[1, 1, 0], [0, 1, 1], [0, 1, 1]],
        )
        self.assertAlmostEqual(result["accuracy"], 0.5)
        self.assertAlmostEqual(result["macro_f1"], 0.5222222222)
        self.assertEqual(result["faults_called_normal"], 2)
        self.assertAlmostEqual(result["fault_to_normal_rate"], 0.5)

    def test_no_faults_has_no_fault_rate(self):
        result = classification_metrics([7, 7], [7, 7])
        self.assertIsNone(result["fault_to_normal_rate"])

    def test_invalid_class_is_rejected(self):
        with self.assertRaises(ValueError):
            classification_metrics([0], [12])

class TinyDataset(Dataset):
    def __init__(self, split="val"):
        self.split = split
        self.rows = [
            {"image_id": "a", "group_id": "g1",
             "image_path": "images/a.jpg", "label": "Cell"},
            {"image_id": "b", "group_id": "g2",
             "image_path": "images/b.jpg", "label": "No-Anomaly"},
        ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return (
            torch.zeros(1, 40, 24),
            CLASS_TO_INDEX[self.rows[index]["label"]],
        )


class ConstantModel(torch.nn.Module):
    def forward(self, images):
        scores = torch.zeros(len(images), len(CLASS_NAMES))
        scores[:, 0] = 5
        return scores


class ValidationPredictionTests(unittest.TestCase):
    def test_keeps_image_ids_and_true_labels(self):
        loader = DataLoader(TinyDataset(), batch_size=2, shuffle=False)
        records = collect_validation_predictions(ConstantModel(), loader, "cpu")

        self.assertEqual([row["image_id"] for row in records], ["a", "b"])
        self.assertEqual([row["true_index"] for row in records], [0, 7])
        self.assertEqual([row["predicted_index"] for row in records], [0, 0])
        self.assertEqual(len(records[0]["probabilities"]), 12)
        self.assertAlmostEqual(sum(records[0]["probabilities"]), 1.0)

    def test_rejects_test_loader(self):
        loader = DataLoader(TinyDataset(split="test"), batch_size=2)
        with self.assertRaisesRegex(ValueError, "validation loader"):
            collect_validation_predictions(ConstantModel(), loader, "cpu")


if __name__ == "__main__":
    unittest.main()
