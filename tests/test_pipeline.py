"""Checks for training and evaluation pipeline behaviour."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pv_fault.evaluate import classification_metrics


class PipelineContractTests(unittest.TestCase):
    @unittest.skip("Pending model and checkpoint implementation")
    def test_checkpoint_inference_matches_saved_predictions(self):
        pass


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


if __name__ == "__main__":
    unittest.main()
