"""Old and new producer histories retain their measured validation cost."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "archive/scripts"))
from summarise_resnet_generalisation import validation_seconds


class ResultTimingTests(unittest.TestCase):
    def test_combined_legacy_and_missing_timings(self):
        old = {"validation_loss_seconds": 2., "validation_prediction_seconds": 3.}
        new = {"validation_seconds": 4.}
        self.assertEqual(validation_seconds([old, new]), 9.)
        self.assertIsNone(validation_seconds([old, {}]))
