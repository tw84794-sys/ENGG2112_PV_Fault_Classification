"""Checks for the tracked, frozen image split."""

import csv
import unittest
from collections import defaultdict
from pathlib import Path


MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "data/splits/infrared_solar_modules_v1.csv"
)


class DataContractTests(unittest.TestCase):
    def test_related_images_never_cross_splits(self):
        with MANIFEST.open(newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))

        self.assertEqual(len(rows), 19922)
        self.assertEqual(len({row["image_id"] for row in rows}), len(rows))

        group_splits = defaultdict(set)
        class_splits = defaultdict(set)
        for row in rows:
            self.assertIn(row["split"], {"train", "val", "test"})
            group_splits[row["group_id"]].add(row["split"])
            class_splits[row["label"]].add(row["split"])

        self.assertTrue(all(len(splits) == 1 for splits in group_splits.values()))
        self.assertTrue(all(splits == {"train", "val", "test"}
                            for splits in class_splits.values()))
        self.assertEqual(len(class_splits), 12)
