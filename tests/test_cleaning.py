"""Cleaning preserves exclusions and transitive groups without copying images."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import csv
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


class CleaningTests(unittest.TestCase):
    def test_conflicts_duplicates_and_transitive_groups(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "scripts").mkdir()
            (root / "data/raw/images").mkdir(parents=True)
            metadata = {}
            cases = [(10, "Cell"), (10, "Cell"), (30, "Cell"), (30, "Diode"),
                     (100, "Cell"), (103, "Cell"), (106, "Cell"),
                     (200, "Cell"), (202, "Diode")]
            for index, (brightness, label) in enumerate(cases):
                image_path = f"images/{index}.png"
                Image.new("L", (24, 40), brightness).save(root / "data/raw" / image_path)
                metadata[str(index)] = {"image_filepath": image_path, "anomaly_class": label}
            (root / "data/raw/module_metadata.json").write_text(json.dumps(metadata))
            for name in ("audit_data.py", "review_near_duplicates.py"):
                shutil.copy2(ROOT / "scripts" / name, root / "scripts" / name)
                subprocess.run([sys.executable, str(root / "scripts" / name)],
                               check=True, capture_output=True, text=True)
            clean = root / "data/processed/near_clean"
            retained = json.loads((clean / "module_metadata.json").read_text())
            self.assertEqual(set(retained), {"0", "4", "5", "6"})
            with (clean / "related_groups.csv").open() as source:
                groups = {row["image_id"]: row for row in csv.DictReader(source)}
            self.assertEqual({groups[i]["group_id"] for i in ("4", "5", "6")}, {"4"})
            self.assertTrue(all(groups[i]["group_size"] == "3" for i in ("4", "5", "6")))
            self.assertFalse((root / "data/processed/images").exists())
            self.assertFalse((clean / "images").exists())
            self.assertEqual(len(list((root / "data/raw/images").glob("*.png"))), 9)
