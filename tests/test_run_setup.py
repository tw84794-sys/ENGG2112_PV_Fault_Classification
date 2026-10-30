"""Shared setup retains run protection, seeds, hashes and producer snapshots."""
import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader
from test_pipeline import TinyDataset
from pv_fault import train


class RunSetupTests(unittest.TestCase):
    def test_recorded_paths_resolve_current_and_archived_recipes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            current = root / "configs/plan.json"
            archived = root / "archive/configs/plan.json"
            archived.parent.mkdir(parents=True)
            archived.write_text("archived")
            self.assertEqual(train.resolve_plan(current), archived)
            self.assertEqual(train.file_hash(current), hashlib.sha256(b"archived").hexdigest())
            current.parent.mkdir()
            current.write_text("current")
            self.assertEqual(train.resolve_plan(current), current)
            self.assertEqual(train.resolve_plan(root / "missing.pt"), root / "missing.pt")

    def test_new_run_records_sources_and_rejects_overwrite(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(train, "ROOT", Path(folder)):
            root = Path(folder)
            plan = {"run_id": "tiny", "seed": 2112, "device": "cpu", "cpu_threads": 1,
                    "manifest_sha256": "frozen", "model": "tiny"}
            (root / "source.py").write_text("# producer\n")
            path = root / "plan.json"
            path.write_text(json.dumps(plan))
            run_dir = train.prepare_run(plan, True)
            first = torch.rand(1)
            train.prepare_run(plan, False)
            self.assertTrue(torch.equal(first, torch.rand(1)))
            model = nn.Sequential(nn.Flatten(), nn.Linear(960, 12))
            metadata = train.record_run(run_dir, path, plan, model, TinyDataset("train"),
                                        TinyDataset(), None, ["source.py"], parent_model="frozen")
            saved = json.loads((run_dir / "provenance.json").read_text())
            self.assertEqual(saved["source_sha256"]["source.py"], train.file_hash(root / "source.py"))
            self.assertEqual((run_dir / "source_snapshot/source.py").read_bytes(), (root / "source.py").read_bytes())
            self.assertEqual(saved["parent_model"], "frozen")
            self.assertEqual(metadata["model"], "tiny")
            with self.assertRaises(FileExistsError): train.prepare_run(plan, True)
            with self.assertRaises(FileExistsError):
                train.record_run(run_dir, path, plan, model, TinyDataset("train"), TinyDataset(), None, [])
            optimizer = torch.optim.SGD(model.parameters(), lr=.01)
            result = train.smoke_check(model, DataLoader(TinyDataset("train")),
                DataLoader(TinyDataset()), optimizer, nn.CrossEntropyLoss(), nn.CrossEntropyLoss(), time.monotonic())
            self.assertEqual(result["train_smoke"]["count"], 2)
            self.assertEqual(result["validation_smoke"]["count"], 2)
            plan["run_id"] = "../outside"
            with self.assertRaises(ValueError): train.prepare_run(plan, True)
