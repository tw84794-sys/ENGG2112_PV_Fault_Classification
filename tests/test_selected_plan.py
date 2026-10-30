"""The compatibility command keeps the original CV-selection safeguards."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import finetune_resnet18 as runner


class SelectedPlanTests(unittest.TestCase):
    def test_selected_plan_requires_winning_cv_and_original_limits(self):
        plan = json.loads((ROOT / "archive/configs/experiment-plan-resnet18-selected.json").read_text())
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            evidence = root / plan["cv_summary"]
            evidence.parent.mkdir(parents=True)
            evidence.write_text(json.dumps({
                "test_evaluated": False, "fold_images": [4649, 4649, 4648],
                "aggregate": {"paper_informed_lr": {"macro_f1": {"mean": .6}},
                              "current_lr": {"macro_f1": {"mean": .5}}}}))
            path = root / "plan.json"
            hashes = {root / plan[p]: plan[h] for p, h in (
                ("manifest", "manifest_sha256"), ("parent_checkpoint", "parent_checkpoint_sha256"),
                ("pretrained_weights", "pretrained_sha256"), ("cv_summary", "cv_summary_sha256"))}
            with patch.object(runner, "ROOT", root), patch.object(runner, "file_hash", side_effect=hashes.__getitem__):
                path.write_text(json.dumps(plan))
                self.assertEqual(runner.load_plan(path), plan)
                plan["max_epochs"] = 7
                path.write_text(json.dumps(plan))
                with self.assertRaisesRegex(ValueError, "original protocol"):
                    runner.load_plan(path)
                plan["max_epochs"] = 6
                path.write_text(json.dumps(plan))
                cv = json.loads(evidence.read_text())
                cv["aggregate"]["paper_informed_lr"]["macro_f1"]["mean"] = .4
                evidence.write_text(json.dumps(cv))
                with self.assertRaisesRegex(ValueError, "did not win"):
                    runner.load_plan(path)
