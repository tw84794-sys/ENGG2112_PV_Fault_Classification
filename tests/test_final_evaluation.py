"""Final policies select on validation; committed choices and one-time test guards stay enforced."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from contextlib import ExitStack

import numpy as np
from torch.utils.data import DataLoader
from test_pipeline import ConstantModel, TinyDataset
from pv_fault.data import CLASS_NAMES
from pv_fault.evaluate import collect_predictions, collect_validation_predictions
from pv_fault.final_evaluation import group_intervals, screening_metrics, select_threshold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_final as workflow


def record(index, true, score, group=None):
    probs = [0.] * len(CLASS_NAMES)
    probs[0], probs[7] = score, 1 - score
    return {"image_id": str(index), "group_id": group or str(index), "true_index": true,
            "predicted_index": int(np.argmax(probs)), "probabilities": probs}


class FinalEvaluationTests(unittest.TestCase):
    def test_validation_collector_keeps_its_boundary(self):
        loader = DataLoader(TinyDataset("test"), batch_size=2)
        with self.assertRaisesRegex(ValueError, "validation loader"):
            collect_validation_predictions(ConstantModel(), loader, "cpu")
        self.assertEqual(len(collect_predictions(ConstantModel(), loader, "cpu", split="test")), 2)
        with self.assertRaises(ValueError):
            collect_predictions(ConstantModel(), loader, "cpu", split="train")

    def test_threshold_minimises_false_alarms_and_reports_infeasible_ties(self):
        rows = [record(0, 0, .9), record(1, 0, .8), record(2, 0, .7), record(3, 7, .6), record(4, 7, .2)]
        selected = select_threshold(rows, min_recall=2/3)
        self.assertAlmostEqual(selected["threshold"], .8)
        self.assertTrue(selected["targets_met"])
        tied = [record(0, 0, .9), record(1, 0, .7), record(2, 0, .7), record(3, 7, .8)]
        selected = select_threshold(tied, min_recall=1, max_false_alarm_rate=.2)
        self.assertAlmostEqual(selected["threshold"], .7)
        self.assertFalse(selected["targets_met"])
        self.assertEqual(selected["validation"]["missed_faults"], 0)
        with self.assertRaises(ValueError): select_threshold([record(0, 0, .9)])
        bad = record(0, 0, .9)
        bad["probabilities"][0] = float("nan")
        with self.assertRaises(ValueError): screening_metrics([bad], .5)

    def test_group_intervals_are_seeded_and_keep_perfect_accuracy(self):
        rows = [record(0, 0, .9, "a"), record(1, 0, .8, "a"), record(2, 7, .1, "b"), record(3, 7, .2, "c")]
        options = {"resamples": 30, "seed": 2112, "confidence": .95}
        interval = group_intervals(rows, .5, **options)
        self.assertEqual(interval, group_intervals(rows, .5, **options))
        self.assertEqual(interval["groups"], 3)
        self.assertEqual(interval["bounds"]["accuracy"]["lower"], 1)
        self.assertEqual(interval["bounds"]["accuracy"]["upper"], 1)

    def test_protocol_must_be_committed_and_code_cannot_change(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "protocol.json"
            metadata = root / "data/splits/infrared_solar_modules_v1.json"
            metadata.parent.mkdir(parents=True)
            metadata.write_text('{}')
            for name in workflow.CODE:
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("producer")
            protocol = {"class_names": CLASS_NAMES, "evaluation_split": "test", "dataset": {},
                        "environment": workflow.ENVIRONMENT,
                        "models": [{"name": name, "run_id": run} for name, run in workflow.SELECTED.items()],
                        "sources": {name: workflow.file_hash(root / name) for name in workflow.CODE}}
            workflow.write_new(path, protocol)
            committed = SimpleNamespace(returncode=0, stdout=path.read_bytes())
            with patch.object(workflow, "ROOT", root), patch.object(workflow, "PROTOCOL", path), patch.object(workflow, "MANIFEST", metadata.with_suffix('.csv')):
                with patch.object(workflow.subprocess, "run", return_value=SimpleNamespace(returncode=1, stdout=b"")):
                    with self.assertRaisesRegex(ValueError, "Commit"):
                        workflow.committed_protocol()
                with patch.object(workflow.subprocess, "run", return_value=committed):
                    self.assertEqual(workflow.committed_protocol()["evaluation_split"], "test")
                    (root / workflow.CODE[0]).write_text("changed")
                    with self.assertRaisesRegex(ValueError, "code changed"):
                        workflow.committed_protocol()
            with self.assertRaises(FileExistsError): workflow.write_new(path, protocol)

    def test_repeat_test_attempt_is_rejected_before_loading_images(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(workflow, "RUN", Path(folder)), patch.object(workflow, "committed_protocol", return_value={}), patch.object(workflow, "load_model") as load:
                with self.assertRaises(FileExistsError): workflow.evaluate_test()
                load.assert_not_called()

    def test_complete_workflow_on_synthetic_data_writes_once(self):
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            root = Path(folder)
            paths = {"ROOT": root, "RUN": root / "run", "RESULTS": root / "results.json",
                     "REPORT": root / "report.md", "FIGURE": root / "figure.png", "PROTOCOL": root / "protocol.json"}
            for name, value in paths.items(): stack.enter_context(patch.object(workflow, name, value))
            protocol = {"cpu_threads": 1, "batch_size": 2, "majority_class": "No-Anomaly",
                        "limit": "Synthetic fixture only.", "screening_targets": {"min_recall": .9, "max_false_alarm_rate": .2},
                        "bootstrap": {"resamples": 30, "seed": 2112, "confidence": .95},
                        "models": [{"name": name, "screening": {"threshold": .5}} for name in workflow.SELECTED]}
            workflow.write_new(paths["PROTOCOL"], protocol)
            stack.enter_context(patch.object(workflow, "committed_protocol", return_value=protocol))
            stack.enter_context(patch.object(workflow, "verify_dataset"))
            load = stack.enter_context(patch.object(workflow, "load_model", side_effect=lambda spec: ConstantModel()))
            stack.enter_context(patch.object(workflow, "SolarModuleDataset", side_effect=lambda split: TinyDataset(split)))
            original_output = workflow.subprocess.check_output
            stack.enter_context(patch.object(workflow.subprocess, "check_output", side_effect=
                lambda args, **kwargs: b"fixture" if args[:2] == ["git", "rev-parse"] else original_output(args, **kwargs)))
            workflow.evaluate_test()
            result = json.loads(paths["RESULTS"].read_text())
            self.assertEqual(result["images"], 2)
            self.assertEqual(set(result["models"]), {"Majority baseline", "SmallCNN", "ResNet18"})
            self.assertEqual(result["models"]["Majority baseline"]["multiclass"]["accuracy"], .5)
            self.assertTrue(paths["FIGURE"].is_file())
            self.assertTrue((paths["RUN"] / "completed.json").is_file())
            with self.assertRaises(FileExistsError): workflow.evaluate_test()
            self.assertEqual(load.call_count, 2)
