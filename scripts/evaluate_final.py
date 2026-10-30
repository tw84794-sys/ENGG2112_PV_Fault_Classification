"""Freeze validation-selected choices, then evaluate them once on the reserved test split."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pv_fault.data import CLASS_NAMES, CLASS_TO_INDEX, SolarModuleDataset
from pv_fault.evaluate import classification_metrics, collect_predictions, collect_validation_predictions
from pv_fault.final_evaluation import group_intervals, screening_metrics, select_threshold
from pv_fault.models.baseline import SmallCNN
from pv_fault.models.resnet import ResNet18Thermal
from pv_fault.resnet_spatial import configure_spatial_variant
from pv_fault.train import file_hash

PROTOCOL = ROOT / "configs/final-evaluation-v1.json"
RESULTS = ROOT / "results/final-evaluation-v1.json"
REPORT = RESULTS.with_suffix(".md")
FIGURE = ROOT / "results/final-evaluation-v1-confusion.png"
RUN = ROOT / "runs/final-evaluation-v1"
MANIFEST = ROOT / "data/splits/infrared_solar_modules_v1.csv"
SELECTED = {"SmallCNN": "smallcnn-depth-gentle60-s2112-v1",
            "ResNet18": "resnet18-generalisation-augmentation-s2112-v1"}
CODE = ("scripts/evaluate_final.py", "scripts/create_split.py", "src/pv_fault/data.py",
        "src/pv_fault/evaluate.py", "src/pv_fault/final_evaluation.py", "src/pv_fault/train.py",
        "src/pv_fault/models/baseline.py", "src/pv_fault/models/resnet.py", "src/pv_fault/resnet_spatial.py",
        "src/pv_fault/plots.py")
ENVIRONMENT = {"python": sys.version, **{name: version(name) for name in ("torch", "torchvision", "numpy", "Pillow")}}


def write_new(path, value):
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as target:
        target.write(text)


def verify_dataset():
    subprocess.run([sys.executable, str(ROOT / "scripts/create_split.py"), "--verify-only"],
                   cwd=ROOT, check=True)


def model_spec(name, run_id):
    folder = ROOT / "runs" / run_id
    plan = json.loads((folder / "experiment-plan.json").read_text())
    provenance = json.loads((folder / "provenance.json").read_text())
    summary = json.loads((folder / "summary.json").read_text())
    if provenance["test_evaluated"] or summary["test_evaluated"]:
        raise ValueError("Expected validation-only producer records.")
    if provenance["manifest_sha256"] != file_hash(MANIFEST) or tuple(provenance["class_names"]) != CLASS_NAMES:
        raise ValueError("Producer uses a different manifest or class order.")
    for source, expected in provenance["source_sha256"].items():
        if file_hash(folder / "source_snapshot" / source) != expected:
            raise ValueError(f"Archived producer source changed: {source}")
    if file_hash(folder / "experiment-plan.json") != provenance["plan_sha256"]:
        raise ValueError("Producer plan fingerprint changed.")
    for path_key, hash_key in (("pretrained_weights", "pretrained_sha256"), ("parent_checkpoint", "parent_checkpoint_sha256")):
        if path_key in plan and file_hash(ROOT / plan[path_key]) != plan[hash_key]:
            raise ValueError(f"Producer dependency fingerprint changed: {path_key}")
    inputs = [folder / filename for filename in ("experiment-plan.json", "provenance.json", "summary.json",
              "best_model.pt", "best_validation_predictions.json", "best_validation_metrics.json")]
    inputs += [ROOT / plan[key] for key in ("pretrained_weights", "parent_checkpoint") if key in plan]
    return {"name": name, "run_id": run_id,
            "files": {str(path.relative_to(ROOT)): file_hash(path) for path in inputs}}


def load_model(spec):
    for name, expected in spec["files"].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError(f"Frozen model input changed: {name}")
    folder = ROOT / "runs" / spec["run_id"]
    plan = json.loads((folder / "experiment-plan.json").read_text())
    checkpoint = torch.load(folder / "best_model.pt", map_location="cpu", weights_only=True)
    summary = json.loads((folder / "summary.json").read_text())
    if (tuple(checkpoint["class_names"]) != CLASS_NAMES or checkpoint["manifest_sha256"] != file_hash(MANIFEST)
            or checkpoint["model"] != plan["model"] or checkpoint["epoch"] != summary["best_epoch"]):
        raise ValueError("Checkpoint metadata differs from the recorded model.")
    if spec["name"] == "SmallCNN" and plan["model"] == "small_cnn":
        options = {key: plan.get(key, default) for key, default in
                   (("batch_norm", False), ("activation", "relu"), ("extra_conv", False))}
        if any(checkpoint.get(key, default) != options[key] for key, default in
               (("batch_norm", False), ("activation", "relu"), ("extra_conv", False))):
            raise ValueError("CNN checkpoint architecture differs from its plan.")
        model = SmallCNN(len(CLASS_NAMES), **options)
    elif spec["name"] == "ResNet18" and plan["model"] == "resnet18_layer4":
        variant = plan.get("spatial_variant", "standard")
        if variant != "standard" or checkpoint.get("spatial_variant", "standard") != variant:
            raise ValueError("The frozen choice is the standard augmented ResNet.")
        model = ResNet18Thermal(len(CLASS_NAMES), ROOT / plan["pretrained_weights"], plan["pretrained_sha256"])
        configure_spatial_variant(model, variant)
    else:
        raise ValueError("Unexpected final model.")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.eval()


def freeze():
    if PROTOCOL.exists() or RUN.exists() or RESULTS.exists():
        raise FileExistsError("Final evaluation already frozen or started; preserve the existing record.")
    torch.set_num_threads(4)
    verify_dataset()
    validation = SolarModuleDataset("val")
    loader = DataLoader(validation, batch_size=64, shuffle=False)
    specifications = []
    for name, run_id in SELECTED.items():
        spec = model_spec(name, run_id)
        records = collect_validation_predictions(load_model(spec), loader, "cpu")
        folder = ROOT / "runs" / run_id
        if records != json.loads((folder / "best_validation_predictions.json").read_text()):
            raise ValueError("Current inference does not reproduce every saved validation probability.")
        metrics = classification_metrics([row["true_index"] for row in records],
                                         [row["predicted_index"] for row in records])
        if metrics != json.loads((folder / "best_validation_metrics.json").read_text()):
            raise ValueError("Saved validation metrics disagree with checkpoint predictions.")
        spec.update(screening=select_threshold(records), validation_macro_f1=metrics["macro_f1"])
        specifications.append(spec)
    with MANIFEST.open(newline="") as source:
        training_labels = Counter(row["label"] for row in csv.DictReader(source) if row["split"] == "train")
    majority = max(CLASS_NAMES, key=lambda label: (training_labels[label], -CLASS_TO_INDEX[label]))
    protocol = {"version": 1, "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "selection_split": "val", "evaluation_split": "test", "class_names": CLASS_NAMES,
        "dataset": json.loads(MANIFEST.with_suffix(".json").read_text()),
        "sources": {name: file_hash(ROOT / name) for name in CODE},
        "cpu_threads": 4, "batch_size": 64, "models": specifications, "environment": ENVIRONMENT,
        "majority_class": majority, "multiclass_policy": "Highest class score (argmax), unchanged by screening.",
        "screening_policy": "Flag anomaly when 1 - P(No-Anomaly) >= the frozen model-specific threshold.",
        "threshold_rule": "Highest validation threshold with at least 90% anomaly recall; this minimises false alarms. Report whether false alarms also remain at most 20%.",
        "screening_targets": {"min_recall": .9, "max_false_alarm_rate": .2},
        "bootstrap": {"resamples": 1000, "seed": 2112, "confidence": .95},
        "limit": "Group intervals use detected related-image groups, not verified sites. Test results will not select models, thresholds or settings."}
    write_new(PROTOCOL, protocol)
    print(f"Frozen choices written to {PROTOCOL}. Commit this protocol and its code before test inference.")
    for spec in specifications:
        print(spec["name"], json.dumps(spec["screening"]))


def committed_protocol():
    content = PROTOCOL.read_bytes()
    recorded = subprocess.run(["git", "show", "HEAD:configs/final-evaluation-v1.json"],
                              cwd=ROOT, capture_output=True, check=False)
    if recorded.returncode or recorded.stdout != content:
        raise ValueError("Commit the unchanged final protocol before opening the test split.")
    protocol = json.loads(content)
    if tuple(protocol["class_names"]) != CLASS_NAMES or protocol["evaluation_split"] != "test":
        raise ValueError("Unexpected final evaluation contract.")
    if (protocol["environment"] != ENVIRONMENT or set(protocol["sources"]) != set(CODE)
            or {spec["name"]: spec["run_id"] for spec in protocol["models"]} != SELECTED):
        raise ValueError("Frozen models, sources or environment changed.")
    for name, expected in protocol["sources"].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError(f"Frozen evaluation code changed: {name}")
    if json.loads(MANIFEST.with_suffix(".json").read_text()) != protocol["dataset"]:
        raise ValueError("Frozen dataset record changed.")
    return protocol


def write_report(result):
    lines = ["# Final reserved-test evaluation", "", "All models, screening thresholds and bootstrap settings were [frozen](../configs/final-evaluation-v1.json) before test inference. These results do not select new settings.", "",
             "| Model | Accuracy | Macro-F1 | Balanced accuracy | Argmax missed faults | Argmax normal false alarms | Screening recall | Screening false-alarm rate |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, record in result["models"].items():
        m, b = record["multiclass"], record["screening"]
        lines.append(f"| {name} | {m['accuracy']:.2%} | {m['macro_f1']:.4f} | {m['balanced_accuracy']:.2%} | {m['faults_called_normal']} | {m['normal_called_fault']} | {b['recall']:.2%} | {b['false_alarm_rate']:.2%} |")
    lines += ["", "Multiclass predictions use argmax. Screening is a separate binary decision using thresholds selected solely on validation. The majority baseline predicts the most frequent training class and has no fitted threshold.", "",
              "## Frozen screening decisions", "",
              "| Model | Threshold | Missed faults | Normal false alarms | Recall | False-alarm rate | Both test targets met |",
              "|---|---:|---:|---:|---:|---:|---|"]
    for name, record in result["models"].items():
        b = record["screening"]
        cutoff = "constant class" if name == "Majority baseline" else f"{b['threshold']:.6f}"
        lines.append(f"| {name} | {cutoff} | {b['missed_faults']} | {b['normal_false_alarms']} | {b['recall']:.2%} | {b['false_alarm_rate']:.2%} | {record['screening_targets_met']} |")
    lines += ["", "Targets are at least 90% anomaly recall and at most 20% normal false alarms. They are research goals from the proposal, not safety standards. No threshold is changed after inspecting test results.", "",
              "## Uncertainty and limitations", "",
              "| Model | Macro-F1 95% interval | Screening recall 95% interval | False-alarm rate 95% interval |",
              "|---|---|---|---|"]
    for name, record in result["models"].items():
        bounds = record["intervals"]["bounds"]
        fields = [f"{bounds[key]['lower']:.4f}–{bounds[key]['upper']:.4f}" for key in ("macro_f1", "screening_recall", "screening_false_alarm_rate")]
        lines.append("| " + name + " | " + " | ".join(fields) + " |")
    lines += ["", "The JSON retains per-class precision, recall, F1, supports, confusion matrices and 95% group-bootstrap intervals. Groups reflect detected visual relationships, not independently sampled sites. These intervals do not establish unseen-site generalisation. The models were selected through repeated validation experiments, mainly with one seed. Predictions cannot certify panel safety or reuse suitability.", "",
              "![Reserved-test confusion matrices](final-evaluation-v1-confusion.png)", "",
              f"Frozen protocol SHA-256: `{result['protocol_sha256']}`. Evaluation code commit: `{result['evaluation_commit']}`.", "",
              "Full image-level predictions remain in the ignored `runs/final-evaluation-v1/` folder, with their hashes in the JSON. The original validation records remain unchanged."]
    with REPORT.open("x", encoding="utf-8") as target:
        target.write("\n".join(lines) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pv_fault.plots import confusion_matrix
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), layout="constrained")
    for ax, name in zip(axes, SELECTED):
        confusion_matrix(result["models"][name]["multiclass"], f"{name}: reserved test", ax=ax)
    fig.savefig(FIGURE, dpi=160, bbox_inches="tight")
    plt.close(fig)


def evaluate_test():
    protocol = committed_protocol()
    if RUN.exists() or any(path.exists() for path in (RESULTS, REPORT, FIGURE)):
        raise FileExistsError("Test evaluation already started or completed. Existing records will not be overwritten.")
    torch.set_num_threads(protocol["cpu_threads"])
    verify_dataset()
    models = [(spec, load_model(spec)) for spec in protocol["models"]]
    RUN.mkdir(parents=True, exist_ok=False)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    write_new(RUN / "started.json", {"protocol_sha256": file_hash(PROTOCOL), "commit": commit,
                                   "started_utc": datetime.now(timezone.utc).isoformat()})
    dataset = SolarModuleDataset("test")
    loader = DataLoader(dataset, batch_size=protocol["batch_size"], shuffle=False)
    majority = CLASS_TO_INDEX[protocol["majority_class"]]
    baseline = [{"image_id": row["image_id"], "group_id": row["group_id"], "image_path": row["image_path"],
                 "true_index": CLASS_TO_INDEX[row["label"]], "predicted_index": majority,
                 "probabilities": [float(index == majority) for index in range(len(CLASS_NAMES))]}
                for row in dataset.rows]
    all_predictions = [("Majority baseline", baseline, .5)]
    for spec, model in models:
        predictions = collect_predictions(model, loader, "cpu", split="test")
        all_predictions.append((spec["name"], predictions, spec["screening"]["threshold"]))
    result = {"protocol_sha256": file_hash(PROTOCOL), "evaluation_commit": commit,
              "completed_utc": datetime.now(timezone.utc).isoformat(), "split": "test",
              "test_evaluated": True, "images": len(dataset), "models": {}, "limitations": protocol["limit"]}
    for name, predictions, threshold in all_predictions:
        binary = screening_metrics(predictions, threshold)
        targets = protocol["screening_targets"]
        path = RUN / (name.replace(" ", "_") + "_predictions.json")
        write_new(path, predictions)
        result["models"][name] = {"multiclass": classification_metrics(
            [row["true_index"] for row in predictions], [row["predicted_index"] for row in predictions]),
            "screening": binary,
            "screening_targets_met": binary["recall"] >= targets["min_recall"]
                                     and binary["false_alarm_rate"] <= targets["max_false_alarm_rate"],
            "intervals": group_intervals(predictions, threshold, **protocol["bootstrap"]),
            "predictions_sha256": file_hash(path)}
    write_new(RESULTS, result)
    write_report(result)
    write_new(RUN / "completed.json", {"results_sha256": file_hash(RESULTS), "test_evaluated": True})
    print(f"Completed reserved-test evaluation: {RESULTS}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "test"))
    args = parser.parse_args()
    freeze() if args.action == "freeze" else evaluate_test()
