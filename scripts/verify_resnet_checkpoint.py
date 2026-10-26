"""Verify a completed ResNet checkpoint against saved original-validation predictions.

Never construct a test loader. Writes a small verification record inside the
ignored run folder; it does not retrain or replace any saved predictions.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pv_fault.data import CLASS_NAMES
from pv_fault.evaluate import classification_metrics, collect_validation_predictions
from pv_fault.models.resnet import ResNet18Thermal, enable_layer4_finetuning
from train_baseline import make_loaders


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    run_dir = run_dir.resolve()
    if not run_dir.is_relative_to((ROOT / "runs").resolve()):
        raise ValueError("Choose a local run folder under runs/.")
    read = lambda name: json.loads((run_dir / name).read_text())
    plan, provenance, summary = map(read, ("experiment-plan.json", "provenance.json", "summary.json"))
    if summary["test_evaluated"] or provenance["test_evaluated"]:
        raise ValueError("Expected training/validation-only development evidence.")
    if file_hash(ROOT / plan["manifest"]) != plan["manifest_sha256"]:
        raise ValueError("Manifest checksum differs from the plan.")
    for name, expected in provenance["source_sha256"].items():
        snapshot = run_dir / "source_snapshot" / name
        if file_hash(snapshot) != expected or file_hash(ROOT / name) != expected:
            raise ValueError(f"Source snapshot/current source differs: {name}")
    if "parent_checkpoint" in plan and file_hash(ROOT / plan["parent_checkpoint"]) != plan["parent_checkpoint_sha256"]:
        raise ValueError("Parent checkpoint checksum differs from the plan.")
    torch.set_num_threads(plan.get("cpu_threads", 4))
    model = ResNet18Thermal(len(CLASS_NAMES), ROOT / plan["pretrained_weights"], plan["pretrained_sha256"])
    if plan["model"] == "resnet18_layer4":
        enable_layer4_finetuning(model)
    elif plan["model"] != "resnet18_frozen":
        raise ValueError("Unknown ResNet stage.")
    checkpoint_path = run_dir / "best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    for key in ("model", "manifest_sha256"):
        if checkpoint[key] != plan[key]:
            raise ValueError(f"Checkpoint metadata differs: {key}")
    if tuple(checkpoint["class_names"]) != CLASS_NAMES or checkpoint["epoch"] != summary["best_epoch"]:
        raise ValueError("Checkpoint class order/selected epoch differs.")
    model.load_state_dict(checkpoint["model_state_dict"])
    _, _, _, validation_loader = make_loaders(plan)
    records = collect_validation_predictions(model, validation_loader, "cpu")
    if records != read("best_validation_predictions.json"):
        raise ValueError("Reloaded checkpoint predictions differ from saved records.")
    metrics = classification_metrics([r["true_index"] for r in records], [r["predicted_index"] for r in records])
    if metrics != read("best_validation_metrics.json"):
        raise ValueError("Recomputed validation metrics differ from saved metrics.")
    result = {"checkpoint_predictions_verified": True, "validation_images": len(records),
              "best_epoch": checkpoint["epoch"], "checkpoint_sha256": file_hash(checkpoint_path),
              "sources_verified": True, "cpu_threads": torch.get_num_threads(),
              "test_evaluated": False}
    (run_dir / "checkpoint_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
