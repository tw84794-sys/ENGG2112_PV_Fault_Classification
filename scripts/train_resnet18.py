"""Run a bounded frozen-backbone ResNet18 trial on train/validation only."""

import argparse
import json
import platform
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import __version__ as pillow_version
from torch import nn

from pv_fault.data import CLASS_NAMES
from pv_fault.evaluate import classification_metrics, collect_validation_predictions
from pv_fault.models.resnet import (
    IMAGENET_MEAN, IMAGENET_STD, RESIZED_SIZE, WEIGHT_SHA256, WEIGHT_URL,
    ResNet18Thermal,
)
from pv_fault.train import evaluate_one_epoch, train_one_epoch
from train_baseline import (
    ROOT, file_hash, make_loaders, make_training_class_weights,
    majority_baseline, save_json,
)


DEFAULT_PLAN = ROOT / "configs/experiment-plan-resnet18.json"


def load_plan(path):
    plan = json.loads(path.read_text(encoding="utf-8"))
    manifest = ROOT / plan["manifest"]
    weights = ROOT / plan["pretrained_weights"]
    if file_hash(manifest) != plan["manifest_sha256"]:
        raise ValueError("Frozen manifest differs from the experiment plan.")
    if plan["pretrained_sha256"] != WEIGHT_SHA256:
        raise ValueError("Plan does not identify the expected ResNet18 weights.")
    if file_hash(weights) != plan["pretrained_sha256"]:
        raise ValueError("Pretrained ResNet18 weights differ from the plan.")
    if plan["model"] != "resnet18_frozen" or plan["device"] != "cpu":
        raise ValueError("This plan is only for frozen ResNet18 on local CPU.")
    if plan["paid_compute_cost"] != 0 or plan["max_trials"] != 1:
        raise ValueError("This plan allows one local, unpaid full trial.")
    if not 1 <= plan["max_epochs"] <= 15:
        raise ValueError("Epoch limit must be between 1 and 15.")
    if not 0 < plan["max_elapsed_seconds"] <= 600:
        raise ValueError("Time limit must be at most 600 seconds.")
    if plan["batch_size"] < 1 or plan["learning_rate"] <= 0:
        raise ValueError("Batch size and learning rate must be positive.")
    if plan["early_stopping_patience"] < 1:
        raise ValueError("Early-stopping patience must be positive.")
    if plan["class_weighting"] != "sqrt_inverse_frequency":
        raise ValueError("The planned comparison uses train-only class weights.")
    if plan["selection_metric"] != "validation_macro_f1":
        raise ValueError("Checkpoint selection must use validation macro-F1.")
    return plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    plan_path = args.plan if args.plan.is_absolute() else ROOT / args.plan
    plan = load_plan(plan_path)

    seed = plan["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    run_id = plan["run_id"]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", run_id):
        raise ValueError("run_id must be a simple folder name.")
    run_dir = ROOT / "runs" / run_id
    if args.full and run_dir.exists():
        raise FileExistsError(f"Existing run will not be overwritten: {run_dir}")

    started = time.monotonic()
    train_data, val_data, train_loader, val_loader = make_loaders(plan)
    majority_class, majority_metrics = majority_baseline(train_data, val_data)
    model = ResNet18Thermal(
        num_classes=len(CLASS_NAMES),
        weights_path=ROOT / plan["pretrained_weights"],
        expected_sha256=plan["pretrained_sha256"],
    ).to("cpu")
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    class_weights = make_training_class_weights(train_data)
    training_criterion = nn.CrossEntropyLoss(weight=class_weights)
    validation_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=plan["learning_rate"],
    )

    if not args.full:
        train_started = time.monotonic()
        train_smoke = train_one_epoch(
            model, train_loader, optimizer, training_criterion, "cpu",
            max_batches=2,
        )
        train_seconds = time.monotonic() - train_started
        validation_started = time.monotonic()
        val_smoke = evaluate_one_epoch(
            model, val_loader, validation_criterion, "cpu", max_batches=2,
        )
        validation_seconds = time.monotonic() - validation_started
        print(json.dumps({
            "model": plan["model"],
            "trainable_parameters": trainable_parameters,
            "majority_class": majority_class,
            "majority_validation_macro_f1": majority_metrics["macro_f1"],
            "train_smoke": train_smoke,
            "train_smoke_seconds": train_seconds,
            "validation_smoke": val_smoke,
            "validation_smoke_seconds": validation_seconds,
            "total_smoke_seconds": time.monotonic() - started,
        }, indent=2))
        return

    run_dir.mkdir(parents=True, exist_ok=False)
    save_json(run_dir / "experiment-plan.json", plan)
    save_json(run_dir / "majority_validation_metrics.json", majority_metrics)
    source_files = (
        "scripts/train_resnet18.py",
        "scripts/train_baseline.py",
        "src/pv_fault/models/resnet.py",
        "src/pv_fault/data.py",
        "src/pv_fault/train.py",
        "src/pv_fault/evaluate.py",
    )
    save_json(run_dir / "provenance.json", {
        "manifest_sha256": plan["manifest_sha256"],
        "plan_sha256": file_hash(plan_path),
        "source_sha256": {
            name: file_hash(ROOT / name) for name in source_files
        },
        "model": plan["model"],
        "trainable_parameters": trainable_parameters,
        "class_names": CLASS_NAMES,
        "pretrained_weights_url": WEIGHT_URL,
        "pretrained_weights_sha256": plan["pretrained_sha256"],
        "pretrained_weight_rights": (
            "TorchVision warns pretrained weights may have dataset-derived terms; "
            "commercial deployment rights have not been verified."
        ),
        "preprocessing": {
            "input": "grayscale float32 [1,40,24] in [0,1]",
            "resize": list(RESIZED_SIZE),
            "repeat_channels": 3,
            "normalization_mean": IMAGENET_MEAN,
            "normalization_std": IMAGENET_STD,
        },
        "class_weighting": plan["class_weighting"],
        "train_class_counts": dict(
            Counter(row["label"] for row in train_data.rows)
        ),
        "class_weights": class_weights.tolist(),
        "train_images": len(train_data),
        "validation_images": len(val_data),
        "seed": seed,
        "python": sys.version,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "pillow": pillow_version,
        "platform": platform.platform(),
        "device": "cpu",
        "test_evaluated": False,
    })

    history = []
    best_score = -1.0
    best_epoch = None
    epochs_without_improvement = 0
    stop_reason = "max_epochs"
    for epoch in range(1, plan["max_epochs"] + 1):
        if time.monotonic() - started >= plan["max_elapsed_seconds"]:
            stop_reason = "time_limit_before_epoch"
            break
        train_stats = train_one_epoch(
            model, train_loader, optimizer, training_criterion, "cpu"
        )
        val_stats = evaluate_one_epoch(
            model, val_loader, validation_criterion, "cpu"
        )
        records = collect_validation_predictions(model, val_loader, "cpu")
        metrics = classification_metrics(
            [row["true_index"] for row in records],
            [row["predicted_index"] for row in records],
        )
        if abs(metrics["accuracy"] - val_stats["accuracy"]) > 1e-9:
            raise ValueError("Validation metrics disagree with saved predictions.")
        elapsed = time.monotonic() - started
        result = {
            "epoch": epoch,
            "train": train_stats,
            "validation_loss": val_stats["loss"],
            "validation_accuracy": metrics["accuracy"],
            "validation_macro_f1": metrics["macro_f1"],
            "fault_to_normal_rate": metrics["fault_to_normal_rate"],
            "normal_false_alarm_rate": metrics["normal_false_alarm_rate"],
            "elapsed_seconds": elapsed,
        }
        history.append(result)
        save_json(run_dir / "history.json", history)
        print(json.dumps(result), flush=True)
        if metrics["macro_f1"] > best_score:
            best_score = metrics["macro_f1"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "model": plan["model"],
                "epoch": epoch,
                "class_names": CLASS_NAMES,
                "manifest_sha256": plan["manifest_sha256"],
                "pretrained_sha256": plan["pretrained_sha256"],
            }, run_dir / "best_model.pt")
            save_json(run_dir / "best_validation_metrics.json", metrics)
            save_json(run_dir / "best_validation_predictions.json", records)
        else:
            epochs_without_improvement += 1
        if elapsed >= plan["max_elapsed_seconds"]:
            stop_reason = "time_limit_after_epoch"
            break
        if epochs_without_improvement >= plan["early_stopping_patience"]:
            stop_reason = "early_stopping"
            break

    save_json(run_dir / "summary.json", {
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_score if best_epoch else None,
        "epochs_completed": len(history),
        "stop_reason": stop_reason,
        "elapsed_seconds": time.monotonic() - started,
        "test_evaluated": False,
    })
    print(f"Saved run to {run_dir}")


if __name__ == "__main__":
    main()
