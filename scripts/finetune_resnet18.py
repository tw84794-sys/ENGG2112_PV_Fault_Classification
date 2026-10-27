"""Fine-tune ResNet18 layer 4 using training and validation only."""

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

from pv_fault.augmentation import ThermalTrainAugmentation
from pv_fault.data import CLASS_NAMES
from pv_fault.finetuning import configure_resnet_finetuning
from pv_fault.evaluate import (
    classification_metrics,
    collect_validation_predictions,
)
from pv_fault.models.resnet import (
    ResNet18Thermal,
    WEIGHT_SHA256,
)
from pv_fault.train import evaluate_one_epoch, train_one_epoch
from train_baseline import (
    ROOT,
    file_hash,
    make_loaders,
    make_training_class_weights,
    make_scheduler,
    save_json,
)


PLAN_PATH = ROOT / "configs/experiment-plan-resnet18-finetune.json"


def load_plan(path):
    plan = json.loads(path.read_text(encoding="utf-8"))

    if file_hash(ROOT / plan["manifest"]) != plan["manifest_sha256"]:
        raise ValueError("The frozen split manifest has changed.")
    if file_hash(ROOT / plan["parent_checkpoint"]) != plan["parent_checkpoint_sha256"]:
        raise ValueError("The frozen ResNet18 checkpoint has changed.")
    if plan["pretrained_sha256"] != WEIGHT_SHA256:
        raise ValueError("The plan names unexpected pretrained weights.")
    if file_hash(ROOT / plan["pretrained_weights"]) != plan["pretrained_sha256"]:
        raise ValueError("The pretrained weights have changed.")

    if plan["model"] not in ("resnet18_layer4", "resnet18_layer34") or plan["device"] != "cpu":
        raise ValueError("This plan must use late-stage ResNet fine-tuning on local CPU.")
    if plan["max_trials"] != 1 or plan["paid_compute_cost"] != 0:
        raise ValueError("This plan allows one local, unpaid full trial.")
    if not 1 <= plan["max_epochs"] <= 12:
        raise ValueError("The epoch limit must be between 1 and 12.")
    if not 0 < plan["max_elapsed_seconds"] <= 1200:
        raise ValueError("The time limit must be at most 1200 seconds.")
    if plan["early_stopping_patience"] < 1:
        raise ValueError("Early-stopping patience must be positive.")
    if plan["batch_size"] < 1:
        raise ValueError("Batch size must be positive.")
    if plan["head_learning_rate"] <= 0 or plan["layer4_learning_rate"] <= 0:
        raise ValueError("Learning rates must be positive.")
    stages = plan.get("trainable_stages", ["layer4"])
    expected_stages = ["layer3", "layer4"] if plan["model"] == "resnet18_layer34" else ["layer4"]
    if stages != expected_stages:
        raise ValueError("Trainable stages differ from the declared model protocol.")
    if "layer3" in stages and not 0 < plan["layer3_learning_rate"] <= plan["layer4_learning_rate"]:
        raise ValueError("Layer3 must have a positive rate no larger than layer4.")
    if plan.get("optimizer", "adam") not in ("adam", "adamw"):
        raise ValueError("Choose Adam or AdamW.")
    if plan.get("weight_decay", 0.0) < 0:
        raise ValueError("Weight decay must be nonnegative.")

    if plan["class_weighting"] != "sqrt_inverse_frequency":
        raise ValueError("This comparison uses training-only class weights.")
    if plan["selection_metric"] != "validation_macro_f1":
        raise ValueError("Checkpoint selection must use validation macro-F1.")

    if plan.get("scheduler", "none") not in ("none", "plateau"):
        raise ValueError("Unknown learning-rate scheduler.")
    if plan.get("scheduler", "none") == "plateau":
        settings = plan["scheduler_settings"]
        if not 0 < settings["factor"] < 1 or settings["patience"] < 0:
            raise ValueError("Invalid plateau scheduler settings.")
        if not 0 < settings["min_lr"] <= min(plan["head_learning_rate"], plan["layer4_learning_rate"], plan.get("layer3_learning_rate", plan["layer4_learning_rate"])):
            raise ValueError("Invalid minimum learning rate.")
        if settings["patience"] + 1 >= plan["early_stopping_patience"]:
            raise ValueError("Early stopping must allow an epoch after LR reduction.")

    augmentation = plan.get("train_augmentation")
    if augmentation is not None:
        expected_keys = {
            "horizontal_flip_probability",
            "contrast_probability",
            "contrast_min",
            "contrast_max",
        }
        if not isinstance(augmentation, dict) or set(augmentation) != expected_keys:
            raise ValueError("Unexpected train_augmentation settings.")

        # Constructing the transform also checks its probability and range limits.
        ThermalTrainAugmentation(
            flip_probability=augmentation["horizontal_flip_probability"],
            contrast_probability=augmentation["contrast_probability"],
            contrast_min=augmentation["contrast_min"],
            contrast_max=augmentation["contrast_max"],
        )

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", plan["run_id"]):
        raise ValueError("run_id must be a simple folder name.")

    return plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run the single bounded full training trial.",
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=PLAN_PATH,
        help="Plan JSON; defaults to the original layer-4 plan.",
    )
    args = parser.parse_args()

    plan_path = args.plan
    if not plan_path.is_absolute():
        plan_path = ROOT / plan_path
    plan = load_plan(plan_path)
    torch.set_num_threads(plan.get("cpu_threads", 4))

    run_dir = ROOT / "runs" / plan["run_id"]

    # Never overwrite an earlier full trial.
    if args.full and run_dir.exists():
        raise FileExistsError(f"Existing run will not be overwritten: {run_dir}")

    random.seed(plan["seed"])
    np.random.seed(plan["seed"])
    torch.manual_seed(plan["seed"])

    augmentation_settings = plan.get("train_augmentation")
    train_transform = None
    if augmentation_settings is not None:
        train_transform = ThermalTrainAugmentation(
            flip_probability=augmentation_settings[
                "horizontal_flip_probability"
            ],
            contrast_probability=augmentation_settings[
                "contrast_probability"
            ],
            contrast_min=augmentation_settings["contrast_min"],
            contrast_max=augmentation_settings["contrast_max"],
        )

    started = time.monotonic()
    train_data, val_data, train_loader, val_loader = make_loaders(
        plan, train_transform=train_transform
    )

    model = ResNet18Thermal(
        num_classes=len(CLASS_NAMES),
        weights_path=ROOT / plan["pretrained_weights"],
        expected_sha256=plan["pretrained_sha256"],
    )
    parent = torch.load(
        ROOT / plan["parent_checkpoint"],
        map_location="cpu",
        weights_only=True,
    )

    if parent["model"] != "resnet18_frozen":
        raise ValueError("Expected a frozen ResNet18 parent checkpoint.")
    if parent["manifest_sha256"] != plan["manifest_sha256"]:
        raise ValueError("Parent checkpoint and plan use different splits.")
    if parent["pretrained_sha256"] != plan["pretrained_sha256"]:
        raise ValueError("Parent checkpoint used different pretrained weights.")
    if tuple(parent["class_names"]) != CLASS_NAMES:
        raise ValueError("Parent checkpoint uses a different class order.")

    model.load_state_dict(parent["model_state_dict"])
    trainable_stages = tuple(plan.get("trainable_stages", ["layer4"]))
    configure_resnet_finetuning(model, trainable_stages)

    parameter_groups = [
        {"params": [p for p in getattr(model.backbone, stage).parameters() if p.requires_grad],
         "lr": plan[stage + "_learning_rate"], "name": stage}
        for stage in trainable_stages
    ]
    parameter_groups.append({"params": list(model.backbone.fc.parameters()),
                             "lr": plan["head_learning_rate"], "name": "head"})
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer_class = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[plan.get("optimizer", "adam")]
    optimizer = optimizer_class(parameter_groups, weight_decay=plan.get("weight_decay", 0.0))

    scheduler = make_scheduler(optimizer, plan)

    class_weights = make_training_class_weights(train_data)
    training_loss = nn.CrossEntropyLoss(weight=class_weights)
    validation_loss = nn.CrossEntropyLoss()

    if not args.full:
        train_started = time.monotonic()
        train_smoke = train_one_epoch(
            model,
            train_loader,
            optimizer,
            training_loss,
            "cpu",
            max_batches=2,
        )
        train_seconds = time.monotonic() - train_started

        validation_started = time.monotonic()
        validation_smoke = evaluate_one_epoch(
            model,
            val_loader,
            validation_loss,
            "cpu",
            max_batches=2,
        )
        validation_seconds = time.monotonic() - validation_started

        print(json.dumps({
            "trainable_parameters": trainable_parameters,
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "optimizer": plan.get("optimizer", "adam"),
            "weight_decay": plan.get("weight_decay", 0.0),
            "trainable_stages": list(trainable_stages),
            "parameter_group_names": [g["name"] for g in optimizer.param_groups],
            "train_smoke": train_smoke,
            "train_smoke_seconds": train_seconds,
            "validation_smoke": validation_smoke,
            "validation_smoke_seconds": validation_seconds,
            "total_seconds": time.monotonic() - started,
        }, indent=2))
        return

    # The test dataset is never created or loaded by this script.
    run_dir.mkdir(parents=True, exist_ok=False)
    save_json(run_dir / "experiment-plan.json", plan)

    source_files = (
        "scripts/finetune_resnet18.py",
        "scripts/train_baseline.py",
        "src/pv_fault/augmentation.py",
        "src/pv_fault/finetuning.py",
        "src/pv_fault/models/resnet.py",
        "src/pv_fault/data.py",
        "src/pv_fault/train.py",
        "src/pv_fault/evaluate.py",
    )
    for name in source_files:
        destination = run_dir / "source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / name).read_bytes())
    save_json(run_dir / "provenance.json", {
        "manifest_sha256": plan["manifest_sha256"],
        "parent_checkpoint_sha256": plan["parent_checkpoint_sha256"],
        "pretrained_sha256": plan["pretrained_sha256"],
        "plan_sha256": file_hash(plan_path),
        "source_sha256": {
            name: file_hash(ROOT / name) for name in source_files
        },
        "model": plan["model"],
        "trainable_parameters": trainable_parameters,
        "class_names": CLASS_NAMES,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "optimizer": plan.get("optimizer", "adam"),
        "weight_decay": plan.get("weight_decay", 0.0),
        "trainable_stages": list(trainable_stages),
        "parameter_group_names": [g["name"] for g in optimizer.param_groups],
        "class_weighting": plan["class_weighting"],
        "train_augmentation": augmentation_settings,
        "scheduler": plan.get("scheduler", "none"),
        "scheduler_settings": plan.get("scheduler_settings"),
        "train_class_counts": dict(
            Counter(row["label"] for row in train_data.rows)
        ),
        "class_weights": class_weights.tolist(),
        "train_images": len(train_data),
        "validation_images": len(val_data),
        "seed": plan["seed"],
        "python": sys.version,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "pillow": pillow_version,
        "platform": platform.platform(),
        "device": "cpu",
        "cpu_threads": torch.get_num_threads(),
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

        learning_rates_used = [group["lr"] for group in optimizer.param_groups]
        phase_started = time.monotonic()
        train_stats = train_one_epoch(
            model, train_loader, optimizer, training_loss, "cpu"
        )
        train_seconds = time.monotonic() - phase_started
        phase_started = time.monotonic()
        validation_stats = evaluate_one_epoch(
            model, val_loader, validation_loss, "cpu"
        )
        validation_loss_seconds = time.monotonic() - phase_started
        phase_started = time.monotonic()
        predictions = collect_validation_predictions(
            model, val_loader, "cpu"
        )
        validation_prediction_seconds = time.monotonic() - phase_started
        metrics = classification_metrics(
            [row["true_index"] for row in predictions],
            [row["predicted_index"] for row in predictions],
        )

        if abs(metrics["accuracy"] - validation_stats["accuracy"]) > 1e-9:
            raise ValueError(
                "Validation metrics disagree with saved predictions."
            )

        if scheduler is not None:
            scheduler.step(metrics["macro_f1"])

        elapsed = time.monotonic() - started
        epoch_result = {
            "train_seconds": train_seconds,
            "validation_loss_seconds": validation_loss_seconds,
            "validation_prediction_seconds": validation_prediction_seconds,
            "learning_rates": learning_rates_used,
            "next_learning_rates": [group["lr"] for group in optimizer.param_groups],
            "epoch": epoch,
            "train": train_stats,
            "validation_loss": validation_stats["loss"],
            "validation_accuracy": metrics["accuracy"],
            "validation_macro_f1": metrics["macro_f1"],
            "fault_to_normal_rate": metrics["fault_to_normal_rate"],
            "normal_false_alarm_rate": metrics["normal_false_alarm_rate"],
            "elapsed_seconds": elapsed,
        }
        history.append(epoch_result)
        save_json(run_dir / "history.json", history)
        print(json.dumps(epoch_result), flush=True)

        if metrics["macro_f1"] > best_score:
            best_score = metrics["macro_f1"]
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save({
                "model_state_dict": model.state_dict(),
                "model": plan["model"],
                "scheduler": plan.get("scheduler", "none"),
                "epoch": epoch,
                "class_names": CLASS_NAMES,
                "manifest_sha256": plan["manifest_sha256"],
                "parent_checkpoint_sha256": plan["parent_checkpoint_sha256"],
            }, run_dir / "best_model.pt")
            save_json(run_dir / "best_validation_metrics.json", metrics)
            save_json(
                run_dir / "best_validation_predictions.json",
                predictions,
            )
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
        "best_validation_macro_f1": (
            best_score if best_epoch is not None else None
        ),
        "epochs_completed": len(history),
        "stop_reason": stop_reason,
        "elapsed_seconds": time.monotonic() - started,
        "test_evaluated": False,
    })
    print(f"Saved run to {run_dir}")


if __name__ == "__main__":
    main()
