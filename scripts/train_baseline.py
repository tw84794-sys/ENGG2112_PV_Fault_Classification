"""Run a planned CNN trial using training and validation only."""

import argparse
import hashlib
import json
import platform
import random
import sys
import time
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import __version__ as pillow_version
from torch import nn
from torch.utils.data import DataLoader

from pv_fault.data import CLASS_NAMES, CLASS_TO_INDEX, SolarModuleDataset
from pv_fault.evaluate import (
    classification_metrics,
    collect_validation_predictions,
)
from pv_fault.models.baseline import SmallCNN, WideCNN
from pv_fault.train import train_one_epoch, evaluate_one_epoch


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "configs/experiment-plan.json"


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_plan(plan_path):
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    manifest = ROOT / plan["manifest"]

    if file_hash(manifest) != plan["manifest_sha256"]:
        raise ValueError("Frozen manifest differs from the experiment plan.")
    if plan["device"] != "cpu" or plan["paid_compute_cost"] != 0:
        raise ValueError("This trial must use local CPU with no paid compute.")
    if plan["max_trials"] != 1:
        raise ValueError("This plan allows exactly one full trial.")
    if not 1 <= plan["max_epochs"] <= 60:
        raise ValueError("Epoch limit must be between 1 and 60.")
    if not 0 < plan["max_elapsed_seconds"] <= 450:
        raise ValueError("Time limit must be at most 450 seconds.")
    if plan["batch_size"] < 1 or plan["learning_rate"] <= 0:
        raise ValueError("Batch size and learning rate must be positive.")
    if plan["early_stopping_patience"] < 1:
        raise ValueError("Early-stopping patience must be positive.")
    if plan["selection_metric"] != "validation_macro_f1":
        raise ValueError("Checkpoint selection must use validation macro-F1.")
    if plan.get("model", "small_cnn") not in ("small_cnn", "wide_cnn"):
        raise ValueError("Unknown CNN model.")
    if plan.get("class_weighting", "none") not in (
        "none", "sqrt_inverse_frequency"
    ):
        raise ValueError("Unknown class-weighting method.")
    if not isinstance(plan.get("batch_norm", False), bool):
        raise ValueError("batch_norm must be a boolean.")
    if plan.get("scheduler", "none") not in ("none", "plateau"):
        raise ValueError("Unknown learning-rate scheduler.")
    if plan.get("scheduler", "none") == "plateau":
        scheduler = plan["scheduler_settings"]
        if not 0 < scheduler["factor"] < 1 or scheduler["patience"] < 0:
            raise ValueError("Invalid plateau scheduler settings.")
        if not 0 < scheduler["min_lr"] <= plan["learning_rate"]:
            raise ValueError("Invalid minimum learning rate.")
        if scheduler["patience"] + 1 >= plan["early_stopping_patience"]:
            raise ValueError("Early stopping must allow time after an LR reduction.")
    if plan.get("activation", "relu") not in ("relu", "gelu"):
        raise ValueError("Unknown activation.")
    if not isinstance(plan.get("extra_conv", False), bool):
        raise ValueError("extra_conv must be a boolean.")
    if plan.get("optimizer", "adam") not in ("adam", "adamw", "sgd"):
        raise ValueError("Unknown optimizer.")
    if plan.get("weight_decay", 0.0) < 0:
        raise ValueError("Weight decay must be nonnegative.")
    if plan.get("augmentation", "none") not in ("none", "mild_thermal"):
        raise ValueError("Unknown training augmentation.")
    return plan


def make_training_transform(plan):
    if plan.get("augmentation", "none") == "none":
        return None
    from pv_fault.augmentation import ThermalTrainAugmentation
    return ThermalTrainAugmentation()


def make_loaders(plan, train_transform=None):
    train_data = SolarModuleDataset(
        "train",
        manifest=ROOT / plan["manifest"],
        transform=train_transform,
    )
    val_data = SolarModuleDataset(
        "val", manifest=ROOT / plan["manifest"]
    )

    train_loader = DataLoader(
        train_data,
        batch_size=plan["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(plan["seed"]),
        num_workers=0,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=plan["batch_size"],
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    return train_data, val_data, train_loader, val_loader


def majority_baseline(train_data, val_data):
    counts = Counter(row["label"] for row in train_data.rows)
    majority_index = max(
        range(len(CLASS_NAMES)),
        key=lambda index: (counts[CLASS_NAMES[index]], -index),
    )
    true_labels = [
        CLASS_TO_INDEX[row["label"]] for row in val_data.rows
    ]
    metrics = classification_metrics(
        true_labels, [majority_index] * len(true_labels)
    )
    return CLASS_NAMES[majority_index], metrics


def make_training_class_weights(train_data):
    if train_data.split != "train":
        raise ValueError("Class weights must come from training data.")

    counts = Counter(row["label"] for row in train_data.rows)
    if any(counts[name] == 0 for name in CLASS_NAMES):
        raise ValueError("Every class needs training examples.")

    weights = torch.tensor(
        [counts[name] ** -0.5 for name in CLASS_NAMES],
        dtype=torch.float32,
    )
    return weights / weights.mean()


def make_optimizer(model, plan):
    name = plan.get("optimizer", "adam")
    kwargs = {"lr": plan["learning_rate"], "weight_decay": plan.get("weight_decay", 0.0)}
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), momentum=plan.get("momentum", 0.9), **kwargs)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), **kwargs)
    return torch.optim.Adam(model.parameters(), **kwargs)


def make_scheduler(optimizer, plan):
    if plan.get("scheduler", "none") == "none":
        return None
    settings = plan["scheduler_settings"]
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=settings["factor"],
        patience=settings["patience"], min_lr=settings["min_lr"],
        threshold=0.0, threshold_mode="abs",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full", action="store_true",
        help="Run the one planned, bounded full training trial.",
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=PLAN_PATH,
        help="Plan JSON; defaults to the original five-epoch plan.",
    )
    args = parser.parse_args()

    plan_path = args.plan
    if not plan_path.is_absolute():
        plan_path = ROOT / plan_path
    plan = load_plan(plan_path)

    torch.set_num_threads(plan.get("cpu_threads", 4))
    seed = plan["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    run_id = plan.get("run_id", f"smallcnn-s{seed}-cpu-v1")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", run_id):
        raise ValueError("run_id must be a simple folder name.")
    run_dir = ROOT / "runs" / run_id
    if args.full and run_dir.exists():
        raise FileExistsError(
            f"{run_dir} already exists; the planned trial will not be overwritten."
        )

    started = time.monotonic()
    train_data, val_data, train_loader, val_loader = make_loaders(
        plan, train_transform=make_training_transform(plan)
    )
    majority_class, majority_metrics = majority_baseline(
        train_data, val_data
    )

    model_name = plan.get("model", "small_cnn")
    model_class = {"small_cnn": SmallCNN, "wide_cnn": WideCNN}[model_name]
    model = model_class(
        num_classes=len(CLASS_NAMES), batch_norm=plan.get("batch_norm", False),
        activation=plan.get("activation", "relu"), extra_conv=plan.get("extra_conv", False),
    ).to("cpu")
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    class_weights = (
        make_training_class_weights(train_data)
        if plan.get("class_weighting", "none") == "sqrt_inverse_frequency"
        else None
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    validation_criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(model, plan)

    scheduler = make_scheduler(optimizer, plan)

    if not args.full:
        train_started = time.monotonic()
        train_smoke = train_one_epoch(
            model, train_loader, optimizer, criterion, "cpu",
            max_batches=2,
        )
        train_smoke_seconds = time.monotonic() - train_started
        validation_started = time.monotonic()
        val_smoke = evaluate_one_epoch(
            model, val_loader, validation_criterion, "cpu",
            max_batches=2,
        )
        validation_smoke_seconds = time.monotonic() - validation_started
        print(json.dumps({
            "model": model_name,
            "trainable_parameters": trainable_parameters,
            "majority_class": majority_class,
            "majority_validation_accuracy": majority_metrics["accuracy"],
            "majority_validation_macro_f1": majority_metrics["macro_f1"],
            "majority_fault_to_normal_rate":
                majority_metrics["fault_to_normal_rate"],
            "cnn_smoke_train": train_smoke,
            "cnn_smoke_train_seconds": train_smoke_seconds,
            "cnn_smoke_validation": val_smoke,
            "cnn_smoke_validation_seconds": validation_smoke_seconds,
            "cnn_smoke_seconds": time.monotonic() - started,
        }, indent=2))
        return

    run_dir.mkdir(parents=True, exist_ok=False)
    save_json(run_dir / "experiment-plan.json", plan)
    save_json(run_dir / "majority_validation_metrics.json", majority_metrics)

    source_files = (
        "scripts/train_baseline.py",
        "src/pv_fault/data.py",
        "src/pv_fault/models/baseline.py",
        "src/pv_fault/train.py",
        "src/pv_fault/evaluate.py",
        "src/pv_fault/augmentation.py",
    )
    for name in source_files:
        destination = run_dir / "source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / name).read_bytes())
    save_json(run_dir / "provenance.json", {
        "manifest_sha256": plan["manifest_sha256"],
        "plan_sha256": file_hash(plan_path),
        "source_sha256": {
            name: file_hash(ROOT / name) for name in source_files
        },
        "class_names": CLASS_NAMES,
        "model": model_name,
        "trainable_parameters": trainable_parameters,
        "class_weighting": plan.get("class_weighting", "none"),
        "train_class_counts": dict(
            Counter(row["label"] for row in train_data.rows)
        ),
        "class_weights": (
            class_weights.tolist() if class_weights is not None else None
        ),
        "train_images": len(train_data),
        "validation_images": len(val_data),
        "seed": seed,
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pillow": pillow_version,
        "platform": platform.platform(),
        "device": "cpu",
        "cpu_threads": torch.get_num_threads(),
        "model_architecture": str(model),
        "activation": plan.get("activation", "relu"),
        "extra_conv": plan.get("extra_conv", False),
        "optimizer": plan.get("optimizer", "adam"),
        "weight_decay": plan.get("weight_decay", 0.0),
        "augmentation": plan.get("augmentation", "none"),
        "batch_norm": plan.get("batch_norm", False),
        "scheduler": plan.get("scheduler", "none"),
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

        learning_rate_used = optimizer.param_groups[0]["lr"]
        train_stats = train_one_epoch(
            model, train_loader, optimizer, criterion, "cpu"
        )
        val_stats = evaluate_one_epoch(
            model, val_loader, validation_criterion, "cpu"
        )
        records = collect_validation_predictions(
            model, val_loader, "cpu"
        )
        metrics = classification_metrics(
            [row["true_index"] for row in records],
            [row["predicted_index"] for row in records],
        )

        if abs(metrics["accuracy"] - val_stats["accuracy"]) > 1e-9:
            raise ValueError("Validation accuracy and saved predictions disagree.")

        if scheduler is not None:
            scheduler.step(metrics["macro_f1"])
        elapsed = time.monotonic() - started
        epoch_result = {
            "learning_rate": learning_rate_used,
            "next_learning_rate": optimizer.param_groups[0]["lr"],
            "epoch": epoch,
            "train": train_stats,
            "validation_loss": val_stats["loss"],
            "validation_accuracy": metrics["accuracy"],
            "validation_macro_f1": metrics["macro_f1"],
            "fault_to_normal_rate": metrics["fault_to_normal_rate"],
            "normal_false_alarm_rate": metrics["normal_false_alarm_rate"],
            "elapsed_seconds": elapsed,
        }
        history.append(epoch_result)
        save_json(run_dir / "history.json", history)
        print(json.dumps(epoch_result))

        if metrics["macro_f1"] > best_score:
            best_score = metrics["macro_f1"]
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save({
                "model_state_dict": model.state_dict(),
                "model": model_name,
                "batch_norm": plan.get("batch_norm", False),
                "scheduler": plan.get("scheduler", "none"),
                "activation": plan.get("activation", "relu"),
                "extra_conv": plan.get("extra_conv", False),
                "trainable_parameters": trainable_parameters,
                "epoch": epoch,
                "class_names": CLASS_NAMES,
                "manifest_sha256": plan["manifest_sha256"],
            }, run_dir / "best_model.pt")
            save_json(run_dir / "best_validation_metrics.json", metrics)
            save_json(
                run_dir / "best_validation_predictions.json", records
            )
        else:
            epochs_without_improvement += 1

        if elapsed >= plan["max_elapsed_seconds"]:
            stop_reason = "time_limit_after_epoch"
            break
        if epochs_without_improvement >= plan["early_stopping_patience"]:
            stop_reason = "early_stopping"
            break

    # Verify that the persisted checkpoint reproduces the selected validation
    # predictions. Measure training fit with unweighted loss for comparison.
    checkpoint_verified = False
    if best_epoch is not None:
        checkpoint = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        restored_records = collect_validation_predictions(model, val_loader, "cpu")
        saved_records = json.loads((run_dir / "best_validation_predictions.json").read_text())
        if restored_records != saved_records:
            raise ValueError("Reloaded checkpoint predictions differ from saved predictions.")
        checkpoint_verified = True
        # Compare selected checkpoints on clean training images, even when
        # stochastic augmentation was used during fitting.
        fit_data = (SolarModuleDataset("train", manifest=ROOT / plan["manifest"])
                    if plan.get("augmentation", "none") != "none" else train_data)
        fit_loader = DataLoader(fit_data, batch_size=plan["batch_size"], shuffle=False)
        train_fit = evaluate_one_epoch(model, fit_loader, validation_criterion, "cpu")
        save_json(run_dir / "selected_training_fit.json", train_fit)

    save_json(run_dir / "summary.json", {
        "checkpoint_predictions_verified": checkpoint_verified,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_score if best_epoch else None,
        "majority_validation_macro_f1": majority_metrics["macro_f1"],
        "epochs_completed": len(history),
        "stop_reason": stop_reason,
        "elapsed_seconds": time.monotonic() - started,
        "test_evaluated": False,
    })
    print(f"Saved run to {run_dir}")


if __name__ == "__main__":
    main()