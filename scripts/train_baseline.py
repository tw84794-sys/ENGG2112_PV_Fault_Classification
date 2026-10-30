"""Run a planned CNN trial using training and validation only."""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from pv_fault.resnet_training import class_weights_from_training
from pv_fault.data import CLASS_NAMES, CLASS_TO_INDEX, SolarModuleDataset
from pv_fault.evaluate import classification_metrics
from pv_fault.models.baseline import SmallCNN, WideCNN
from pv_fault.train import prepare_run, smoke_check, record_run, resolve_plan, file_hash, fit, save_json, evaluate_one_epoch


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "configs/experiment-plan-smallcnn-depth-gentle60.json"


def load_plan(plan_path):
    plan = json.loads(resolve_plan(plan_path).read_text(encoding="utf-8"))
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
        help="Plan JSON; defaults to the selected SmallCNN reference.",
    )
    args = parser.parse_args()

    plan_path = args.plan
    if not plan_path.is_absolute():
        plan_path = ROOT / plan_path
    plan_path = resolve_plan(plan_path)
    plan = load_plan(plan_path)

    run_dir = prepare_run(plan, args.full)

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
        class_weights_from_training(train_data)
        if plan.get("class_weighting", "none") == "sqrt_inverse_frequency"
        else None
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    validation_criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(model, plan)

    scheduler = make_scheduler(optimizer, plan)

    if not args.full:
        print(json.dumps({"model": plan.get("model", "small_cnn"),
                          "trainable_parameters": trainable_parameters,
                          "majority_class": majority_class, "majority_validation_macro_f1": majority_metrics["macro_f1"],
                          **smoke_check(model, train_loader, val_loader, optimizer,
                                        criterion, validation_criterion, started)}, indent=2))
        return

    source_files = (
        "scripts/train_baseline.py",
        "src/pv_fault/data.py",
        "src/pv_fault/models/baseline.py",
        "src/pv_fault/resnet_training.py",
        "src/pv_fault/train.py",
        "src/pv_fault/evaluate.py",
        "src/pv_fault/augmentation.py",
    )
    checkpoint_metadata = record_run(
        run_dir, plan_path, plan, model, train_data, val_data, class_weights, source_files,
        model_architecture=str(model),
        optimizer=plan.get("optimizer", "adam"),
        weight_decay=plan.get("weight_decay", 0.0),
        augmentation=plan.get("augmentation", "none"))
    save_json(run_dir / "majority_validation_metrics.json", majority_metrics)
    summary = fit(model, train_loader, val_loader, optimizer, criterion,
                  validation_criterion, plan, run_dir, checkpoint_metadata,
                  started, scheduler=scheduler)
    if summary["best_epoch"] is not None:
        fit_data = SolarModuleDataset("train", manifest=ROOT / plan["manifest"])
        fit_loader = DataLoader(fit_data, batch_size=plan["batch_size"], shuffle=False)
        save_json(run_dir / "selected_training_fit.json",
                  evaluate_one_epoch(model, fit_loader, validation_criterion, "cpu"))
    summary["majority_validation_macro_f1"] = majority_metrics["macro_f1"]
    save_json(run_dir / "summary.json", summary)
    print(f"Saved run to {run_dir}")


if __name__ == "__main__":
    main()
