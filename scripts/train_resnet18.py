"""Run a bounded frozen-backbone ResNet18 trial on train/validation only."""

import argparse
import json
import time
from pathlib import Path

import torch
import torchvision
from torch import nn

from pv_fault.resnet_training import class_weights_from_training
from pv_fault.data import CLASS_NAMES
from pv_fault.models.resnet import (
    IMAGENET_MEAN, IMAGENET_STD, RESIZED_SIZE, WEIGHT_SHA256, WEIGHT_URL,
    ResNet18Thermal,
)
from pv_fault.train import prepare_run, smoke_check, record_run, resolve_plan, file_hash, fit, save_json
from train_baseline import (
    ROOT, make_loaders,
    majority_baseline,
)


DEFAULT_PLAN = ROOT / "configs/experiment-plan-resnet18-frozen-repro.json"


def load_plan(path):
    plan = json.loads(resolve_plan(path).read_text(encoding="utf-8"))
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
    plan_path = resolve_plan(plan_path)
    plan = load_plan(plan_path)
    run_dir = prepare_run(plan, args.full)

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
    class_weights = class_weights_from_training(train_data)
    training_criterion = nn.CrossEntropyLoss(weight=class_weights)
    validation_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=plan["learning_rate"],
    )

    if not args.full:
        print(json.dumps({"model": plan.get("model", "small_cnn"),
                          "trainable_parameters": trainable_parameters,
                          "majority_class": majority_class, "majority_validation_macro_f1": majority_metrics["macro_f1"],
                          **smoke_check(model, train_loader, val_loader, optimizer,
                                        training_criterion, validation_criterion, started)}, indent=2))
        return

    source_files = (
        "scripts/train_resnet18.py",
        "scripts/train_baseline.py",
        "src/pv_fault/models/resnet.py",
        "src/pv_fault/data.py",
        "src/pv_fault/resnet_training.py",
        "src/pv_fault/train.py",
        "src/pv_fault/evaluate.py",
    )
    checkpoint_metadata = record_run(
        run_dir, plan_path, plan, model, train_data, val_data, class_weights, source_files,
        pretrained_weights_url=WEIGHT_URL,
        pretrained_weights_sha256=plan["pretrained_sha256"],
        pretrained_weight_rights="TorchVision warns pretrained weights may have dataset-derived terms; "
            "commercial deployment rights have not been verified.",
        preprocessing={
            "input": "grayscale float32 [1,40,24] in [0,1]",
            "resize": list(RESIZED_SIZE),
            "repeat_channels": 3,
            "normalization_mean": IMAGENET_MEAN,
            "normalization_std": IMAGENET_STD,
        },
        torchvision=torchvision.__version__)
    save_json(run_dir / "majority_validation_metrics.json", majority_metrics)
    summary = fit(model, train_loader, val_loader, optimizer, training_criterion,
                  validation_criterion, plan, run_dir, checkpoint_metadata,
                  started, scheduler=None)
    print(f"Saved run to {run_dir}")


if __name__ == "__main__":
    main()
