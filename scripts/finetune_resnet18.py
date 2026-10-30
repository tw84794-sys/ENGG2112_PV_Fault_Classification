"""Run bounded late-stage ResNet experiments using training and validation only."""

import argparse
import json
import re
import time
from pathlib import Path

import torch
import torchvision
from torch import nn

from pv_fault.augmentation import ThermalTrainAugmentation
from pv_fault.data import CLASS_NAMES
from pv_fault.finetuning import configure_resnet_finetuning
from pv_fault.resnet_training import class_weights_from_training, validate_parent_metadata
from pv_fault.resnet_spatial import configure_spatial_variant, VARIANTS
from pv_fault.models.resnet import (
    ResNet18Thermal,
    WEIGHT_SHA256,
)
from pv_fault.train import prepare_run, smoke_check, record_run, resolve_plan, file_hash, fit
from train_baseline import (
    ROOT,
    make_loaders,
    make_scheduler,
)


PLAN_PATH = ROOT / "configs/experiment-plan-resnet18-generalisation-augmentation.json"


def load_plan(path):
    plan = json.loads(resolve_plan(path).read_text(encoding="utf-8"))

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
    variant = plan.get("spatial_variant", "standard")
    if variant not in VARIANTS:
        raise ValueError("Unknown ResNet spatial variant.")
    time_cap = 3600 if variant != "standard" else 1200
    if not 0 < plan["max_elapsed_seconds"] <= time_cap:
        raise ValueError(f"The time limit must be at most {time_cap} seconds.")
    if variant != "standard" and plan.get("parent_model", "resnet18_frozen") != "resnet18_frozen":
        raise ValueError("Spatial experiments must start from the common frozen parent.")
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

    weighting = plan["class_weighting"]
    if weighting not in ("sqrt_inverse_frequency", "power_inverse_frequency"):
        raise ValueError("This comparison uses training-only class weights.")
    exponent = plan.get("class_weight_exponent", 0.5)
    if not 0 <= exponent <= 1 or (weighting == "sqrt_inverse_frequency" and exponent != 0.5):
        raise ValueError("Invalid inverse-frequency exponent.")
    if not 0 <= plan.get("label_smoothing", 0.0) < 1:
        raise ValueError("Label smoothing must be in [0, 1).")
    if plan.get("parent_model", "resnet18_frozen") not in ("resnet18_frozen", "resnet18_layer4"):
        raise ValueError("Unsupported parent model.")
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

    if "cv_summary" in plan:
        cv_path = ROOT / plan["cv_summary"]
        if file_hash(cv_path) != plan["cv_summary_sha256"]:
            raise ValueError("The cross-validation summary has changed.")
        cv = json.loads(cv_path.read_text())
        if cv["test_evaluated"] or cv["fold_images"] != [4649, 4649, 4648]:
            raise ValueError("Unexpected cross-validation evidence.")
        if cv["aggregate"]["paper_informed_lr"]["macro_f1"]["mean"] <= cv["aggregate"]["current_lr"]["macro_f1"]["mean"]:
            raise ValueError("The chosen learning rate did not win cross-validation.")
        if (plan["head_learning_rate"], plan["layer4_learning_rate"]) != (0.0003, 0.00003):
            raise ValueError("Learning rates differ from the CV-selected arm.")
        if (plan["model"] != "resnet18_layer4" or plan["max_epochs"] > 6
                or plan["max_elapsed_seconds"] > 720 or plan["class_weighting"] != "sqrt_inverse_frequency"):
            raise ValueError("CV-selected plan exceeds its original protocol.")

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


def main(default_plan=PLAN_PATH):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run the single bounded full training trial.",
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=default_plan,
        help="Plan JSON; defaults to the standard augmented ResNet reference.",
    )
    args = parser.parse_args()

    plan_path = args.plan
    if not plan_path.is_absolute():
        plan_path = ROOT / plan_path
    plan_path = resolve_plan(plan_path)
    plan = load_plan(plan_path)
    run_dir = prepare_run(plan, args.full)

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

    parent_provenance = None
    if plan.get("parent_model", "resnet18_frozen") == "resnet18_layer4":
        parent_dir = (ROOT / plan["parent_checkpoint"]).parent
        parent_provenance = json.loads((parent_dir / "provenance.json").read_text())
        verification = json.loads((parent_dir / "checkpoint_verification.json").read_text())
        if not verification["checkpoint_predictions_verified"] or verification["checkpoint_sha256"] != plan["parent_checkpoint_sha256"]:
            raise ValueError("Warm-start parent must have verified saved validation predictions.")
    validate_parent_metadata(parent, plan, parent_provenance)

    model.load_state_dict(parent["model_state_dict"])
    configure_spatial_variant(model, plan.get("spatial_variant", "standard"))
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

    class_weights = class_weights_from_training(train_data, plan.get("class_weight_exponent", 0.5))
    training_loss = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=plan.get("label_smoothing", 0.0))
    validation_loss = nn.CrossEntropyLoss()

    if not args.full:
        print(json.dumps({"model": plan.get("model", "small_cnn"),
                          "trainable_parameters": trainable_parameters,
                          **smoke_check(model, train_loader, val_loader, optimizer,
                                        training_loss, validation_loss, started)}, indent=2))
        return

    source_files = (
        "scripts/finetune_resnet18.py",
        "scripts/train_baseline.py",
        "src/pv_fault/augmentation.py",
        "src/pv_fault/finetuning.py",
        "src/pv_fault/resnet_training.py",
        "src/pv_fault/resnet_spatial.py",
        "src/pv_fault/models/resnet.py",
        "src/pv_fault/data.py",
        "src/pv_fault/train.py",
        "src/pv_fault/evaluate.py",
    )
    checkpoint_metadata = record_run(
        run_dir, plan_path, plan, model, train_data, val_data, class_weights, source_files,
        cv_summary_sha256=plan.get("cv_summary_sha256"),
        optimizer=plan.get("optimizer", "adam"),
        weight_decay=plan.get("weight_decay", 0.0),
        trainable_stages=list(trainable_stages),
        parameter_group_names=[g["name"] for g in optimizer.param_groups],
        class_weight_exponent=plan.get("class_weight_exponent", 0.5),
        label_smoothing=plan.get("label_smoothing", 0.0),
        parent_model=plan.get("parent_model", "resnet18_frozen"),
        optimizer_state_restored=False,
        train_augmentation=augmentation_settings,
        scheduler_settings=plan.get("scheduler_settings"),
        torchvision=torchvision.__version__)
    summary = fit(model, train_loader, val_loader, optimizer, training_loss,
                  validation_loss, plan, run_dir, checkpoint_metadata,
                  started, scheduler=scheduler)
    print(f"Saved run to {run_dir}")


if __name__ == "__main__":
    main()
