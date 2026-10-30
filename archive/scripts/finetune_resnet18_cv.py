"""Bounded 3-fold ResNet18 learning-rate comparison on original train only.

Run without --full for a short smoke test, then inspect its timing before
running --full. The original validation and test partitions are never loaded.
"""

import argparse
import hashlib
import json
import platform
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import __version__ as pillow_version
from torch import nn
from torch.utils.data import DataLoader, Dataset

from pv_fault.resnet_training import class_weights_from_training
from pv_fault.data import CLASS_NAMES, SolarModuleDataset
from pv_fault.evaluate import classification_metrics, collect_validation_predictions
from pv_fault.finetuning import configure_resnet_finetuning
from pv_fault.models.resnet import ResNet18Thermal, WEIGHT_SHA256
from pv_fault.train import train_one_epoch, fit, resolve_plan


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


class FoldView(Dataset):
    """Give existing train images a fold-specific train or validation view."""

    def __init__(self, base, indices, split):
        self.base = base
        self.indices = indices
        self.rows = [base.rows[i] for i in indices]
        self.split = split

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.base[self.indices[index]]


def load_plan(root, path):
    plan = json.loads(path.read_text())
    if plan["device"] != "cpu" or plan["paid_compute_cost"] != 0:
        raise ValueError("This experiment is local CPU only.")
    if plan["folds"] != 3 or plan["max_trials"] != 6:
        raise ValueError("Exactly three folds and two arms are planned.")
    if plan["selection_metric"] != "fold_mean_macro_f1":
        raise ValueError("Unexpected selection metric.")
    if plan["class_weighting"] != "sqrt_inverse_frequency_from_fold_train_only":
        raise ValueError("Unexpected class weighting.")
    if plan["augmentation"] != "none_for_this_learning_rate_comparison":
        raise ValueError("This run isolates learning rates without augmentation.")
    if not 0 < plan["max_total_seconds"] <= 3600:
        raise ValueError("Time budget must be at most one hour.")
    if not 1 <= plan["batch_size"] <= 128 or not 1 <= plan["cpu_threads"] <= 8:
        raise ValueError("Unsupported batch size or CPU thread count.")
    if not 1 <= plan["frozen_head"]["max_epochs"] <= 5:
        raise ValueError("Frozen-head epoch cap exceeded.")
    if not 1 <= plan["fine_tuning"]["max_epochs"] <= 6:
        raise ValueError("Fine-tuning epoch cap exceeded.")
    if len(plan["fine_tuning"]["arms"]) != 2:
        raise ValueError("Expected exactly two learning-rate arms.")
    names = [arm["name"] for arm in plan["fine_tuning"]["arms"]]
    if names != ["current_lr", "paper_informed_lr"]:
        raise ValueError("The two predeclared arms changed.")
    for stage in (plan["frozen_head"], plan["fine_tuning"]):
        if not 1 <= stage["early_stopping_patience"] <= 3:
            raise ValueError("Invalid early-stopping patience.")
    for arm in plan["fine_tuning"]["arms"]:
        if arm["head_learning_rate"] <= 0 or arm["layer4_learning_rate"] <= 0:
            raise ValueError("Learning rates must be positive.")
    for filename, expected in (
        (plan["manifest"], plan["manifest_sha256"]),
        (plan["fold_manifest"], plan["fold_manifest_sha256"]),
        (plan["pretrained_weights"], plan["pretrained_sha256"]),
    ):
        if digest(root / filename) != expected:
            raise ValueError(f"Unexpected file checksum: {filename}")
    if plan["pretrained_sha256"] != WEIGHT_SHA256:
        raise ValueError("Unexpected ImageNet weight version.")
    return plan


def fold_membership(root, plan, base):
    import csv
    fold_path = root / plan["fold_manifest"]
    with fold_path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != ["image_id", "image_path", "label", "group_id", "fold"]:
            raise ValueError("Unexpected fold columns.")
        fold_rows = list(reader)
    if len(fold_rows) != len(base):
        raise ValueError("Folds must cover original train exactly.")
    by_id = {row["image_id"]: row for row in fold_rows}
    if len(by_id) != len(base) or set(by_id) != {row["image_id"] for row in base.rows}:
        raise ValueError("Fold image IDs differ from original train.")
    group_folds = defaultdict(set)
    indices = [[] for _ in range(plan["folds"])]
    for index, row in enumerate(base.rows):
        fold_row = by_id[row["image_id"]]
        if any(fold_row[key] != row[key]
               for key in ("image_path", "label", "group_id")):
            raise ValueError("Fold row disagrees with original train.")
        fold = int(fold_row["fold"])
        if fold not in range(plan["folds"]):
            raise ValueError("Fold number outside plan.")
        group_folds[row["group_id"]].add(fold)
        indices[fold].append(index)
    if any(len(value) != 1 for value in group_folds.values()):
        raise ValueError("Related-image group crosses CV folds.")
    for fold_indices in indices:
        if {base.rows[i]["label"] for i in fold_indices} != set(CLASS_NAMES):
            raise ValueError("A validation fold lacks a class.")
    return indices


def loaders(base, fold_indices, fold, batch_size, seed):
    validation = FoldView(base, fold_indices[fold], "val")
    training = FoldView(
        base,
        [i for index, members in enumerate(fold_indices) if index != fold
         for i in members],
        "train",
    )
    if set(r["group_id"] for r in training.rows) & set(
        r["group_id"] for r in validation.rows
    ):
        raise ValueError("Training and validation groups overlap.")
    train_loader = DataLoader(
        training, batch_size=batch_size, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(seed), drop_last=False,
    )
    val_loader = DataLoader(
        validation, batch_size=batch_size, shuffle=False, num_workers=0,
        drop_last=False,
    )
    return training, validation, train_loader, val_loader


def metrics_for(model, loader, max_batches=None):
    predictions = collect_validation_predictions(
        model, loader, "cpu", max_batches=max_batches
    )
    metrics = classification_metrics(
        [row["true_index"] for row in predictions],
        [row["predicted_index"] for row in predictions],
    )
    return metrics, predictions


def new_model(root, plan):
    return ResNet18Thermal(
        num_classes=len(CLASS_NAMES),
        weights_path=root / plan["pretrained_weights"],
        expected_sha256=plan["pretrained_sha256"],
    )


def frozen_optimizer(model, plan):
    return torch.optim.Adam(
        model.backbone.fc.parameters(),
        lr=plan["frozen_head"]["learning_rate"],
    )


def fine_optimizer(model, arm):
    return torch.optim.Adam([
        {"params": [p for p in model.backbone.layer4.parameters()
                    if p.requires_grad], "lr": arm["layer4_learning_rate"]},
        {"params": model.backbone.fc.parameters(),
         "lr": arm["head_learning_rate"]},
    ])


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def smoke(root, plan, plan_path, base, fold_indices, smoke_path):
    fold = 0
    seed = plan["seed"]
    seed_all(seed)
    training, validation, train_loader, val_loader = loaders(
        base, fold_indices, fold, plan["batch_size"], seed
    )
    model = new_model(root, plan)
    criterion = nn.CrossEntropyLoss(weight=class_weights_from_training(training))

    start = time.monotonic()
    train_one_epoch(
        model, train_loader, frozen_optimizer(model, plan), criterion, "cpu",
        max_batches=2,
    )
    frozen_train_seconds = time.monotonic() - start
    start = time.monotonic()
    metrics_for(model, val_loader, max_batches=2)
    frozen_val_seconds = time.monotonic() - start

    configure_resnet_finetuning(model)
    start = time.monotonic()
    train_one_epoch(
        model, train_loader, fine_optimizer(model, plan["fine_tuning"]["arms"][0]),
        criterion, "cpu", max_batches=2,
    )
    fine_train_seconds = time.monotonic() - start
    start = time.monotonic()
    metrics_for(model, val_loader, max_batches=2)
    fine_val_seconds = time.monotonic() - start

    train_batches = (len(training) + plan["batch_size"] - 1) // plan["batch_size"]
    val_batches = (len(validation) + plan["batch_size"] - 1) // plan["batch_size"]
    frozen_epoch = frozen_train_seconds * train_batches / 2 + frozen_val_seconds * val_batches / 2
    fine_epoch = fine_train_seconds * train_batches / 2 + fine_val_seconds * val_batches / 2
    projected_seconds = plan["folds"] * (
        plan["frozen_head"]["max_epochs"] * frozen_epoch
        + len(plan["fine_tuning"]["arms"])
        * plan["fine_tuning"]["max_epochs"] * fine_epoch
    )
    result = {
        "plan_sha256": digest(plan_path),
        "fold_manifest_sha256": plan["fold_manifest_sha256"],
        "train_images_in_smoke_fold": len(training),
        "validation_images_in_smoke_fold": len(validation),
        "batches_per_stage": 2,
        "frozen_train_seconds": frozen_train_seconds,
        "frozen_validation_seconds": frozen_val_seconds,
        "fine_train_seconds": fine_train_seconds,
        "fine_validation_seconds": fine_val_seconds,
        "projected_full_seconds_before_warmup_or_early_stop": projected_seconds,
        "hard_limit_seconds": plan["max_total_seconds"],
        "warning": "Two-batch projection is approximate; the hard cap is enforced between epochs.",
    }
    save_json(smoke_path, result)
    print(json.dumps(result, indent=2))


def fit_stage(model, train_loader, val_loader, optimizer, criterion,
              max_epochs, patience, started, time_cap, output_dir, identity):
    summary = fit(model, train_loader, val_loader, optimizer, criterion,
                  nn.CrossEntropyLoss(),
                  {"max_epochs": max_epochs, "early_stopping_patience": patience,
                   "max_elapsed_seconds": time_cap}, output_dir,
                  {"stage": identity}, started)
    if summary["best_epoch"] is None:
        raise TimeoutError(f"No completed epoch for {identity}; partial artifacts remain.")
    return json.loads((output_dir / "best_validation_metrics.json").read_text())


def full(root, plan, plan_path, base, fold_indices, smoke_path):
    smoke_result = json.loads(smoke_path.read_text())
    if smoke_result["plan_sha256"] != digest(plan_path):
        raise ValueError("Plan changed since the smoke test. Rerun smoke.")
    if smoke_result["fold_manifest_sha256"] != plan["fold_manifest_sha256"]:
        raise ValueError("Fold manifest changed since smoke.")
    if (smoke_result["projected_full_seconds_before_warmup_or_early_stop"]
            > plan["max_total_seconds"]):
        raise ValueError("Smoke projection exceeds the local time cap.")
    run_dir = root / "runs" / plan["run_id"]
    if run_dir.exists():
        raise FileExistsError("Run exists; refusing to overwrite it.")
    run_dir.mkdir(parents=True)
    save_json(run_dir / "experiment-plan.json", plan)
    save_json(run_dir / "smoke-timing.json", smoke_result)
    save_json(run_dir / "provenance.json", {
        "plan_sha256": digest(plan_path),
        "source_manifest_sha256": plan["manifest_sha256"],
        "fold_manifest_sha256": plan["fold_manifest_sha256"],
        "pretrained_sha256": plan["pretrained_sha256"],
        "runner_sha256": digest(Path(__file__)),
        "source_sha256": {
            name: digest(root / name) for name in (
                "src/pv_fault/data.py", "src/pv_fault/train.py",
                "src/pv_fault/evaluate.py", "src/pv_fault/models/resnet.py",
                "src/pv_fault/resnet_training.py", "src/pv_fault/finetuning.py",
            )
        },
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "numpy_version": np.__version__,
        "pillow_version": pillow_version,
        "platform": platform.platform(),
        "seed": plan["seed"],
        "cpu_threads": plan["cpu_threads"],
        "class_names": CLASS_NAMES,
        "test_evaluated": False,
    })
    started = time.monotonic()
    results = []
    for fold in range(plan["folds"]):
        fold_seed = plan["seed"] + fold
        seed_all(fold_seed)
        training, validation, train_loader, val_loader = loaders(
            base, fold_indices, fold, plan["batch_size"], fold_seed
        )
        criterion = nn.CrossEntropyLoss(weight=class_weights_from_training(training))
        fold_dir = run_dir / f"fold-{fold}"
        fold_dir.mkdir()
        save_json(fold_dir / "fold-data.json", {
            "train_images": len(training),
            "validation_images": len(validation),
            "train_class_counts": dict(Counter(r["label"] for r in training.rows)),
            "class_weights": class_weights_from_training(training).tolist(),
        })

        # Every fold starts from ImageNet, never the old full-train checkpoint.
        parent = new_model(root, plan)
        parent_dir = fold_dir / "frozen-head"
        parent_dir.mkdir()
        fit_stage(
            parent, train_loader, val_loader,
            frozen_optimizer(parent, plan), criterion,
            plan["frozen_head"]["max_epochs"],
            plan["frozen_head"]["early_stopping_patience"],
            started, plan["max_total_seconds"], parent_dir,
            f"fold-{fold}-frozen-head",
        )
        parent_state = torch.load(
            parent_dir / "best_model.pt", map_location="cpu", weights_only=True
        )["model_state_dict"]
        del parent
        for arm in plan["fine_tuning"]["arms"]:
            seed_all(fold_seed)
            train_loader.generator.manual_seed(fold_seed)
            model = new_model(root, plan)
            model.load_state_dict(parent_state)
            configure_resnet_finetuning(model)
            arm_dir = fold_dir / arm["name"]
            arm_dir.mkdir()
            metrics = fit_stage(
                model, train_loader, val_loader,
                fine_optimizer(model, arm), criterion,
                plan["fine_tuning"]["max_epochs"],
                plan["fine_tuning"]["early_stopping_patience"],
                started, plan["max_total_seconds"], arm_dir,
                f"fold-{fold}-{arm['name']}",
            )
            results.append({"fold": fold, "arm": arm["name"], "metrics": metrics})
            save_json(run_dir / "completed-fold-results.json", results)
            del model

    aggregate = {}
    for arm in plan["fine_tuning"]["arms"]:
        arm_results = [r["metrics"] for r in results if r["arm"] == arm["name"]]
        aggregate[arm["name"]] = {
            metric: {
                "mean": float(np.mean([r[metric] for r in arm_results])),
                "sd_across_folds": float(np.std([r[metric] for r in arm_results], ddof=1)),
                "per_fold": [r[metric] for r in arm_results],
            }
            for metric in (
                "macro_f1", "accuracy", "balanced_accuracy",
                "fault_to_normal_rate", "normal_false_alarm_rate",
            )
        }
        aggregate[arm["name"]]["per_class_recall"] = {
            label: [r["per_class"][label]["recall"] for r in arm_results]
            for label in CLASS_NAMES
        }
    summary = {
        "elapsed_seconds": time.monotonic() - started,
        "fold_images": [len(members) for members in fold_indices],
        "aggregate": aggregate,
        "test_evaluated": False,
        "note": "Fold variation is not unseen-site uncertainty because real site IDs are unavailable.",
    }
    save_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--plan", type=Path,
                        default=Path("configs/experiment-plan-resnet18-cv.json"))
    parser.add_argument("--smoke-out", type=Path, default=None)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    plan_path = resolve_plan(plan_path)
    plan = load_plan(root, plan_path)
    torch.set_num_threads(plan["cpu_threads"])

    # The loader verifies source manifest and metadata; it loads train rows only.
    base = SolarModuleDataset("train", manifest=root / plan["manifest"])
    fold_indices = fold_membership(root, plan, base)
    smoke_path = args.smoke_out or root / "runs" / f"{plan['run_id']}-smoke.json"
    if args.full:
        full(root, plan, plan_path, base, fold_indices, smoke_path)
    else:
        smoke(root, plan, plan_path, base, fold_indices, smoke_path)


if __name__ == "__main__":
    main()
