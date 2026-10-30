"""Reusable training and validation steps for the fault classifier."""

import json
import hashlib
import platform
import random
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import __version__ as pillow_version
import time

import torch

from pv_fault.data import CLASS_NAMES, ROOT
from pv_fault.evaluate import classification_metrics, collect_validation_predictions


def train_one_batch(model, images, labels, optimizer, criterion):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    scores = model(images)
    loss = criterion(scores, labels)

    loss.backward()
    optimizer.step()

    correct = (scores.argmax(dim=1) == labels).sum().item()
    return {
        "loss": loss.item(),
        "correct": int(correct),
        "count": labels.numel(),
    }


@torch.inference_mode()
def evaluate_one_batch(model, images, labels, criterion):
    model.eval()

    scores = model(images)
    loss = criterion(scores, labels)

    correct = (scores.argmax(dim=1) == labels).sum().item()
    return {
        "loss": loss.item(),
        "correct": int(correct),
        "count": labels.numel(),
    }

def train_one_epoch(model, loader, optimizer, criterion, device, max_batches=None):
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for batch_number, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_number >= max_batches:
            break

        result = train_one_batch(
            model, images.to(device), labels.to(device), optimizer, criterion
        )
        total_loss += result["loss"] * result["count"]
        total_correct += result["correct"]
        total_count += result["count"]

    if total_count == 0:
        raise ValueError("Training loader produced no batches.")

    return {
        "loss": total_loss / total_count,
        "accuracy": total_correct / total_count,
        "count": total_count,
    }


def evaluate_one_epoch(model, loader, criterion, device, max_batches=None):
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for batch_number, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_number >= max_batches:
            break

        result = evaluate_one_batch(
            model, images.to(device), labels.to(device), criterion
        )
        total_loss += result["loss"] * result["count"]
        total_correct += result["correct"]
        total_count += result["count"]

    if total_count == 0:
        raise ValueError("Validation loader produced no batches.")

    return {
        "loss": total_loss / total_count,
        "accuracy": total_correct / total_count,
        "count": total_count,
    }


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def fit(model, train_loader, val_loader, optimizer, training_loss, validation_loss,
        plan, run_dir, checkpoint_metadata, started, scheduler=None):
    """Fit one bounded trial, save its best validation epoch, and verify reload."""
    history = []
    best_score, best_epoch, stale = -1.0, None, 0
    stop_reason = "max_epochs"
    for epoch in range(1, plan["max_epochs"] + 1):
        if time.monotonic() - started >= plan["max_elapsed_seconds"]:
            stop_reason = "time_limit_before_epoch"
            break
        rates = [group["lr"] for group in optimizer.param_groups]
        phase = time.monotonic()
        train_stats = train_one_epoch(model, train_loader, optimizer, training_loss, "cpu")
        train_seconds = time.monotonic() - phase
        phase = time.monotonic()
        predictions, validation = collect_validation_predictions(
            model, val_loader, "cpu", criterion=validation_loss)
        validation_seconds = time.monotonic() - phase
        metrics = classification_metrics([row["true_index"] for row in predictions],
                                         [row["predicted_index"] for row in predictions])
        if abs(metrics["accuracy"] - validation["accuracy"]) > 1e-9:
            raise ValueError("Validation accuracy and saved predictions disagree.")
        if scheduler is not None:
            scheduler.step(metrics["macro_f1"])
        elapsed = time.monotonic() - started
        following = [group["lr"] for group in optimizer.param_groups]
        record = {"epoch": epoch, "train": train_stats,
                  "validation_loss": validation["loss"],
                  "validation_accuracy": metrics["accuracy"],
                  "validation_macro_f1": metrics["macro_f1"],
                  "fault_to_normal_rate": metrics["fault_to_normal_rate"],
                  "normal_false_alarm_rate": metrics["normal_false_alarm_rate"],
                  "learning_rate": rates[0], "next_learning_rate": following[0],
                  "learning_rates": rates, "next_learning_rates": following,
                  "train_seconds": train_seconds, "validation_seconds": validation_seconds,
                  "elapsed_seconds": elapsed}
        history.append(record)
        save_json(run_dir / "history.json", history)
        print(json.dumps(record), flush=True)
        if metrics["macro_f1"] > best_score:
            best_score, best_epoch, stale = metrics["macro_f1"], epoch, 0
            torch.save({**checkpoint_metadata, "model_state_dict": model.state_dict(),
                        "epoch": epoch, "class_names": CLASS_NAMES}, run_dir / "best_model.pt")
            save_json(run_dir / "best_validation_metrics.json", metrics)
            save_json(run_dir / "best_validation_predictions.json", predictions)
        else:
            stale += 1
        if elapsed >= plan["max_elapsed_seconds"]:
            stop_reason = "time_limit_after_epoch"
            break
        if stale >= plan["early_stopping_patience"]:
            stop_reason = "early_stopping"
            break
    verified = False
    if best_epoch is not None:
        checkpoint = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        restored = collect_validation_predictions(model, val_loader, "cpu")
        if restored != json.loads((run_dir / "best_validation_predictions.json").read_text()):
            raise ValueError("Reloaded checkpoint predictions differ from saved predictions.")
        verified = True
    summary = {"best_epoch": best_epoch,
               "best_validation_macro_f1": best_score if best_epoch is not None else None,
               "epochs_completed": len(history), "stop_reason": stop_reason,
               "elapsed_seconds": time.monotonic() - started,
               "checkpoint_predictions_verified": verified, "test_evaluated": False}
    save_json(run_dir / "summary.json", summary)
    return summary



def resolve_plan(path):
    """Keep recorded config paths usable after older recipes move to archive."""
    path = Path(path)
    if not path.is_file() and path.parent.name == "configs":
        archived = path.parent.parent / "archive/configs" / path.name
        if archived.is_file():
            return archived
    return path


def file_hash(path):
    return hashlib.sha256(resolve_plan(path).read_bytes()).hexdigest()


def prepare_run(plan, full):
    run_id = plan.get("run_id", f"smallcnn-s{plan['seed']}-cpu-v1")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", run_id):
        raise ValueError("run_id must be a simple folder name.")
    run_dir = ROOT / "runs" / run_id
    if full and run_dir.exists():
        raise FileExistsError(f"Existing run will not be overwritten: {run_dir}")
    torch.set_num_threads(plan.get("cpu_threads", 4))
    random.seed(plan["seed"])
    np.random.seed(plan["seed"])
    torch.manual_seed(plan["seed"])
    return run_dir


def smoke_check(model, train_loader, val_loader, optimizer, training_loss, validation_loss, started):
    phase = time.monotonic()
    training = train_one_epoch(model, train_loader, optimizer, training_loss, "cpu", max_batches=2)
    train_seconds = time.monotonic() - phase
    phase = time.monotonic()
    validation = evaluate_one_epoch(model, val_loader, validation_loss, "cpu", max_batches=2)
    return {"train_smoke": training, "train_smoke_seconds": train_seconds,
            "validation_smoke": validation, "validation_smoke_seconds": time.monotonic() - phase,
            "total_seconds": time.monotonic() - started}


def record_run(run_dir, plan_path, plan, model, train_data, val_data, weights, source_files, **details):
    """Archive the exact producer and record the inputs for one new full run."""
    run_dir.mkdir(parents=True, exist_ok=False)
    save_json(run_dir / "experiment-plan.json", plan)
    for name in source_files:
        destination = run_dir / "source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / name).read_bytes())
    checkpoint = {"model": plan.get("model", "small_cnn"),
                  "manifest_sha256": plan["manifest_sha256"],
                  "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                  "batch_norm": plan.get("batch_norm", False),
                  "activation": plan.get("activation", "relu"), "extra_conv": plan.get("extra_conv", False),
                  **{key: plan[key] for key in ("scheduler", "parent_checkpoint_sha256", "pretrained_sha256", "spatial_variant") if key in plan}}
    save_json(run_dir / "provenance.json", {
        **checkpoint, "class_names": CLASS_NAMES,
        "plan_sha256": file_hash(plan_path),
        "source_sha256": {name: file_hash(ROOT / name) for name in source_files},
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "class_weighting": plan.get("class_weighting", "none"),
        "train_class_counts": dict(Counter(row["label"] for row in train_data.rows)),
        "class_weights": weights.tolist() if weights is not None else None,
        "train_images": len(train_data), "validation_images": len(val_data), "seed": plan["seed"],
        "python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
        "pillow": pillow_version, "platform": platform.platform(),
        "device": plan["device"], "cpu_threads": torch.get_num_threads(), "test_evaluated": False,
        **details})
    return checkpoint
