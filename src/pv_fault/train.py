"""Reusable training and validation steps for the fault classifier."""

import torch


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