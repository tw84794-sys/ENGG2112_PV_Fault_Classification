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
