"""Reusable training steps for fault classifier."""

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
