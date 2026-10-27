"""Training-only loss settings and parent-lineage checks for ResNet experiments."""

from collections import Counter
import torch
from pv_fault.data import CLASS_NAMES


def class_weights_from_training(train_data, exponent=0.5):
    """Mean-one inverse-frequency weights, computed exclusively from train rows."""
    if train_data.split != "train":
        raise ValueError("Class weights must come from training data.")
    if not 0 <= exponent <= 1:
        raise ValueError("The inverse-frequency exponent must be in [0, 1].")
    counts = Counter(row["label"] for row in train_data.rows)
    if any(counts[name] == 0 for name in CLASS_NAMES):
        raise ValueError("Every class needs training examples.")
    weights = torch.tensor([counts[name] ** -exponent for name in CLASS_NAMES], dtype=torch.float32)
    return weights / weights.mean()


def validate_parent_metadata(parent, plan, provenance=None):
    """Require explicit warm-start opt-in and matching split, weights and classes."""
    expected = plan.get("parent_model", "resnet18_frozen")
    if expected not in ("resnet18_frozen", "resnet18_layer4") or parent["model"] != expected:
        raise ValueError("Parent model differs from the declared starting point.")
    if parent["manifest_sha256"] != plan["manifest_sha256"]:
        raise ValueError("Parent checkpoint and plan use different splits.")
    if tuple(parent["class_names"]) != CLASS_NAMES:
        raise ValueError("Parent checkpoint uses a different class order.")
    if expected == "resnet18_layer4":
        if provenance is None or provenance["model"] != expected or provenance["test_evaluated"]:
            raise ValueError("Warm-start parent needs training/validation-only provenance.")
        if provenance["manifest_sha256"] != plan["manifest_sha256"]:
            raise ValueError("Parent provenance uses a different split.")
        pretrained = provenance["pretrained_sha256"]
        if parent.get("pretrained_sha256", pretrained) != pretrained:
            raise ValueError("Parent checkpoint and provenance disagree on pretrained weights.")
    else:
        pretrained = parent["pretrained_sha256"]
    if pretrained != plan["pretrained_sha256"]:
        raise ValueError("Parent used different pretrained weights.")
