"""Frozen ImageNet ResNet18 adapted to 24 x 40 grayscale module crops."""

import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18


WEIGHT_URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
WEIGHT_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
INPUT_SIZE = (40, 24)
RESIZED_SIZE = (80, 48)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def preprocess_thermal(images, mean, std):
    """Repeat grayscale into RGB-sized channels and apply ImageNet scaling.

    Bilinear resizing doubles each dimension but does not add image detail.
    The same deterministic operation is used during training and inference.
    """
    if images.ndim != 4 or tuple(images.shape[1:]) != (1, *INPUT_SIZE):
        raise ValueError("Expected a batch of grayscale [N, 1, 40, 24] images.")
    resized = F.interpolate(
        images, size=RESIZED_SIZE, mode="bilinear", align_corners=False
    )
    repeated = resized.repeat(1, 3, 1, 1)
    return (repeated - mean) / std


class ResNet18Thermal(nn.Module):
    """Train only a 12-class head on verified TorchVision ImageNet weights."""

    def __init__(self, num_classes: int, weights_path: Path,
                 expected_sha256: str = WEIGHT_SHA256):
        super().__init__()
        weights_path = Path(weights_path)
        if not weights_path.is_file():
            raise FileNotFoundError(weights_path)
        digest = hashlib.sha256(weights_path.read_bytes()).hexdigest()
        if digest != expected_sha256:
            raise ValueError("ResNet18 pretrained weights have an unexpected SHA-256.")

        backbone = resnet18(weights=None)
        pretrained_state = torch.load(
            weights_path, map_location="cpu", weights_only=True
        )
        backbone.load_state_dict(pretrained_state, strict=True)
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
        backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
        self.backbone = backbone
        self.register_buffer(
            "mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        )
        self.train()

    def train(self, mode: bool = True):
        super().train(mode)
        # Freeze BatchNorm running statistics as well as backbone weights.
        self.backbone.eval()
        self.backbone.fc.train(mode)
        return self

    def forward(self, images):
        adapted = preprocess_thermal(images, self.mean, self.std)
        return self.backbone(adapted)
