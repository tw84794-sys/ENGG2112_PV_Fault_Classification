"""Configure optional late-stage ResNet adaptation without changing its forward path."""

from torch import nn


def configure_resnet_finetuning(model, stages=("layer4",)):
    """Train layer4 or layer3+layer4 and the head; keep BatchNorm fixed.

    ResNet18Thermal.train() keeps running statistics frozen. This function
    additionally freezes BatchNorm affine parameters and earlier stage weights.
    """
    stages = tuple(stages)
    if stages not in (("layer4",), ("layer3", "layer4")):
        raise ValueError("Choose layer4 or layer3+layer4 in their forward order.")
    for parameter in model.backbone.parameters():
        parameter.requires_grad_(False)
    for stage in stages:
        for parameter in getattr(model.backbone, stage).parameters():
            parameter.requires_grad_(True)
    for module in model.backbone.modules():
        if isinstance(module, nn.BatchNorm2d):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
    for parameter in model.backbone.fc.parameters():
        parameter.requires_grad_(True)
