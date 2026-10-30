"""Optional early downsampling experiments; pretrained tensors stay unchanged."""
from torch import nn

VARIANTS = ("standard", "stride1", "no_maxpool")


def configure_spatial_variant(model, variant="standard"):
    """Apply exactly one declared stem change to a newly constructed ResNet.

    Neither stride nor max pooling has learned tensors. State dictionaries alone
    cannot identify these changes; inference must restore the recorded variant.
    """
    if variant not in VARIANTS:
        raise ValueError("Unknown ResNet spatial variant.")
    model.backbone.conv1.stride = (1, 1) if variant == "stride1" else (2, 2)
    model.backbone.maxpool = (nn.Identity() if variant == "no_maxpool" else
                              nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
