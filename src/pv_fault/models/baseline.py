from torch import nn


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 12, *, batch_norm: bool = False, activation: str = "relu", extra_conv: bool = False):
        super().__init__()
        if activation not in ("relu", "gelu"):
            raise ValueError("activation must be relu or gelu")
        activation_class = {"relu": nn.ReLU, "gelu": nn.GELU}[activation]

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            activation_class(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            activation_class(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        if extra_conv:
            channels = next(layer.out_channels for layer in reversed(self.features)
                            if isinstance(layer, nn.Conv2d))
            original = list(self.features)
            self.features = nn.Sequential(
                *original[:-1], nn.Conv2d(channels, channels, 3, padding=1),
                activation_class(), original[-1],
            )

        # Reuse the original layers so enabling BatchNorm does not change
        # convolution/classifier initialisation under the same seed.
        if batch_norm:
            layers = []
            for layer in self.features:
                layers.append(layer)
                if isinstance(layer, nn.Conv2d):
                    layers.append(nn.BatchNorm2d(layer.out_channels))
            self.features = nn.Sequential(*layers)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64),
            activation_class(),
            nn.Linear(64, num_classes),
        )

    def forward(self, images):
        features = self.features(images)
        return self.classifier(features)


class WideCNN(nn.Module):
    """Wider version of SmallCNN for a controlled capacity comparison."""

    def __init__(self, num_classes: int = 12, *, batch_norm: bool = False, activation: str = "relu", extra_conv: bool = False):
        super().__init__()
        if activation not in ("relu", "gelu"):
            raise ValueError("activation must be relu or gelu")
        activation_class = {"relu": nn.ReLU, "gelu": nn.GELU}[activation]

        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            activation_class(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            activation_class(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        if extra_conv:
            channels = next(layer.out_channels for layer in reversed(self.features)
                            if isinstance(layer, nn.Conv2d))
            original = list(self.features)
            self.features = nn.Sequential(
                *original[:-1], nn.Conv2d(channels, channels, 3, padding=1),
                activation_class(), original[-1],
            )

        # Reuse the original layers so enabling BatchNorm does not change
        # convolution/classifier initialisation under the same seed.
        if batch_norm:
            layers = []
            for layer in self.features:
                layers.append(layer)
                if isinstance(layer, nn.Conv2d):
                    layers.append(nn.BatchNorm2d(layer.out_channels))
            self.features = nn.Sequential(*layers)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            activation_class(),
            nn.Linear(128, num_classes),
        )

    def forward(self, images):
        features = self.features(images)
        return self.classifier(features)
