"""Small, training-only changes to thermal module images."""

import torch


class ThermalTrainAugmentation:
    def __init__(
        self,
        flip_probability=0.5,
        contrast_probability=0.5,
        contrast_min=0.9,
        contrast_max=1.1,
    ):
        for probability in (flip_probability, contrast_probability):
            if not 0.0 <= probability <= 1.0:
                raise ValueError("Probabilities must be between 0 and 1.")
        if not 0.0 < contrast_min <= contrast_max:
            raise ValueError("Invalid contrast range.")

        self.flip_probability = flip_probability
        self.contrast_probability = contrast_probability
        self.contrast_min = contrast_min
        self.contrast_max = contrast_max

    def __call__(self, image):
        if image.shape != (1, 40, 24):
            raise ValueError("Expected one grayscale 40 × 24 image.")

        result = image.clone()

        if torch.rand(()) < self.flip_probability:
            result = torch.flip(result, dims=(-1,))

        if torch.rand(()) < self.contrast_probability:
            factor = self.contrast_min + (
                self.contrast_max - self.contrast_min
            ) * torch.rand(())
            mean = result.mean()
            result = (mean + (result - mean) * factor).clamp(0.0, 1.0)

        return result