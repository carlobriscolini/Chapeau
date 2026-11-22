"""Optional augmentation that preserves the geometry of composed training images.

Enable in research.train_seg with --post-augment-prob 0.25. The training loop
applies it to online composites and pre-generated COCO images. It logs the
seed and probability with the experiment settings.
"""
from __future__ import annotations

import numpy as np
import torch
from torchvision.transforms import functional as image_ops


_TRANSFORMS = ("blur", "noise", "tone")


class PostWatermarkAugment:
    """Apply at most one change: blur (σ=0.5 px), noise (σ=0.5/255), or tone."""

    def __init__(self, *, seed: int, probability: float = 0.25):
        if not 0 <= probability <= 1:
            raise ValueError("probability must be between 0 and 1")
        self.probability = probability
        self.rng = np.random.default_rng(seed)

    @torch.no_grad()
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """Return B × 3 × H × W floating RGB images in [0, 1] on the input device."""
        if self.probability == 0:
            return images
        chosen = np.flatnonzero(self.rng.random(len(images)) < self.probability)
        if not len(chosen):
            return images

        output = images.clone()
        for index, operation in zip(chosen.tolist(), self.rng.choice(_TRANSFORMS, size=len(chosen))):
            output[index] = self._apply(images[index], operation)
        return output

    def _apply(self, image: torch.Tensor, operation: str) -> torch.Tensor:
        if operation == "blur":
            return image_ops.gaussian_blur(image, kernel_size=[3, 3], sigma=[0.5, 0.5])
        if operation == "noise":
            noise_seed = int(self.rng.integers(0, 2**63 - 1))
            generator = torch.Generator(device=image.device).manual_seed(noise_seed)
            noise = torch.randn(image.shape, generator=generator, device=image.device, dtype=image.dtype)
            return (image + noise * (0.5 / 255)).clamp_(0, 1)
        strength = self.rng.uniform(-1, 1)
        toned = image_ops.adjust_brightness(image, 1 + 0.08 * strength)
        return image_ops.adjust_contrast(toned, 1 + 0.05 * strength)
