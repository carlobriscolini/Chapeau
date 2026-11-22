"""Check for identical RGB pixels and perceptually similar views of an image."""
from __future__ import annotations

from hashlib import sha256

import numpy as np
from PIL import Image


PHASH_DISTANCE = 6
_DCT = np.cos(np.pi / 32 * np.arange(8)[:, None] * (np.arange(32) + 0.5))
_BIT_COUNTS = np.array([value.bit_count() for value in range(256)], dtype=np.uint8)


def pixel_hash(image: Image.Image) -> str:
    return sha256(f"{image.width}x{image.height}:".encode() + image.convert("RGB").tobytes()).hexdigest()


def perceptual_hash(image: Image.Image) -> np.ndarray:
    gray = image.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    coefficients = (_DCT @ np.asarray(gray) @ _DCT.T).ravel()[1:]
    return np.packbits(coefficients > np.median(coefficients))


class ImageIndex:
    """Search labelled images using RGB SHA-256 and 63-bit DCT hashes."""

    def __init__(self, references):
        self.exact = {}
        self.labels = []
        hashes = []
        for label, image in references:
            self.exact.setdefault(pixel_hash(image), []).append(label)
            self.labels.append(label)
            hashes.append(perceptual_hash(image))
        self.hashes = np.stack(hashes)

    def matches(self, image: Image.Image) -> list[dict]:
        exact = set(self.exact.get(pixel_hash(image), ()))
        distances = _BIT_COUNTS[self.hashes ^ perceptual_hash(image)].sum(axis=1)
        return [
            {"reference": self.labels[index], "exact_rgb": self.labels[index] in exact,
             "phash_distance": int(distances[index])}
            for index in np.flatnonzero(distances <= PHASH_DISTANCE)
        ]
