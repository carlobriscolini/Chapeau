"""Local geometric variations of the minority region of a mask."""
from __future__ import annotations

import math

import numpy as np
from PIL import Image

from research.synth import bbox


def cropped_shape(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    """Return the shape's dimensions and packed pixels, independent of position."""
    if not mask.any():
        return (0, 0), b""
    y0, y1, x0, x1 = bbox(mask)
    shape = mask[y0:y1, x0:x1]
    return shape.shape, np.packbits(shape).tobytes()


def _resize_shape(
    shape: np.ndarray, frame: tuple[int, int], rng: np.random.Generator
) -> np.ndarray:
    height, width = shape.shape
    original_area = np.count_nonzero(shape)
    rectangular = shape.all()
    image = None if rectangular else Image.fromarray(shape)
    for _ in range(12):
        area_factor = rng.uniform(0.82, 1.18)
        aspect_factor = math.exp(rng.uniform(-0.4, 0.4))
        new_height = min(frame[0], max(1, round(height * math.sqrt(area_factor / aspect_factor))))
        new_width = min(frame[1], max(1, round(width * math.sqrt(area_factor * aspect_factor))))
        if (new_height, new_width) == shape.shape:
            continue

        if image is None:
            candidate = np.ones((new_height, new_width), dtype=np.uint8)
        else:
            candidate = np.asarray(image.resize((new_width, new_height), Image.Resampling.NEAREST))
        area = np.count_nonzero(candidate)
        if not 0.8 * original_area <= area <= 1.2 * original_area:
            continue
        if image is not None:
            y0, y1, x0, x1 = bbox(candidate)
            candidate = candidate[y0:y1, x0:x1]
            if candidate.all() or np.array_equal(candidate, shape):
                continue
        return candidate

    # Thin or tiny raster shapes can have no acceptable resize at this resolution.
    return shape


def _shifted_start(start: int, end: int, size: int, limit: int, rng: np.random.Generator) -> int:
    last_start = limit - size
    centered = max(0, min(last_start, (start + end - size) // 2))
    displacement = max(12, min(80, (end - start) // 2))
    first = max(0, centered - displacement)
    last = min(last_start, centered + displacement)
    return int(rng.integers(first, last + 1))


def perturb_minority(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Resize and move a nonempty binary minority region within the image."""
    y0, y1, x0, x1 = bbox(mask)
    shape = (mask[y0:y1, x0:x1] != 0).astype(np.uint8)
    variant = _resize_shape(shape, mask.shape, rng)
    height, width = variant.shape
    y = _shifted_start(y0, y1, height, mask.shape[0], rng)
    x = _shifted_start(x0, x1, width, mask.shape[1], rng)
    result = np.zeros(mask.shape, dtype=np.uint8)
    result[y:y + height, x:x + width] = variant
    return result
