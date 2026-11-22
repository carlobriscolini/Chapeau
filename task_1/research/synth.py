"""Synthetic composites that follow the documented generation process:
  full_watermark = False: a rectangular patch of a clean image is watermarked, the marked region is a mask inside it;
  full_watermark = True:  the whole image is watermarked and the region outside the mask is restored to the original.
In both cases image = where(mask, watermarked, clean), so the mask is exact by construction."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_clean(path: Path, rng: np.random.Generator, size: int = 512) -> np.ndarray:
    """Random square crop of a clean photo, resized to size x size uint8 RGB."""
    img = Image.open(path).convert("RGB")
    w, h = img.size; s = min(w, h)
    x0 = int(rng.integers(0, w - s + 1)); y0 = int(rng.integers(0, h - s + 1))
    return np.asarray(img.crop((x0, y0, x0 + s, y0 + s)).resize((size, size), Image.Resampling.BICUBIC), np.uint8)


def bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def watermark_patch(wm, clean: np.ndarray, rect: tuple[int, int, int, int], bits) -> np.ndarray:
    """Clean image with only `rect` (y0, y1, x0, x1) watermarked by the official embedder at the patch's own size."""
    y0, y1, x0, x1 = rect
    out = clean.copy()
    out[y0:y1, x0:x1] = wm.embed(np.ascontiguousarray(clean[y0:y1, x0:x1]), bits)
    return out


def expand_rect(mask: np.ndarray, rng: np.random.Generator, max_aspect: float) -> tuple[int, int, int, int]:
    """Random rectangle that contains the mask's bbox (an object cut from a larger watermarked patch)."""
    h, w = mask.shape
    y0, y1, x0, x1 = bbox(mask)
    for _ in range(50):
        a, b = int(rng.integers(0, y0 + 1)), int(rng.integers(y1, h + 1))
        c, d = int(rng.integers(0, x0 + 1)), int(rng.integers(x1, w + 1))
        if max(b - a, d - c) / max(1, min(b - a, d - c)) <= max_aspect:
            return a, b, c, d
    return 0, h, 0, w


def is_rectangle(mask: np.ndarray) -> bool:
    y0, y1, x0, x1 = bbox(mask)
    return bool(mask[y0:y1, x0:x1].all())


def patch_rect(mask: np.ndarray, full_watermark: bool, mode: str, rng: np.random.Generator,
               max_aspect: float) -> tuple[int, int, int, int]:
    """Watermarked patch for a mask. Rectangular regions on a clean background are the patch itself (outpainting);
    irregular ones are cut from the whole image (mode 'whole'), from a larger random rectangle ('expand') or,
    as in the first simulation, from their own bbox ('bbox')."""
    h, w = mask.shape
    if full_watermark:
        return 0, h, 0, w
    if is_rectangle(mask) or mode == "bbox":
        return bbox(mask)
    return (0, h, 0, w) if mode == "whole" else expand_rect(mask, rng, max_aspect)


def composite(wm, clean: np.ndarray, mask512: np.ndarray, full_watermark: bool, bits,
              rect: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """where(mask, watermarked, clean). For full_watermark=False the watermarked patch defaults to the mask's bbox."""
    h, w = mask512.shape
    rect = (0, h, 0, w) if full_watermark else (rect or bbox(mask512))
    marked = watermark_patch(wm, clean, rect, bits)
    return np.where(mask512[..., None].astype(bool), marked, clean)
