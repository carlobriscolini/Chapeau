"""Vectorised scoring of soft 128 x 128 maps against the official grid (same mIoU as wmloc.utils.miou_image)."""
from __future__ import annotations

import numpy as np


def miou_batch(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Per-image official mIoU for N x H x W binary arrays (an empty union counts as 1)."""
    p = pred.astype(bool).reshape(len(pred), -1); g = gt.astype(bool).reshape(len(gt), -1)
    out = np.zeros(len(p))
    for pc, gc in ((p, g), (~p, ~g)):
        inter = (pc & gc).sum(1); union = (pc | gc).sum(1)
        out += np.where(union == 0, 1.0, inter / np.maximum(union, 1))
    return out / 2


def auc_per_image(score: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """ROC AUC of the map for the watermarked cells, per image (NaN if one class is absent)."""
    out = np.full(len(score), np.nan)
    for i, (s, g) in enumerate(zip(score.reshape(len(score), -1), gt.reshape(len(gt), -1).astype(bool))):
        n1 = g.sum(); n0 = g.size - n1
        if n1 and n0:
            ranks = np.empty(g.size); ranks[np.argsort(s, kind="stable")] = np.arange(1, g.size + 1)
            out[i] = (ranks[g].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return out


def zscore(maps: np.ndarray) -> np.ndarray:
    flat = maps.reshape(len(maps), -1)
    return ((flat - flat.mean(1, keepdims=True)) / (flat.std(1, keepdims=True) + 1e-8)).reshape(maps.shape)


def best_global_threshold(maps: np.ndarray, gt: np.ndarray, grid=np.linspace(-3, 3, 61)) -> tuple[float, float]:
    """Best single threshold on per-image z-scored maps (fitted in-sample: a diagnostic, not a test score)."""
    z = zscore(maps)
    scores = [miou_batch(z > t, gt).mean() for t in grid]
    k = int(np.argmax(scores))
    return float(grid[k]), float(scores[k])


def oracle_fraction_miou(maps: np.ndarray, gt: np.ndarray) -> float:
    """Upper bound of a ranking: label as watermarked the top cells, in the true watermarked proportion."""
    flat = maps.reshape(len(maps), -1); g = gt.reshape(len(gt), -1)
    pred = np.zeros_like(g, dtype=bool)
    for i in range(len(flat)):
        k = int(g[i].sum())
        if k:
            pred[i, np.argsort(-flat[i], kind="stable")[:k]] = True
    return float(miou_batch(pred, g).mean())
