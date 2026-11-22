"""Edge-aware refinement of the ensemble probabilities with a guided filter (He, Sun, Tang, ECCV 2010 / TPAMI 2013).

Ground-truth edges often follow object edges (pasted objects have visible borders), while the segmenter output is
blobby. The guided filter makes the probability map locally linear in the image intensity, so the 0.5 contour snaps
to image edges. Evaluated out-of-fold on train (CPU, cached logits), parameters chosen on the other folds and applied
to the held-out fold. Scope variants: all images, or only images whose background is clean (transplants/rectangles;
at test time the polarity comes from the whole-image bit accuracy, 96-98 % correct).

    python -m research.guided_eval --method trustmark"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter

from research.analyze_cache import category, miou, oof_logits
from research.decode_eval import nested
from wmloc.data import LocalisationDataset

GRID = [(r, eps, T) for r in (1, 2, 4, 8) for eps in (1e-3, 1e-2, 3e-2) for T in (1.0, 3.0)]


def guided(I: np.ndarray, p: np.ndarray, r: int, eps: float) -> np.ndarray:
    box = lambda x: uniform_filter(x, 2 * r + 1, mode="reflect")
    mI, mp = box(I), box(p)
    a = (box(I * p) - mI * mp) / (box(I * I) - mI * mI + eps)
    b = mp - a * mI
    return box(a) * I + box(b)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--method", required=True)
    a = ap.parse_args()
    z = np.load(f"outputs/cache/{a.method}.npz")
    lg, cov, _ = oof_logits(z, a.method, flip=False)
    gt, full, folds, names = z["gt_train"], z["full_train"], z["folds_train"], z["names_train"].tolist()
    ds = LocalisationDataset("release", "train", methods=[a.method]); pos = {ds.meta[i]["image_name"]: i for i in range(len(ds))}
    idx = np.nonzero(cov)[0]
    guide = {}
    for i in idx:
        im = Image.fromarray(ds[pos[names[i]]]["image"]).convert("L").resize((128, 128), Image.Resampling.BOX)
        guide[i] = np.asarray(im, np.float32) / 255
    base = np.array([miou(lg[i] > 0, gt[i]) for i in idx])
    cats = [category(gt[i], bool(full[i])) for i in idx]
    clean = np.array([not full[i] for i in idx])
    sc = {}
    for r, eps, T in GRID:
        s = []
        for i in idx:
            p = 1 / (1 + np.exp(-lg[i] / T))
            s.append(miou(guided(guide[i], p, r, eps) > 0.5, gt[i]))
        sc[(r, eps, T)] = np.array(s)
    F = folds[idx]
    print(f"=== {a.method}: {len(idx)} OOF images, base {base.mean():.4f}")
    for scope, mask in (("all images", np.ones(len(idx), bool)), ("clean background only", clean)):
        scs = {p: np.where(mask, v, base) for p, v in sc.items()}
        best = max(scs, key=lambda p: scs[p].mean()); d = scs[best] - base
        nest, picks = nested(scs, base, F)
        print(f"  {scope}: in-sample best {best} {d.mean():+.4f} (better/worse {(d > 1e-9).sum()}/{(d < -1e-9).sum()}); "
              f"NESTED {(nest - base).mean():+.4f} picks {picks}")
        by = defaultdict(list)
        for c, v in zip(cats, nest - base):
            by[c].append(v)
        print("    nested by category: " + ", ".join(f"{c} n={len(v)} {np.mean(v):+.4f}" for c, v in sorted(by.items())))


if __name__ == "__main__":
    main()
