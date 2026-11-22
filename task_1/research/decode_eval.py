"""Metric-aware decoding of the ensemble logits (CPU, out-of-fold on train).

The official score is the per-image mean of the two class IoUs. A mask that misses the minority class entirely
(e.g. a small hole predicted as all watermarked) scores at most 0.5, while any overlap earns partial credit, so plain
thresholding at 0 is not the score-optimal decision. Two decoders are compared with threshold 0:
  A  minimum-class rule: if the predicted class c has < kmin cells, set the k cells most likely to be c to c.
  B  expected-mIoU top-k (Nowozin, CVPR 2014): p = sigmoid(logit / T); among the top-k sets of p choose the one that
     maximises the ratio-of-expectations approximation of E[mIoU].
Parameters are chosen on the other folds and applied to the held-out fold (nested, no leakage). Images replaced by a
message rectangle (same rule as build_sub_rect) are left unchanged.

    python -m research.decode_eval --method pixelseal --rect outputs/rect/pixelseal.json:0.85:0.5"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from research.analyze_cache import category, miou, oof_logits
from research.build_sub_rect import iou1, rect_mask

N = 128 * 128
A_GRID = [(kmin, k) for kmin in (1, 16, 64, 256) for k in (16, 64, 128, 256, 512, 1024) if k >= kmin]
B_GRID = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]


def rule_a(lg: np.ndarray, kmin: int, k: int) -> np.ndarray:
    pred = lg > 0
    for c in (False, True):
        if (pred == c).sum() < kmin:
            order = np.argsort(lg if not c else -lg)        # most likely class c first
            pred = pred.copy(); pred[order[:k]] = c
            return pred
    return pred


def rule_b(lg: np.ndarray, order: np.ndarray, T: float) -> np.ndarray:
    ps = 1 / (1 + np.exp(-lg[order].astype(np.float64) / T))
    P = ps.sum(); s1 = np.concatenate([[0.0], np.cumsum(ps)]); k = np.arange(N + 1)
    i1 = s1 / np.maximum(k + P - s1, 1e-9)
    tp0 = (N - k) - (P - s1)
    i0 = tp0 / np.maximum((N - k) + (N - P) - tp0, 1e-9)
    out = np.zeros(N, bool); out[order[:int(np.argmax(i1 + i0))]] = True
    return out


def nested(scores: dict, base: np.ndarray, folds: np.ndarray) -> tuple[np.ndarray, list]:
    """Pick the best parameter on the other folds, apply it to the held-out fold."""
    out, picks = base.copy(), []
    for f in sorted(set(folds.tolist())):
        tr, te = folds != f, folds == f
        if tr.sum() == 0:
            continue
        best = max(scores, key=lambda p: scores[p][tr].mean())
        out[te] = scores[best][te]; picks.append((f, best))
    return out, picks


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--method", required=True)
    ap.add_argument("--rect", default=None, help="rects.json:tau:agree (train rows, as in build_sub_rect)")
    a = ap.parse_args()
    z = np.load(f"outputs/cache/{a.method}.npz")
    lg, cov, _ = oof_logits(z, a.method, flip=False)
    gt, full, folds, names = z["gt_train"], z["full_train"], z["folds_train"], z["names_train"].tolist()
    keep = np.nonzero(cov)[0]; fixed = set()
    if a.rect:
        path, tau, agree = a.rect.rsplit(":", 2)
        rows = {r["name"]: r for r in json.loads(Path(path).read_text())["rows"] if r["split"] == "train"}
        for i in keep:
            r = rows.get(names[i])
            if r and r["score"] >= float(tau) and iou1(lg[i] > 0, rect_mask(r["rect"])) < float(agree):
                fixed.add(int(i))
    idx = np.array([i for i in keep if int(i) not in fixed])
    g = gt[idx].reshape(len(idx), N).astype(bool); L = lg[idx].reshape(len(idx), N); F = folds[idx]
    cats = [category(gt[i], bool(full[i])) for i in idx]
    base = np.array([miou(L[j] > 0, g[j]) for j in range(len(idx))])
    gt_empty = np.array([g[j].all() or not g[j].any() for j in range(len(idx))])
    npos = (L > 0).sum(1); small = np.minimum(npos, N - npos)
    print(f"=== {a.method}: {len(idx)} OOF images (+{len(fixed)} rect-replaced, unchanged). base mIoU {base.mean():.4f}")
    print(f"  GT with an empty class: {gt_empty.mean():.3f}. Predicted minority class = 0 cells: {(small == 0).mean():.3f} "
          f"(mIoU {base[small == 0].mean() if (small == 0).any() else float('nan'):.3f}), < 16 cells: {(small < 16).mean():.3f}, "
          f"< 64 cells: {(small < 64).mean():.3f} (mIoU {base[small < 64].mean() if (small < 64).any() else float('nan'):.3f})")
    sa = {p: np.array([miou(rule_a(L[j], *p), g[j]) for j in range(len(idx))]) for p in A_GRID}
    orders = [np.argsort(-L[j]) for j in range(len(idx))]
    sb = {T: np.array([miou(rule_b(L[j], orders[j], T), g[j]) for j in range(len(idx))]) for T in B_GRID}
    for tag, sc in (("A minimum-class", sa), ("B expected-mIoU", sb)):
        best = max(sc, key=lambda p: sc[p].mean()); d = sc[best] - base
        nest, picks = nested(sc, base, F)
        print(f"  {tag}: in-sample best {best} delta {d.mean():+.4f} (better/worse {(d > 1e-9).sum()}/{(d < -1e-9).sum()}); "
              f"NESTED delta {(nest - base).mean():+.4f}, picks {picks}")
        by = defaultdict(list)
        for c, v in zip(cats, d):
            by[c].append(v)
        print("    by category (in-sample best): " + ", ".join(f"{c} n={len(v)} {np.mean(v):+.4f}" for c, v in sorted(by.items())))
    for T in B_GRID:
        print(f"    B T={T}: {sb[T].mean() - base.mean():+.4f}", end="")
    print()


if __name__ == "__main__":
    main()
