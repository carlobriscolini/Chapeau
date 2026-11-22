"""Evaluate observed-vs-expected contribution maps (research.contrib_maps) on marked-background train images (holes).

Part 1: per-image AUC of "this cell is the hole" for each map (higher score = more hole-like), 0.5 = chance.
Part 2: does adding the maps to the OOF ensemble logits improve mIoU? Per-cell logistic regression on
[ensemble logit, z-scored maps], fitted on the other folds and applied to the held-out fold (no leakage), compared
with the ensemble alone on the same images.

    python -m research.contrib_eval --method pixelseal"""
from __future__ import annotations

import argparse

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize
from scipy.stats import rankdata

from research.analyze_cache import miou, oof_logits


def auc(score: np.ndarray, label: np.ndarray) -> float:
    pos, n1 = label.astype(bool), int(label.sum()); n0 = label.size - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(score)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def z(m: np.ndarray) -> np.ndarray:
    m = m.astype(np.float32)
    return (m - m.mean()) / (m.std() + 1e-6)


def fit_logreg(X: np.ndarray, y: np.ndarray, l2: float = 1e-3) -> np.ndarray:
    X1 = np.c_[X, np.ones(len(X))]
    def f(w):
        t = X1 @ w
        loss = np.mean(np.logaddexp(0, t) - y * t) + l2 * (w[:-1] ** 2).sum()
        g = X1.T @ (1 / (1 + np.exp(-t)) - y) / len(y); g[:-1] += 2 * l2 * w[:-1]
        return loss, g
    return minimize(f, np.zeros(X1.shape[1]), jac=True, method="L-BFGS-B").x


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--method", required=True)
    a = ap.parse_args()
    z_c = np.load(f"outputs/cache/{a.method}.npz"); m = np.load(f"outputs/contrib/{a.method}.npz")
    assert z_c["names_train"].tolist() == m["names_train"].tolist()
    lg, cov, _ = oof_logits(z_c, a.method, flip=False)
    gt, full, folds = z_c["gt_train"], z_c["full_train"], z_c["folds_train"]
    sm = lambda x: gaussian_filter(x.astype(np.float32), 2)
    maps = {}
    for tag in ("native", "512"):
        O, E = m[f"O_{tag}_train"].astype(np.float32), m[f"E_{tag}_train"].astype(np.float32)
        maps[f"-O_{tag}"] = -O
        maps[f"E-O_{tag} (z)"] = np.stack([z(e) - z(o) for o, e in zip(O, E)])
        maps[f"E-O_{tag} (z, smooth)"] = np.stack([sm(z(e) - z(o)) for o, e in zip(O, E)])
        maps[f"(E-O)/E_{tag}"] = (E - O) / (np.abs(E) + 1e-3)
    R = m["R_train"].astype(np.float32)
    maps["R-O_512 (z, smooth)"] = np.stack([sm(z(r) - z(o)) for r, o in zip(R, m["O_512_train"].astype(np.float32))])
    holes = np.nonzero(full & cov)[0]                      # marked background, OOF ensemble available
    maps["-ensemble (OOF)"] = -lg
    print(f"=== {a.method}: {len(holes)} marked-background OOF images. Per-image AUC for 'cell is in the hole'")
    for k, v in maps.items():
        aucs = [auc(v[i].ravel(), 1 - gt[i].ravel()) for i in holes]
        print(f"  {k:26s} mean AUC {np.nanmean(aucs):.3f}")
    # Part 2: fusion, leave-one-fold-out over the folds that have OOF ensemble logits
    feats = lambda i: np.stack([lg[i], z(m["O_512_train"][i]), z(m["E_512_train"][i]),
                                maps["E-O_512 (z, smooth)"][i], maps["E-O_native (z, smooth)"][i]], -1).reshape(-1, 5)
    rng = np.random.default_rng(0)
    base_s, fused_s = [], []
    for k in sorted(set(folds[holes].tolist())):
        tr = [i for i in holes if folds[i] != k]; te = [i for i in holes if folds[i] == k]
        X = np.concatenate([feats(i) for i in tr]); y = np.concatenate([gt[i].ravel() for i in tr]).astype(np.float32)
        sel = rng.choice(len(X), min(len(X), 400000), replace=False)
        w = fit_logreg(X[sel], y[sel])
        for i in te:
            p = (np.c_[feats(i), np.ones(128 * 128)] @ w).reshape(128, 128) > 0
            base_s.append(miou(lg[i] > 0, gt[i])); fused_s.append(miou(p, gt[i]))
        print(f"  fold {k}: weights [ens, O, E, E-O_512, E-O_native, bias] = {np.round(w, 3).tolist()}")
    d = np.array(fused_s) - np.array(base_s)
    print(f"  fusion on {len(d)} held-out marked-bg images: ensemble {np.mean(base_s):.4f} -> fused {np.mean(fused_s):.4f} "
          f"(delta {d.mean():+.4f}, better/worse {(d > 1e-9).sum()}/{(d < -1e-9).sum()})")


if __name__ == "__main__":
    main()
