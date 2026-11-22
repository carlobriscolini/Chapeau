"""TrustMark removal-network residual as a local presence map (TrustMark only; Bui et al. 2023, "TrustMark-RM").

The official package ships a remover (trustmark_rm_Q) trained to take the watermark out: res = remover(x256) - x256.
Where the image is watermarked, res must cancel the watermark (energy ~ watermark energy); where it is clean there is
nothing to cancel. Unlike the decoder's per-position logits, the remover is a local image-to-image network, so its
residual should be spatially precise. Maps on the 128 grid (float16, ONE .npz):
  E   mean |res| over channels            N   E / (local high-pass energy of the image + eps)
    python -m research.rm_maps --out outputs/rm/trustmark.npz --device cpu      # compute (train + test)
    python -m research.rm_maps --out outputs/rm/trustmark.npz --eval           # AUC + leave-fold-out fusion (CPU)"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from wmloc.data import LocalisationDataset
from wmloc.watermarks import load_watermark


def hp_energy(x: torch.Tensor) -> torch.Tensor:
    k = torch.tensor([1., 4., 6., 4., 1.]); k = (k[:, None] * k[None, :]) / 256
    blur = F.conv2d(F.pad(x, (2,) * 4, mode="reflect"), k.to(x)[None, None].repeat(3, 1, 1, 1), groups=3)
    return (x - blur).abs().mean(1, keepdim=True)


@torch.no_grad()
def compute(a):
    torch.set_num_threads(a.threads)
    tm = load_watermark("trustmark", "models", a.device).tm
    out = {}
    for split in ("train", "test"):
        ds = LocalisationDataset("release", split, methods=["trustmark"]); E, N, names = [], [], []
        for b in range(0, len(ds), a.bs):
            items = [ds[i] for i in range(b, min(b + a.bs, len(ds)))]
            x = torch.from_numpy(np.stack([it["image"] for it in items])).permute(0, 3, 1, 2).float().to(a.device) / 255
            x256 = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False) * 2 - 1
            res = tm.removal(x256).clamp(-1, 1) - x256
            e = F.avg_pool2d(res.abs().mean(1, keepdim=True), 2)                       # B x 1 x 128 x 128
            h = F.avg_pool2d(hp_energy(x256), 2)
            E.append(e[:, 0].cpu().numpy().astype(np.float16)); N.append((e / (h + 0.01))[:, 0].cpu().numpy().astype(np.float16))
            names += [it["image_name"] for it in items]
        out[f"E_{split}"], out[f"N_{split}"], out[f"names_{split}"] = np.concatenate(E), np.concatenate(N), np.array(names)
        print(f"{split}: {len(names)} images", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); np.savez(a.out, **out)


def evaluate(a):
    from scipy.ndimage import gaussian_filter
    from research.analyze_cache import category, miou, oof_logits
    from research.contrib_eval import auc, fit_logreg, z
    zc, m = np.load("outputs/cache/trustmark.npz"), np.load(a.out)
    assert zc["names_train"].tolist() == m["names_train"].tolist()
    lg, cov, _ = oof_logits(zc, "trustmark", flip=False)
    gt, full, folds = zc["gt_train"], zc["full_train"], zc["folds_train"]
    sm = lambda v: gaussian_filter(v.astype(np.float32), 2)
    maps = {"E": m["E_train"].astype(np.float32), "N": m["N_train"].astype(np.float32)}
    maps["E smooth"] = np.stack([sm(v) for v in maps["E"]]); maps["N smooth"] = np.stack([sm(v) for v in maps["N"]])
    maps["ensemble (OOF)"] = lg
    cats = np.array([category(gt[i], bool(full[i])).rsplit("/", 1)[0] for i in range(len(gt))])
    for fam in ("marked_bg", "clean_bg/irregular", "clean_bg/rect"):
        idx = [i for i in np.nonzero(cov)[0] if cats[i] == fam and 0 < gt[i].mean() < 1]
        print(f"{fam} (n={len(idx)}): " + "  ".join(
            f"{k} AUC {np.nanmean([auc(v[i].ravel(), gt[i].ravel()) for i in idx]):.3f}" for k, v in maps.items()))
    feats = lambda i: np.stack([lg[i], z(maps["E smooth"][i]), z(maps["N smooth"][i]), z(maps["E"][i])], -1).reshape(-1, 4)
    ids = np.nonzero(cov)[0]; rng = np.random.default_rng(0); base, fused = [], []
    for k in sorted(set(folds[ids].tolist())):
        tr = [i for i in ids if folds[i] != k]; te = [i for i in ids if folds[i] == k]
        if not tr:
            continue
        X = np.concatenate([feats(i) for i in tr]); y = np.concatenate([gt[i].ravel() for i in tr]).astype(np.float32)
        sel = rng.choice(len(X), min(len(X), 400000), replace=False); w = fit_logreg(X[sel], y[sel])
        for i in te:
            p = (np.c_[feats(i), np.ones(128 * 128)] @ w).reshape(128, 128) > 0
            base.append(miou(lg[i] > 0, gt[i])); fused.append(miou(p, gt[i]))
        print(f"  fold {k}: weights [ens, E_s, N_s, E, bias] = {np.round(w, 3).tolist()}")
    d = np.array(fused) - np.array(base)
    print(f"fusion on {len(d)} OOF images: ensemble {np.mean(base):.4f} -> fused {np.mean(fused):.4f} ({d.mean():+.4f}, "
          f"better/worse {(d > 1e-9).sum()}/{(d < -1e-9).sum()})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--eval", action="store_true")
    ap.add_argument("--device", default="cpu"); ap.add_argument("--bs", type=int, default=8); ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(); t0 = time.time()
    evaluate(a) if a.eval else compute(a)
    print(f"done {time.time() - t0:.0f}s")
