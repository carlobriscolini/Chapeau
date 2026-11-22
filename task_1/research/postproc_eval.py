"""Out-of-fold evaluation of mask post-processing, one variant at a time (research.analyze_cache protocol).

Variants (applied to the MINORITY region, i.e. the class opposite to the predicted background):
  cc_min<k>   drop minority connected components smaller than k cells (nnU-Net style component removal)
  largest     keep only the largest minority component (nnU-Net "keep largest connected component")
  fill        fill holes inside minority regions
Each variant is adopted per method only if it improves the OOF mean, as nnU-Net decides post-processing on CV
(Isensee et al., Nature Methods 2021).

Also reports agreement between the mask's background polarity and the whole-image bit-accuracy test.

    python -m research.postproc_eval outputs/cache/*.npz"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import ndimage

from research.analyze_cache import category, miou, oof_logits

POLARITY_THR = {"pixelseal": 0.7988, "trustmark": 0.78, "stegastamp": 0.755}   # research/results/polarity_fill.json


def background(pred: np.ndarray) -> int:
    return int(pred.mean() >= 0.5)


def postprocess(pred: np.ndarray, variant: str) -> np.ndarray:
    bg = background(pred)
    minority = pred != bg
    if not minority.any():
        return pred
    if variant.startswith("cc_min"):
        k = int(variant[6:])
        lab, n = ndimage.label(minority)
        sizes = ndimage.sum(minority, lab, range(1, n + 1))
        keep = np.isin(lab, 1 + np.nonzero(sizes >= k)[0])
        minority = keep
    elif variant == "largest":
        lab, n = ndimage.label(minority)
        sizes = ndimage.sum(minority, lab, range(1, n + 1))
        minority = lab == 1 + int(np.argmax(sizes))
    elif variant == "fill":
        minority = ndimage.binary_fill_holes(minority)
    out = np.full_like(pred, bg)
    out[minority] = 1 - bg
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("caches", nargs="+"); a = ap.parse_args()
    variants = ["none", "cc_min4", "cc_min16", "cc_min64", "largest", "fill"]
    for path in a.caches:
        z = np.load(path); method = Path(path).stem
        lg, cov, _ = oof_logits(z, method, flip=False)
        gt, full, folds, acc = z["gt_train"], z["full_train"], z["folds_train"], z["bitacc_train"]
        idx = np.nonzero(cov)[0]
        pred = {i: (lg[i] > 0).astype(np.uint8) for i in idx}
        print(f"\n=== {method} ({len(idx)} OOF images, folds {sorted(set(folds[idx].tolist()))})")
        base = {i: miou(pred[i], gt[i]) for i in idx}
        for v in variants:
            sc = {i: miou(postprocess(pred[i], v), gt[i]) for i in idx}
            d = np.array([sc[i] - base[i] for i in idx])
            per_fold = [np.mean([sc[i] - base[i] for i in idx if folds[i] == k]) for k in sorted(set(folds[idx].tolist()))]
            cats = {}
            for i in idx:
                cats.setdefault(category(gt[i], bool(full[i])).split("/")[0], []).append(sc[i] - base[i])
            print(f"  {v:9s} mean {np.mean(list(sc.values())):.4f}  delta {d.mean():+.4f}  better/worse {int((d > 1e-9).sum())}/{int((d < -1e-9).sum())}"
                  f"  per-fold " + " ".join(f"{x:+.4f}" for x in per_fold)
                  + "  | " + " ".join(f"{c} {np.mean(v_):+.4f}" for c, v_ in sorted(cats.items())))
        # polarity agreement: mask background vs bit-accuracy test vs truth
        bit_full = acc[idx] > POLARITY_THR[method]
        mask_full = np.array([background(pred[i]) == 1 for i in idx])
        true_full = full[idx].astype(bool)
        dis = np.nonzero(bit_full != mask_full)[0]
        print(f"  polarity: mask right {np.mean(mask_full == true_full):.3f}, bit-acc right {np.mean(bit_full == true_full):.3f}, "
              f"disagree on {len(dis)} images, of which bit-acc right on {int(np.sum(bit_full[dis] == true_full[dis]))}")
        for j in dis:
            i = idx[j]
            print(f"    {z['names_train'][i]} bitacc {acc[i]:.3f} true_full {bool(full[i])} mask_fg_frac {pred[i].mean():.3f} "
                  f"gt_fg_frac {gt[i].mean():.3f} mIoU {base[i]:.3f}")


if __name__ == "__main__":
    main()
