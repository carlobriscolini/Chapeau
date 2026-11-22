"""Hysteresis + shape verification: rescue edits whose evidence is present but below the decision threshold.

The oracle test showed that for ~30 % of the missed small PS/TM edits the ensemble evidence IS concentrated in the true
region, just below 0. Lowering the threshold everywhere only adds noise blobs. As in Canny's hysteresis, weak but
image-anomalous evidence (standardised per image) proposes candidate components; each candidate is then verified by
Segment Anything: it is accepted only if SAM finds an object there whose mask agrees with the candidate (IoU >= tau)
and is not huge. Noise blobs rarely have an object's shape; an edit is an object. Candidates never overlap the
current prediction, which is left untouched. Out-of-fold report over (k, tau).

    python -m research.sam_weak --method pixelseal --rect outputs/rect/pixelseal.json:0.85:0.5"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from research.analyze_cache import miou
from research.sam_refine import iou, to128

KS = (1.5, 2.0, 3.0)
TAUS = (0.5, 0.6, 0.7)
MIN_CELLS, MAX_FRAC = 6, 0.10


def candidates(lg: np.ndarray, predictor, image: np.ndarray, k_min: float):
    pred = lg > 0
    bg = pred.mean() > 0.5
    ev = -lg if bg else lg
    evn = (ev - np.median(ev)) / (np.percentile(ev, 75) - np.percentile(ev, 25) + 1e-6)
    edited = pred != bg
    weak = ((evn > k_min) & ~cv2.dilate(edited.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(weak, connectivity=8)
    comps = [lab == j for j in range(1, n) if stats[j, cv2.CC_STAT_AREA] >= MIN_CELLS]
    out = []
    if not comps:
        return bg, evn, out
    predictor.set_image(image)
    for c in comps:
        ys, xs = np.nonzero(c)
        box = np.array([xs.min() * 4 - 6, ys.min() * 4 - 6, xs.max() * 4 + 10, ys.max() * 4 + 10]).clip(0, 511)
        cy, cx = np.unravel_index(np.argmax(np.where(c, evn, -np.inf)), c.shape)
        masks, _, _ = predictor.predict(point_coords=np.array([[cx * 4 + 2, cy * 4 + 2]]), point_labels=np.array([1]),
                                        box=box, multimask_output=True)
        cand = [to128(m) for m in masks]
        ious = [iou(m, c) for m in cand]; j = int(np.argmax(ious))
        out.append((float(evn[c].mean()), cand[j], ious[j]))
    return bg, evn, out


def apply(pred, bg, cands, k, tau):
    edited = pred != bg
    for score, m, s in cands:
        if score >= k and s >= tau and m.mean() <= MAX_FRAC:
            edited = edited | m
    return np.where(edited, not bg, bg).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True); ap.add_argument("--data", default="release")
    ap.add_argument("--logits", default="outputs/logits"); ap.add_argument("--out", default="outputs/sam")
    ap.add_argument("--ckpt", default="outputs/sam/sam_vit_h_4b8939.pth")
    a = ap.parse_args()
    from segment_anything import SamPredictor, sam_model_registry
    sam = sam_model_registry["vit_h"]()
    sam.load_state_dict(torch.load(a.ckpt, map_location="cpu", weights_only=True))
    predictor = SamPredictor(sam.cuda().eval())
    z = np.load(Path(a.logits) / f"{a.method}.npz")
    res, t0 = {}, time.time()
    for split, names, logits in (("train", z["names_train"].tolist(), z["oof_logit"]), ("test", z["names_test"].tolist(), z["test_logit"])):
        rows = []
        for name, lg in zip(names, logits.astype(np.float32)):
            img = np.asarray(Image.open(Path(a.data) / split / "images" / f"{name}.png").convert("RGB"))
            with torch.no_grad():
                bg, _, cands = candidates(lg, predictor, img, min(KS))
            rows.append((lg > 0, bg, cands))
        res[split] = (names, rows)
        print(f"{split}: {sum(len(r[2]) for r in rows)} candidates, {time.time() - t0:.0f}s", flush=True)
    names, rows = res["train"]
    gt, folds = z["gt"], z["folds"]
    sb = np.array([miou(p.astype(np.uint8), g) for (p, _, _), g in zip(rows, gt)])
    report = {"base_no_rect": round(float(sb.mean()), 4)}
    for k in KS:
        for t in TAUS:
            st = np.array([miou(apply(p, bg, c, k, t), g) for (p, bg, c), g in zip(rows, gt)])
            d = st - sb
            report[f"k{k}_tau{t}"] = {"delta": round(float(d.mean()), 4), "changed": int((d != 0).sum()),
                                      "hits": int((d > 0.05).sum()), "losses": int((d < -0.02).sum()),
                                      "per_fold": [round(float(d[folds == f].mean()), 4) for f in sorted(set(folds.tolist()))]}
    print(json.dumps({"method": a.method, "report": report}), flush=True)
    tn, trows = res["test"]
    keys = [(k, t) for k in KS for t in TAUS]
    path = Path(a.out) / f"{a.method}_weak.npz"
    with open(path, "r+b" if path.exists() else "wb") as fh:
        fh.truncate(0)
        np.savez_compressed(fh, keys=np.array([f"k{k}_tau{t}" for k, t in keys]), names_test=np.array(tn),
                            add=np.stack([np.stack([(apply(p, bg, c, k, t) != p.astype(np.uint8)) for p, bg, c in trows])
                                          for k, t in keys]).astype(bool))


if __name__ == "__main__":
    main()
