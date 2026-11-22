"""Rescue missed small edits with SAM object proposals (decision-theoretic bet on the per-image mIoU).

When the ensemble predicts a single class, the image scores ~0.49 if it has a small edit. Adding a WRONG small region
costs ~0.01 (the background IoU drops by its area), adding the RIGHT one gains ~0.3-0.5, so the bet pays off above a
~3 % hit rate. Earlier "force the top-k cells" attempts used blobby cells; here the candidates are SAM's automatic
object masks (the edits are objects), scored by the ensemble evidence for "edited" inside the mask relative to the
image. Only images predicted single-class are touched. Out-of-fold report over a grid of score thresholds.

    python -m research.sam_small --method pixelseal --out outputs/sam"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from research.analyze_cache import miou
from research.sam_refine import to128

THETAS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
MIN_AREA, MAX_AREA = 0.0005, 0.10


def proposals(gen, image: np.ndarray) -> list[np.ndarray]:
    out = []
    for r in gen.generate(image):
        m = to128(r["segmentation"])
        if MIN_AREA <= m.mean() <= MAX_AREA:
            out.append(m)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True); ap.add_argument("--data", default="release")
    ap.add_argument("--logits", default="outputs/logits"); ap.add_argument("--out", default="outputs/sam")
    ap.add_argument("--ckpt", default="outputs/sam/sam_vit_h_4b8939.pth")
    a = ap.parse_args()
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    sam = sam_model_registry["vit_h"]()
    sam.load_state_dict(torch.load(a.ckpt, map_location="cpu", weights_only=True))
    gen = SamAutomaticMaskGenerator(sam.cuda().eval(), points_per_side=24, pred_iou_thresh=0.8,
                                    stability_score_thresh=0.85, min_mask_region_area=0)
    z = np.load(Path(a.logits) / f"{a.method}.npz")
    res, t0 = {}, time.time()
    for split, names, logits in (("train", z["names_train"].tolist(), z["oof_logit"]), ("test", z["names_test"].tolist(), z["test_logit"])):
        best = []   # per image: (score, mask) of the best proposal, or None if not single-class
        for name, lg in zip(names, logits.astype(np.float32)):
            pred = lg > 0
            if 0.001 < pred.mean() < 0.999:
                best.append(None); continue
            bg = pred.mean() > 0.5
            ev = -lg if bg else lg                     # evidence for "edited"
            evn = (ev - np.median(ev)) / (np.percentile(ev, 75) - np.percentile(ev, 25) + 1e-6)
            img = np.asarray(Image.open(Path(a.data) / split / "images" / f"{name}.png").convert("RGB"))
            with torch.no_grad():
                props = proposals(gen, img)
            if not props:
                best.append((-np.inf, None, bg)); continue
            sc = [float(evn[m].mean()) for m in props]
            k = int(np.argmax(sc)); best.append((sc[k], props[k], bg))
        res[split] = (names, logits, best)
        print(f"{split}: {sum(b is not None for b in best)} single-class images, {time.time() - t0:.0f}s", flush=True)
    names, logits, best = res["train"]
    gt, folds = z["gt"], z["folds"]
    base = (logits.astype(np.float32) > 0).astype(np.uint8)

    def apply(preds, best, theta):
        out = preds.copy()
        for i, b in enumerate(best):
            if b is not None and b[1] is not None and b[0] >= theta:
                out[i] = np.where(b[1], not b[2], b[2])
        return out
    sb = np.array([miou(p, g) for p, g in zip(base, gt)])
    report = {"base_no_rect": round(float(sb.mean()), 4), "n_single": int(sum(b is not None for b in best))}
    for th in THETAS:
        st = np.array([miou(p, g) for p, g in zip(apply(base, best, th), gt)])
        d = st - sb
        report[f"theta{th}"] = {"delta": round(float(d.mean()), 4), "n_changed": int((d != 0).sum()),
                                "n_hit": int((d > 0.05).sum()),
                                "per_fold_delta": [round(float(d[folds == k].mean()), 4) for k in sorted(set(folds.tolist()))]}
    print(json.dumps({"method": a.method, "report": report}), flush=True)
    tn, tl, tb = res["test"]
    tbase = (tl.astype(np.float32) > 0).astype(np.uint8)
    path = Path(a.out) / f"{a.method}_small.npz"
    with open(path, "r+b" if path.exists() else "wb") as fh:
        fh.truncate(0)
        np.savez_compressed(fh, thetas=np.array(THETAS), names_test=np.array(tn), names_train=np.array(names),
                            test=np.stack([apply(tbase, tb, th) for th in THETAS]).astype(bool),
                            test_changed=np.stack([(apply(tbase, tb, th) != tbase).any((1, 2)) for th in THETAS]))


if __name__ == "__main__":
    main()
