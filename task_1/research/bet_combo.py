"""Bets on images where the ensemble found (almost) nothing: SAM object vs max-evidence rectangle.

Per-image mIoU makes "add one region" an asymmetric bet on images predicted (almost) single-class: a wrong small
region costs ~0.01, a right one gains ~0.3-0.5. Two proposal families:
  sam   the SAM automatic-mask object with the highest standardised ensemble evidence (research.sam_small);
  rect  the axis-aligned rectangle maximising sum(evidence - C) (exact 2D max-subarray), aspect within the method's
        limit: the task watermarks a RECTANGULAR subset, and many misses on clean backgrounds are tiny watermarked
        patches with no further edit (seen on train), where scattered per-cell evidence adds up inside the right box.
Rules compared out of fold: sam only, rect only, split (rect if the predicted background is clean, sam if marked),
best (higher mean evidence). Candidate images: predicted minority < PMAX of the image.

    python -m research.bet_combo --method stegastamp --logits outputs/logits13 --out outputs/sam13"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from research.analyze_cache import miou
from research.sam_small import proposals

MAXA = {"pixelseal": 16, "trustmark": 2, "stegastamp": 16}
C, PMAX, MAX_RECT = 2.0, 0.005, 0.05
RULES = ("none", "sam", "rect", "split", "best")


def best_rect(s: torch.Tensor):
    S = torch.cat([torch.zeros(1, 128), torch.cumsum(s, 0)])
    col = S[None, :, :] - S[:, None, :]
    P = torch.cat([torch.zeros(129, 129, 1), torch.cumsum(col, 2)], 2)
    pmin, imin = torch.cummin(P, 2)
    gain = P[:, :, 1:] - pmin[:, :, :-1]
    valid = (torch.arange(129)[None, :] > torch.arange(129)[:, None])[:, :, None]
    gain = torch.where(valid, gain, torch.tensor(-1e9))
    v, flat = gain.reshape(-1).max(0); t, b1, j = np.unravel_index(int(flat), gain.shape)
    return int(t), int(b1), int(imin[t, b1, j]), int(j + 1)


def decide(lg: np.ndarray, img_fn, gen, method: str):
    pred = (lg > 0)
    bg = pred.mean() > 0.5
    minority = (pred != bg).mean()
    if minority >= PMAX:
        return None
    ev = -lg if bg else lg
    evn = (ev - np.median(ev)) / (np.percentile(ev, 75) - np.percentile(ev, 25) + 1e-6)
    out = {"bg": bg, "sam": None, "rect": None}
    t, b, l, r = best_rect(torch.from_numpy(evn - C).float())
    h, w = b - t, r - l
    if 4 <= h * w <= MAX_RECT * 128 * 128 and max(h, w) / max(min(h, w), 1) <= MAXA[method]:
        m = np.zeros((128, 128), bool); m[t:b, l:r] = True
        out["rect"] = (float(evn[m].mean()), m)
    with torch.no_grad():
        props = proposals(gen, img_fn())
    if props:
        sc = [float(evn[m].mean()) for m in props]; k = int(np.argmax(sc))
        out["sam"] = (sc[k], props[k])
    return out


def apply(pred: np.ndarray, d, rule: str) -> np.ndarray:
    if d is None or rule == "none":
        return pred
    choice = None
    if rule == "sam":
        choice = d["sam"]
    elif rule == "rect":
        choice = d["rect"]
    elif rule == "split":
        choice = d["sam"] if d["bg"] else (d["rect"] or d["sam"])
    elif rule == "best":
        cands = [c for c in (d["sam"], d["rect"]) if c is not None]
        choice = max(cands, key=lambda c: c[0]) if cands else None
    if choice is None:
        return pred
    bg = d["bg"]
    return np.where(choice[1], not bg, bg).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True); ap.add_argument("--data", default="release")
    ap.add_argument("--logits", default="outputs/logits13"); ap.add_argument("--out", default="outputs/sam13")
    ap.add_argument("--ckpt", default="outputs/sam/sam_vit_h_4b8939.pth")
    a = ap.parse_args()
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    sam = sam_model_registry["vit_h"]()
    sam.load_state_dict(torch.load(a.ckpt, map_location="cpu", weights_only=True))
    gen = SamAutomaticMaskGenerator(sam.cuda().eval(), points_per_side=24, pred_iou_thresh=0.8,
                                    stability_score_thresh=0.85, min_mask_region_area=0)
    z = np.load(Path(a.logits) / f"{a.method}.npz"); t0 = time.time(); res = {}
    for split, names, logits in (("train", z["names_train"].tolist(), z["oof_logit"]), ("test", z["names_test"].tolist(), z["test_logit"])):
        ds = []
        for name, lg in zip(names, logits.astype(np.float32)):
            fn = lambda: np.asarray(Image.open(Path(a.data) / split / "images" / f"{name}.png").convert("RGB"))
            ds.append(decide(lg, fn, gen, a.method))
        res[split] = (names, logits.astype(np.float32), ds)
        print(f"{split}: {sum(d is not None for d in ds)} candidate images, {time.time() - t0:.0f}s", flush=True)
    names, logits, ds = res["train"]; gt, folds = z["gt"], z["folds"]
    base = [(l > 0).astype(np.uint8) for l in logits]
    sb = np.array([miou(p, g) for p, g in zip(base, gt)])
    report = {}
    for rule in RULES[1:]:
        st = np.array([miou(apply(p, d, rule), g) for p, d, g in zip(base, ds, gt)]); dl = st - sb
        report[rule] = {"delta": round(float(dl.mean()), 4), "hits": int((dl > 0.05).sum()), "losses": int((dl < -0.005).sum()),
                        "per_fold": [round(float(dl[folds == k].mean()), 4) for k in sorted(set(folds.tolist()))]}
    print(json.dumps({"method": a.method, "report": report}), flush=True)
    tn, tl, tds = res["test"]
    tbase = [(l > 0).astype(np.uint8) for l in tl]
    path = Path(a.out) / f"{a.method}_bet.npz"
    with open(path, "r+b" if path.exists() else "wb") as fh:
        fh.truncate(0)
        np.savez_compressed(fh, rules=np.array(RULES[1:]), names_test=np.array(tn),
                            test=np.stack([np.stack([apply(p, d, r) for p, d in zip(tbase, tds)]) for r in RULES[1:]]).astype(bool),
                            changed=np.stack([np.array([d is not None for d in tds]) for _ in RULES[1:]]))


if __name__ == "__main__":
    main()
