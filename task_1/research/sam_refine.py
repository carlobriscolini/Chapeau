"""Boundary refinement of the predicted edited regions with Segment Anything (Kirillov et al., ICCV 2023).

Why: on LARGE edits the PixelSeal / TrustMark ensembles find the right region with the right area but blurry borders
(OOF recall 0.76-0.85, precision 0.82-0.87; StegaStamp 0.94/0.95), because those watermarks live at 256 px. The edits
are objects with sharp outlines (transplanted / added / altered objects), which SAM recovers from the image alone.
Each connected component of the predicted edited region (the cells whose class differs from the predicted background
polarity) prompts SAM with its box (+ its most confident cell as a positive point); among SAM's 3 masks the one that
agrees best with the component replaces it if their IoU >= tau. Missed edits (no component) are untouched; the
rectangle rule is applied afterwards as in build_sub_v2. No training: tau is the only knob, reported per fold.

    python -m research.sam_refine --method trustmark --out outputs/sam"""
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
from research.build_sub_v2 import apply_rect

TAUS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
MIN_CELLS = 12


def to128(m512: np.ndarray) -> np.ndarray:   # official rule: a cell is 1 if >= 50 % of its 4 x 4 pixels are
    return m512.reshape(128, 4, 128, 4).mean((1, 3)) >= 0.5


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0


VARIANTS = ("A", "B", "C", "D")


def soft_iou(m: np.ndarray, prob: np.ndarray, box128) -> float:
    x0, y0, x1, y1 = box128
    mm, pp = m[y0:y1, x0:x1].astype(np.float32), prob[y0:y1, x0:x1]
    return float((mm * pp).sum() / (mm + pp - mm * pp).sum().clip(1e-6))


def refine(pred, logit, image, predictor):
    """pred: 128 bool. Returns bg and, per variant, a list of (component, chosen SAM mask on the 128 grid, score):
    A box + top point, pick by IoU with the component; B box + 3 positive / 3 negative points; C box + our logit map
    as SAM's mask prompt; D as A but pick by soft IoU with the ensemble probability."""
    bg = pred.mean() > 0.5
    edited = (pred != bg).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(edited, connectivity=8)
    comps = [(lab == k) for k in range(1, n) if stats[k, cv2.CC_STAT_AREA] >= MIN_CELLS]
    out = {v: [] for v in VARIANTS}
    if not comps:
        return bg, out
    predictor.set_image(image)
    ev = -logit if bg else logit                       # evidence for "edited" (positive inside the edited region)
    prob = 1 / (1 + np.exp(-ev))
    for c in comps:
        ys, xs = np.nonzero(c)
        b128 = (max(xs.min() - 2, 0), max(ys.min() - 2, 0), min(xs.max() + 3, 128), min(ys.max() + 3, 128))
        box = np.array([xs.min() * 4 - 6, ys.min() * 4 - 6, xs.max() * 4 + 10, ys.max() * 4 + 10]).clip(0, 511)
        inside = np.where(c, ev, -np.inf); order = np.argsort(inside, axis=None)[::-1]
        pos = [np.unravel_index(i, c.shape) for i in order[:3]]
        ring = np.zeros_like(c); ring[b128[1]:b128[3], b128[0]:b128[2]] = True; ring &= ~c & (lab == 0)
        outside = np.where(ring, -ev, -np.inf); oo = np.argsort(outside, axis=None)[::-1]
        neg = [np.unravel_index(i, c.shape) for i in oo[:3] if np.isfinite(outside.flat[i])]
        pt = lambda cells: np.array([[x * 4 + 2, y * 4 + 2] for y, x in cells])
        mA, _, _ = predictor.predict(point_coords=pt(pos[:1]), point_labels=np.array([1]), box=box, multimask_output=True)
        cA = [to128(m) for m in mA]
        iA = [iou(m, c) for m in cA]; k = int(np.argmax(iA)); out["A"].append((c, cA[k], iA[k]))
        sD = [soft_iou(m, prob, b128) for m in cA]; k = int(np.argmax(sD)); out["D"].append((c, cA[k], iou(cA[k], c)))
        cells = pos + neg
        mB, _, _ = predictor.predict(point_coords=pt(cells), point_labels=np.array([1] * len(pos) + [0] * len(neg)),
                                     box=box, multimask_output=True)
        cB = [to128(m) for m in mB]; iB = [iou(m, c) for m in cB]; k = int(np.argmax(iB)); out["B"].append((c, cB[k], iB[k]))
        prompt = np.full((128, 128), -8.0, np.float32)
        prompt[b128[1]:b128[3], b128[0]:b128[2]] = np.clip(ev[b128[1]:b128[3], b128[0]:b128[2]], -8, 8)
        prompt = cv2.resize(prompt, (256, 256), interpolation=cv2.INTER_LINEAR)[None]
        mC, _, _ = predictor.predict(box=box, mask_input=prompt, multimask_output=False)
        cC = to128(mC[0]); out["C"].append((c, cC, iou(cC, c)))
    return bg, out


def apply_refine(pred, bg, comps, tau):
    edited = pred != bg
    for c, m, s in comps:
        if s >= tau:
            edited = (edited & ~c) | m
    return np.where(edited, not bg, bg).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True); ap.add_argument("--data", default="release")
    ap.add_argument("--logits", default="outputs/logits"); ap.add_argument("--out", default="outputs/sam")
    ap.add_argument("--ckpt", default="outputs/sam/sam_vit_h_4b8939.pth")
    ap.add_argument("--rect", default=None, help="rects.json:tau:agree")
    ap.add_argument("--tag", default="", help="suffix of the output npz")
    a = ap.parse_args()
    from segment_anything import SamPredictor, sam_model_registry
    sam = sam_model_registry["vit_h"]()
    sam.load_state_dict(torch.load(a.ckpt, map_location="cpu", weights_only=True))
    predictor = SamPredictor(sam.cuda().eval())
    rule = None
    if a.rect:
        path, tau, agree = a.rect.rsplit(":", 2); rows = {"train": {}, "test": {}}
        for r in json.loads(Path(path).read_text())["rows"]:
            rows[r["split"]][r["name"]] = r
        rule = {"tau": float(tau), "agree": float(agree), "rows": rows}
    z = np.load(Path(a.logits) / f"{a.method}.npz")
    res, t0 = {}, time.time()
    for split, names, logits in (("train", z["names_train"].tolist(), z["oof_logit"]), ("test", z["names_test"].tolist(), z["test_logit"])):
        outs = {(v, t): [] for v in VARIANTS for t in TAUS}; base = []; ncomp = 0
        for name, lg in zip(names, logits.astype(np.float32)):
            img = np.asarray(Image.open(Path(a.data) / split / "images" / f"{name}.png").convert("RGB"))
            pred = lg > 0
            with torch.no_grad():
                bg, comps = refine(pred, lg, img, predictor)
            ncomp += len(comps["A"]); base.append(pred.astype(np.uint8))
            for v in VARIANTS:
                for t in TAUS:
                    outs[(v, t)].append(apply_refine(pred, bg, comps[v], t))
        base, _ = apply_rect(np.stack(base), names, rule, split)
        outs = {t: apply_rect(np.stack(v), names, rule, split)[0] for t, v in outs.items()}
        res[split] = (names, base, outs)
        print(f"{split}: {len(names)} images, {ncomp} components refined-candidates, {time.time() - t0:.0f}s", flush=True)
    names, base, outs = res["train"]
    gt, folds = z["gt"], z["folds"]
    sb = np.array([miou(p, g) for p, g in zip(base, gt)])
    report = {"base": round(float(sb.mean()), 4)}
    for v, t in outs:
        st = np.array([miou(p, g) for p, g in zip(outs[(v, t)], gt)])
        report[f"{v}_tau{t}"] = {"oof": round(float(st.mean()), 4),
                             "per_fold_delta": [round(float((st - sb)[folds == k].mean()), 4) for k in sorted(set(folds.tolist()))]}
    print(json.dumps({"method": a.method, "report": report}), flush=True)
    path = Path(a.out) / f"{a.method}{a.tag}.npz"
    with open(path, "r+b" if path.exists() else "wb") as fh:
        fh.truncate(0)
        keys = list(outs)
        np.savez_compressed(fh, keys=np.array([f"{v}_tau{t}" for v, t in keys]), names_train=np.array(names),
                            names_test=np.array(res["test"][0]),
                            train=np.stack([outs[k] for k in keys]).astype(bool),
                            test=np.stack([res["test"][2][k] for k in keys]).astype(bool))


if __name__ == "__main__":
    main()
