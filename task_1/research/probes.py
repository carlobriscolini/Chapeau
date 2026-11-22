"""Three probes that decide the design of the trained localiser (train split, in-sample diagnostics):

  crop_decode  For clean-background images, decode the crop of the ground-truth marked region (its bounding box)
               and a same-size crop elsewhere. If the crop decodes the message, a rectangle search can localise.
  template     For every image, R = embed(image, message) - image is a message-specific template. The local
               normalised correlation between high-pass(image) and high-pass(R) is a candidate presence map.
  sim_to_real  Composites on clean COCO photos with real train masks, scored with the same zero-shot maps as the
               real images. Similar AUCs mean synthetic training data should transfer.

    python -m research.probes --coco .downloads/coco/val2017 --out outputs/probes"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from research.decoders import Decoder, signed
from research.metrics import auc_per_image
from research.synth import bbox, composite, load_clean, patch_rect
from research.zero_shot import METHODS, evaluate, gaussian_blur, markdown, to_grid
from wmloc.data import LocalisationDataset

MIN_CROP = 16


def crop_decode(dec: Decoder, items: list[dict], rng: np.random.Generator) -> dict:
    rows = []
    for it in items:
        if it["full_watermark"]:
            continue
        y0, y1, x0, x1 = bbox(it["mask"])
        if min(y1 - y0, x1 - x0) < MIN_CROP:
            continue
        h, w = y1 - y0, x1 - x0
        cy = int(rng.integers(0, 512 - h + 1)); cx = int(rng.integers(0, 512 - w + 1))
        crops = [it["image"][y0:y1, x0:x1], it["image"][cy:cy + h, cx:cx + w]]
        s = signed(torch.from_numpy(it["message"]))[None].cuda()
        accs = []
        for c in crops:
            x = torch.from_numpy(np.ascontiguousarray(c)).permute(2, 0, 1)[None].float().cuda() / 255
            with torch.no_grad():
                accs.append(float(((dec.logits(x) > 0).float() == (s > 0).float()).float().mean()))
        fill = it["mask"][y0:y1, x0:x1].mean()
        rows.append({"name": it["image_name"], "bbox_acc": accs[0], "random_acc": accs[1], "fill": float(fill),
                     "area": h * w / 512 ** 2})
    acc = np.array([r["bbox_acc"] for r in rows]); rnd = np.array([r["random_acc"] for r in rows])
    rect = np.array([r["fill"] > 0.999 for r in rows])
    summ = lambda a: {"n": int(len(a)), "mean": float(a.mean()) if len(a) else None,
                      "share_ge_0.9": float((a >= 0.9).mean()) if len(a) else None}
    return {"bbox_all": summ(acc), "bbox_exact_rectangle": summ(acc[rect]), "bbox_irregular": summ(acc[~rect]),
            "random_same_size": summ(rnd), "rows": rows}


def template_maps(wm, items: list[dict], sigma_cells: float = 2.0) -> np.ndarray:
    out = []
    for it in items:
        img = it["image"]
        r = wm.embed(img, it["message"]).astype(np.float32) - img.astype(np.float32)
        x = torch.from_numpy(np.stack([img.astype(np.float32), r])).permute(0, 3, 1, 2).cuda() / 255
        hp = x - gaussian_blur(x, 2.0)
        a, b = hp[0:1], hp[1:2]
        k = 4 * sigma_cells
        ab = gaussian_blur((a * b).sum(1, keepdim=True), k); aa = gaussian_blur((a * a).sum(1, keepdim=True), k)
        bb = gaussian_blur((b * b).sum(1, keepdim=True), k)
        out.append(to_grid((ab / (aa * bb).sqrt().clamp_min(1e-12))[:, 0]).cpu().numpy()[0])
    return np.stack(out)


def template(wm, items: list[dict]) -> dict:
    maps = template_maps(wm, items)
    gt = np.stack([it["mask128"] for it in items]); full = np.array([bool(it["full_watermark"]) for it in items])
    auc = auc_per_image(maps, gt)
    return {"auc_mean": float(np.nanmean(auc)), "auc_bg_clean": float(np.nanmean(auc[~full])),
            "auc_bg_marked": float(np.nanmean(auc[full]))}


MAX_ASPECT = {"pixelseal": 16.0, "trustmark": 2.0, "stegastamp": 16.0}


def sim_items(wm, items: list[dict], coco: list[Path], rng: np.random.Generator, n: int, method: str,
              patch: str = "bbox") -> list[dict]:
    from wmloc.utils import downsample_mask
    out = []
    for k, it in enumerate(rng.choice(len(items), size=n, replace=False)):
        it = items[it]
        clean = load_clean(coco[int(rng.integers(len(coco)))], rng)
        bits = rng.integers(0, 2, size=len(it["message"])).astype(np.uint8)
        full = bool(it["full_watermark"])
        rect = patch_rect(it["mask"], full, patch, rng, MAX_ASPECT[method])
        img = composite(wm, clean, it["mask"], full, bits, rect)
        out.append({"image": img, "image_name": f"sim{k}", "message": bits, "mask": it["mask"],
                    "mask128": downsample_mask(it["mask"]), "full_watermark": it["full_watermark"]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="release"); ap.add_argument("--coco", default=".downloads/coco/val2017")
    ap.add_argument("--out", default="outputs/probes"); ap.add_argument("--sim", type=int, default=60)
    ap.add_argument("--methods", nargs="*", default=list(METHODS))
    ap.add_argument("--patch", nargs="*", default=["bbox"], choices=["bbox", "whole", "expand"],
                    help="how irregular clean-background regions are cut from a watermarked patch")
    ap.add_argument("--sim-only", action="store_true", help="skip the probes on the real images")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    coco = sorted(Path(a.coco).glob("*.jpg"))
    rng = np.random.default_rng(0)
    report = {}
    for method in a.methods:
        t0 = time.time()
        ds = LocalisationDataset(a.data, "train", methods=[method])
        items = [ds[i] for i in range(len(ds))]
        dec = Decoder(method)
        wm = dec.wm if dec.wm is not None else _embedder(method)
        r = {}
        if not a.sim_only:
            r["crop_decode"] = crop_decode(dec, items, rng)
            print(method, "crop_decode", {k: v for k, v in r["crop_decode"].items() if k != "rows"}, flush=True)
            r["template"] = template(wm, items)
            print(method, "template", r["template"], flush=True)
        for patch in a.patch:
            sim = sim_items(wm, items, coco, rng, a.sim, method, patch)
            key = "" if patch == "bbox" else f"_{patch}"
            r[f"sim_to_real{key}"], _, _ = evaluate(method, sim, dec)
            r[f"sim_crop_decode{key}"] = crop_decode(dec, sim, rng)
            print(markdown({f"{method} (synthetic, patch={patch})": r[f"sim_to_real{key}"]}), flush=True)
            print(method, patch, "crop", {k: v for k, v in r[f"sim_crop_decode{key}"].items() if k != "rows"}, flush=True)
        r["seconds"] = round(time.time() - t0, 1)
        report[method] = r
        torch.cuda.empty_cache()
    (out / "probes.json").write_text(json.dumps(report, indent=2))


_EMB = {}


def _embedder(method: str):
    from wmloc.watermarks import load_watermark
    if method not in _EMB:
        _EMB[method] = load_watermark(method, device="cuda")
    return _EMB[method]


if __name__ == "__main__":
    main()
