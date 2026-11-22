"""Rectangle search for clean-background images whose watermarked region is a rectangle (outpainting case).

The crop of the true rectangle decodes the known message (TrustMark 0.94, StegaStamp 0.83, PixelSeal 0.73 mean bit
accuracy on train), a random crop does not. `tolerance` measures how fast that accuracy decays when the crop is
shifted or rescaled, which sets the search grid.

    python -m research.rect_search tolerance --out outputs/rect_search"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from research.decoders import Decoder, signed
from research.synth import bbox, is_rectangle
from research.zero_shot import METHODS
from wmloc.data import LocalisationDataset

MIN_AREA = 0.01
SHIFTS = (0, 4, 8, 16, 32)
SCALES = (0.8, 0.9, 1.0, 1.1, 1.25)


@torch.no_grad()
def decode_crops(dec: Decoder, img: torch.Tensor, rects: list[tuple[int, int, int, int]], s: torch.Tensor,
                 bs: int = 128) -> np.ndarray:
    """Bit accuracy of every crop (y0, y1, x0, x1) of one image (1 x 3 x H x W on GPU), batched by resizing each
    crop to the decoder's working resolution first (the decoder would resize anyway)."""
    res = {"pixelseal": 256, "trustmark": 245, "stegastamp": 400}[dec.method]
    out = []
    for b in range(0, len(rects), bs):
        crops = [F.interpolate(img[:, :, y0:y1, x0:x1], size=(res, res), mode="bilinear", align_corners=False,
                               antialias=True) for y0, y1, x0, x1 in rects[b:b + bs]]
        lg = dec.logits(torch.cat(crops))
        out.append(((lg > 0) == (s > 0)).float().mean(1).cpu().numpy())
    return np.concatenate(out)


def clip_rect(y0, y1, x0, x1, n=512):
    y0, x0 = max(0, y0), max(0, x0); y1, x1 = min(n, y1), min(n, x1)
    return (y0, y1, x0, x1) if y1 - y0 >= 8 and x1 - x0 >= 8 else None


def tolerance(data: str, out: Path) -> dict:
    report = {}
    for method in METHODS:
        dec = Decoder(method)
        ds = LocalisationDataset(data, "train", methods=[method])
        rows = []
        for i in range(len(ds)):
            it = ds[i]
            if it["full_watermark"] or not is_rectangle(it["mask"]) or it["mask"].mean() < MIN_AREA:
                continue
            y0, y1, x0, x1 = bbox(it["mask"])
            img = torch.from_numpy(it["image"]).permute(2, 0, 1)[None].float().cuda() / 255
            s = signed(torch.from_numpy(it["message"]))[None].cuda()
            rects, keys = [], []
            for d in SHIFTS:
                for dy, dx in ((d, 0), (0, d), (-d, 0), (0, -d)) if d else ((0, 0),):
                    r = clip_rect(y0 + dy, y1 + dy, x0 + dx, x1 + dx)
                    if r: rects.append(r); keys.append(f"shift{d}")
            cy, cx, h, w = (y0 + y1) / 2, (x0 + x1) / 2, y1 - y0, x1 - x0
            for sc in SCALES:
                r = clip_rect(int(cy - h * sc / 2), int(cy + h * sc / 2), int(cx - w * sc / 2), int(cx + w * sc / 2))
                if r: rects.append(r); keys.append(f"scale{sc}")
            accs = decode_crops(dec, img, rects, s)
            row = {}
            for k, a in zip(keys, accs):
                row.setdefault(k, []).append(float(a))
            rows.append({k: float(np.mean(v)) for k, v in row.items()})
        keys = sorted({k for r in rows for k in r})
        report[method] = {"images": len(rows), **{k: float(np.mean([r[k] for r in rows if k in r])) for k in keys}}
        print(method, json.dumps(report[method]), flush=True)
        del dec; torch.cuda.empty_cache()
    out.mkdir(parents=True, exist_ok=True)
    (out / "tolerance.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tolerance"]); ap.add_argument("--data", default="release")
    ap.add_argument("--out", default="outputs/rect_search")
    a = ap.parse_args()
    tolerance(a.data, Path(a.out))
