"""Message-guided rectangle search for clean-background images (outpainting: the watermarked region is a rectangle).

The official decoder recovers the known message from a crop of the exact embedding rectangle (bit accuracy 0.94
TrustMark, 0.88 StegaStamp, 0.80 PixelSeal on train) and much worse from shifted crops, so the rectangle that
maximises agreement with the known message locates the watermark. For every image whose whole-image bit accuracy says
"clean background", we search:

  1. a coarse grid of rectangles (edges on a 64 px lattice, aspect ratio within the method's limit);
     plus a finer lattice (--fine-step) restricted to small rectangles (--fine-max), the common case;
  2. starts from the segmenter ensemble (bbox of its mask and of its largest component), from the cache;
  3. greedy refinement of the best starts: move each edge / translate / grow / shrink by 32, 16, 8, 4, 2 px while the
     score improves;
  4. growth: push edges outwards while the score stays within --grow-tol of the optimum (crops slightly inside the
     true rectangle still decode perfectly, so step 3 alone undershoots). Both rectangles are saved.

Score = bit accuracy + 0.01 * mean clipped signed logit (tie-breaker). Output: one .npz per method with the best
rectangle and its score per image (JSON); whether to trust it is decided offline on train (research.rect_eval).

    python -m research.rect_refine --method trustmark --cache outputs/cache/trustmark.npz --out outputs/rect/trustmark.json"""
from __future__ import annotations

import argparse
import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from research.analyze_cache import oof_logits
from research.decoders import Decoder, signed
from wmloc.data import LocalisationDataset

RES = {"pixelseal": 256, "trustmark": 245, "stegastamp": 400}
MAX_ASPECT = {"pixelseal": 16.0, "trustmark": 2.0, "stegastamp": 16.0}
MIN_SIDE = 16
STEPS = (32, 16, 8, 4, 2)


class Scorer:
    def __init__(self, method: str, bs: int, device: str = "cuda"):
        self.dec, self.method, self.res, self.bs = Decoder(method, device=device), method, RES[method], bs
        self.mode = "bicubic" if method == "stegastamp" else "bilinear"
        self.evals = 0

    @torch.no_grad()
    def __call__(self, img: torch.Tensor, s: torch.Tensor, rects: list[tuple]) -> np.ndarray:
        """img 1 x 3 x 512 x 512 on GPU in [0, 1]; s 1 x nbits in {-1, +1}; rects (y0, y1, x0, x1)."""
        out = []
        for b in range(0, len(rects), self.bs):
            crops = torch.cat([F.interpolate(img[:, :, y0:y1, x0:x1], size=(self.res, self.res), mode=self.mode,
                                             align_corners=False, antialias=True) for y0, y1, x0, x1 in rects[b:b + self.bs]])
            lg = self.dec.logits(crops.clamp(0, 1)).float()
            agree = lg * s
            acc = (agree > 0).float().mean(1)
            soft = agree.clamp(-5, 5).mean(1) / 5
            out.append((acc + 0.01 * soft).cpu().numpy())
        self.evals += len(rects)
        return np.concatenate(out)


def valid(r, max_aspect: float, n: int = 512) -> bool:
    y0, y1, x0, x1 = r
    h, w = y1 - y0, x1 - x0
    return 0 <= y0 and 0 <= x0 and y1 <= n and x1 <= n and h >= MIN_SIDE and w >= MIN_SIDE and max(h / w, w / h) <= max_aspect


def coarse_grid(max_aspect: float, step: int = 64, n: int = 512, max_side: int = 512) -> list[tuple]:
    edges = list(range(0, n + 1, step))
    spans = [(a, b) for a, b in combinations(edges, 2) if b - a <= max_side]
    return [(y0, y1, x0, x1) for y0, y1 in spans for x0, x1 in spans if valid((y0, y1, x0, x1), max_aspect)]


def neighbours(r, d: int) -> list[tuple]:
    y0, y1, x0, x1 = r
    out = [(y0 + a, y1, x0, x1) for a in (-d, d)] + [(y0, y1 + a, x0, x1) for a in (-d, d)]
    out += [(y0, y1, x0 + a, x1) for a in (-d, d)] + [(y0, y1, x0, x1 + a) for a in (-d, d)]
    out += [(y0 + a, y1 + a, x0, x1) for a in (-d, d)] + [(y0, y1, x0 + a, x1 + a) for a in (-d, d)]
    out += [(y0 - d, y1 + d, x0 - d, x1 + d), (y0 + d, y1 - d, x0 + d, x1 - d)]
    return out


def refine(score: Scorer, img, s, start, max_aspect: float) -> tuple[tuple, float]:
    best, best_v = start, float(score(img, s, [start])[0])
    for d in STEPS:
        while True:
            cands = [c for c in neighbours(best, d) if valid(c, max_aspect)]
            if not cands:
                break
            v = score(img, s, cands); j = int(np.argmax(v))
            if v[j] > best_v + 1e-9:
                best, best_v = cands[j], float(v[j])
            else:
                break
    return best, best_v


def grow(score: Scorer, img, s, r, ref: float, max_aspect: float, tol: float) -> tuple[tuple, float]:
    """A crop slightly inside the true rectangle still decodes perfectly, so the refined optimum tends to be too small.
    Push each edge outwards (16, 8, 4, 2 px) as long as the score stays within `tol` of the refined optimum `ref`."""
    v_r = ref
    for d in (16, 8, 4, 2):
        while True:
            y0, y1, x0, x1 = r
            cands = [c for c in ((y0 - d, y1, x0, x1), (y0, y1 + d, x0, x1), (y0, y1, x0 - d, x1), (y0, y1, x0, x1 + d))
                     if valid(c, max_aspect)]
            if not cands:
                break
            v = score(img, s, cands); j = int(np.argmax(v))
            if v[j] >= ref - tol:
                r, v_r = cands[j], float(v[j])
            else:
                break
    return r, v_r


def mask_starts(prob: np.ndarray) -> list[tuple]:
    """Rectangles (512 px) from a 128 x 128 ensemble mask: bbox of the whole mask and of its largest component."""
    m = prob > 0
    if not m.any():
        return []
    out = []
    lab, n = ndimage.label(m)
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    for sel in (m, lab == (1 + int(np.argmax(sizes)))):
        ys, xs = np.nonzero(sel)
        out.append((int(ys.min()) * 4, int(ys.max() + 1) * 4, int(xs.min()) * 4, int(xs.max() + 1) * 4))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=list(RES)); ap.add_argument("--data", default="release")
    ap.add_argument("--cache", required=True, help="research.cache_logits output (ensemble starts, bit accuracy)")
    ap.add_argument("--members", default="", help="comma-separated cache names averaged for the TEST starts (default: all);"
                    " train starts are out-of-fold (research.analyze_cache.oof_logits) so train scores stay honest")
    ap.add_argument("--max-bitacc", type=float, default=0.85, help="only images whose whole-image bit accuracy is below")
    ap.add_argument("--top", type=int, default=3, help="coarse-grid winners refined"); ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--fine-step", type=int, default=32); ap.add_argument("--fine-max", type=int, default=192)
    ap.add_argument("--grid-step", type=int, default=64); ap.add_argument("--limit", type=int, help="images per split (smoke test)")
    ap.add_argument("--grow-tol", type=float, default=0.015, help="score drop allowed while growing (about 1 bit at 100)")
    ap.add_argument("--splits", default="train,test"); ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    t0 = time.time()
    cache = np.load(a.cache)
    names = [k[:-len("/test")] for k in cache.files if k.endswith("/test")]
    members = a.members.split(",") if a.members else names
    score = Scorer(a.method, a.bs, a.device)
    grid = coarse_grid(MAX_ASPECT[a.method], a.grid_step)
    if a.fine_step:
        grid = sorted(set(grid) | set(coarse_grid(MAX_ASPECT[a.method], a.fine_step, max_side=a.fine_max)))
    print(f"{a.method}: {len(grid)} grid rectangles per image", flush=True)
    rows = []
    for split in a.splits.split(","):
        ds = LocalisationDataset(a.data, split, methods=[a.method])
        assert [ds.meta[i]["image_name"] for i in range(len(ds))] == cache[f"names_{split}"].tolist()
        if split == "train":
            prob = np.nan_to_num(oof_logits(cache, a.method, flip=False)[0], nan=-1.0)
        else:
            prob = np.mean([cache[f"{m}/test"].astype(np.float32) for m in members], axis=0)
        acc = cache[f"bitacc_{split}"]
        for i in np.nonzero(acc < a.max_bitacc)[0][:a.limit]:
            it = ds[int(i)]
            img = torch.from_numpy(it["image"]).permute(2, 0, 1)[None].float().to(a.device) / 255
            s = signed(torch.from_numpy(it["message"]))[None].to(a.device)
            v = score(img, s, grid)
            starts = [grid[j] for j in np.argsort(-v)[:a.top]] + mask_starts(prob[i])
            results = [refine(score, img, s, st, MAX_ASPECT[a.method]) for st in starts]
            k = int(np.argmax([r[1] for r in results]))
            grown, grown_v = grow(score, img, s, results[k][0], results[k][1], MAX_ASPECT[a.method], a.grow_tol)
            rows.append({"split": split, "name": it["image_name"], "rect": list(results[k][0]), "score": results[k][1],
                         "rect_grown": list(grown), "score_grown": grown_v,
                         "start": list(starts[k]), "from_mask": k >= a.top, "grid_best": float(v.max()),
                         "image_bitacc": float(acc[i])})
        print(f"{a.method} {split}: {sum(r['split'] == split for r in rows)} images, {score.evals} decodes, "
              f"{time.time() - t0:.0f}s", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"method": a.method, "members": members, "rows": rows}))
    print(f"{a.method}: wrote {a.out} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
