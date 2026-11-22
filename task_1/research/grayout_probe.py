"""Gray-out probe: is the known message readable from the watermarked region ALONE, kept in place (no crop/resize)?

avert1's crop probe decoded the bounding box resized to the working resolution, which rescales the watermark. Here the
image keeps its geometry and everything outside a region is replaced (gray 0.5 or a blurred copy):
  obj / rect (clean background): keep the GT region / GT dilated 16 px / GT bounding box / GT shifted (control)
  hole (marked background): gray the GT hole vs gray the same shape shifted elsewhere (soft agreement)
If keep-GT decodes far better than the full image and the shifted control, a message-guided search over candidate
object masks (like rect_refine for rectangles) becomes possible. CPU is enough (one image = one small batch).

    python -m research.grayout_probe --method trustmark --device cpu"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from research.analyze_cache import is_rect
from research.decoders import Decoder, signed
from wmloc.data import LocalisationDataset


def blur(x: torch.Tensor, s: float = 4.0) -> torch.Tensor:
    k = int(4 * s) | 1; t = torch.arange(k, dtype=x.dtype) - k // 2
    g = torch.exp(-t ** 2 / (2 * s * s)); g = g / g.sum()
    x = F.conv2d(F.pad(x, (k // 2,) * 4, mode="reflect"), g.view(1, 1, 1, k).repeat(3, 1, 1, 1), groups=3)
    return F.conv2d(x, g.view(1, 1, k, 1).repeat(3, 1, 1, 1), groups=3)


def dilate(m: torch.Tensor, r: int) -> torch.Tensor:
    return F.max_pool2d(m, 2 * r + 1, 1, r)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--method", required=True)
    ap.add_argument("--device", default="cpu"); ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(); torch.set_grad_enabled(False); torch.set_num_threads(a.threads)
    z = np.load(f"outputs/cache/{a.method}.npz")
    full = dict(zip(z["names_train"].tolist(), z["full_train"].tolist()))
    ds = LocalisationDataset("release", "train", methods=[a.method]); dec = Decoder(a.method, device=a.device)
    res: dict = {}
    for i in range(len(ds)):
        it = ds[i]; g128 = it["mask128"]
        x = torch.from_numpy(it["image"]).permute(2, 0, 1)[None].float() / 255
        M = torch.from_numpy(it["mask"])[None, None].float()
        s = signed(torch.from_numpy(it["message"])[None]).to(a.device)
        if full[it["image_name"]]:
            fam, H = "hole", 1 - M
            Hs = torch.roll(H, shifts=(171, 171), dims=(2, 3))
            vs = {"full": x, "gray_hole": x * (1 - H) + 0.5 * H, "gray_shift": x * (1 - Hs) + 0.5 * Hs}
        else:
            fam = "rect" if is_rect(g128) else "obj"
            ys, xs = torch.nonzero(M[0, 0], as_tuple=True)
            B = torch.zeros_like(M); B[..., ys.min():ys.max() + 1, xs.min():xs.max() + 1] = 1
            Ms, D = torch.roll(M, shifts=(171, 171), dims=(2, 3)), dilate(M, 16)
            vs = {"full": x, "keep_gray": x * M + 0.5 * (1 - M), "keep_blur": x * M + blur(x) * (1 - M),
                  "keep_dil16": x * D + 0.5 * (1 - D), "keep_bbox": x * B + 0.5 * (1 - B),
                  "keep_shift": x * Ms + 0.5 * (1 - Ms)}
        size = "small" if min(g128.mean(), 1 - g128.mean()) < 0.05 else "large"
        lg = dec.logits(torch.cat(list(vs.values())).to(a.device))
        S = s.expand(len(lg), -1)
        acc = (torch.where(lg > 0, 1.0, -1.0) == S).float().mean(1).cpu().numpy()
        soft = (torch.tanh(lg) * S).mean(1).cpu().numpy()
        for k, ac, so in zip(vs, acc, soft):
            res.setdefault((fam, size, k), []).append((ac, so))
    for fam in ("obj", "rect", "hole"):
        for size in ("small", "large"):
            keys = [k for (f, sz, k) in res if f == fam and sz == size]
            if not keys:
                continue
            r = {k: np.array(res[(fam, size, k)]) for k in keys}
            print(f"{a.method} {fam}/{size} n={len(r['full'])}: " +
                  "  ".join(f"{k} {v[:, 0].mean():.3f}/{v[:, 1].mean():+.3f}" for k, v in r.items()), flush=True)
            if fam == "hole":
                print(f"   soft(gray hole) > soft(gray shifted) in {np.mean(r['gray_hole'][:, 1] > r['gray_shift'][:, 1]):.0%}")
            else:
                g, sh = r["keep_gray"][:, 0], r["keep_shift"][:, 0]
                print(f"   keep_gray acc >= 0.8: {np.mean(g >= .8):.0%}, >= 0.9: {np.mean(g >= .9):.0%}; "
                      f"beats shifted control: {np.mean(g > sh):.0%}; beats full image: {np.mean(g > r['full'][:, 0]):.0%}")


if __name__ == "__main__":
    main()
