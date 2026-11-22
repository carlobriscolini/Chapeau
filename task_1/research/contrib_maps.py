"""Observed vs expected watermark contribution maps (exploratory; PixelSeal and TrustMark).

  O  observed: per-position agreement of the official decoder with the KNOWN message (mean over bits of
     sign(bit) * per-position logit, before the decoder's global pooling), on the image as given.
  E  expected: the same agreement on embed(image) (official embedder, known message): how much this area WOULD
     contribute if it were watermarked. Texture-poor areas (sky, walls) carry little watermark, so a low O there is
     normal; a low O where E is high is suspicious (a hole).
  R  residual energy of embed(image) - image, pooled per cell (a cruder expected-strength proxy).
Maps are computed at the decoder's working resolution and at 512 input, resized to the official 128 grid, stored as
float16 in ONE .npz per method (keys <map>_<split>).

    python -m research.contrib_maps --method pixelseal --out outputs/contrib/pixelseal.npz"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from research.decoders import Decoder, signed
from research.gpu_embed import GpuEmbedder
from wmloc.data import LocalisationDataset


def to128(m: torch.Tensor) -> np.ndarray:
    return F.interpolate(m[:, None].float(), size=(128, 128), mode="bilinear", align_corners=False)[:, 0].cpu().numpy().astype(np.float16)


@torch.no_grad()
def agreement(dec: Decoder, x: torch.Tensor, s: torch.Tensor, size: int | None) -> torch.Tensor:
    lg = dec.per_position_logits(x, size=size)                     # B x nbits x h x w
    return (lg * s[:, :, None, None]).mean(1)                        # B x h x w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["pixelseal", "trustmark"])
    ap.add_argument("--data", default="release"); ap.add_argument("--out", required=True)
    ap.add_argument("--bs", type=int, default=16)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    t0 = time.time()
    dec, emb = Decoder(a.method), GpuEmbedder(a.method)
    out = {}
    for split in ("train", "test"):
        ds = LocalisationDataset(a.data, split, methods=[a.method])
        maps = {k: [] for k in ("O_native", "O_512", "E_native", "E_512", "R")}
        names = []
        for b in range(0, len(ds), a.bs):
            items = [ds[i] for i in range(b, min(b + a.bs, len(ds)))]
            x = torch.from_numpy(np.stack([it["image"] for it in items])).permute(0, 3, 1, 2).float().cuda() / 255
            s = signed(torch.from_numpy(np.stack([it["message"] for it in items]))).cuda()
            xw = emb.embed(x, torch.from_numpy(np.stack([it["message"] for it in items])))
            for size, tag in ((None, "native"), (512, "512")):
                maps[f"O_{tag}"].append(to128(agreement(dec, x, s, size)))
                maps[f"E_{tag}"].append(to128(agreement(dec, xw, s, size)))
            r = (xw - x).abs().mean(1)                                   # B x 512 x 512
            maps["R"].append(F.avg_pool2d(r[:, None], 4)[:, 0].cpu().numpy().astype(np.float16))
            names += [it["image_name"] for it in items]
        for k, v in maps.items():
            out[f"{k}_{split}"] = np.concatenate(v)
        out[f"names_{split}"] = np.array(names)
        print(f"{a.method} {split}: {len(names)} images {time.time() - t0:.0f}s", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, **out)
    print(f"{a.method}: wrote {a.out} ({Path(a.out).stat().st_size / 1e6:.0f} MB) {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
