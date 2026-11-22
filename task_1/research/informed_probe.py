"""Informed (known-message) local watermark detection, differential template.

research/probes.py correlated high-pass(image) with R = embed(image, m) - image and found AUC ~0.5. That test is
confounded: the embedders adapt the residual to the content (JND / edge-weighted strength), so R correlates with the
image's own details everywhere, watermarked or not. The classical fix of informed detection: subtract the residual of
other messages, D = R_m - mean_j R_mj. The content-driven envelope is common to all messages and cancels; what is
left is the message-specific pattern, which correlates with the image only where THAT watermark is present
(Cox et al., Digital Watermarking; matched filtering with a known payload).

For every train and test image (the known message is given for both) the normalised local correlation
corr_s = G_s(hp(x) * hp(D)) / sqrt(G_s(hp(x)^2) G_s(hp(D)^2)) is pooled to the 128 grid for a few scales s and saved to
<out>/<method>.npz; on train it prints per-image AUC (watermarked vs clean cells) by edit family and size.

    python -m research.informed_probe --method trustmark --out outputs/informed"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from research.gpu_embed import GpuEmbedder
from research.vae_probe import auc
from research.zero_shot import gaussian_blur, to_grid

SIGMAS = (2.0, 4.0, 8.0)
RATIO = False


def load(data: str, method: str):
    items = []
    for split in ("train", "test"):
        for m in json.loads((Path(data) / split / "metadata.json").read_text()):
            if m["method"] == method:
                items.append((split, m["image_name"], np.array([int(c) for c in m["message"]], np.uint8),
                              m.get("full_watermark")))
    imgs = np.stack([np.asarray(Image.open(Path(data) / s / "images" / f"{n}.png").convert("RGB")) for s, n, _, _ in items])
    return items, imgs


@torch.no_grad()
def maps(emb: GpuEmbedder, imgs: np.ndarray, msgs: np.ndarray, n_other: int, bs: int, seed: int) -> np.ndarray:
    g = torch.Generator(device="cuda"); g.manual_seed(seed)
    out = np.zeros((len(imgs), len(SIGMAS), 128, 128), np.float16)
    for b in range(0, len(imgs), bs):
        x = torch.from_numpy(imgs[b:b + bs]).cuda().permute(0, 3, 1, 2).float() / 255
        m = torch.from_numpy(msgs[b:b + bs]).cuda().long()
        r_true = emb.embed(x, m) - x
        r_other = torch.zeros_like(r_true)
        for _ in range(n_other):
            r_other += emb.embed(x, torch.randint(0, 2, m.shape, device="cuda", generator=g)) - x
        if RATIO:   # idempotence test: does re-embedding the TRUE message change already-marked pixels less?
            e_true = (r_true ** 2).sum(1, keepdim=True); e_oth = torch.zeros_like(e_true)
            g2 = torch.Generator(device="cuda"); g2.manual_seed(seed + 7)
            for _ in range(n_other):
                e_oth += ((emb.embed(x, torch.randint(0, 2, m.shape, device="cuda", generator=g2)) - x) ** 2).sum(1, keepdim=True)
            for j, s in enumerate(SIGMAS):
                ratio = gaussian_blur(e_true, s) / gaussian_blur(e_oth / n_other, s).clamp_min(1e-12)
                out[b:b + bs, j] = to_grid(-torch.log(ratio.clamp_min(1e-6))[:, 0]).cpu().numpy()
            continue
        d = r_true - r_other / n_other
        hx, hd = x - gaussian_blur(x, 2.0), d - gaussian_blur(d, 2.0)
        for j, s in enumerate(SIGMAS):
            ab = gaussian_blur((hx * hd).sum(1, keepdim=True), s)
            aa = gaussian_blur((hx * hx).sum(1, keepdim=True), s); bb = gaussian_blur((hd * hd).sum(1, keepdim=True), s)
            out[b:b + bs, j] = to_grid((ab / (aa * bb).sqrt().clamp_min(1e-12))[:, 0]).cpu().numpy()
    return out


def report(items, M: np.ndarray, data: str) -> dict:
    z = np.load(Path(data) / "train" / "masks_128.npz"); gt = dict(zip(z["names"].tolist(), z["masks"]))
    groups: dict = {}
    for i, (split, n, _, full) in enumerate(items):
        if split != "train":
            continue
        g = gt[n].astype(bool)
        if g.all() or not g.any():
            continue
        edited = g != bool(full); ys, xs = np.nonzero(edited)
        fill = edited.sum() / ((np.ptp(ys) + 1) * (np.ptp(xs) + 1))
        key = ("rect" if fill > 0.97 else ("hole" if full else "transplant")) + ("/small" if edited.mean() < 0.05 else "/large")
        groups.setdefault(key, []).append([auc(M[i, j].astype(np.float32)[g], M[i, j].astype(np.float32)[~g])
                                           for j in range(len(SIGMAS))])
    return {k: {"n": len(v), **{f"auc_s{s:g}": round(float(np.mean(np.array(v)[:, j])), 3) for j, s in enumerate(SIGMAS)},
                **{f"frac>0.8_s{s:g}": round(float((np.array(v)[:, j] > 0.8).mean()), 2) for j, s in enumerate(SIGMAS)}}
            for k, v in sorted(groups.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True); ap.add_argument("--data", default="release")
    ap.add_argument("--out", default="outputs/informed"); ap.add_argument("--n-other", type=int, default=4)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--sanity", action="store_true", help="re-watermark the LEFT half of 64 train images ourselves "
                                                          "(known message) and report the AUC left vs right")
    ap.add_argument("--ratio", action="store_true", help="energy ratio |R_m|^2 / mean |R_m'|^2 instead of correlation")
    a = ap.parse_args()
    global RATIO
    RATIO = a.ratio
    t0 = time.time(); items, imgs = load(a.data, a.method)
    emb = GpuEmbedder(a.method)
    if a.sanity:
        sel = [i for i, it in enumerate(items) if it[0] == "train"][:64]
        x = torch.from_numpy(imgs[sel]).cuda().permute(0, 3, 1, 2).float() / 255
        bits = torch.randint(0, 2, (len(sel), emb.nbits), device="cuda")
        comp = []
        for b in range(0, len(sel), a.bs):
            xw = emb.embed(x[b:b + a.bs], bits[b:b + a.bs])
            xc = x[b:b + a.bs].clone(); xc[..., :256] = xw[..., :256]
            comp.append((xc * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy())
        M = maps(emb, np.concatenate(comp), bits.cpu().numpy().astype(np.uint8), a.n_other, a.bs, seed=1)
        left = np.zeros((128, 128), bool); left[:, :64] = True
        res = {f"s{s:g}": round(float(np.mean([auc(m[j].astype(np.float32)[left], m[j].astype(np.float32)[~left])
                                                 for m in M])), 3) for j, s in enumerate(SIGMAS)}
        print(json.dumps({"method": a.method, "sanity_auc_left_vs_right": res}), flush=True)
        return
    M = maps(emb, imgs, np.stack([it[2] for it in items]), a.n_other, a.bs, seed=0)
    path = Path(a.out) / f"{a.method}{'_ratio' if a.ratio else ''}.npz"
    with open(path, "r+b" if path.exists() else "wb") as fh:
        fh.truncate(0)
        np.savez(fh, names=np.array([it[1] for it in items]), splits=np.array([it[0] for it in items]), maps=M,
                 sigmas=np.array(SIGMAS))
    print(json.dumps({"method": a.method, "seconds": round(time.time() - t0), "auc": report(items, M, a.data)}), flush=True)


if __name__ == "__main__":
    main()
