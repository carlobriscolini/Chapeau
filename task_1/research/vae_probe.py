"""Autoencoder-reconstruction probe for edited regions (AEROBLADE idea, Ricker et al., CVPR 2024).

Pixels decoded by a latent-diffusion autoencoder are (almost) a fixed point of that autoencoder, so the per-pixel
reconstruction error |x - D(E(x))| is low where an LDM generated / inpainted the pixels with that AE and higher where
the pixels were changed afterwards (e.g. by a watermark residual) or come from another generator. The official images
are AI-generated and their small edits (object removal / addition, transplantation) are generative, which is exactly
the cue the watermark-only models cannot see.

For every train and test image and every AE: error map pooled to the official 128 grid (float16), saved to
<out>/<vae>.npz (names, err, hf = local high-frequency energy of the image for normalisation). On train it prints how
well the map separates the edited (non-background) region from the background: per-image AUC, by method, edit family
(hole = marked background, transplant, rect) and size.

    python -m research.vae_probe --out outputs/vae --vaes sd"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.stats import rankdata

VAES = {"sd": ("stabilityai/sd-vae-ft-mse", None), "sdxl": ("madebyollin/sdxl-vae-fp16-fix", None),
        "flux": ("ostris/Flex.1-alpha", "vae")}


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def load_items(data: str):
    items = []
    for split in ("train", "test"):
        meta = json.loads((Path(data) / split / "metadata.json").read_text())
        items += [(split, m["image_name"], m["method"], m.get("full_watermark")) for m in meta]
    imgs = np.stack([np.asarray(Image.open(Path(data) / s / "images" / f"{n}.png").convert("RGB")) for s, n, _, _ in items])
    return items, imgs


@torch.no_grad()
def error_maps(name: str, imgs: np.ndarray, bs: int, dev: str):
    from research.ae_kl import load_ae
    vae = load_ae(*VAES[name]).to(dev)
    err = np.zeros((len(imgs), 128, 128), np.float16); hf = np.zeros_like(err)
    lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=dev).view(1, 1, 3, 3)
    for b in range(0, len(imgs), bs):
        x = torch.from_numpy(imgs[b:b + bs]).to(dev).permute(0, 3, 1, 2).float() / 255
        rec = (vae.reconstruct(x * 2 - 1).clamp(-1, 1) + 1) / 2
        e = (x - rec).abs().mean(1, keepdim=True)
        h = F.conv2d(x.mean(1, keepdim=True), lap, padding=1).abs()
        err[b:b + bs] = F.avg_pool2d(e, 4)[:, 0].cpu().numpy()
        hf[b:b + bs] = F.avg_pool2d(h, 4)[:, 0].cpu().numpy()
    return err, hf


def evaluate(name: str, items, err: np.ndarray, hf: np.ndarray, data: str) -> dict:
    z = np.load(Path(data) / "train" / "masks_128.npz"); gt = dict(zip(z["names"].tolist(), z["masks"]))
    rows = []
    for i, (split, n, method, full) in enumerate(items):
        if split != "train":
            continue
        g = gt[n].astype(bool); edited = g != bool(full)          # region whose watermark status differs from the bg
        if edited.all() or not edited.any():
            continue
        ys, xs = np.nonzero(edited); fill = edited.sum() / ((np.ptp(ys) + 1) * (np.ptp(xs) + 1))
        fam = "rect" if fill > 0.97 else ("hole" if full else "transplant")
        size = "small" if edited.mean() < 0.05 else "large"
        e = err[i].astype(np.float32); r = e / (hf[i].astype(np.float32) + 1e-3)
        rows.append({"method": method, "fam": fam, "size": size, "auc": auc(e[edited], e[~edited]),
                     "auc_norm": auc(r[edited], r[~edited]), "err_in": float(e[edited].mean()),
                     "err_out": float(e[~edited].mean())})
    out = {"vae": name, "mean_err_all_images": float(err.astype(np.float32).mean()), "groups": {}}
    for key in sorted({(r["method"], r["fam"], r["size"]) for r in rows}):
        sel = [r for r in rows if (r["method"], r["fam"], r["size"]) == key]
        a = np.array([r["auc"] for r in sel]); an = np.array([r["auc_norm"] for r in sel])
        out["groups"]["/".join(key)] = {"n": len(sel), "auc": round(float(a.mean()), 3),
                                        "frac_auc<0.25": round(float((a < 0.25).mean()), 2),
                                        "frac_auc>0.75": round(float((a > 0.75).mean()), 2),
                                        "auc_norm": round(float(an.mean()), 3),
                                        "err_in/out": round(float(np.mean([r["err_in"] / r["err_out"] for r in sel])), 3)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="release"); ap.add_argument("--out", default="outputs/vae")
    ap.add_argument("--vaes", nargs="+", default=list(VAES)); ap.add_argument("--bs", type=int, default=8)
    a = ap.parse_args()
    t0 = time.time(); items, imgs = load_items(a.data)
    print(f"loaded {len(imgs)} images in {time.time() - t0:.0f}s", flush=True)
    for name in a.vaes:
        t0 = time.time(); err, hf = error_maps(name, imgs, a.bs, "cuda")
        path = Path(a.out) / f"{name}.npz"
        with open(path, "r+b" if path.exists() else "wb") as fh:   # pre-created file: write in place (inode quota)
            fh.truncate(0)
            np.savez(fh, names=np.array([n for _, n, _, _ in items]), splits=np.array([s for s, _, _, _ in items]),
                     methods=np.array([m for _, _, m, _ in items]), err=err, hf=hf)
        print(json.dumps(evaluate(name, items, err, hf, a.data)), f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
