"""Synthetic retain anchors by model inversion (Zero-Shot unlearning / RUAGO idea): optimise images from noise until the
ORIGINAL model classifies them confidently as a target class. For the model they are 'its own data', unlike the
out-of-distribution mini-ImageNet photos. We invert only classes absent from the forget set (default), where the gold
model behaves like the original, so any drift there is pure damage.

Priors to avoid adversarial noise: optimise a low-resolution image (upsampled), random jitter + flip + scale at
every step (the image must work under augmentation), total-variation and L2 penalties.

Run (GPU job, ~10-15 min for 2000 images on 4 GPUs):
    sbatch inv.sh                      # 4 shards in parallel, then  python experiments/invert.py --merge
Output: ../output/inverted.npz (images uint8 [N,224,224,3], cls [N]) and ../output/inverted_preview.jpg
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # shared modules (common, diagnostics, ...)

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from common import DEVICE, OUTPUT_DIR, load_forget_images, load_model, predict

SHARD = OUTPUT_DIR / "inv_shard{}.npz"


def target_classes(teacher, which):
    labels = predict(teacher, load_forget_images()).argmax(1)
    counts = torch.bincount(labels, minlength=100)
    if which == "untouched":
        return torch.nonzero(counts == 0).flatten()
    return torch.arange(100)


def tv(x):
    return (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean() + (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()


def jitter(x):
    """Random flip, shift (roll) and rescale+crop, applied identically to the whole batch (cheap)."""
    if torch.rand(1).item() < 0.5:
        x = x.flip(3)
    dx, dy = torch.randint(-16, 17, (2,)).tolist()
    x = torch.roll(x, shifts=(dy, dx), dims=(2, 3))
    s = float(torch.empty(1).uniform_(0.9, 1.1))
    size = int(224 * s)
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    if size >= 224:
        o = (size - 224) // 2
        return x[:, :, o:o + 224, o:o + 224]
    p = 224 - size
    return F.pad(x, (p // 2, p - p // 2, p // 2, p - p // 2), mode="reflect")


def invert_batch(teacher, cls, args):
    b = len(cls)
    low = args.low_res
    z = (torch.randn(b, 3, low, low, device=DEVICE) * 0.2).requires_grad_(True)
    opt = torch.optim.Adam([z], lr=args.lr)
    for step in range(args.steps):
        x = torch.tanh(F.interpolate(z, size=(224, 224), mode="bicubic", align_corners=False))  # in [-1, 1]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
            logits = teacher(jitter(x)).float()
        loss = F.cross_entropy(logits, cls) + args.tv * tv(x) + args.l2 * x.pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        x = torch.tanh(F.interpolate(z, size=(224, 224), mode="bicubic", align_corners=False))
        img = ((x * 0.5 + 0.5).clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu()
    return img


def run_shard(args):
    torch.manual_seed(args.seed + 1000 * args.shard)
    teacher = load_model()
    for p in teacher.parameters():
        p.requires_grad_(False)
    classes = target_classes(teacher, args.classes)
    per_shard = int(np.ceil(args.n / args.nshards))
    cls_all = classes[torch.arange(per_shard * args.nshards) % len(classes)]  # round-robin over classes
    cls_shard = cls_all[args.shard * per_shard:(args.shard + 1) * per_shard]
    imgs, t0 = [], time.time()
    for i in range(0, len(cls_shard), args.batch_size):
        c = cls_shard[i:i + args.batch_size].to(DEVICE)
        imgs.append(invert_batch(teacher, c, args))
        print(f"shard {args.shard}: {i + len(c)}/{len(cls_shard)} images ({time.time() - t0:.0f}s)", flush=True)
    imgs = torch.cat(imgs)
    p = F.softmax(predict(teacher, imgs), 1)
    conf = p.gather(1, cls_shard[:, None]).squeeze(1)
    print(f"shard {args.shard}: original model confidence on its target class: mean {conf.mean():.3f}, "
          f"argmax = target {(p.argmax(1) == cls_shard).float().mean():.3f}", flush=True)
    tmp = OUTPUT_DIR / f"inv_shard{args.shard}.tmp{os.getpid()}.npz"
    np.savez(tmp, images=imgs.numpy(), cls=cls_shard.numpy())
    os.replace(tmp, str(SHARD).format(args.shard))


def merge(args):
    files = sorted(OUTPUT_DIR.glob("inv_shard[0-9].npz"))
    assert files, "no inv_shard*.npz"
    imgs, cls = [], []
    for f in files:
        with np.load(f) as d:
            imgs.append(d["images"]); cls.append(d["cls"])
    imgs, cls = np.concatenate(imgs), np.concatenate(cls)
    out = OUTPUT_DIR / "inverted.npz"
    tmp = OUTPUT_DIR / f"inverted.tmp{os.getpid()}.npz"
    np.savez(tmp, images=imgs, cls=cls)
    os.replace(tmp, out)
    print(f"saved {out}: {imgs.shape}")
    tile, ncol = 112, 8
    sel = imgs[:ncol * 4]
    canvas = Image.new("RGB", (ncol * tile, ((len(sel) + ncol - 1) // ncol) * tile), "white")
    for k, im in enumerate(sel):
        canvas.paste(Image.fromarray(im).resize((tile, tile)), ((k % ncol) * tile, (k // ncol) * tile))
    canvas.save(OUTPUT_DIR / "inverted_preview.jpg", quality=85)
    print(f"preview: {OUTPUT_DIR / 'inverted_preview.jpg'}")
    for f in files:
        f.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--n", type=int, default=2000, help="total images")
    ap.add_argument("--classes", default="untouched", choices=["untouched", "all"])
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--low-res", type=int, default=56, help="optimised resolution (upsampled to 224)")
    ap.add_argument("--tv", type=float, default=0.05)
    ap.add_argument("--l2", type=float, default=0.01)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    merge(args) if args.merge else run_shard(args)


if __name__ == "__main__":
    main()
