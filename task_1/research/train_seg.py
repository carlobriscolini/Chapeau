"""Train a per-method segmenter that predicts the official 128 x 128 grid from the 512 x 512 image.

Each batch mixes on-GPU synthetic composites (research.gpu_synth) with real train images outside the validation
fold. Everything runs on the GPU: bf16 autocast, channels_last, fused AdamW and torch.compile in
"reduce-overhead" mode (CUDA graphs, static batch shape).

    python -m research.train_seg --method trustmark --steps 3000 --out outputs/seg/trustmark"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from research import folds as folds_mod
from research.gpu_synth import CleanPool, SynthGenerator, add_official_hosts
from research.metrics import miou_batch
from research.post_watermark import PostWatermarkAugment
from wmloc.data import LocalisationDataset

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
ImageMaskBatch = tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
MESSAGE_ENCODER_PREFIX = "wmmsg:"


class SegNet(nn.Module):
    """timm encoder (strides 2..32) + FPN decoder, logits at stride 4 = the official 128 grid for 512 inputs."""

    requires_messages = False

    def __init__(self, encoder: str = "resnet34", width: int = 128, pretrained: bool = True):
        super().__init__()
        self.enc = timm.create_model(encoder, pretrained=pretrained, features_only=True, out_indices=(0, 1, 2, 3, 4))
        chans = self.enc.feature_info.channels()
        self.lat = nn.ModuleList(nn.Conv2d(c, width, 1) for c in chans)
        self.down0 = nn.Conv2d(width, width, 3, stride=2, padding=1)          # stride 2 -> 4 (fine residual detail)
        self.head = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(inplace=True),
                                  nn.Conv2d(width, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(inplace=True),
                                  nn.Conv2d(width, 1, 1))
        self.register_buffer("mean", MEAN.clone(), persistent=False)
        self.register_buffer("std", STD.clone(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.enc((x - self.mean) / self.std)
        p = self.lat[4](feats[4])
        for i in (3, 2, 1):
            p = self.lat[i](feats[i]) + F.interpolate(p, size=feats[i].shape[-2:], mode="bilinear", align_corners=False)
        p = p + self.down0(self.lat[0](feats[0]))
        return self.head(p)[:, 0]


def build_model(encoder: str, pretrained: bool = True, *, device: str = "cuda") -> nn.Module:
    """Build a timm, wm:<method>, high-resolution wmhr:<method>, or message-conditioned wmmsg:<method> segmenter."""
    if encoder.startswith("wmhr:"):
        from research.wm_encoder import HighResWatermarkSegNet
        return HighResWatermarkSegNet(encoder.split(":", 1)[1], pretrained=pretrained, device=device)
    if encoder.startswith(("wm:", MESSAGE_ENCODER_PREFIX)):
        from research.wm_encoder import WatermarkSegNet
        kind, method = encoder.split(":", 1)
        return WatermarkSegNet(method, message_conditioned=kind == "wmmsg", device=device)
    return SegNet(encoder, pretrained=pretrained)


def loss_fn(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """BCE + (1 - soft per-image mIoU over both classes), the official metric made differentiable."""
    logits = logits.float(); t = target.float()
    bce = F.binary_cross_entropy_with_logits(logits, t)
    p = torch.sigmoid(logits).flatten(1); t = t.flatten(1)
    iou1 = (p * t).sum(1) / (p + t - p * t).sum(1).clamp_min(1e-6)
    q, u = 1 - p, 1 - t
    iou0 = (q * u).sum(1) / (q + u - q * u).sum(1).clamp_min(1e-6)
    return bce + 1 - ((iou0 + iou1) / 2).mean()


def stack_messages(items: list[dict], device: str | torch.device) -> torch.Tensor:
    """Pack metadata messages in image order, from either release arrays or generated bit strings."""
    bits = np.stack([np.fromiter(item["message"], dtype=np.uint8) for item in items])
    return torch.from_numpy(bits).to(device)


def real_split(data: str, method: str, val_fold: int, device: str, *, return_messages: bool = False):
    fold_of = folds_mod.load()
    ds = LocalisationDataset(data, "train", methods=[method])
    items = [ds[i] for i in range(len(ds))]
    def pack(sel):
        x = torch.from_numpy(np.stack([it["image"] for it in sel])).permute(0, 3, 1, 2).contiguous().to(device)
        y = torch.from_numpy(np.stack([it["mask128"] for it in sel])).bool().to(device)
        f = torch.tensor([bool(it["full_watermark"]) for it in sel], device=device)
        return (x, y, f, stack_messages(sel, device)) if return_messages else (x, y, f)
    tr = [it for it in items if fold_of[it["image_name"]] != val_fold]
    va = [it for it in items if fold_of[it["image_name"]] == val_fold]
    return pack(tr), pack(va)


def pseudo_split(path: str, data: str, method: str, val_fold: int, device: str):
    """Pseudo-labelled real images (self-training, Lee 2013; Xie et al., CVPR 2020): every TEST image of the method with
    the mask of the current best ensemble, plus the train images of the VALIDATION fold with their out-of-fold ensemble
    mask (predicted by models that never saw that fold). The validation images thus enter training exactly like test
    images do (image + pseudo label, never their true label), so the validation score stays an honest estimate.
    Organizers, 22/11 05:58: "you can use any to you available data"."""
    z = np.load(Path(path) / f"{method}.npz" if Path(path).is_dir() else path)
    lab = dict(zip(z["names_test"].tolist(), z["masks_test"]))
    fold_of = dict(zip(z["names_train"].tolist(), z["folds_train"].tolist()))
    lab.update({n: m for n, m, f in zip(z["names_train"].tolist(), z["masks_train"], z["folds_train"]) if f == val_fold})
    items = [it for split in ("test", "train") for it in LocalisationDataset(data, split, methods=[method]).meta
             if (split == "test" or fold_of.get(it["image_name"]) == val_fold)]
    from PIL import Image
    root, test_names = Path(data), set(z["names_test"].tolist())
    x = np.stack([np.asarray(Image.open(root / ("test" if it["image_name"] in test_names else "train") / "images"
                                        / f"{it['image_name']}.png").convert("RGB")) for it in items])
    y = np.stack([lab[it["image_name"]] for it in items])
    return (torch.from_numpy(x).permute(0, 3, 1, 2).contiguous().to(device), torch.from_numpy(y).bool().to(device))


def extra_split(extra_dir: str, method: str, val_fold: int, device: str, *, return_messages: bool = False):
    """Static augmented rows (research.coco_augment output) for one method, without the original train images and
    without rows whose reused mask comes from the validation fold."""
    root = Path(extra_dir)
    meta = json.loads((root / "train" / "metadata.json").read_text())
    rows = [m for m in meta if m["method"] == method and "source_fold" in m and m["source_fold"] != val_fold]
    z = np.load(root / "train" / "masks_128.npz"); m128 = dict(zip(z["names"].tolist(), z["masks"]))
    from PIL import Image
    x = torch.from_numpy(np.stack([np.asarray(Image.open(root / "train" / "images" / f"{r['image_name']}.png").convert("RGB"))
                                   for r in rows])).permute(0, 3, 1, 2).contiguous().to(device)
    y = torch.from_numpy(np.stack([m128[r["image_name"]] for r in rows])).bool().to(device)
    return (x, y, stack_messages(rows, device)) if return_messages else (x, y)


@torch.no_grad()
def predict(model: nn.Module, x_uint8: torch.Tensor, bs: int = 32, *, messages: torch.Tensor | None = None) -> torch.Tensor:
    if messages is not None and len(messages) != len(x_uint8):
        raise ValueError("expected one message per inference image")
    model.eval(); out = []
    for b in range(0, len(x_uint8), bs):
        x = (x_uint8[b:b + bs].float() / 255).contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.device.type == "cuda"):
            logits = model(x) if messages is None else model(x, messages[b:b + bs])
            out.append(logits.float())
    model.train()
    return torch.cat(out)


def real_cutmix(x: torch.Tensor, y: torch.Tensor, prob: float) -> tuple[torch.Tensor, torch.Tensor]:
    """CutMix between real train images (Yun et al., ICCV 2019; French et al., BMVC 2020 for segmentation): with
    probability `prob` a box of another real image of the batch is pasted AT THE SAME POSITION (no shift, so the
    watermark pattern keeps its geometry) together with its mask. Fights memorisation of the few real images."""
    if prob <= 0 or len(x) < 2:
        return x, y
    x, y = x.clone(), y.clone(); src = torch.roll(torch.arange(len(x), device=x.device), 1)
    xs, ys = x[src].clone(), y[src].clone()
    for i in range(len(x)):
        if torch.rand(()) >= prob:
            continue
        h, w = (int(v) for v in torch.randint(16, 97, (2,)))         # box on the 128 grid: 12-75 % of the side
        t, l = int(torch.randint(0, 128 - h + 1, ())), int(torch.randint(0, 128 - w + 1, ()))
        y[i, t:t + h, l:l + w] = ys[i, t:t + h, l:l + w]
        x[i, :, 4 * t:4 * (t + h), 4 * l:4 * (l + w)] = xs[i, :, 4 * t:4 * (t + h), 4 * l:4 * (l + w)]
    return x, y


def real_classmix(x: torch.Tensor, y: torch.Tensor, prob: float) -> tuple[torch.Tensor, torch.Tensor]:
    """ClassMix / Copy-Paste between real train images (Olsson et al., WACV 2021; Ghiasi et al., CVPR 2021): with
    probability `prob`, the edited (minority-class) region of another real image of the batch, with its exact shape,
    is pasted AT THE SAME POSITION together with its label. Creates many small, irregular, genuinely edited regions."""
    if prob <= 0 or len(x) < 2:
        return x, y
    x, y = x.clone(), y.clone(); src = torch.roll(torch.arange(len(x), device=x.device), 2)
    xs, ys = x[src].clone(), y[src].clone()
    for i in range(len(x)):
        if torch.rand(()) >= prob:
            continue
        c = bool(ys[i].float().mean() < 0.5)                          # minority class of the source image
        m = ys[i] == c
        if not m.any():
            continue
        y[i][m] = c
        m512 = m.repeat_interleave(4, 0).repeat_interleave(4, 1)
        x[i][:, m512] = xs[i][:, m512]
    return x, y


def save_atomic(obj, path: Path) -> None:
    """Write then rename, so a reader (predict, another agent) never loads a half-written checkpoint."""
    if os.environ.get("WMLOC_INPLACE_SAVE") and path.exists():   # inode quota full: reuse the pre-created file
        torch.save(obj, path)
        return
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def mix_training_sources(
    online: ImageMaskBatch,
    real: ImageMaskBatch | None,
    extra: ImageMaskBatch | None,
    post_augment: PostWatermarkAugment,
) -> ImageMaskBatch:
    """Mix online composites with optional uint8 real and pre-generated images, preserving message order."""
    online_image, online_mask, *messages = online
    if any(len(source) != len(online) for source in (real, extra) if source is not None):
        raise ValueError("provide messages for every training source")
    images, masks = [post_augment(online_image)], [online_mask]
    if real is not None:
        image, mask, *bits = real
        images.append(image.float() / 255)
        masks.append(mask)
        messages.extend(bits)
    if extra is not None:
        image, mask, *bits = extra
        images.append(post_augment(image.float() / 255))
        masks.append(mask)
        messages.extend(bits)
    if len(images) == 1:
        return images[0], masks[0], *messages
    mixed = torch.cat(images), torch.cat(masks)
    return (*mixed, torch.cat(messages)) if messages else mixed


def scores(logits: torch.Tensor, y: torch.Tensor, full: torch.Tensor) -> dict:
    pred = (logits > 0).cpu().numpy(); gt = y.cpu().numpy(); f = full.cpu().numpy()
    m = miou_batch(pred, gt)
    minority_iou = []
    for p, g, fw in zip(pred, gt, f):
        c = 0 if fw else 1                                   # the minority class is the opposite of the background
        inter = ((p == c) & (g == c)).sum(); union = ((p == c) | (g == c)).sum()
        minority_iou.append(inter / union if union else 1.0)
    fill = miou_batch(np.broadcast_to(f[:, None, None], gt.shape), gt)
    return {"miou": float(m.mean()), "minority_iou": float(np.mean(minority_iou)),
            "polarity_fill_same_images": float(fill.mean()), "n": int(len(gt))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["pixelseal", "trustmark", "stegastamp"])
    ap.add_argument("--data", default="release"); ap.add_argument("--out", required=True)
    ap.add_argument("--coco", default=".downloads/coco/val2017")
    ap.add_argument("--ann", default=".downloads/coco/annotations/instances_val2017.json")
    ap.add_argument("--native-contours", action="store_true", help="compose COCO masks on the native 512 grid")
    ap.add_argument("--pool", type=int, default=1500); ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=24); ap.add_argument("--real-per-batch", type=int, default=6)
    ap.add_argument("--extra-dir", help="static augmented split (research.coco_augment), mixed into each batch")
    ap.add_argument("--extra-per-batch", type=int, default=0)
    ap.add_argument("--post-augment-prob", type=float, default=0.0,
                    help="probability of one mild transformation per synthetic training image")
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--real-classmix", type=float, default=0.0, help="ClassMix probability between real train images")
    ap.add_argument("--official-hosts", action="store_true",
                    help="add the official images of the other methods (FLUX content) as synthetic hosts")
    ap.add_argument("--pseudo", help="dir (or npz) from build_sub_v2 --save-pseudo: pseudo-labelled test + val-fold images")
    ap.add_argument("--pseudo-per-batch", type=int, default=0, help="pseudo-labelled images per batch (taken from synth)")
    ap.add_argument("--real-cutmix", type=float, default=0.0, help="CutMix probability between real train images")
    ap.add_argument("--encoder", default="resnet34", help="timm, wm:<method>, or wmmsg:<method>")
    ap.add_argument("--val-fold", type=int, default=0); ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--no-compile", action="store_true"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    conditioned = a.encoder.startswith(MESSAGE_ENCODER_PREFIX)
    if conditioned and (a.method == "stegastamp" or a.encoder != f"{MESSAGE_ENCODER_PREFIX}{a.method}"):
        ap.error("wmmsg encoders require the matching --method (pixelseal or trustmark)")
    post_augment = PostWatermarkAugment(seed=a.seed, probability=a.post_augment_prob)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed); torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    dev = "cuda"
    torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    pool = CleanPool(a.coco, a.ann, a.pool, seed=a.seed, device=dev, native_contours=a.native_contours)
    n_hosts = add_official_hosts(pool, a.data, a.method, a.val_fold, seed=a.seed) if a.official_hosts else 0
    gen = SynthGenerator(a.method, pool, seed=a.seed)
    rt, vt = real_split(a.data, a.method, a.val_fold, dev, return_messages=conditioned)
    rx, ry, _, *rm = rt
    vx, vy, vf, *vm = vt
    ex, ey, *em = (extra_split(a.extra_dir, a.method, a.val_fold, dev, return_messages=conditioned)
                  if a.extra_dir and a.extra_per_batch else (None, None))
    if a.pseudo and a.pseudo_per_batch and conditioned:
        ap.error("--pseudo is not wired for message-conditioned encoders")
    px, py = pseudo_split(a.pseudo, a.data, a.method, a.val_fold, dev) if a.pseudo and a.pseudo_per_batch else (None, None)
    sx, sy, sf, *sm = gen.sample(64, return_messages=conditioned)
    sx = (sx * 255).round().to(torch.uint8)    # fixed synthetic validation batch
    print(f"setup {time.time() - t0:.1f}s pool={len(pool)} real_train={len(rx)} real_val={len(vx)} extra={0 if ex is None else len(ex)} pseudo={0 if px is None else len(px)} official_hosts={n_hosts}", flush=True)

    model = build_model(a.encoder).to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4, fused=True)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    fwd = model if a.no_compile else torch.compile(model, mode="reduce-overhead")
    n_syn = a.batch - a.real_per_batch - (a.extra_per_batch if ex is not None else 0) - (a.pseudo_per_batch if px is not None else 0)
    log, best = [], -1.0
    t0 = time.time()
    for step in range(1, a.steps + 1):
        x, y, _, *bits = gen.sample(n_syn, return_messages=conditioned)
        real = extra = None
        if a.real_per_batch:
            j = torch.randint(0, len(rx), (a.real_per_batch,), device=dev)
            bx, by = rx[j], ry[j]
            if px is not None:      # pseudo-labelled images join the real ones (same CutMix/ClassMix treatment)
                jp = torch.randint(0, len(px), (a.pseudo_per_batch,), device=dev)
                bx, by = torch.cat([bx, px[jp]]), torch.cat([by, py[jp]])
            real = (*real_cutmix(*real_classmix(bx, by, a.real_classmix), a.real_cutmix), *(m[j] for m in rm))
        if ex is not None:
            j = torch.randint(0, len(ex), (a.extra_per_batch,), device=dev)
            extra = ex[j], ey[j], *(m[j] for m in em)
        x, y, *messages = mix_training_sources((x, y, *bits), real, extra, post_augment)
        x = x.contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = fwd(x, *messages)
        loss = loss_fn(logits, y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if step % 50 == 0:
            print(f"::progress {step}/{a.steps} loss {loss.item():.4f} {(time.time() - t0) / step:.3f}s/step", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            rec = {"step": step, "loss": float(loss.item()), "elapsed_s": round(time.time() - t0, 1),
                   "real_val": scores(predict(model, vx, messages=vm[0] if vm else None), vy, vf),
                   "synthetic_val": scores(predict(model, sx, messages=sm[0] if sm else None), sy, sf)}
            log.append(rec); print(json.dumps(rec), flush=True)
            if rec["real_val"]["miou"] > best:
                best = rec["real_val"]["miou"]; save_atomic(model.state_dict(), out / "best.pt")
    save_atomic(model.state_dict(), out / "last.pt")
    peak_memory = {"allocated_bytes": torch.cuda.max_memory_allocated(),
                   "reserved_bytes": torch.cuda.max_memory_reserved()}
    (out / "log.json").write_text(json.dumps({"args": vars(a), "online_coco_ids": list(pool.image_ids),
                                             "peak_cuda_memory": peak_memory,
                                             "log": log, "best_real_val_miou": best}, indent=2))
    print(json.dumps({"best_real_val_miou": best, "final": log[-1]}), flush=True)


if __name__ == "__main__":
    main()
