"""Batched, per-sample image augmentations on GPU tensors already preprocessed to [-1, 1] ([B, 3, H, W]).

Levels
  light   - mimics the server's hidden "altered copies": small zoom/shift, flip, colour, a bit of noise.
            Used on the forget branch during training (and sometimes skipped, so exact images are seen too).
  medium / strong - destroy pixel-exact memorisation while keeping class content.
            Used to estimate the "never seen it" target q(x) from the frozen original model.
Mixing (mixup / cutmix of two different images) builds the retain-anchor images.
"""
import math

import torch
import torch.nn.functional as F

LEVELS = {
    #          zoom range     extra shift  blur sigma   brightness contrast   noise std  p_skip
    "light":  dict(scale=(0.85, 1.0), shift=0.08, blur=(0.0, 0.0), bright=0.10, contrast=0.10, noise=0.02, p_skip=0.25),
    "medium": dict(scale=(0.55, 0.85), shift=0.0,  blur=(0.0, 1.0), bright=0.15, contrast=0.20, noise=0.04, p_skip=0.0),
    "strong": dict(scale=(0.35, 0.65), shift=0.0,  blur=(0.0, 1.5), bright=0.20, contrast=0.30, noise=0.06, p_skip=0.0),
}


def _rand(n, lo, hi, device, g=None):
    return torch.rand(n, device=device, generator=g) * (hi - lo) + lo


def random_affine(x, scale, shift, flip=True, g=None):
    """Per-sample random crop-and-resize (zoom in by `scale`), translation and horizontal flip."""
    b, dev = x.shape[0], x.device
    s = _rand(b, *scale, dev, g)
    max_t = (1 - s) + shift  # stay inside the crop range, plus an extra shift (reflection padding beyond)
    tx = (torch.rand(b, device=dev, generator=g) * 2 - 1) * max_t
    ty = (torch.rand(b, device=dev, generator=g) * 2 - 1) * max_t
    fx = torch.where(torch.rand(b, device=dev, generator=g) < 0.5, -1.0, 1.0) if flip else torch.ones(b, device=dev)
    theta = torch.zeros(b, 2, 3, device=dev)
    theta[:, 0, 0] = s * fx
    theta[:, 1, 1] = s
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


def gaussian_blur(x, sigma: float):
    if sigma < 0.3:
        return x
    k = int(2 * math.ceil(3 * sigma) + 1)
    t = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2
    w = torch.exp(-t ** 2 / (2 * sigma ** 2))
    w = w / w.sum()
    c = x.shape[1]
    x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="reflect"), w.view(1, 1, 1, k).repeat(c, 1, 1, 1), groups=c)
    x = F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="reflect"), w.view(1, 1, k, 1).repeat(c, 1, 1, 1), groups=c)
    return x


def augment(x, level: str = "light", g=None, p_skip=None):
    """x: float [B,3,H,W] in [-1,1]. Returns an augmented copy (same shape, clamped to [-1,1])."""
    cfg = LEVELS[level]
    b, dev = x.shape[0], x.device
    out = random_affine(x, cfg["scale"], cfg["shift"], g=g)
    if cfg["blur"][1] > 0:
        out = gaussian_blur(out, float(_rand(1, *cfg["blur"], dev, g)))
    mean = out.mean(dim=(1, 2, 3), keepdim=True)
    contrast = 1 + (torch.rand(b, 1, 1, 1, device=dev, generator=g) * 2 - 1) * cfg["contrast"]
    bright = (torch.rand(b, 1, 1, 1, device=dev, generator=g) * 2 - 1) * cfg["bright"] * 2  # [-1,1] range is 2 wide
    out = (out - mean) * contrast + mean + bright
    if cfg["noise"] > 0:
        out = out + torch.randn(out.shape, device=dev, generator=g) * cfg["noise"] * 2
    out = out.clamp(-1, 1)
    p_skip = cfg["p_skip"] if p_skip is None else p_skip
    if p_skip > 0:
        keep = (torch.rand(b, device=dev, generator=g) < p_skip).view(b, 1, 1, 1)
        out = torch.where(keep, x, out)
    return out


def mix(x, g=None, lam_range=(0.35, 0.65)):
    """Retain-anchor images: each image mixed with a DIFFERENT image of the batch.
    Half the batch uses mixup (blend), half cutmix (paste a large box)."""
    b, _, h, w = x.shape
    dev = x.device
    shift = int(torch.randint(1, max(b, 2), (1,), device=dev, generator=g)) if b > 1 else 0
    other = x.roll(shift, dims=0)
    lam = _rand(b, *lam_range, dev, g).view(b, 1, 1, 1)
    mixed = lam * x + (1 - lam) * other
    # cutmix for the second half: box covering (1 - lam) of the area
    cut = torch.zeros(b, 1, h, w, device=dev)
    for i in range(b // 2, b):
        area = float(1 - lam[i])
        bh, bw = int(h * math.sqrt(area)), int(w * math.sqrt(area))
        y0 = int(torch.randint(0, h - bh + 1, (1,), device=dev, generator=g))
        x0 = int(torch.randint(0, w - bw + 1, (1,), device=dev, generator=g))
        cut[i, :, y0:y0 + bh, x0:x0 + bw] = 1
    is_cut = torch.zeros(b, 1, 1, 1, device=dev, dtype=torch.bool)
    is_cut[b // 2:] = True
    cutmixed = x * (1 - cut) + other * cut
    return torch.where(is_cut, cutmixed, mixed)


def patch_shuffle(x, grid: int = 8, g=None):
    """Cut each image into grid x grid tiles and shuffle them (per sample): natural colours/textures,
    but the object (and so the memorised image) is no longer recognisable."""
    b, c, h, w = x.shape
    ph, pw = h // grid, w // grid
    x = x[:, :, :ph * grid, :pw * grid]
    tiles = x.reshape(b, c, grid, ph, grid, pw).permute(0, 2, 4, 1, 3, 5).reshape(b, grid * grid, c, ph, pw)
    perm = torch.argsort(torch.rand(b, grid * grid, device=x.device, generator=g), dim=1)
    tiles = tiles[torch.arange(b, device=x.device)[:, None], perm]
    out = tiles.reshape(b, grid, grid, c, ph, pw).permute(0, 3, 1, 4, 2, 5).reshape(b, c, grid * ph, grid * pw)
    if out.shape[-2:] != (h, w):
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
    return out


def smooth_noise(b, h=224, w=224, device="cpu", g=None):
    """Random coloured, spatially smooth noise images in [-1, 1] (no data needed): sum of a few octaves."""
    out = torch.zeros(b, 3, h, w, device=device)
    for res, amp in ((4, 1.0), (8, 0.6), (16, 0.35), (32, 0.2), (64, 0.1)):
        z = torch.randn(b, 3, res, res, device=device, generator=g)
        out += amp * F.interpolate(z, size=(h, w), mode="bicubic", align_corners=False)
    out = out / out.flatten(1).std(dim=1).view(b, 1, 1, 1).clamp_min(1e-6) * 0.5
    shift = (torch.rand(b, 3, 1, 1, device=device, generator=g) - 0.5)  # random colour cast
    return (out + shift).clamp(-1, 1)


def retain_images(x, kind: str = "mix", g=None):
    """Anchor images for the retain branch. kind: mix | shuffle | noise | shuffle+noise (half/half)."""
    if kind == "mix":
        return mix(x, g)
    if kind == "shuffle":
        return patch_shuffle(x, grid=int(torch.randint(4, 9, (1,)).item()), g=g)
    if kind == "noise":
        return smooth_noise(x.shape[0], x.shape[2], x.shape[3], x.device, g)
    if kind == "shuffle+noise":
        half = x.shape[0] // 2
        return torch.cat([patch_shuffle(x[:half], grid=8, g=g),
                          smooth_noise(x.shape[0] - half, x.shape[2], x.shape[3], x.device, g)])
    raise ValueError(kind)


def patch_mask(x, frac=(0.5, 0.7), patch=16, g=None):
    """Hide a random 50-70% of the 16x16 patches (set to 0 = mid-grey in [-1,1]): breaks exact-instance
    memorisation in a ViT while keeping enough semantic evidence (MAE-style)."""
    b, c, h, w = x.shape
    gh, gw = h // patch, w // patch
    f = _rand(b, *frac, x.device, g).view(b, 1)
    keep = (torch.rand(b, gh * gw, device=x.device, generator=g) >= f).float().view(b, 1, gh, gw)
    keep = F.interpolate(keep, scale_factor=patch, mode="nearest")
    return x * keep
