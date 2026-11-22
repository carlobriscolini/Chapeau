"""On-GPU generator of training composites that follows the generation process checked in research/probes.py:

  marked background (full_watermark=True):  whole image watermarked, holes restored to the clean original;
  clean background, rectangle (outpainting): only a rectangle watermarked, at its own size;
  clean background, object (transplant):     object cut from the whole watermarked image.

Region shapes are COCO instance masks of the same photo (object-aligned, like the real masks) or random blobs.
Clean photos and their instance masks are decoded once on CPU into GPU tensors; everything per batch (messages,
embedding, shapes, compositing, the official 128 grid) runs on the GPU."""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from research.gpu_embed import GpuEmbedder

MAX_ASPECT = {"pixelseal": 16.0, "trustmark": 2.0, "stegastamp": 16.0}
MAX_OBJECTS = 8
MIN_FRAC, MAX_FRAC = 0.003, 0.45


def _native_contour(segmentation, cropped: np.ndarray | None, x0: int, y0: int, side: int) -> np.ndarray:
    from pycocotools import mask as mask_utils

    if cropped is not None:
        # RLE has only source pixel coverage. Preserve it when moving to the 512 grid.
        return np.asarray(Image.fromarray(cropped).resize((512, 512), Image.Resampling.NEAREST))

    origin = np.array([x0, y0])
    polygons = [((np.asarray(polygon).reshape(-1, 2) - origin) * (512 / side)).ravel().tolist()
                for polygon in segmentation]
    rles = mask_utils.frPyObjects(polygons, 512, 512)
    return mask_utils.decode(mask_utils.merge(rles))


def _load_one(args):
    path, anns, seed, native_contours = args
    from pycocotools import mask as mask_utils
    rng = np.random.default_rng(seed)
    img = Image.open(path).convert("RGB"); w, h = img.size; s = min(w, h)
    x0 = int(rng.integers(0, w - s + 1)); y0 = int(rng.integers(0, h - s + 1))
    out = np.asarray(img.crop((x0, y0, x0 + s, y0 + s)).resize((512, 512), Image.Resampling.BICUBIC), np.uint8)
    masks = []
    for a in anns:
        if a.get("iscrowd"):
            continue
        segmentation = a["segmentation"]
        is_polygon = isinstance(segmentation, list)
        m = mask_utils.decode(mask_utils.frPyObjects(segmentation, h, w) if is_polygon else segmentation)
        m = m.max(axis=2) if m.ndim == 3 else m
        cropped = m[y0:y0 + s, x0:x0 + s]
        grid = np.asarray(Image.fromarray(cropped * 255).resize((128, 128), Image.Resampling.BILINEAR)) > 127
        if MIN_FRAC <= grid.mean() <= 0.6:
            masks.append((grid, segmentation, cropped if native_contours and not is_polygon else None))
    # Select and rank exactly as in the legacy pool so contour resolution cannot change object draws.
    masks = sorted(masks, key=lambda entry: -entry[0].sum())[:MAX_OBJECTS]
    if native_contours:
        pad = np.zeros((MAX_OBJECTS, 512, 64), np.uint8)
        for i, (_, segmentation, cropped) in enumerate(masks):
            contour = _native_contour(segmentation, cropped, x0, y0, s)
            pad[i] = np.packbits(contour, axis=-1, bitorder="little")
    else:
        pad = np.zeros((MAX_OBJECTS, 128, 128), bool)
        for i, (grid, _, _) in enumerate(masks):
            pad[i] = grid
    return out, pad, len(masks)


class CleanPool:
    """Clean 512 x 512 photos with COCO IDs and legacy 128-grid or packed 512-pixel instance masks."""

    def __init__(self, coco_dir: str | Path, ann_file: str | Path, n: int, seed: int = 0, device: str = "cuda",
                 *, native_contours: bool = False):
        import json
        coco_dir = Path(coco_dir)
        meta = json.loads(Path(ann_file).read_text())
        by_img: dict[int, list] = {}
        for a in meta["annotations"]:
            by_img.setdefault(a["image_id"], []).append(a)
        files = {im["id"]: im["file_name"] for im in meta["images"]}
        ids = sorted(i for i in files if i in by_img)
        rng = np.random.default_rng(seed); rng.shuffle(ids)
        self.image_ids = tuple(ids[:n])
        jobs = [(coco_dir / files[i], by_img[i], seed + k, native_contours) for k, i in enumerate(self.image_ids)]
        with ThreadPoolExecutor(8) as ex:
            res = list(ex.map(_load_one, jobs))
        self.images = torch.from_numpy(np.stack([r[0] for r in res])).permute(0, 3, 1, 2).contiguous().to(device)
        self.objects = torch.from_numpy(np.stack([r[1] for r in res])).to(device)
        self.n_objects = torch.tensor([r[2] for r in res], device=device)
        self.device = device
        self.native_contours = native_contours

    def __len__(self):
        return len(self.images)


def blobs(b: int, frac: torch.Tensor, gen: torch.Generator, res: int, *, size: int = 128) -> torch.Tensor:
    """Random smooth blobs, thresholded at the requested size to cover about `frac` of each mask."""
    noise = torch.randn(b, 1, res, res, device=frac.device, generator=gen)
    field = F.interpolate(noise, size=(size, size), mode="bicubic", align_corners=False)[:, 0].reshape(b, -1)
    k = (frac * field.shape[1]).long().clamp(1, field.shape[1] - 1)
    thr = field.sort(dim=1, descending=True).values.gather(1, k[:, None])
    return (field > thr).reshape(b, size, size)


def rect_mask(frac: float, max_aspect: float, rng: np.random.Generator) -> tuple[int, int, int, int]:
    """Random rectangle (pixel coordinates at 512) with area fraction ~frac and aspect <= max_aspect."""
    area = frac * 512 * 512
    aspect = math.exp(rng.uniform(-math.log(max_aspect), math.log(max_aspect)))
    h = int(np.clip(round(math.sqrt(area * aspect)), 16, 512)); w = int(np.clip(round(area / h), 16, 512))
    y0 = int(rng.integers(0, 512 - h + 1)); x0 = int(rng.integers(0, 512 - w + 1))
    return y0, y0 + h, x0, x0 + w


def to_grid128(m512: torch.Tensor) -> torch.Tensor:
    """Official rule: a cell is watermarked if at least 8 of its 16 pixels are."""
    return F.avg_pool2d(m512[:, None].float(), 4)[:, 0] >= 0.5


class SynthGenerator:
    def __init__(self, method: str, pool: CleanPool, embedder: GpuEmbedder | None = None, seed: int = 0,
                 p_rect: float = 0.35, p_object: float = 0.6):
        self.method, self.pool = method, pool
        self.emb = embedder or GpuEmbedder(method)
        self.dev = pool.device
        self.gen = torch.Generator(device=self.dev); self.gen.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.p_rect, self.p_object = p_rect, p_object
        self.shape_size = 512 if pool.native_contours else 128
        if pool.native_contours:
            self.bit_shifts = torch.arange(8, device=self.dev, dtype=torch.uint8)

    def _fracs(self, b: int) -> torch.Tensor:
        u = torch.rand(b, device=self.dev, generator=self.gen)
        return torch.exp(math.log(MIN_FRAC) + u * (math.log(MAX_FRAC) - math.log(MIN_FRAC)))

    def _shapes(self, idx: torch.Tensor, frac: torch.Tensor) -> torch.Tensor:
        """Object mask of the same photo (one instance, or a union of two) or a blob."""
        b = len(idx)
        out = blobs(b, frac, self.gen, int(self.rng.integers(4, 17)), size=self.shape_size)
        n_obj = self.pool.n_objects[idx]
        use_obj = (torch.rand(b, device=self.dev, generator=self.gen) < self.p_object) & (n_obj > 0)
        k = (torch.rand(b, device=self.dev, generator=self.gen) * n_obj.clamp_min(1)).long()
        obj = self.pool.objects[idx, k]
        k2 = (torch.rand(b, device=self.dev, generator=self.gen) * n_obj.clamp_min(1)).long()
        two = torch.rand(b, device=self.dev, generator=self.gen) < 0.3
        other = self.pool.objects[idx, k2]
        if self.pool.native_contours:
            packed = obj | torch.where(two[:, None, None], other, 0)
            obj = ((packed[..., None] >> self.bit_shifts) & 1).bool().reshape_as(out)
        else:
            obj = obj | (other & two[:, None, None])
        return torch.where(use_obj[:, None, None], obj, out)

    @torch.no_grad()
    def sample(self, b: int, *, return_messages: bool = False) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        """Return float images, official Boolean 128 masks, full-watermark flags, and optionally the embedded bits."""
        idx = torch.randint(0, len(self.pool), (b,), device=self.dev, generator=self.gen)
        clean = self.pool.images[idx].float() / 255
        bits = torch.randint(0, 2, (b, self.emb.nbits), device=self.dev, generator=self.gen)
        full = torch.rand(b, device=self.dev, generator=self.gen) < 0.5
        frac = self._fracs(b)
        marked = self.emb.embed(clean, bits)
        shapes = self._shapes(idx, frac)
        region = torch.where(full[:, None, None], ~shapes, shapes)
        region512 = region if self.pool.native_contours else (
            F.interpolate(region[:, None].float(), size=(512, 512), mode="nearest")[:, 0] > 0.5
        )
        rect = (~full) & (torch.rand(b, device=self.dev, generator=self.gen) < self.p_rect)
        for i in torch.nonzero(rect).flatten().tolist():
            y0, y1, x0, x1 = rect_mask(float(frac[i]), MAX_ASPECT[self.method], self.rng)
            patch = self.emb.embed(clean[i:i + 1, :, y0:y1, x0:x1], bits[i:i + 1])
            marked[i] = clean[i]; marked[i, :, y0:y1, x0:x1] = patch[0]
            region512[i] = False; region512[i, y0:y1, x0:x1] = True
        images = torch.where(region512[:, None], marked, clean)
        result = images, to_grid128(region512), full
        return (*result, bits) if return_messages else result


HOST_METHODS = {"pixelseal": ("trustmark",), "trustmark": ("pixelseal",), "stegastamp": ("pixelseal", "trustmark")}


def add_official_hosts(pool: CleanPool, data: str | Path, method: str, val_fold: int, seed: int = 0) -> int:
    """Append the official images (train AND test, organizers 22/11: any available data may be used) watermarked
    with ANOTHER method as clean hosts: same FLUX-generated content statistics as the target images, while the
    other method's residual is just content for this method's detector. Shapes: real edit masks (the minority region)
    of official train images outside the validation fold, so the composite regions look like the real edits.
    Returns the number of hosts added. Only for legacy 128-grid pools (native_contours False)."""
    import json
    from research import folds as folds_mod
    assert not pool.native_contours
    root = Path(data); fold_of = folds_mod.load(); rng = np.random.default_rng(seed)
    z = np.load(root / "train" / "masks_128.npz"); m128 = dict(zip(z["names"].tolist(), z["masks"]))
    meta_tr = json.loads((root / "train" / "metadata.json").read_text())
    shapes = []
    for m in meta_tr:
        if fold_of[m["image_name"]] == val_fold:
            continue
        g = m128[m["image_name"]].astype(bool); r = g != bool(m["full_watermark"])
        if MIN_FRAC <= r.mean() <= 0.6:
            shapes.append(r)
    paths = [root / split / "images" / f"{m['image_name']}.png"
             for split in ("train", "test") for m in json.loads((root / split / "metadata.json").read_text())
             if m["method"] in HOST_METHODS[method]]
    with ThreadPoolExecutor(8) as ex:
        imgs = list(ex.map(lambda p: np.asarray(Image.open(p).convert("RGB"), np.uint8), paths))
    objs = np.zeros((len(imgs), MAX_OBJECTS, 128, 128), bool)
    for i in range(len(imgs)):
        for j, s in enumerate(rng.choice(len(shapes), MAX_OBJECTS, replace=False)):
            objs[i, j] = shapes[s]
    dev = pool.device
    pool.images = torch.cat([pool.images, torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).contiguous().to(dev)])
    pool.objects = torch.cat([pool.objects, torch.from_numpy(objs).to(dev)])
    pool.n_objects = torch.cat([pool.n_objects, torch.full((len(imgs),), MAX_OBJECTS, device=dev, dtype=pool.n_objects.dtype)])
    pool.image_ids = tuple(pool.image_ids) + tuple(f"official:{p.stem}" for p in paths)
    return len(imgs)
