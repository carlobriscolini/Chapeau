"""'Sister images': new, never-seen images that look like the training data, made with an image-to-image
diffusion model (SD-Turbo, an external pretrained generator).

  siblings - K variations of every forget image (same object and style, different instance).
             The original model's prediction on them ~ what a model that never saw the image would say
             -> better gold target (run_unlearn.py --sib-weight, targets.py prints the statistics).
  extgen   - external mini-ImageNet photos re-rendered in the generator's style: images that look like the
             training data but contain NOTHING from the forget set -> retain anchor (--retain extgen)
             and drift check (retain_proxy/gen).

Run:
  1) once, on the login node (needs internet):   python experiments/gen_siblings.py --download
  2) GPU job (4 GPUs, ~10-15 min):               sbatch gen.sh
     (4 shards in parallel, then  python experiments/gen_siblings.py --merge)
Outputs: ../output/siblings.npz (images uint8 [N*K,224,224,3], src [N*K]), ../output/extgen.npz,
         ../output/gen/preview.jpg (grid: each forget image followed by its siblings; photo/extgen pairs).
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

from common import EXTERNAL_HOLDOUT, OUTPUT_DIR, holdout_size, load_external_images, load_forget_images

GEN_DIR = OUTPUT_DIR / "gen"
PROMPT = "a high quality photograph"
DEFAULT_MODEL = "stabilityai/sd-turbo"


def download(model):
    from huggingface_hub import snapshot_download
    path = snapshot_download(model, allow_patterns=["model_index.json", "*/*.json", "*/*.txt", "*.fp16.safetensors"])
    print(f"downloaded {model} to {path}")


def make_pipe(model, device):
    from diffusers import AutoPipelineForImage2Image
    kw = dict(torch_dtype=torch.float16, variant="fp16") if device == "cuda" else {}
    pipe = AutoPipelineForImage2Image.from_pretrained(model, safety_checker=None, requires_safety_checker=False, **kw)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


@torch.no_grad()
def img2img(pipe, imgs_uint8, strength, steps, seed, batch_size, size, device):
    """uint8 [B,224,224,3] -> uint8 [B,224,224,3] re-generated with the diffusion model."""
    out = []
    for i in range(0, len(imgs_uint8), batch_size):
        pil = [Image.fromarray(a).resize((size, size), Image.BICUBIC) for a in imgs_uint8[i:i + batch_size].numpy()]
        g = torch.Generator(device=device).manual_seed(seed + i)
        res = pipe(prompt=[PROMPT] * len(pil), image=pil, strength=strength, num_inference_steps=steps,
                   guidance_scale=0.0, generator=g, output_type="pt").images
        res = F.interpolate(res.float(), size=(224, 224), mode="bicubic", antialias=True, align_corners=False)
        out.append((res.clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu())
    return torch.cat(out) if out else torch.zeros(0, 224, 224, 3, dtype=torch.uint8)


def run_shard(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = make_pipe(args.model, device)
    forget = load_forget_images()
    idx = np.array_split(np.arange(len(forget)), args.nshards)[args.shard]
    t0 = time.time()
    sib, sib_src = [], []
    for k in range(args.K):
        sib.append(img2img(pipe, forget[torch.from_numpy(idx)], args.strength, args.steps,
                           args.seed + 100003 * k + 7919 * args.shard, args.batch_size, args.size, device))
        sib_src.append(idx)
        print(f"shard {args.shard}: siblings k={k} done ({len(idx)} images, {time.time() - t0:.0f}s)", flush=True)

    ext_imgs = np.zeros((0, 224, 224, 3), np.uint8)
    ext_src = np.zeros(0, np.int64)
    ext = load_external_images()
    if ext is not None and args.n_ext > 0:
        pool = np.arange(min(args.n_ext, len(ext) - holdout_size(len(ext), EXTERNAL_HOLDOUT)))  # never held-out photos
        eidx = np.array_split(pool, args.nshards)[args.shard]
        ext_imgs = img2img(pipe, ext[torch.from_numpy(eidx)], args.ext_strength, args.steps,
                           args.seed + 55555 + 7919 * args.shard, args.batch_size, args.size, device).numpy()
        ext_src = eidx
        print(f"shard {args.shard}: extgen done ({len(eidx)} images, {time.time() - t0:.0f}s)", flush=True)

    GEN_DIR.mkdir(parents=True, exist_ok=True)
    out = GEN_DIR / f"shard{args.shard}.npz"
    tmp = GEN_DIR / f"shard{args.shard}.tmp{os.getpid()}.npz"
    np.savez(tmp, sib=torch.cat(sib).numpy(), sib_src=np.concatenate(sib_src), ext=ext_imgs, ext_src=ext_src,
             strength=args.strength, ext_strength=args.ext_strength)
    os.replace(tmp, out)
    print(f"shard {args.shard}: saved {out} ({time.time() - t0:.0f}s)")


def _save_npz(path, **arrays):
    tmp = path.with_name(path.stem + f".tmp{os.getpid()}.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)
    print(f"saved {path}: {arrays['images'].shape} ({path.stat().st_size / 1e9:.2f} GB)")


def preview(forget, sib, src, photos, ext, ext_src, path, tile=112, rows=6):
    k = int(np.bincount(src).max()) if len(src) else 0
    ncol = max(k + 1, 4)
    rng = np.random.default_rng(0)
    grid_rows = []
    for s in rng.choice(np.unique(src), min(rows, len(np.unique(src))), replace=False):
        grid_rows.append([forget[s]] + list(sib[src == s]))
    for r in range(2):
        pairs = rng.choice(len(ext), ncol // 2, replace=False) if len(ext) >= ncol // 2 else []
        row = []
        for j in pairs:
            row += [photos[ext_src[j]], ext[j]]
        if row:
            grid_rows.append(row)
    canvas = Image.new("RGB", (ncol * tile, len(grid_rows) * tile), "white")
    for r, row in enumerate(grid_rows):
        for c, img in enumerate(row[:ncol]):
            canvas.paste(Image.fromarray(np.asarray(img)).resize((tile, tile)), (c * tile, r * tile))
    canvas.save(path, quality=85)
    print(f"preview: {path}  (rows 1-{rows}: forget image + its siblings; last rows: photo, re-rendered photo, ...)")


def merge(args):
    files = sorted(GEN_DIR.glob("shard[0-9]*.npz"))
    files = [f for f in files if ".tmp" not in f.name]
    assert files, f"no shards in {GEN_DIR}"
    sib, src, ext, esrc = [], [], [], []
    for f in files:
        with np.load(f, allow_pickle=False) as d:
            sib.append(d["sib"]); src.append(d["sib_src"]); ext.append(d["ext"]); esrc.append(d["ext_src"])
    sib, src = np.concatenate(sib), np.concatenate(src)
    order = np.argsort(src, kind="stable")
    sib, src = sib[order], src[order]
    ext, esrc = np.concatenate(ext), np.concatenate(esrc)
    _save_npz(OUTPUT_DIR / "siblings.npz", images=sib, src=src)
    if len(ext):
        _save_npz(OUTPUT_DIR / "extgen.npz", images=ext, src=esrc)
    photos = load_external_images()
    preview(load_forget_images().numpy(), sib, src, photos.numpy() if photos is not None else None,
            ext, esrc, GEN_DIR / "preview.jpg")
    for f in files:
        f.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true", help="download the generator (login node)")
    ap.add_argument("--merge", action="store_true", help="merge shards into siblings.npz / extgen.npz")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--K", type=int, default=4, help="siblings per forget image")
    ap.add_argument("--strength", type=float, default=0.6, help="how much a sibling differs (0..1)")
    ap.add_argument("--n-ext", type=int, default=4000, help="external photos to re-render (0 = none)")
    ap.add_argument("--ext-strength", type=float, default=0.6)
    ap.add_argument("--steps", type=int, default=4, help="denoising steps x strength = steps actually run")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.download:
        download(args.model)
    elif args.merge:
        merge(args)
    else:
        run_shard(args)


if __name__ == "__main__":
    main()
