"""Download an external, unlabeled natural-image set for the retain anchor (labels are NOT used).

Default: timm/mini-imagenet (100 ImageNet classes, real photos) from HuggingFace.
Images are centre-cropped and resized to 224x224, stored as uint8 [N, 224, 224, 3] (same layout as the forget set)
in ../output/external_224.npz.

Run ONCE on the login node (needs internet), in a throw-away env so the shared .venv is not modified:
    uv run --no-project --with pyarrow --with pillow --with numpy --with huggingface_hub \
        python data_prep/prepare_external.py --n 10000
"""
import argparse
import io
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image

REPO = "timm/mini-imagenet"
FILES = ["data/test-00000-of-00002.parquet", "data/test-00001-of-00002.parquet",
         "data/validation-00000-of-00003.parquet", "data/validation-00001-of-00003.parquet"]
OUT = Path(__file__).resolve().parents[2] / "output" / "external_224.npz"


def to_224(img_bytes: bytes) -> np.ndarray:
    im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((224, 224), Image.BICUBIC)
    return np.asarray(im, dtype=np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--files", nargs="+", default=FILES)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    imgs = []
    for f in args.files:
        path = hf_hub_download(REPO, f, repo_type="dataset")
        table = pq.read_table(path, columns=["image"])
        col = table.column("image").to_pylist()
        print(f"{f}: {len(col)} images", flush=True)
        for item in col:
            imgs.append(to_224(item["bytes"]))
            if len(imgs) >= args.n:
                break
        if len(imgs) >= args.n:
            break
    arr = np.stack(imgs)
    rng = np.random.default_rng(0)
    arr = arr[rng.permutation(len(arr))]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.stem + f".tmp{os.getpid()}.npz")
    np.savez(tmp, images=arr)
    os.replace(tmp, args.out)
    print(f"saved {args.out}: {arr.shape} {arr.dtype} ({args.out.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
