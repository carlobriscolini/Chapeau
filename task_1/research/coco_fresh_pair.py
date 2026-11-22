"""Build two equal COCO splits with identical new masks and messages but different photos."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from io import TextIOWrapper
from itertools import chain
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import time
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
from PIL import Image
import torch

from research.coco_augment import (
    METHOD_BITS, copy_originals, cover_image, cover_views, image_path, mask_from_disk,
    read_json, reference_images, snapshot, verify_dataset,
)
from research.overlap import PHASH_DISTANCE, ImageIndex, pixel_hash
from research.synth import composite
from wmloc.utils import downsample_mask
from wmloc.watermarks import load_watermark


COCO_URL = "https://s3.amazonaws.com/images.cocodataset.org/train2017/"
ANNOTATIONS = Path(".downloads/coco/annotations_trainval2017.zip")
BASE = Path("outputs/coco_augmented")
V2 = Path("outputs/coco_augmented_v2")
VAL_COCO = Path(".downloads/coco/val2017")
SEED = 20261123


def paired_templates(rows: list[dict], count: int, rng) -> list[tuple[dict, dict]]:
    """Assign an old cover distinct from the template photo to each of the same recipes."""
    templates = list(rng.permutation(rows)[:count])
    if len(templates) < 2 or len({row["source_coco_id"] for row in templates}) != len(templates):
        raise ValueError("Need at least two distinct template covers")
    offset = int(rng.integers(1, len(templates)))
    return list(zip(templates, templates[offset:] + templates[:offset], strict=True))


def raw_jpeg(image: dict):
    url = COCO_URL + image["file_name"]
    for attempt in range(3):
        try:
            with urlopen(url, timeout=12) as response:
                payload = response.read(6 * 1024 * 1024 + 1)
            if payload[:2] == b"\xff\xd8" and len(payload) <= 6 * 1024 * 1024:
                return image, payload
            return image, None
        except OSError:
            if attempt < 2:
                time.sleep(.25 * (attempt + 1))
    return image, None


def old_views():
    for path in sorted(VAL_COCO.glob("*.jpg")):
        original, _, crop = cover_image(path, SEED)
        for view, image in enumerate(cover_views(original, crop)):
            yield (path.name, view), image


def image_index_records(stream):
    """Read the COCO image index without materialising the much larger polygon annotations."""
    decoder = json.JSONDecoder()
    prefix = ""
    with TextIOWrapper(stream, encoding="utf-8") as text:
        for _ in range(8):
            prefix += text.read(8 * 1024 * 1024)
            start = re.search(r'"images"\s*:\s*(?=\[)', prefix)
            if start:
                try:
                    rows, _ = decoder.raw_decode(prefix, idx=start.end())
                except json.JSONDecodeError:
                    continue
                return rows
    raise ValueError("COCO images index could not be read from the first 64 MiB")


def fetch_unseen(root: Path, count: int, reference: ImageIndex, rng):
    """Fetch originals from COCO train2017 under the same cap as the previous dataset."""
    with ZipFile(ANNOTATIONS) as archive, archive.open("annotations/instances_train2017.json") as stream:
        images = image_index_records(stream)
    # Use the documented COCO licence IDs for CC BY and CC BY-SA originals.
    sources = [image for image in images if image["license"] in (4, 5)
               and min(image["width"], image["height"]) >= 256]
    del images
    existing = ImageIndex(old_views())
    chosen, pixel_ids, failures = [], set(), Counter()
    photos = root / "fresh_covers"
    photos.mkdir()
    order = rng.permutation(sources)
    transferred = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        for start in range(0, len(order), 32):
            for item, payload in pool.map(raw_jpeg, order[start:start + 32]):
                if payload is None:
                    failures["download"] += 1
                    continue
                transferred += len(payload)
                if transferred > 10_000_000_000:
                    raise ValueError("COCO input exceeds the 10 GB download limit")
                if len(chosen) == count:
                    continue
                path = photos / item["file_name"]
                path.write_bytes(payload)
                try:
                    original, crop, cropped = cover_image(path, SEED)
                except OSError:
                    path.unlink()
                    failures["invalid_jpeg"] += 1
                    continue
                pixels = pixel_hash(cropped)
                if pixels in pixel_ids or any(reference.matches(view) or existing.matches(view)
                                             for view in cover_views(original, cropped)):
                    path.unlink()
                    failures["overlap"] += 1
                    continue
                pixel_ids.add(pixels)
                chosen.append((item, path, crop))
                if len(chosen) % 100 == 0 or len(chosen) == count:
                    print(f"{len(chosen)}/{count} nouvelles photos contrôlées", flush=True)
            if len(chosen) == count:
                return chosen, dict(failures), transferred
    raise ValueError(f"Could audit only {len(chosen)} independent COCO photos")


def render(watermark, template, source_path, mask):
    _, crop, cover = cover_image(source_path, SEED)
    clean = np.asarray(cover)
    bits = np.fromiter(template["message"], dtype=np.uint8)
    full = template["full_watermark"]
    result = composite(watermark, clean, mask, full, bits, tuple(template["embedding_rect"]))
    changed = np.any(result != clean, axis=-1)
    if not changed.any() or np.any(changed & (mask == 0)):
        raise ValueError(f"Invalid compositing residual for {template['image_name']} on {source_path.name}")
    return Image.fromarray(result), crop, pixel_hash(cover)


def save_split(split, metadata, grids):
    names = [row["image_name"] for row in metadata]
    (split / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    np.savez_compressed(split / "masks_128.npz", names=names, masks=np.stack(grids))


def build(out: Path, count: int, device: str):
    if out.exists():
        raise FileExistsError(out)
    if not 2 <= count <= 1020:
        raise ValueError("count must be between 2 and 1020")
    rng = np.random.default_rng(SEED)
    base_rows = read_json(BASE / "train/metadata.json")
    v2_rows = read_json(V2 / "train/metadata.json")
    if v2_rows[:len(base_rows)] != base_rows:
        raise ValueError("V2 must extend V1 without modifying its rows")
    pairs = paired_templates(v2_rows[len(base_rows):], count, rng)
    train_dir, test_dir = Path("release/train"), Path("release/test")
    train_rows, test_rows = [read_json(part / "metadata.json") for part in (train_dir, test_dir)]
    folds = read_json(Path("research/folds.json"))
    test_before = snapshot(test_dir)
    if test_before != read_json(V2 / "audit.json")["official_test_snapshot_sha256"]:
        raise ValueError("Official test differs from the previous audit")
    reference = ImageIndex(chain(reference_images(train_dir, train_rows), reference_images(test_dir, test_rows)))
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{out.name}-", dir=out.parent) as staging:
        root = Path(staging) / "data"
        root.mkdir()
        photos, exclusions, downloaded = fetch_unseen(root, count, reference, rng)
        arms = {kind: root / kind / "train" for kind in ("reused", "fresh")}
        grids = {kind: copy_originals(base_rows, BASE / "train", split) for kind, split in arms.items()}
        rows = {kind: base_rows.copy() for kind in arms}
        before = {item["source_coco_id"] for item in base_rows if "source_coco_id" in item}
        provenance = []
        seen = {kind: set() for kind in arms}
        for method in METHOD_BITS:
            tasks = [(pair, photo) for pair, photo in zip(pairs, photos, strict=True) if pair[0]["method"] == method]
            if not tasks:
                continue
            wm = load_watermark(method, device=device)
            for ((template, old), (new, fresh_path, _)) in tasks:
                name = template["image_name"]
                mask = mask_from_disk(V2 / "train", name)
                grid = downsample_mask(mask)
                if template["source_fold"] != folds[template["source_train_recipe"]]:
                    raise ValueError(f"Wrong validation-fold provenance for {name}")
                ids = {"reused": old["source_coco_id"], "fresh": new["id"]}
                if ids["fresh"] in before or ids["reused"] not in before:
                    raise ValueError("Fresh and reused photos are not in the required source pools")
                paths = {"reused": VAL_COCO / f"{ids['reused']:012d}.jpg", "fresh": fresh_path}
                for kind, split in arms.items():
                    image, crop, cover_digest = render(wm, template, paths[kind], mask)
                    pixels = pixel_hash(image)
                    if pixels in seen[kind] or reference.matches(image):
                        raise ValueError(f"Unexpected match in {kind} arm for {name}")
                    seen[kind].add(pixels)
                    row = template.copy()
                    row.pop("parent_image_name", None)
                    row.update({
                        "image_name": f"pair_{kind}_{name}", "source_coco_id": ids[kind],
                        "source_crop": crop, "source_coco_annotation": None, "matched_recipe_id": name,
                        "source_cover_rgb_sha256": cover_digest,
                        "source_coco_split": "val2017" if kind == "reused" else "train2017",
                    })
                    image.save(image_path(split, row), compress_level=1)
                    Image.fromarray(mask * 255).save(split / "masks" / f"{row['image_name']}.png")
                    rows[kind].append(row)
                    grids[kind].append(grid)
                provenance.append({"recipe": name, "reused_id": ids["reused"], "fresh_id": ids["fresh"],
                                   "reused_jpeg_sha256": sha256(paths["reused"].read_bytes()).hexdigest(),
                                   "fresh_license_id": new["license"], "fresh_url": COCO_URL + new["file_name"],
                                   "fresh_jpeg_sha256": sha256(fresh_path.read_bytes()).hexdigest()})
                if len(provenance) % 25 == 0 or len(provenance) == count:
                    print(f"::progress {len(provenance)}/{count} paires tatouées", flush=True)
            del wm
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        for kind, split in arms.items():
            save_split(split, rows[kind], grids[kind])
            verify_dataset(split.parent, train_rows, folds)
        if snapshot(test_dir) != test_before:
            raise ValueError("Official test changed during comparative dataset generation")
        paired = list(zip(rows["reused"][len(base_rows):], rows["fresh"][len(base_rows):], strict=True))
        paired_fields = ("matched_recipe_id", "method", "full_watermark", "message", "source_fold", "source_train_recipe")
        if len(paired) != count or any(left[key] != right[key] for left, right in paired for key in paired_fields):
            raise ValueError("Unmatched mask or message settings in experimental arms")
        report = {"base_examples": len(base_rows), "paired_additions": count, "data_source": COCO_URL,
                  "fresh_download_bytes": downloaded, "fresh_rgb_or_phash_hits_retained": 0,
                  "phash_distance_limit": PHASH_DISTANCE, "cover_exclusions": exclusions,
                  "official_test_sha256": test_before,
                  "paired_fields": paired_fields, "matched_mask_geometry": True,
                  "validation_counts_equal_after_source_fold_exclusion": True,
                  "input_shapes": {"fresh": "new COCO train2017 covers", "reused": "deranged COCO val2017 covers"},
                  "provenance": provenance}
        (root / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
        root.rename(out)
        print(f"Verified two splits of {len(base_rows) + count} examples in {out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("outputs/protocol_v3_datasets"))
    parser.add_argument("--count", type=int, default=1020)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    build(args.out, args.count, args.device)


if __name__ == "__main__":
    main()
