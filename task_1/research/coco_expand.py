"""Append new COCO mask geometries to an already audited training split."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
from itertools import chain
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image
import torch

from research.coco_augment import (
    METHOD_BITS, coco_annotations, copy_originals, cover_image, cover_views,
    embed_masked, image_path, mask_from_disk, object_mask, read_json, reference_images,
    snapshot, verify_dataset,
)
from research.mask_variants import cropped_shape, perturb_minority
from research.overlap import ImageIndex, PHASH_DISTANCE, pixel_hash
from research.synth import bbox, is_rectangle
from wmloc.utils import downsample_mask
from wmloc.watermarks import load_watermark


@dataclass
class MaskVariant:
    mask: np.ndarray
    grid: np.ndarray
    kind: str
    annotation: int | None = None


def minority_mask(mask: np.ndarray, full: bool) -> np.ndarray:
    return 1 - mask if full else mask


def size_band(fraction: float) -> str:
    if fraction < .01:
        return "below_1pct"
    return "1_to_5pct" if fraction < .05 else "at_least_5pct"


def quotas(counts: dict[str, int], total: int) -> dict[str, int]:
    """Preserve the proportions of the parent split, rounded to whole examples."""
    denominator = sum(counts.values())
    target = {key: value * total // denominator for key, value in counts.items()}
    order = sorted(counts, key=lambda key: (-(counts[key] * total % denominator), key))
    for key in order[:total - sum(target.values())]:
        target[key] += 1
    return target


def groups_for(rows, grids):
    groups = defaultdict(lambda: defaultdict(list))
    for row, grid in zip(rows, grids, strict=True):
        if "source_coco_id" not in row:
            continue
        full = row["full_watermark"]
        area = float(minority_mask(grid, full).mean())
        groups[row["method"], full][size_band(area)].append(row)
    return groups


def grid_key(grid: np.ndarray) -> bytes:
    return np.packbits(grid).tobytes()


def diversity(rows, grids):
    all_stats = {}
    for full, name in ((False, "clean_background"), (True, "watermarked_background")):
        selected = [minority_mask(grid, full) for row, grid in zip(rows, grids, strict=True)
                    if "source_coco_id" in row and row["full_watermark"] == full]
        small = [grid for grid in selected if 0 < grid.mean() < .05]
        centers = np.zeros((4, 4), dtype=int)
        for grid in small:
            y0, y1, x0, x1 = bbox(grid)
            centers[(y0 + y1) // 64, (x0 + x1) // 64] += 1
        all_stats[name] = {"n": len(selected), "small_n": len(small),
                           "small_unique_grids": len({grid_key(grid) for grid in small}),
                           "small_unique_cropped_shapes": len({cropped_shape(grid) for grid in small}),
                           "small_centres_4x4": centers.tolist()}
    return all_stats


def choose_mask(parent, parent_mask, parent_grid, crop, annotations, rng, used_grids, used_shapes):
    full = parent["full_watermark"]
    template = minority_mask(parent_mask, full)
    previous = minority_mask(parent_grid, full).astype(bool)
    if not template.any() or not previous.any():
        return None
    band = size_band(float(previous.mean()))
    rectangle = is_rectangle(template)
    parent_annotation = parent.get("source_coco_annotation")
    objects = [obj for obj in annotations.get(parent["source_coco_id"], ()) if obj["id"] != parent_annotation]
    aligned = object_mask(objects, crop, int(template.sum()), rng) if not rectangle else None
    for attempt in range(14):
        annotation = None
        if attempt == 0 and aligned is not None:
            region, annotation = aligned
            kind = "coco_instance"
        else:
            # An aligned positive remains tied to a real COCO object on this photo.
            if not full and parent_annotation is not None:
                break
            region = perturb_minority(template, rng)
            kind = "rectangle" if rectangle else "contour"
        if rectangle and not full and parent["method"] == "trustmark":
            y0, y1, x0, x1 = bbox(region)
            if max(y1 - y0, x1 - x0) > 2 * min(y1 - y0, x1 - x0):
                continue
        mask = minority_mask(region, full)
        grid = downsample_mask(mask)
        minority = minority_mask(grid, full).astype(bool)
        fraction = float(minority.mean())
        if not 0 < fraction < .5 or size_band(fraction) != band:
            continue
        union = (minority | previous).sum()
        if (minority & previous).sum() / union > .8:
            continue
        if grid_key(grid) in used_grids or cropped_shape(minority) in used_shapes[full]:
            continue
        return MaskVariant(mask, grid, kind, annotation)
    return None


def make_example(watermark, parent, variant, clean, rng):
    method = parent["method"]
    bits = rng.integers(0, 2, size=watermark.nbits, dtype=np.uint8)
    message = "".join(map(str, bits))
    if message == parent["message"]:
        bits[0] ^= 1
        message = "".join(map(str, bits))
    name = f"{parent['image_name']}_mask2"
    image, rect = embed_masked(watermark, method, np.asarray(clean), variant.mask, parent["full_watermark"],
                               bits, rng, name)
    row = {**parent, "image_name": name, "message": message, "embedding_rect": rect,
           "source_coco_annotation": variant.annotation, "parent_image_name": parent["image_name"],
           "mask_variant": variant.kind}
    return image, row


def build(args):
    parent_dir = args.base / "train"
    if args.out.exists():
        raise FileExistsError(args.out)
    base_rows = read_json(parent_dir / "metadata.json")
    parent_audit = read_json(args.base / "audit.json")
    train_rows = read_json(args.data / "train" / "metadata.json")
    test_dir = args.data / "test"
    test_rows = read_json(test_dir / "metadata.json")
    folds = read_json(args.folds)
    if base_rows[:len(train_rows)] != train_rows or len(base_rows) != parent_audit["total_train"]:
        raise ValueError("Base dataset does not match its audit and the official train")
    before = snapshot(test_dir)
    if before != parent_audit["official_test_snapshot_sha256"]:
        raise ValueError("Official test does not match the base dataset audit")
    reference = ImageIndex(chain(reference_images(args.data / "train", train_rows), reference_images(test_dir, test_rows)))
    annotations = coco_annotations(args.annotations)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    goal = 6 * args.per_stratum
    additions, new_grids = [], []
    rejected = Counter()
    overlaps = []
    counts = Counter()
    shapes = Counter()
    emitted = set()
    with TemporaryDirectory(prefix=f".{args.out.name}-", dir=args.out.parent) as temporary:
        root = Path(temporary) / "dataset"
        split = root / "train"
        grids = copy_originals(base_rows, parent_dir, split)
        by_name = {row["image_name"]: grid for row, grid in zip(base_rows, grids, strict=True)}
        groups = groups_for(base_rows, grids)
        expected = {(method, full) for method in METHOD_BITS for full in (False, True)}
        if set(groups) != expected:
            raise ValueError("Missing method or background group in base dataset")
        used_grids = {grid_key(grid) for grid in grids}
        used_shapes = {False: set(), True: set()}
        for row, grid in zip(base_rows, grids, strict=True):
            full = row["full_watermark"]
            minor = minority_mask(grid, full)
            if minor.any():
                used_shapes[full].add(cropped_shape(minor))
        for method in METHOD_BITS:
            watermark = load_watermark(method, device=args.device)
            for full, background in ((False, "clean"), (True, "marked")):
                buckets = groups[method, full]
                targets = quotas({band: len(rows) for band, rows in buckets.items()}, args.per_stratum)
                for band, candidates in buckets.items():
                    key = f"{method}:{background}:{band}"
                    for parent in rng.permutation(candidates):
                        if counts[key] == targets[band]:
                            break
                        name = parent["image_name"]
                        path = args.coco / f"{parent['source_coco_id']:012d}.jpg"
                        original, crop, clean = cover_image(path, parent_audit["seed"])
                        if list(crop) != parent["source_crop"]:
                            raise ValueError(f"Parent crop does not reproduce for {name}")
                        recipe = mask_from_disk(parent_dir, name)
                        mask_rng = np.random.default_rng([args.seed, parent["source_coco_id"], 0])
                        variant = choose_mask(parent, recipe, by_name[name], crop, annotations,
                                              mask_rng, used_grids, used_shapes)
                        if variant is None:
                            rejected["no_new_shape"] += 1
                            continue
                        hits = [hit for view in cover_views(original, clean) for hit in reference.matches(view)]
                        if hits:
                            overlaps.append({"parent": name, "kind": "cover", "matches": hits})
                            continue
                        message_rng = np.random.default_rng([args.seed, parent["source_coco_id"], 1])
                        image, row = make_example(watermark, parent, variant, clean, message_rng)
                        hits = reference.matches(image)
                        if hits:
                            overlaps.append({"parent": name, "kind": "composite", "matches": hits})
                            continue
                        pixels = pixel_hash(image)
                        with Image.open(image_path(parent_dir, parent)) as prior:
                            identical_to_parent = pixels == pixel_hash(prior)
                        if identical_to_parent or pixels in emitted:
                            rejected["exact_duplicate_rgb"] += 1
                            continue
                        emitted.add(pixels)
                        used_grids.add(grid_key(variant.grid))
                        used_shapes[full].add(cropped_shape(minority_mask(variant.grid, full)))
                        image.save(image_path(split, row), compress_level=1)
                        Image.fromarray(variant.mask * 255).save(split / "masks" / f"{row['image_name']}.png")
                        additions.append(row)
                        new_grids.append(variant.grid)
                        counts[key] += 1
                        shapes[variant.kind] += 1
                        if len(additions) % 25 == 0 or len(additions) == goal:
                            print(f"::progress {len(additions)}/{goal} génération {method} {background}", flush=True)
                    if counts[key] != targets[band]:
                        raise ValueError(f"Only {counts[key]} unique examples for {key}, expected {targets[band]}")
            del watermark
        rows = base_rows + additions
        (split / "metadata.json").write_text(json.dumps(rows, indent=2) + "\n")
        np.savez_compressed(split / "masks_128.npz", names=[row["image_name"] for row in rows], masks=np.stack(grids + new_grids))
        verify_dataset(root, train_rows, folds)
        if before != snapshot(test_dir):
            raise ValueError("Official test changed during generation")
        report = {
            "base": str(args.base), "base_train_count": len(base_rows), "total_train": len(rows),
            "original_train": len(train_rows), "official_test": len(test_rows), "official_test_snapshot_sha256": before,
            "base_metadata_sha256": sha256((parent_dir / "metadata.json").read_bytes()).hexdigest(),
            "seed": args.seed, "device": args.device, "added": len(additions), "phash_threshold": PHASH_DISTANCE,
            "before": diversity(base_rows, grids), "added_diversity": diversity(additions, new_grids),
            "after": diversity(rows, grids + new_grids), "added_by_stratum": dict(sorted(counts.items())),
            "added_by_shape_recipe": dict(sorted(shapes.items())), "rejected": dict(rejected),
            "overlap_exclusions": overlaps,
            "validation": "Exclude rows with source_fold equal to the held-out official fold from model training.",
        }
        (root / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
        root.rename(args.out)
        print(f"{report['total_train']} images, dont {len(additions)} nouveaux masques, dans {args.out}", flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--base", type=Path, default=Path("outputs/coco_augmented"))
    cli.add_argument("--data", type=Path, default=Path("release"))
    cli.add_argument("--coco", type=Path, default=Path(".downloads/coco/val2017"))
    cli.add_argument("--annotations", type=Path, default=Path(".downloads/coco/annotations_trainval2017.zip"))
    cli.add_argument("--folds", type=Path, default=Path("research/folds.json"))
    cli.add_argument("--out", type=Path, default=Path("outputs/coco_augmented_v2"))
    cli.add_argument("--device", default="cpu")
    cli.add_argument("--seed", type=int, default=20261122)
    cli.add_argument("--per-stratum", type=int, default=170, help="Added images per method and background pair")
    args = cli.parse_args()
    if args.per_stratum < 1:
        cli.error("--per-stratum must be positive")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    build(args)


if __name__ == "__main__":
    main()
