"""Build a train split with COCO val2017 covers and audit against the official splits."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import chain, cycle
import json
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw
import torch

from research.overlap import ImageIndex, PHASH_DISTANCE, pixel_hash
from research.synth import composite, expand_rect, is_rectangle, patch_rect
from wmloc.data import LocalisationDataset
from wmloc.utils import downsample_mask
from wmloc.watermarks import load_watermark


METHOD_BITS = {"pixelseal": 256, "trustmark": 100, "stegastamp": 100}


def read_json(path: Path):
    return json.loads(path.read_text())


def snapshot(directory: Path) -> str:
    result = sha256()
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        with path.open("rb") as stream:
            content_hash = sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                content_hash.update(block)
        result.update(path.relative_to(directory).as_posix().encode() + b"\0" + content_hash.digest())
    return result.hexdigest()


def image_path(split: Path, row: dict) -> Path:
    return split / "images" / f"{row['image_name']}.png"


def reference_images(split: Path, metadata: list[dict]):
    for row in metadata:
        with Image.open(image_path(split, row)) as image:
            yield (split.name, row["image_name"]), image


def cover_image(path: Path, seed: int) -> tuple[Image.Image, tuple[int, int, int, int], Image.Image]:
    """Return the source and a deterministic square crop with its coordinates on the source."""
    with Image.open(path) as source:
        image = source.convert("RGB")
    side = min(image.size)
    rng = np.random.default_rng([seed, int(path.stem)])
    x = int(rng.integers(image.width - side + 1))
    y = int(rng.integers(image.height - side + 1))
    crop = (x, y, x + side, y + side)
    return image, crop, image.crop(crop).resize((512, 512), Image.Resampling.BICUBIC)


def cover_views(image: Image.Image, cropped: Image.Image):
    side = min(image.size)
    x, y = (image.width - side) // 2, (image.height - side) // 2
    return image, cropped, image.crop((x, y, x + side, y + side))


def check_covers(paths, reference: ImageIndex, seed: int) -> tuple[list[Path], list[dict]]:
    retained = []
    excluded = []
    for index, path in enumerate(paths, start=1):
        image, _, cropped = cover_image(path, seed)
        hits = []
        for view, candidate in zip(("original", "training_crop", "center_crop"), cover_views(image, cropped)):
            hits.extend({"view": view, **hit} for hit in reference.matches(candidate))
        if hits:
            excluded.append({"source": path.name, "matches": hits})
        else:
            retained.append(path)
        if index % 500 == 0:
            print(f"Contrôle COCO {index}/{len(paths)}", flush=True)
    return retained, excluded


def official_overlap(train_dir: Path, train_rows: list[dict], reference: ImageIndex) -> list[dict]:
    found = []
    for label, image in reference_images(train_dir, train_rows):
        hits = [hit for hit in reference.matches(image) if hit["reference"][0] == "test"]
        if hits:
            found.append({"train": label[1], "matches": hits})
    return found


def coco_annotations(path: Path) -> dict:
    with ZipFile(path) as archive:
        with archive.open("annotations/instances_val2017.json") as stream:
            annotations = json.load(stream)["annotations"]
    polygons = defaultdict(list)
    for annotation in annotations:
        if not annotation["iscrowd"]:
            polygons[annotation["image_id"]].append(annotation)
    return polygons


def object_mask(annotations, crop, desired_area: int, rng) -> tuple[np.ndarray, int] | None:
    left, top, right, _ = crop
    side = right - left
    scale = 512 / side
    candidates = []
    for annotation in annotations:
        estimated_area = annotation["area"] * scale**2
        if not 0.55 * desired_area <= estimated_area <= 1.8 * desired_area:
            continue
        shape = Image.new("L", (512, 512))
        draw = ImageDraw.Draw(shape)
        for polygon in annotation["segmentation"]:
            vertices = [((x - left) * scale, (y - top) * scale)
                        for x, y in zip(polygon[::2], polygon[1::2])]
            if len(vertices) >= 3:
                draw.polygon(vertices, fill=1)
        mask = np.asarray(shape)
        area = int(mask.sum())
        if 0.72 * desired_area <= area <= 1.38 * desired_area and not is_rectangle(mask):
            candidates.append((mask, annotation["id"]))
    if candidates:
        return candidates[int(rng.integers(len(candidates)))]
    return None


def mask_from_disk(split: Path, name: str) -> np.ndarray:
    with Image.open(split / "masks" / f"{name}.png") as source:
        return (np.asarray(source) > 127).astype(np.uint8)


def schedule(covers: list[Path], train_rows: list[dict], rng) -> dict:
    strata = defaultdict(list)
    for row in train_rows:
        strata[row["method"], row["full_watermark"]].append(row)
    donors = {key: cycle(rng.permutation(rows)) for key, rows in strata.items()}
    jobs = defaultdict(list)
    for path, key in zip(rng.permutation(covers), cycle(sorted(strata))):
        jobs[key[0]].append((path, next(donors[key])))
    return jobs


def load_recipes(train_dir: Path, train_rows: list[dict]) -> dict:
    return {row["image_name"]: mask_from_disk(train_dir, row["image_name"]) for row in train_rows}


def embed_masked(watermark, method, clean, mask, full, bits, rng, source_name):
    """Apply the hackathon's patch or whole-image recipe to a binary presence mask."""
    rect = patch_rect(mask, full, "whole", rng, 2.0)
    y0, y1, x0, x1 = rect
    h, w = y1 - y0, x1 - x0
    if method == "trustmark" and max(h, w) > 2 * min(h, w):
        rect = expand_rect(mask, rng, 2.0)
    marked = composite(watermark, clean, mask, full, bits, rect)
    changed = np.any(marked != clean, axis=2)
    if not changed.any() or np.any(changed & (mask == 0)):
        raise ValueError(f"Unexpected watermark residual outside the intended composition for {source_name}")
    return Image.fromarray(marked), rect


def synthesise(watermark, method, path, donor, base_mask, annotations, seed):
    image_id = int(path.stem)
    _, crop, cover = cover_image(path, seed)
    clean = np.asarray(cover)
    rng = np.random.default_rng([seed, image_id, 1])
    full = donor["full_watermark"]
    mask = base_mask
    annotation = None
    if not full and not is_rectangle(mask):
        selected = object_mask(annotations.get(image_id, ()), crop, int(mask.sum()), rng)
        if selected is not None:
            mask, annotation = selected
    bits = rng.integers(0, 2, size=watermark.nbits, dtype=np.uint8)
    marked, rect = embed_masked(watermark, method, clean, mask, full, bits, rng, path.name)
    row = {
        "image_name": f"coco_val2017_{path.stem}", "method": method,
        "message": "".join(map(str, bits)), "full_watermark": full,
        "source_coco_id": image_id, "source_crop": crop, "embedding_rect": rect,
        "source_train_recipe": donor["image_name"], "source_coco_annotation": annotation,
    }
    return marked, mask, row


def copy_originals(rows: list[dict], source: Path, target: Path) -> list:
    for directory in ("images", "masks"):
        (target / directory).mkdir(parents=True)
        for row in rows:
            filename = f"{row['image_name']}.png"
            copyfile(source / directory / filename, target / directory / filename)
    with np.load(source / "masks_128.npz") as official:
        grids = dict(zip(official["names"].tolist(), official["masks"]))
    return [grids[row["image_name"]] for row in rows]


def verify_dataset(root: Path, original_rows: list[dict], folds: dict) -> None:
    dataset = LocalisationDataset(root, "train")
    names = [row["image_name"] for row in dataset.meta]
    if list(dataset.meta[:len(original_rows)]) != original_rows or len(names) != len(set(names)):
        raise ValueError("Original metadata changed or example names are duplicated")
    for index, row in enumerate(dataset.meta):
        name = row["image_name"]
        if len(row["message"]) != METHOD_BITS[row["method"]] or set(row["message"]) - {"0", "1"}:
            raise ValueError(f"Invalid message for {name}")
        if not np.array_equal(downsample_mask(mask_from_disk(dataset.dir, name)), dataset.m128[name]):
            raise ValueError(f"Invalid 128×128 grid for {name}")
        if index >= len(original_rows) and row["source_fold"] != folds[row["source_train_recipe"]]:
            raise ValueError(f"Invalid source fold for {name}")
    for method in METHOD_BITS:
        sample_index = next(i for i, row in enumerate(dataset.meta[len(original_rows):], len(original_rows))
                            if row["method"] == method)
        item = dataset[sample_index]
        if item["image"].shape != (512, 512, 3) or item["mask"].shape != (512, 512):
            raise ValueError(f"Invalid sample shape for {method}")


def generate(args) -> None:
    original = args.data / "train"
    test_dir = args.data / "test"
    train_rows = read_json(original / "metadata.json")
    test_rows = read_json(test_dir / "metadata.json")
    folds = read_json(args.folds)
    if not train_rows or not test_rows or {row["image_name"] for row in train_rows} != set(folds):
        raise ValueError("The official splits or the official validation folds are incomplete")
    if args.out.exists():
        raise FileExistsError(args.out)
    before = snapshot(test_dir)
    known = ImageIndex(chain(reference_images(original, train_rows), reference_images(test_dir, test_rows)))
    official_matches = official_overlap(original, train_rows, known)
    photos = sorted(args.coco.glob("*.jpg"))
    if args.limit is not None:
        photos = list(np.random.default_rng(args.seed).permutation(photos)[:args.limit])
    covers, source_matches = check_covers(photos, known, args.seed)
    print(f"COCO : {len(photos)} contrôlées, {len(source_matches)} écartées. "
          f"Candidats train officiel ↔ test : {len(official_matches)}.", flush=True)
    jobs = schedule(covers, train_rows, np.random.default_rng(args.seed))
    if set(jobs) != set(METHOD_BITS):
        raise ValueError("Select enough COCO covers to include all three watermark methods")
    annotations = coco_annotations(args.annotations)
    masks = load_recipes(original, train_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{args.out.name}-", dir=args.out.parent) as pending:
        root = Path(pending) / "dataset"
        split = root / "train"
        grids = copy_originals(train_rows, original, split)
        rows = train_rows.copy()
        generated_matches = []
        emitted_pixels = set()
        counts = Counter()
        completed = 0
        for method, group in jobs.items():
            watermark = load_watermark(method, device=args.device)
            for path, donor in group:
                name = donor["image_name"]
                image, mask, row = synthesise(watermark, method, path, donor, masks[name], annotations, args.seed)
                hits = known.matches(image)
                pixels = pixel_hash(image)
                if hits or pixels in emitted_pixels:
                    generated_matches.append({"source": path.name, "matches": hits,
                                              "exact_generated_duplicate": pixels in emitted_pixels})
                else:
                    emitted_pixels.add(pixels)
                    row["source_fold"] = folds[name]
                    image.save(image_path(split, row), compress_level=1)
                    Image.fromarray(mask * 255).save(split / "masks" / f"{row['image_name']}.png")
                    grids.append(downsample_mask(mask))
                    rows.append(row)
                    counts[f"{method}:{'marked' if row['full_watermark'] else 'clean'}"] += 1
                completed += 1
                if completed % 50 == 0 or completed == len(covers):
                    print(f"::progress {completed}/{len(covers)} génération {method}", flush=True)
            del watermark
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
        (split / "metadata.json").write_text(json.dumps(rows, indent=2) + "\n")
        np.savez_compressed(split / "masks_128.npz", names=[row["image_name"] for row in rows], masks=np.stack(grids))
        verify_dataset(root, train_rows, folds)
        after = snapshot(test_dir)
        if before != after:
            raise ValueError("The official test changed during generation")
        report = {
            "source": "COCO val2017", "seed": args.seed, "original_train": len(train_rows),
            "official_test": len(test_rows), "official_test_snapshot_sha256": after,
            "coco_checked": len(photos), "phash_threshold": PHASH_DISTANCE,
            "official_train_vs_test_candidates": official_matches,
            "source_exclusions": source_matches, "generated_exclusions": generated_matches,
            "added": len(rows) - len(train_rows), "total_train": len(rows),
            "added_by_method_and_background": dict(sorted(counts.items())),
            "added_with_coco_object_mask": sum(row.get("source_coco_annotation") is not None for row in rows),
            "validation": "Use only original images for evaluation. Exclude augmented rows whose source_fold "
                          "equals the held-out fold when training for that fold.",
        }
        (root / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
        root.rename(args.out)
        print(f"{report['total_train']} images dans {args.out}, dont {report['added']} ajouts. "
              f"Rapport : {args.out / 'audit.json'}", flush=True)


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data", type=Path, default=Path("release"))
    cli.add_argument("--coco", type=Path, default=Path(".downloads/coco/val2017"))
    cli.add_argument("--annotations", type=Path, default=Path(".downloads/coco/annotations_trainval2017.zip"))
    cli.add_argument("--folds", type=Path, default=Path("research/folds.json"))
    cli.add_argument("--out", type=Path, default=Path("outputs/coco_augmented"))
    cli.add_argument("--device", default="cuda")
    cli.add_argument("--seed", type=int, default=42)
    cli.add_argument("--limit", type=int, help="Number of COCO source photos to audit and process for a smoke test.")
    generate(cli.parse_args())


if __name__ == "__main__":
    main()
