"""Build an offline candidate that changes only the PixelSeal masks of sub01."""
from __future__ import annotations

import argparse
from hashlib import file_digest
import json
from pathlib import Path

import numpy as np
import torch

from research import submission
from research.coco_augment import read_json
from research.decoders import Decoder, signed
from research.metrics import zscore
from research.oracle_bench import best_rank_rule, decoder_maps, prior_table, rank_minorities
from research.polarity_fill import fit_threshold
from research.zero_shot import maps_for_batch
from wmloc.data import LocalisationDataset


def fit_pixel_rank(map_path: Path):
    with np.load(map_path) as archive:
        names = archive["pixelseal/names"].tolist()
        maps = decoder_maps(archive, "pixelseal", slice(None))
        bit_accuracy = archive["pixelseal/global_bit_acc"]
    train = LocalisationDataset("release", "train", methods=["pixelseal"])
    backgrounds = {row["image_name"]: row["full_watermark"] for row in train.meta}
    if len(set(names)) != len(train) or set(names) != set(backgrounds):
        raise ValueError("Decoder maps must cover exactly the official PixelSeal training images")
    true_background = np.array([backgrounds[name] for name in names], dtype=bool)
    masks = np.stack([train.m128[name] for name in names]).astype(bool)
    threshold = float(fit_threshold(bit_accuracy, true_background))
    full = bit_accuracy > threshold
    table = prior_table(maps, masks)
    return threshold, {background: best_rank_rule(maps, table, full == background, background)
                       for background in (False, True)}


@torch.no_grad()
def predict_pixel_batch(decoder, images, messages, threshold, rules):
    expected = signed(messages)
    observed = decoder.logits(images)
    full = (((observed > 0) == (expected > 0)).float().mean(dim=1) > threshold).cpu().numpy()
    output = np.broadcast_to(full[:, None, None], (len(full), 128, 128)).copy()
    for background, rule in rules.items():
        indices = np.flatnonzero(full == background)
        if not len(indices) or rule["fraction"] == 0:
            continue
        name = rule["map"]
        requested = {"native_work", "native_512"} if name == "native_mean" else {name}
        batch_indices = torch.as_tensor(indices, device=images.device)
        maps, _ = maps_for_batch(decoder, images[batch_indices], expected[batch_indices], requested=requested)
        # Match the numeric precision of the cached maps used for rule selection.
        maps = {key: value.astype(np.float16).astype(np.float32) for key, value in maps.items()}
        score = (zscore(maps["native_work"]) + zscore(maps["native_512"])) / 2 if name == "native_mean" else maps[name]
        if not np.isfinite(score).all():
            raise ValueError("Nonfinite PixelSeal map")
        for i, grid in zip(indices, score, strict=True):
            output[i] = rank_minorities(grid, background, rule["fraction"])
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps", type=Path, default=Path("outputs/zero_shot/maps.npz"))
    parser.add_argument("--out", type=Path, default=Path("outputs/oracle_v1/sub02_pixelseal_rank.pt"))
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(2)
    sub01 = read_json(Path("research/submissions/sub01_seg_v1.json"))
    source = Path(sub01["file"]["path"])
    with source.open("rb") as stream:
        if file_digest(stream, "sha256").hexdigest() != sub01["file"]["sha256"]:
            raise ValueError("The local sub01 file differs from its recorded SHA-256")
    base = torch.load(source, map_location="cpu", weights_only=True)
    names = base["image_names"]
    masks = base["masks"].numpy().copy()
    positions = {name: i for i, name in enumerate(names)}
    threshold, rules = fit_pixel_rank(args.maps)
    decoder = Decoder("pixelseal")
    test = LocalisationDataset("release", "test", methods=["pixelseal"])
    changed = 0
    for start in range(0, len(test), args.batch_size):
        items = [test[i] for i in range(start, min(start + args.batch_size, len(test)))]
        images = torch.from_numpy(np.stack([item["image"] for item in items])).permute(0, 3, 1, 2).float().cuda() / 255
        messages = torch.from_numpy(np.stack([item["message"] for item in items])).cuda()
        predictions = predict_pixel_batch(decoder, images, messages, threshold, rules)
        index = [positions[item["image_name"]] for item in items]
        changed += int((masks[index] != predictions).any(axis=(1, 2)).sum())
        masks[index] = predictions
        print(f"::progress {start + len(items)}/{len(test)} PixelSeal", flush=True)
    audit = submission.write(names, masks, args.out, "release")
    with args.out.open("rb") as stream:
        audit["sha256"] = file_digest(stream, "sha256").hexdigest()
    report = {"source_sha256": sub01["file"]["sha256"], "train_images_for_calibration": 170,
              "polarity_threshold": threshold, "clean": rules[False], "marked": rules[True],
              "changed_pixel_masks": changed, "file": audit}
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
