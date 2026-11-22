"""Run fixed-step watermark data and message ablations on two official training folds."""
from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
import subprocess
import sys

import numpy as np

from research.coco_fresh_pair import BASE, VAL_COCO
from research.folds import load as load_folds
from research.metrics import miou_batch
from research.trustmark_holdout import digest, read_json, write_json


FOLDS = (1, 2)
ARMS = {"pixelseal": ("reused", "native", "fresh", "message"), "trustmark": ("reused", "message")}
STEPS = 1000
TRAIN = {"data": "release", "steps": STEPS, "eval_every": STEPS, "pool": 1000, "batch": 16,
         "real_per_batch": 4, "extra_per_batch": 6, "post_augment_prob": 0.0, "seed": 0,
         "no_compile": True}
LEARNING_RATE = {"pixelseal": 1e-4, "trustmark": 1e-4}
DATA_ROOT = Path("outputs/protocol_v3_datasets")
RUN_ROOT = Path("outputs/protocol_v3")
CODE = (*sorted(Path("research").glob("*.py")), *sorted(Path("wmloc").glob("*.py")))
COMMON_READS = (Path("release"), Path("release/train"), Path(".downloads/coco"), Path("models"))


def queued(label, command, *, reads, writes, gpu=True):
    argv = [sys.executable, "-m", "research.gpuq", "run", "--label", label]
    if not gpu:
        argv.append("--no-gpu")
    for option, paths in (("--reads", reads), ("--writes", writes)):
        for path in paths:
            argv.extend((option, str(path)))
    subprocess.run([*argv, "--", sys.executable, "-u", "-m", *map(str, command)], check=True)


def arm_dir(data_root, arm):
    return data_root / ("fresh" if arm == "fresh" else "reused")


def run_dir(out, method, fold, arm):
    return out / method / f"fold{fold}" / arm


def training_settings(data_root, method, fold, arm):
    prefix = "wmmsg:" if arm == "message" else "wm:"
    return {**TRAIN, "method": method, "val_fold": fold, "encoder": prefix + method,
            "lr": LEARNING_RATE[method], "extra_dir": str(arm_dir(data_root, arm)),
            "native_contours": arm == "native"}


def data_checks(data_root, count):
    audit = read_json(data_root / "audit.json")
    base = audit["base_examples"]
    reused, fresh = [read_json(data_root / kind / "train/metadata.json") for kind in ("reused", "fresh")]
    if audit["paired_additions"] != count or len(reused) != len(fresh) or len(fresh) != base + count:
        raise ValueError("Unexpected size of either paired dataset")
    if reused[:base] != fresh[:base] or reused[:base] != read_json(BASE / "train/metadata.json"):
        raise ValueError("Both arms must share the original V1 dataset")
    fold_of = load_folds()
    keep = {method: {str(fold): 0 for fold in FOLDS} for method in ARMS}
    for left, right in zip(reused, fresh, strict=True):
        if "source_fold" not in left:
            continue
        truth = fold_of[left["source_train_recipe"]]
        if left["source_fold"] != truth or right["source_fold"] != truth:
            raise ValueError("A static mask has invalid fold provenance")
        method = left["method"]
        if method in keep:
            for fold in FOLDS:
                keep[method][str(fold)] += int(truth != fold)
    new_ids = {item["source_coco_id"] for item in fresh[base:]}
    old_ids = {item["source_coco_id"] for item in reused if "source_coco_id" in item}
    if len(new_ids) != count or not new_ids.isdisjoint(old_ids):
        raise ValueError("Fresh photos must be unique and absent from the reused arm")
    provenance = {row["recipe"]: row for row in audit["provenance"]}
    keys = (*audit["paired_fields"], "embedding_rect")
    for left, right in zip(reused[base:], fresh[base:], strict=True):
        if any(left[key] != right[key] for key in keys):
            raise ValueError("Matched examples disagree on their encoding recipe")
        sources = provenance[left["matched_recipe_id"]]
        for kind, row, covers in (("reused", left, VAL_COCO), ("fresh", right, data_root / "fresh_covers")):
            source_id = row["source_coco_id"]
            if source_id != sources[f"{kind}_id"] or "parent_image_name" in row:
                raise ValueError("Static metadata disagrees with the source manifest")
            if digest(covers / f"{source_id:012d}.jpg") != sources[f"{kind}_jpeg_sha256"]:
                raise ValueError(f"The {kind} cover JPEG changed after augmentation")
        mask_left = data_root / "reused/train/masks" / f"{left['image_name']}.png"
        mask_right = data_root / "fresh/train/masks" / f"{right['image_name']}.png"
        if digest(mask_left) != digest(mask_right):
            raise ValueError("Matched examples disagree on their 512-pixel mask")
    with np.load(data_root / "reused/train/masks_128.npz") as left, np.load(data_root / "fresh/train/masks_128.npz") as right:
        if not np.array_equal(left["masks"], right["masks"]):
            raise ValueError("Matched splits disagree on 128 grid masks")
    return keep, new_ids


class InvertedMessages:
    """Expose the same evaluation pixels with every expected bit flipped."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        row = self.dataset[index]
        return {**row, "message": 1 - np.fromiter(row["message"], dtype=np.uint8)}


def cell(out, data_root, method, fold, arm):
    import torch
    from research.oracle_bench import breakdown, checkpoint_logits
    from wmloc.data import LocalisationDataset

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    dest = run_dir(out, method, fold, arm)
    settings = training_settings(data_root, method, fold, arm)
    weights, log_path = dest / "last.pt", dest / "log.json"
    if not (weights.is_file() and log_path.is_file()):
        argv = [sys.executable, "-u", "-m", "research.train_seg", "--out", str(dest)]
        for key, value in settings.items():
            if value is False:
                continue
            argv.append("--" + key.replace("_", "-"))
            if value is not True:
                argv.append(str(value))
        subprocess.run(argv, check=True)
    log = read_json(log_path)
    if any(log["args"].get(key) != value for key, value in settings.items()):
        raise ValueError(f"Unexpected arguments in {log_path}")
    if [rec["step"] for rec in log["log"]] != [STEPS]:
        raise ValueError("Only the fixed last step belongs to the comparison")
    dataset = LocalisationDataset("release", "train", methods=[method])
    fold_of = load_folds()
    indices = [i for i, row in enumerate(dataset.meta) if fold_of[row["image_name"]] == fold]
    if len(indices) != 34 or len(dataset) - len(indices) != 136:
        raise ValueError(f"Unexpected validation size for {method} fold {fold}")
    records = [dataset.meta[i] for i in indices]
    names = [row["image_name"] for row in records]
    masks = np.stack([dataset.m128[name] for name in names]).astype(bool)
    full = np.array([row["full_watermark"] for row in records], dtype=bool)
    predictions = checkpoint_logits(dataset, indices, weights, settings["encoder"]) > 0
    summary = breakdown(predictions, masks, full)
    # Inference batches differ from training-time validation, which can change low precision ties at zero.
    if abs(summary["miou"] - log["log"][0]["real_val"]["miou"]) > .003:
        raise ValueError(f"Checkpoint failed to reproduce validation scores in {dest}")
    result = {"args": settings, "checkpoint_sha256": digest(weights), "scores": summary,
              "per_image_miou": dict(zip(names, miou_batch(predictions, masks).tolist(), strict=True)),
              "online_coco_ids": log["online_coco_ids"], "peak_cuda_memory": log["peak_cuda_memory"]}
    if arm == "message":
        inverted = checkpoint_logits(InvertedMessages(dataset), indices, weights, settings["encoder"]) > 0
        result["inverted_messages"] = breakdown(inverted, masks, full)
    write_json(dest / "evaluation.json", result)


def comparisons(out, results):
    changes = {}
    for method, arms in ARMS.items():
        for arm in arms[1:]:
            by_fold = {}
            for fold in FOLDS:
                baseline, candidate = results.get((method, fold, "reused")), results.get((method, fold, arm))
                if baseline is None or candidate is None:
                    continue
                old, new = baseline["per_image_miou"], candidate["per_image_miou"]
                if old.keys() != new.keys():
                    raise ValueError("Comparison changed official validation images")
                by_fold[str(fold)] = {"baseline": baseline["scores"], "candidate": candidate["scores"],
                                      "per_image_delta_miou": {name: new[name] - old[name] for name in old}}
                if "inverted_messages" in candidate:
                    by_fold[str(fold)]["candidate_inverted_messages"] = candidate["inverted_messages"]
            if by_fold:
                deltas = [list(row["per_image_delta_miou"].values()) for row in by_fold.values()]
                change = {"folds": by_fold, "pooled_n": sum(map(len, deltas)),
                          "pooled_delta_miou": float(np.concatenate(deltas).mean()),
                          "positive_on_both_folds": len(deltas) == len(FOLDS) and all(np.mean(d) > 0 for d in deltas)}
                changes[f"{method}:{arm}"] = change
    write_json(out / "report.json", {"completed_cells": len(results), "comparisons": changes})


def register(out, data_root):
    out.mkdir(parents=True, exist_ok=True)
    counts, fresh_ids = data_checks(data_root, 1020)
    paths = [*CODE, Path("research/folds.json"), Path("release/train/metadata.json"),
             Path("release/train/masks_128.npz"), data_root / "audit.json"]
    for arm in ("reused", "fresh"):
        paths.extend(arm_dir(data_root, arm) / "train" / name for name in ("metadata.json", "masks_128.npz"))
    protocol = {"folds": list(FOLDS), "arms": {key: list(value) for key, value in ARMS.items()},
                "training": TRAIN, "learning_rates": LEARNING_RATE, "checkpoint": "last.pt",
                "static_training_counts_in_both_arms": counts, "fresh_coco_ids": sorted(fresh_ids),
                "criterion": "Positive paired mIoU in both held-out folds at the fixed last step.",
                "inputs_sha256": {str(path): digest(path) for path in paths}}
    protocol_file = out / "protocol.json"
    if protocol_file.exists() and read_json(protocol_file) != protocol:
        raise ValueError("This output uses a different frozen protocol")
    write_json(protocol_file, protocol)


def run(out, data_root):
    queued("v3-register", ["research.protocol_v3", "--out", out, "--data-root", data_root, "--register"],
           reads=(*COMMON_READS, BASE, data_root, arm_dir(data_root, "reused"), arm_dir(data_root, "fresh")),
           writes=(out / "protocol.json",), gpu=False)
    protocol = read_json(out / "protocol.json")
    fresh_ids = set(protocol["fresh_coco_ids"])
    results = {}
    online_ids = None
    pairs = [(method, fold, arm) for fold, (method, arms) in product(FOLDS, ARMS.items()) for arm in arms]
    for method, fold, arm in pairs:
        if any(digest(path) != protocol["inputs_sha256"][str(path)] for path in CODE):
            raise ValueError("Training code changed after the protocol was fixed")
        dest = run_dir(out, method, fold, arm)
        evaluation = dest / "evaluation.json"
        if not evaluation.is_file():
            queued(f"v3-{method}-f{fold}-{arm}", ["research.protocol_v3", "--out", out, "--data-root", data_root,
                   "--cell", method, fold, arm], reads=(*COMMON_READS, data_root, arm_dir(data_root, arm)), writes=(dest,))
        result = read_json(evaluation)
        if (result["args"] != training_settings(data_root, method, fold, arm)
                or result["checkpoint_sha256"] != digest(dest / "last.pt")):
            raise ValueError(f"Existing evaluation does not match {dest}")
        ids = result["online_coco_ids"]
        if len(set(ids)) != TRAIN["pool"] or not fresh_ids.isdisjoint(ids):
            raise ValueError("An online pool overlaps the fresh arm or has the wrong size")
        if online_ids is not None and ids != online_ids:
            raise ValueError("The comparisons must share the same online photo pool")
        online_ids = ids
        results[method, fold, arm] = result
        comparisons(out, results)
        print(f"::progress {len(results)}/{len(pairs)} {method} pli {fold} {arm} mIoU {result['scores']['miou']:.4f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=RUN_ROOT)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT, help="paired output from research.coco_fresh_pair")
    parser.add_argument("--register", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--cell", nargs=3, metavar=("METHOD", "FOLD", "ARM"))
    args = parser.parse_args()
    if args.cell:
        method, fold, arm = args.cell
        if int(fold) not in FOLDS or arm not in ARMS.get(method, ()):
            parser.error("unknown comparison cell")
        cell(args.out, args.data_root, method, int(fold), arm)
    elif args.register:
        register(args.out, args.data_root)
    else:
        run(args.out, args.data_root)


if __name__ == "__main__":
    main()
