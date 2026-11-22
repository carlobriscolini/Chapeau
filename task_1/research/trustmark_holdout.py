"""Confirm the fixed TrustMark area rule on two separately held-out official folds."""
from __future__ import annotations

import argparse
from hashlib import file_digest
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from research.folds import load as load_folds
from research.metrics import miou_batch
from research.oracle_bench import breakdown, checkpoint_logits, constrain_predicted_area
from research.polarity_fill import fit_threshold
from wmloc.data import LocalisationDataset


FOLDS = (1, 2)
AREA_QUANTILES = (.10, .95)
EXTRA = Path("outputs/coco_augmented_v2")
MAPS = Path("outputs/zero_shot/maps.npz")
TRAIN_SETTINGS = {
    "method": "trustmark", "data": "release", "encoder": "resnet50", "steps": 1000,
    "eval_every": 1000, "pool": 1000, "batch": 16, "real_per_batch": 4,
    "extra_dir": str(EXTRA), "extra_per_batch": 6, "post_augment_prob": 0., "seed": 0, "no_compile": True,
}


def read_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, item):
    pending = path.with_suffix(".pending")
    pending.write_text(json.dumps(item, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def fit_area_rule(masks, backgrounds, bit_accuracy, calibration):
    area = (masks != backgrounds[:, None, None]).mean(axis=(1, 2))
    threshold = float(fit_threshold(bit_accuracy[calibration], backgrounds[calibration]))
    bounds = {full: np.quantile(area[calibration & (backgrounds == full)], AREA_QUANTILES).tolist()
              for full in (False, True)}
    return threshold, bounds


def train_checkpoint(root: Path, fold: int) -> tuple[Path, dict]:
    run = root / f"fold{fold}"
    log_path = run / "log.json"
    weights = run / "last.pt"
    if not log_path.is_file() or not weights.is_file():
        argv = [sys.executable, "-u", "-m", "research.train_seg", "--out", str(run), "--val-fold", str(fold)]
        for key, value in TRAIN_SETTINGS.items():
            argv.append("--" + key.replace("_", "-"))
            if key != "no_compile":
                argv.append(str(value))
        print(f"TrustMark pli {fold} · {TRAIN_SETTINGS['steps']} pas fixés", flush=True)
        subprocess.run(argv, check=True)
    log = read_json(log_path)
    if any(log["args"].get(key) != value for key, value in TRAIN_SETTINGS.items()) or log["args"]["val_fold"] != fold:
        raise ValueError(f"Checkpoint configuration does not match the protocol for fold {fold}")
    if [row["step"] for row in log["log"]] != [TRAIN_SETTINGS["steps"]]:
        raise ValueError(f"Expected a single validation at the fixed last step of fold {fold}")
    return weights, log["log"][0]["real_val"]


def run(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    protocol_file = root / "protocol.json"
    code_paths = [Path("research/train_seg.py"), Path("research/gpu_synth.py"), Path("research/gpu_embed.py"),
                  Path("research/oracle_bench.py"), Path("research/polarity_fill.py"), Path(__file__)]
    provenance_paths = [*code_paths, Path("research/folds.json"),
                        Path("release/train/metadata.json"), Path("release/train/masks_128.npz"),
                        EXTRA / "train/metadata.json", EXTRA / "train/masks_128.npz", MAPS]
    protocol = {"folds": list(FOLDS), "area_quantiles": list(AREA_QUANTILES), "training": TRAIN_SETTINGS,
                "checkpoint": "last.pt", "criterion": "Positive paired mIoU change on both held-out folds.",
                "calibration": "Per-fold polarity and area bounds fitted exclusively on the other four folds.",
                "inputs_sha256": {str(path): digest(path) for path in provenance_paths}}
    if protocol_file.is_file() and read_json(protocol_file) != protocol:
        raise ValueError("This folder contains a different protocol; use a new output folder")
    write_json(protocol_file, protocol)

    dataset = LocalisationDataset("release", "train", methods=["trustmark"])
    names = [row["image_name"] for row in dataset.meta]
    fold_of = load_folds()
    folds = np.array([fold_of[name] for name in names])
    masks = np.stack([dataset.m128[name] for name in names]).astype(bool)
    backgrounds = np.array([row["full_watermark"] for row in dataset.meta], dtype=bool)
    with np.load(MAPS) as maps:
        positions = {name: i for i, name in enumerate(maps["trustmark/names"].tolist())}
        bit_acc = maps["trustmark/global_bit_acc"][[positions[name] for name in names]]
    extra = [row for row in read_json(EXTRA / "train/metadata.json")
             if row["method"] == "trustmark" and "source_fold" in row]
    if not extra or any(row["source_fold"] != fold_of[row["source_train_recipe"]] for row in extra):
        raise ValueError("The augmented split has missing or inconsistent fold provenance")

    all_deltas = []
    results = {}
    for fold in FOLDS:
        evaluation = folds == fold
        if evaluation.sum() != 34 or (~evaluation).sum() != 136:
            raise ValueError(f"Unexpected TrustMark split size for fold {fold}")
        threshold, bounds = fit_area_rule(masks, backgrounds, bit_acc, ~evaluation)
        if any(digest(path) != protocol["inputs_sha256"][str(path)] for path in code_paths):
            raise ValueError("Training code changed after the protocol was fixed")
        weights, logged = train_checkpoint(root, fold)
        logits = checkpoint_logits(dataset, np.flatnonzero(evaluation), weights, TRAIN_SETTINGS["encoder"])
        torch.cuda.empty_cache()
        targets, truth_background = masks[evaluation], backgrounds[evaluation]
        predicted_background = bit_acc[evaluation] > threshold
        baseline = logits > 0
        adjusted = constrain_predicted_area(logits, predicted_background, bounds, floor=True, cap=True)
        raw = breakdown(baseline, targets, truth_background)
        if abs(raw["miou"] - logged["miou"]) > .003:
            raise ValueError(f"Last checkpoint did not reproduce the baseline mIoU of fold {fold}")
        deltas = miou_batch(adjusted, targets) - miou_batch(baseline, targets)
        val_names = np.array(names)[evaluation]
        results[str(fold)] = {
            "checkpoint_path": str(weights), "checkpoint_sha256": digest(weights),
            "original_training_count": int((~evaluation).sum()), "original_validation_count": int(evaluation.sum()),
            "coco_training_count": sum(row["source_fold"] != fold for row in extra),
            "coco_excluded_count": sum(row["source_fold"] == fold for row in extra),
            "polarity_threshold": threshold, "area_bounds": {str(key): val for key, val in bounds.items()},
            "baseline": raw, "area_rule": breakdown(adjusted, targets, truth_background),
            "paired_delta_miou": float(deltas.mean()),
            "per_image_delta_miou": dict(zip(val_names.tolist(), deltas.tolist(), strict=True)),
        }
        all_deltas.append(deltas)
        print(f"TrustMark pli {fold}: Δ mIoU {deltas.mean():+.6f}", flush=True)
    result = {"protocol": str(protocol_file), "folds": results, "pooled_n": sum(len(d) for d in all_deltas),
              "pooled_paired_delta_miou": float(np.concatenate(all_deltas).mean()),
              "positive_on_both_folds": all(d.mean() > 0 for d in all_deltas)}
    write_json(root / "report.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "folds"}), flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--out", type=Path, default=Path("outputs/trustmark_area_holdout"))
    args = cli.parse_args()
    torch.set_num_threads(2)
    run(args.out)


if __name__ == "__main__":
    main()
