"""Measure mask-selection headroom and test train-calibrated decoder maps on the official folds."""
from __future__ import annotations

import argparse
from hashlib import file_digest
import json
from pathlib import Path

import numpy as np
import torch

from research.coco_augment import read_json
from research.folds import load as load_folds
from research.fuse import fuse
from research.metrics import miou_batch, oracle_fraction_miou, zscore
from research.polarity_fill import fit_threshold
from research.predict import predict_items
from research.train_seg import build_model
from wmloc.data import LocalisationDataset


METHODS = ("pixelseal", "trustmark", "stegastamp")
REGION_FRACTIONS = (0., .01, .02, .04, .06, .09, .13, .18, .25, .35)
MAPS = ("native_work", "native_512", "occlusion", "hp_energy")


def miou_from_counts(tp, fp, positives, total):
    tn = total - positives - fp
    iou1_union = positives + fp
    iou0_union = total - tp
    iou1 = np.divide(tp, iou1_union, out=np.ones_like(tp, dtype=float), where=iou1_union != 0)
    iou0 = np.divide(tn, iou0_union, out=np.ones_like(tp, dtype=float), where=iou0_union != 0)
    return (iou1 + iou0) / 2


def best_threshold(score, target):
    """Exact per-image mIoU oracle, including both constant masks and tied scores."""
    values = score.ravel()
    order = np.argsort(-values, kind="stable")
    positives = target.ravel()[order].astype(bool)
    ordered = values[order]
    eligible = np.flatnonzero(np.r_[ordered[:-1] != ordered[1:], True]) + 1
    sizes = np.r_[0, eligible]
    tp = np.r_[0, np.cumsum(positives)[eligible - 1]]
    return float(miou_from_counts(tp, sizes - tp, positives.sum(), positives.size).max())


def rank_minorities(score, full, fraction):
    result = np.full(score.size, full, dtype=bool)
    size = round(score.size * fraction)
    if size:
        order = np.argsort(score.ravel(), kind="stable")
        result[order[:size] if full else order[-size:]] = not full
    return result.reshape(score.shape)


def constrain_predicted_area(logits, backgrounds, bounds, *, floor, cap):
    predictions = logits > 0
    for i, full in enumerate(backgrounds):
        low, high = bounds[bool(full)]
        area = (predictions[i] != full).mean()
        if floor and area < low:
            predictions[i] = rank_minorities(logits[i], full, low)
        elif cap and area > high:
            predictions[i] = rank_minorities(logits[i], full, high)
    return predictions


def decoder_maps(archive, method, indices):
    maps = {name: archive[f"{method}/{name}"][indices].astype(np.float32)
            for name in MAPS if f"{method}/{name}" in archive.files}
    if "native_work" in maps:
        maps["native_mean"] = (zscore(maps["native_work"]) + zscore(maps["native_512"])) / 2
    if any(not np.isfinite(values).all() for values in maps.values()):
        raise ValueError(f"Nonfinite local decoder scores for {method}")
    return maps


def prior_table(maps, masks):
    """mIoU per image for each map, area and hypothesised background, precomputed once."""
    table = {}
    n, height, width = masks.shape
    total = height * width
    target = masks.reshape(n, total)
    for name, values in maps.items():
        order = np.argsort(values.reshape(n, total), axis=1, kind="stable")
        sorted_target = np.take_along_axis(target, order, axis=1)
        for full in (False, True):
            minor = ~sorted_target if full else sorted_target[:, ::-1]
            cumulative = minor.cumsum(axis=1)
            positives = cumulative[:, -1]
            for fraction in REGION_FRACTIONS:
                k = round(total * fraction)
                tp = cumulative[:, k - 1] if k else np.zeros(n, dtype=int)
                table[name, fraction, full] = miou_from_counts(tp, k - tp, positives, total)
    return table


def best_rank_rule(maps, table, calibration, full):
    candidates = [(name, fraction, full) for name in maps for fraction in REGION_FRACTIONS]
    name, fraction, _ = max(candidates, key=lambda key: float(table[key][calibration].mean()))
    return {"map": name, "fraction": fraction}


def calibrate_maps(maps, masks, true_bg, bit_acc, folds):
    """Select maps and areas outside each scored fold, using image-level mIoU."""
    table = prior_table(maps, masks)
    rank_predictions = np.empty_like(masks)
    estimated_bg = np.empty_like(true_bg)
    settings = {}
    for fold in sorted(set(folds.tolist())):
        chosen = folds == fold
        threshold = fit_threshold(bit_acc[~chosen], true_bg[~chosen])
        estimate = bit_acc > threshold
        estimated_bg[chosen] = estimate[chosen]
        settings[fold] = {"polarity_threshold": float(threshold)}
        for full in (False, True):
            cal = (~chosen) & (estimate == full)
            if not cal.any():
                raise ValueError(f"Empty calibration partition for fold {fold} and background {full}")
            rule = best_rank_rule(maps, table, cal, full)
            settings[fold]["marked" if full else "clean"] = rule
            for i in np.flatnonzero(chosen & (estimate == full)):
                rank_predictions[i] = rank_minorities(maps[rule["map"]][i], full, rule["fraction"])
    return rank_predictions, estimated_bg, settings


def breakdown(predictions, masks, full):
    scores = miou_batch(predictions, masks)
    minor_gt = masks != full[:, None, None]
    minor_pred = predictions != full[:, None, None]
    intersect = (minor_gt & minor_pred).sum(axis=(1, 2))
    union = (minor_gt | minor_pred).sum(axis=(1, 2))
    local_iou = np.divide(intersect, union, out=np.ones(len(scores)), where=union != 0)
    small = minor_gt.mean(axis=(1, 2)) < .05
    return {"n": len(scores), "miou": float(scores.mean()), "minority_iou": float(local_iou.mean()),
            "small_n": int(small.sum()), "small_minority_iou": float(local_iou[small].mean()),
            "small_no_overlap": int((small & (intersect == 0)).sum())}


def checkpoint_logits(dataset, indices, checkpoint, encoder):
    model = build_model(encoder, pretrained=False).cuda().to(memory_format=torch.channels_last)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    outputs = []
    for start in range(0, len(indices), 4):
        items = [dataset[i] for i in indices[start:start + 4]]
        outputs.append(predict_items(model, items, bs=4).cpu().numpy())
    return np.concatenate(outputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, default=Path("outputs/seg_v1"))
    parser.add_argument("--maps", type=Path, default=Path("outputs/zero_shot/maps.npz"))
    parser.add_argument("--out", type=Path, default=Path("outputs/oracle_v1"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.out.mkdir(parents=True, exist_ok=True)
    fold_by_name = load_folds()
    submitted = read_json(Path("research/submissions/sub01_seg_v1.json"))
    previous_path = args.out / "report.json"
    previous = read_json(previous_path)["methods"] if previous_path.is_file() else {}
    report = {"methods": {}, "limitations": "Checkpoint selection used the reported validation fold; official test excluded."}
    with np.load(args.maps) as archive:
        for method in METHODS:
            ds = LocalisationDataset("release", "train", methods=[method])
            names = [r["image_name"] for r in ds.meta]
            indices_by_name = {name: i for i, name in enumerate(archive[f"{method}/names"].tolist())}
            order = [indices_by_name[name] for name in names]
            assert len(order) == 170 and len(set(order)) == len(order)
            maps = decoder_maps(archive, method, order)
            bit_acc = archive[f"{method}/global_bit_acc"][order]
            masks = np.stack([ds.m128[name] for name in names]).astype(bool)
            full = np.array([r["full_watermark"] for r in ds.meta], dtype=bool)
            folds = np.array([fold_by_name[name] for name in names])
            rank_maps, bg, settings = calibrate_maps(maps, masks, full, bit_acc, folds)
            checkpoint = args.checkpoints / method / "best.pt"
            with checkpoint.open("rb") as stream:
                sha = file_digest(stream, "sha256").hexdigest()
            config = read_json(checkpoint.with_name("log.json"))["args"]
            fold = config["val_fold"]
            validation = folds == fold
            val_names = np.array(names)[validation]
            cache = args.out / f"{method}.npz"
            logits = None
            if cache.is_file() and previous.get(method, {}).get("checkpoint_sha256") == sha:
                with np.load(cache) as stored:
                    if np.array_equal(stored["names"], val_names):
                        logits = stored["logits"]
            if logits is None:
                logits = checkpoint_logits(ds, np.flatnonzero(validation), checkpoint, config["encoder"])
                torch.cuda.empty_cache()
            targets, backgrounds, predicted_backgrounds = masks[validation], full[validation], bg[validation]
            score_maps = rank_maps[validation]
            candidates = {strategy: fuse(logits, predicted_backgrounds, strategy)
                          for strategy in ("model", "fill", "gate", "bias_0.5", "bias_1.0", "bias_2.0", "bias_4.0")}
            candidates["decoder_rank"] = score_maps
            minority_areas = (masks != full[:, None, None]).mean(axis=(1, 2))
            bounds = {background: np.quantile(minority_areas[(~validation) & (full == background)], [.1, .95]).tolist()
                      for background in (False, True)}
            for name, floor, cap in (("area_floor", True, False), ("area_cap", False, True), ("area_bounds", True, True)):
                candidates[name] = constrain_predicted_area(logits, predicted_backgrounds, bounds, floor=floor, cap=cap)
            primary = submitted["methods"][method]
            candidates["submitted_strategy"] = candidates["fill" if primary["source"] == "polarity_fill" else "model"]
            if "sha256" in primary and primary["sha256"] != sha:
                raise ValueError(f"The {method} checkpoint differs from the recorded submission")
            by_strategy = {key: breakdown(value, targets, backgrounds) for key, value in candidates.items()}
            if abs(by_strategy["submitted_strategy"]["miou"] - primary["fold0_miou"]) > .002:
                raise ValueError(f"Could not reproduce the submitted {method} validation score")
            family = {key: miou_batch(value, targets) for key, value in candidates.items()}
            thresholds = np.array([best_threshold(x, y) for x, y in zip(logits, targets)])
            upper = {"best_of_model_or_fill": float(np.maximum(family["model"], family["fill"]).mean()),
                     "best_of_candidate_family": float(np.max(list(family.values()), axis=0).mean()),
                     "true_area": oracle_fraction_miou(logits, targets),
                     "best_threshold_per_image": float(thresholds.mean())}
            small = minority_areas[validation] < .05
            report["methods"][method] = {
                "fold": fold, "checkpoint": str(checkpoint), "checkpoint_sha256": sha,
                "calibration": settings, "area_prior_q10_q95": {str(key): val for key, val in bounds.items()},
                "validation": by_strategy, "oracles": upper,
                "threshold_oracle_by_size": {
                    label: {"n": int(group.sum()), "model_miou": float(family["model"][group].mean()),
                            "oracle_miou": float(thresholds[group].mean())}
                    for label, group in (("below_5pct", small), ("at_least_5pct", ~small))
                },
                "decoder_map_five_fold": breakdown(rank_maps, masks, full),
                "fill_five_fold": breakdown(np.broadcast_to(bg[:, None, None], masks.shape), masks, full),
                "rank_vs_fill_by_fold": {
                    index: {"rank": float(miou_batch(rank_maps[folds == index], masks[folds == index]).mean()),
                            "fill": float(miou_batch(np.broadcast_to(bg[:, None, None], masks.shape)[folds == index],
                                                     masks[folds == index]).mean())}
                    for index in settings
                },
            }
            np.savez_compressed(cache, logits=logits, masks=targets, true_background=backgrounds,
                                decoder_background=predicted_backgrounds, names=val_names,
                                decoder_rank=score_maps, submitted=candidates["submitted_strategy"],
                                bit_accuracy=bit_acc[validation])
            print(method, json.dumps({key: round(val["miou"], 4) for key, val in by_strategy.items()}), upper, flush=True)
    methods = report["methods"]
    report["summary"] = {
        "submitted_strategy_macro": float(np.mean([row["validation"]["submitted_strategy"]["miou"]
                                                   for row in methods.values()])),
        "pixel_rank_candidate_macro": float(np.mean([
            row["validation"]["decoder_rank" if method == "pixelseal" else "submitted_strategy"]["miou"]
            for method, row in methods.items()
        ])),
        "threshold_oracle_macro": float(np.mean([row["oracles"]["best_threshold_per_image"]
                                                 for row in methods.values()])),
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
