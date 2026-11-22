"""Combine a trained segmenter with the decoder's background polarity (whole-image bit accuracy vs the message).

Strategies, scored on the real images of the validation fold (the polarity threshold is fitted on the other folds):
  model     the segmenter alone (logits > 0)
  fill      the polarity alone (whole grid = predicted background)
  gate      the segmenter, unless its majority class contradicts the polarity, then fill
  bias_b    logits shifted by +b on predicted marked backgrounds and -b on clean ones

    python -m research.fuse --method pixelseal --ckpt outputs/seg_v1/pixelseal/best.pt"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from research import folds as folds_mod
from research.decoders import Decoder
from research.metrics import miou_batch
from research.polarity_fill import bit_accuracy, fit_threshold
from research.predict import predict_items
from research.train_seg import build_model
from wmloc.data import LocalisationDataset

BIASES = (0.5, 1.0, 2.0, 4.0)


def fuse(logits: np.ndarray, marked: np.ndarray, strategy: str) -> np.ndarray:
    pred = logits > 0
    if strategy == "model":
        return pred
    fill = np.broadcast_to(marked[:, None, None], pred.shape)
    if strategy == "fill":
        return fill
    if strategy == "gate":
        agree = (pred.mean(axis=(1, 2)) > 0.5) == marked
        return np.where(agree[:, None, None], pred, fill)
    b = float(strategy.split("_")[1])
    return logits + np.where(marked, b, -b)[:, None, None] > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--encoder", default="resnet34"); ap.add_argument("--data", default="release")
    ap.add_argument("--val-fold", type=int, default=0)
    a = ap.parse_args()
    fold_of = folds_mod.load()
    ds = LocalisationDataset(a.data, "train", methods=[a.method]); items = [ds[i] for i in range(len(ds))]
    f = np.array([fold_of[it["image_name"]] for it in items]); val = f == a.val_fold
    full = np.array([bool(it["full_watermark"]) for it in items])
    acc = bit_accuracy(Decoder(a.method), items, a.method)
    marked = acc[val] > fit_threshold(acc[~val], full[~val])
    model = build_model(a.encoder, pretrained=False).cuda().to(memory_format=torch.channels_last)
    model.load_state_dict(torch.load(a.ckpt, map_location="cuda", weights_only=True))
    validation = [it for it, chosen in zip(items, val) if chosen]
    logits = predict_items(model, validation).cpu().numpy()
    gt = np.stack([it["mask128"] for it in validation])
    res = {s: float(miou_batch(fuse(logits, marked, s), gt).mean())
           for s in ["model", "fill", "gate"] + [f"bias_{b}" for b in BIASES]}
    res["polarity_accuracy"] = float((marked == full[val]).mean()); res["n"] = int(val.sum())
    print(json.dumps({a.method: res}))


if __name__ == "__main__":
    main()
