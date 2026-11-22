"""Cache the 128 x 128 logits of many segmenter checkpoints (plus horizontal-flip TTA) and the whole-image bit
accuracy against the known message, for every train and test image of one method, in ONE .npz file.

All later tuning (thresholds, ensemble weights, post-processing, rectangle-search initialisation) then runs on CPU
from this file, without the GPU queue.

    python -m research.cache_logits --method trustmark --out outputs/cache/trustmark.npz \
        --ckpt heavy1_r50=outputs/.../best.pt@resnet50 --ckpt heavy_f1_wm=outputs/.../best.pt@wm:trustmark

Keys: names_{train,test}; bitacc_{train,test}; full_train (train polarity); folds_train;
      <ckpt>/{train,test}[_flip] (float16 N x 128 x 128); ckpts (json: name -> path, encoder)."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from research import folds as folds_mod
from research.decoders import Decoder
from research.polarity_fill import bit_accuracy
from research.train_seg import build_model, predict as seg_predict
from wmloc.data import LocalisationDataset


def uint8_batch(items: list[dict]) -> torch.Tensor:
    return torch.from_numpy(np.stack([it["image"] for it in items])).permute(0, 3, 1, 2).contiguous().cuda()


@torch.no_grad()
def logits_with_flip(net, x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    plain = seg_predict(net, x).cpu().numpy().astype(np.float16)
    flip = seg_predict(net, x.flip(-1)).flip(-1).cpu().numpy().astype(np.float16)
    return plain, flip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["pixelseal", "trustmark", "stegastamp"])
    ap.add_argument("--data", default="release"); ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", action="append", default=[], help="name=checkpoint.pt@encoder")
    a = ap.parse_args()
    t0 = time.time()
    fold_of = folds_mod.load()
    tr = LocalisationDataset(a.data, "train", methods=[a.method]); tr = [tr[i] for i in range(len(tr))]
    te = LocalisationDataset(a.data, "test", methods=[a.method]); te = [te[i] for i in range(len(te))]
    out = {"names_train": np.array([it["image_name"] for it in tr]), "names_test": np.array([it["image_name"] for it in te]),
           "full_train": np.array([bool(it["full_watermark"]) for it in tr]),
           "folds_train": np.array([fold_of[it["image_name"]] for it in tr]),
           "gt_train": np.stack([it["mask128"] for it in tr]).astype(np.uint8)}
    dec = Decoder(a.method)
    out["bitacc_train"], out["bitacc_test"] = bit_accuracy(dec, tr, a.method), bit_accuracy(dec, te, a.method)
    del dec; torch.cuda.empty_cache()
    print(f"{a.method}: data + bit accuracy {time.time() - t0:.0f}s", flush=True)
    xtr, xte = uint8_batch(tr), uint8_batch(te)
    meta = {}
    for spec in a.ckpt:
        name, _, rest = spec.partition("="); path, _, enc = rest.partition("@")
        net = build_model(enc, pretrained=False).cuda().to(memory_format=torch.channels_last)
        net.load_state_dict(torch.load(path, map_location="cuda", weights_only=True)); net.eval()
        out[f"{name}/train"], out[f"{name}/train_flip"] = logits_with_flip(net, xtr)
        out[f"{name}/test"], out[f"{name}/test_flip"] = logits_with_flip(net, xte)
        meta[name] = {"path": path, "encoder": enc}
        del net; torch.cuda.empty_cache()
        print(f"{a.method}: {name} done {time.time() - t0:.0f}s", flush=True)
    out["ckpts"] = np.array(json.dumps(meta))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, **out)
    print(f"{a.method}: wrote {a.out} ({Path(a.out).stat().st_size / 1e6:.0f} MB) in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
