"""Safety-net predictor: fill the whole grid with the predicted background polarity.

The whole-image bit accuracy against the known message separates marked backgrounds (about 0.97) from clean
backgrounds (about 0.53). A per-method threshold is fitted on train; its score is reported out-of-fold.

    python -m research.polarity_fill --data release --out outputs/submissions/polarity_fill.pt"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from research import folds as folds_mod
from research import submission
from research.decoders import Decoder, signed
from research.metrics import miou_batch
from research.zero_shot import BATCH, METHODS
from wmloc.data import LocalisationDataset


@torch.no_grad()
def bit_accuracy(dec: Decoder, items: list[dict], method: str) -> np.ndarray:
    out = []
    for b in range(0, len(items), BATCH[method]):
        chunk = items[b:b + BATCH[method]]
        x = torch.from_numpy(np.stack([it["image"] for it in chunk])).permute(0, 3, 1, 2).float().cuda() / 255
        s = signed(torch.from_numpy(np.stack([it["message"] for it in chunk]))).cuda()
        out.append(((dec.logits(x) > 0) == (s > 0)).float().mean(1).cpu().numpy())
    return np.concatenate(out)


def fit_threshold(acc: np.ndarray, full: np.ndarray) -> float:
    """Midpoint of the best-accuracy interval between sorted values."""
    v = np.sort(np.unique(acc)); cand = np.concatenate([[v[0] - 1e-3], (v[:-1] + v[1:]) / 2, [v[-1] + 1e-3]])
    accs = np.array([((acc > t) == full).mean() for t in cand])
    best = cand[accs == accs.max()]
    return float(np.median(best))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="release"); ap.add_argument("--out", default="outputs/submissions/polarity_fill.pt")
    a = ap.parse_args()
    fold_of = folds_mod.load()
    report, test_names, test_masks = {}, [], []
    for method in METHODS:
        dec = Decoder(method)
        tr = LocalisationDataset(a.data, "train", methods=[method]); tr = [tr[i] for i in range(len(tr))]
        te = LocalisationDataset(a.data, "test", methods=[method]); te = [te[i] for i in range(len(te))]
        acc_tr, acc_te = bit_accuracy(dec, tr, method), bit_accuracy(dec, te, method)
        full = np.array([bool(it["full_watermark"]) for it in tr]); gt = np.stack([it["mask128"] for it in tr])
        f = np.array([fold_of[it["image_name"]] for it in tr])
        pred = np.zeros(len(tr), bool)
        for k in range(folds_mod.K):
            t = fit_threshold(acc_tr[f != k], full[f != k]); pred[f == k] = acc_tr[f == k] > t
        thr = fit_threshold(acc_tr, full)
        fill = np.broadcast_to(pred[:, None, None], gt.shape)
        hist = np.histogram(acc_te, bins=10, range=(0, 1))[0].tolist()
        report[method] = {"threshold": thr, "oof_polarity_accuracy": float((pred == full).mean()),
                          "oof_miou": float(miou_batch(fill, gt).mean()),
                          "test_marked_background_share": float((acc_te > thr).mean()),
                          "test_bit_acc_histogram_0_to_1": hist}
        test_names += [it["image_name"] for it in te]
        test_masks.append(np.broadcast_to((acc_te > thr)[:, None, None], (len(te), 128, 128)))
        print(method, report[method], flush=True)
        del dec; torch.cuda.empty_cache()
    report["submission"] = submission.write(test_names, np.concatenate(test_masks), Path(a.out), a.data)
    report["oof_miou_all"] = float(np.mean([report[m]["oof_miou"] for m in METHODS]))
    Path(a.out).with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in METHODS}, indent=2))


if __name__ == "__main__":
    main()
