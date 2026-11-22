"""Fixed stratified K-fold split of the train set, shared by every experiment (research/folds.json is committed).
Strata: method x background polarity (full_watermark) x minority region under / over 5 % of the grid.

    python -m research.folds --data release"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from wmloc.data import LocalisationDataset

K = 5
PATH = Path(__file__).with_name("folds.json")


def stratum(item: dict) -> str:
    frac = float(item["mask128"].mean())
    minority = frac if not item["full_watermark"] else 1 - frac
    return f"{item['method']}|{bool(item['full_watermark'])}|{'small' if minority < 0.05 else 'large'}"


def make(data: str, k: int = K, seed: int = 0) -> dict[str, int]:
    ds = LocalisationDataset(data, "train")
    groups: dict[str, list[str]] = {}
    for i in range(len(ds)):
        it = ds[i]
        groups.setdefault(stratum(it), []).append(it["image_name"])
    rng = np.random.default_rng(seed)
    folds, offset = {}, 0
    for key in sorted(groups):
        names = sorted(groups[key]); rng.shuffle(names)
        for j, name in enumerate(names):
            folds[name] = (offset + j) % k
        offset += len(names)
    return dict(sorted(folds.items()))


def load() -> dict[str, int]:
    return json.loads(PATH.read_text())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="release")
    folds = make(ap.parse_args().data)
    PATH.write_text(json.dumps(folds, indent=0) + "\n")
    print("wrote", PATH, len(folds), "images,", np.bincount(list(folds.values())).tolist(), "per fold")
