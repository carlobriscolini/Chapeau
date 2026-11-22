"""Diagnostics for already-saved submission files (no training). Works on the login-node GPU (~1 min per file).

    python experiments/evaluate.py ../output/best/*.pt ../output/sub_xxx.pt
Prints forget-set stats and drift vs the original model on mixes / shuffles / noise / held-out external photos
(the last one estimates the server's retain_kl_similarity = 1 / (1 + KL)).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # shared modules (common, diagnostics, ...)

import sys

import torch

from common import load_forget_images, load_model, predict
from diagnostics import drift_sets, report


def main():
    files = sys.argv[1:]
    if not files:
        print("Usage: python experiments/evaluate.py <sub.pt> [more.pt ...]")
        return
    forget = load_forget_images()
    teacher = load_model()
    orig_logits = predict(teacher, forget)
    sets = drift_sets()
    for f in files:
        print(f"\n===== {f}")
        model = load_model(f)
        report(model, forget, orig_logits, teacher=teacher, sets=sets)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
