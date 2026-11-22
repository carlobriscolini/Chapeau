"""Logit-temperature test on a submission: head.weight and head.bias multiplied by gamma (= logits * gamma, i.e.
temperature 1/gamma). Training-free; prints the usual diagnostics per gamma, --save writes the files.

    python src/headscale.py --model ../output/best/score0.2534_reinit6_PPext_mse.pt --gammas 1 1.15 1.25 1.35
"""
import argparse
from pathlib import Path

import torch

from common import OUTPUT_DIR, load_forget_images, load_model, predict, save_submission
from diagnostics import drift_sets, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gammas", nargs="+", type=float, default=[1.0, 1.15, 1.25, 1.35])
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    forget = load_forget_images()
    teacher = load_model()
    orig_logits = predict(teacher, forget)
    sets = drift_sets()
    for g in args.gammas:
        model = load_model(Path(args.model))
        with torch.no_grad():
            model.head.weight.mul_(g)
            model.head.bias.mul_(g)
        print(f"\n===== head scale {g:g}")
        report(model, forget, orig_logits, teacher=teacher, sets=sets)
        if args.save:
            save_submission(model, OUTPUT_DIR / f"{Path(args.model).stem}_hs{g:g}.pt")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
