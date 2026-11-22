"""Weight-space tricks on already-trained submissions (no training, login-node GPU is enough).

  θ(α) = θ_orig + α · (mean(θ_1..θ_k) − θ_orig)

  - one model, several α: trade-off curve. α < 1 = closer to the original (less drift, less forgetting),
    α > 1 = extrapolate (more forgetting).
  - several models (e.g. different seeds / configs): "model soup", averages away random drift.

Usage:
    python experiments/soup.py --models ../output/best/score0.1115_*.pt --alphas 0.6 0.8 1.0 1.2
    python experiments/soup.py --models A.pt B.pt --alphas 1.0 --save          # writes ../output/soup_*.pt
Prints the usual diagnostics for every α (forget/exact, altered, drift on external photos).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # shared modules (common, diagnostics, ...)

import argparse
from pathlib import Path

import torch

from common import (OUTPUT_DIR, build_model, load_forget_images, load_model, predict, save_submission, DEVICE)
from diagnostics import drift_sets, report


def load_sd(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--alphas", nargs="+", type=float, default=[1.0])
    ap.add_argument("--save", action="store_true", help="save every evaluated θ(α) as a submission file")
    ap.add_argument("--name", default=None, help="name prefix for saved files")
    args = ap.parse_args()

    forget = load_forget_images()
    teacher = load_model()
    orig_logits = predict(teacher, forget)
    sets = drift_sets()
    orig = {k: v.detach().cpu().float() for k, v in teacher.state_dict().items()}

    sds = [load_sd(m) for m in args.models]
    mean = {k: sum(sd[k].float() for sd in sds) / len(sds) for k in orig}
    delta = {k: mean[k] - orig[k] for k in orig}
    name = args.name or ("soup" if len(sds) > 1 else Path(args.models[0]).stem)

    for a in args.alphas:
        sd = {k: orig[k] + a * delta[k] for k in orig}
        model = build_model()
        model.load_state_dict(sd, strict=True)
        model = model.to(DEVICE).eval()
        print(f"\n===== {len(sds)} model(s), alpha = {a:g}")
        report(model, forget, orig_logits, teacher=teacher, sets=sets)
        if args.save:
            save_submission(model, OUTPUT_DIR / f"{name}_alpha{a:g}.pt")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
