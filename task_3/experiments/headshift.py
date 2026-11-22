"""'Head surgery' for class-level forgetting (no training, login-node GPU is enough).

The forget set removes 9 classes entirely (129-130 images each) and part of 36 others. A model retrained without
these images:
  - never saw a removed class -> its logit for that class is very low on every image
    (on those images it predicts its second choice);
  - saw fewer images of a partially removed class -> lower class prior: logit shift ~ log(remaining / class_size)
    (standard label-shift correction).
We emulate this by shifting the bias of the classifier head:
  full classes:     b_c -= full_shift
  partial classes:  b_c += partial_scale * log((class_size - n_c) / class_size)
applied on top of any submission (e.g. our best distilled model).

Usage:
  python experiments/headshift.py --model ../output/best/score0.1243_*.pt --full-shifts 4 8 12 --partial-scales 0 1
  python experiments/headshift.py --model ... --full-shifts 8 --partial-scales 1 --save --name best_hs
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # shared modules (common, diagnostics, ...)

import argparse
import math
from pathlib import Path

import torch

from common import OUTPUT_DIR, load_forget_images, load_model, predict, save_submission
from diagnostics import drift_sets, report


def shift_head(model, counts, full_shift, partial_scale, class_size):
    full = counts >= class_size - 1
    part = (counts > 0) & ~full
    with torch.no_grad():
        b = model.head.bias
        b[full.to(b.device)] -= full_shift
        if partial_scale != 0:
            remaining = (class_size - counts[part]).clamp(min=1).float()
            b[part.to(b.device)] += partial_scale * torch.log(remaining / class_size).to(b.device)
    return int(full.sum()), int(part.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="submission file to modify (or ../model.pt)")
    ap.add_argument("--full-shifts", nargs="+", type=float, default=[8.0])
    ap.add_argument("--partial-scales", nargs="+", type=float, default=[0.0, 1.0])
    ap.add_argument("--class-size", type=int, default=130)
    ap.add_argument("--head-scales", nargs="+", type=float, default=None,
                    help="instead of bias shifts: multiply head weight and bias by c (undo a global logit shrink)")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    forget = load_forget_images()
    teacher = load_model()
    orig_logits = predict(teacher, forget)
    counts = torch.bincount(orig_logits.argmax(1), minlength=orig_logits.shape[1])
    sets = drift_sets()
    name = args.name or Path(args.model).stem

    if args.head_scales:
        ext = sets.get("ext")
        if ext is not None:
            m0 = load_model(Path(args.model))
            lt, ls = predict(teacher, ext), predict(m0, ext)
            ft, fs = orig_logits, predict(m0, forget)
            print(f"logit std on photos: original {lt.std(1).mean():.3f}  model {ls.std(1).mean():.3f}  "
                  f"(ratio {ls.std(1).mean() / lt.std(1).mean():.3f});  on forget images: original "
                  f"{ft.std(1).mean():.3f}  model {fs.std(1).mean():.3f}")
            del m0
        for c in args.head_scales:
            model = load_model(Path(args.model))
            with torch.no_grad():
                model.head.weight.mul_(c)
                model.head.bias.mul_(c)
            print(f"\n===== head scale {c:g}")
            report(model, forget, orig_logits, teacher=teacher, sets=sets)
            if args.save:
                save_submission(model, OUTPUT_DIR / f"{name}_hscale{c:g}.pt")
            del model
            torch.cuda.empty_cache()
        return

    for fs in args.full_shifts:
        for ps in args.partial_scales:
            model = load_model(Path(args.model))
            nf, npart = shift_head(model, counts, fs, ps, args.class_size)
            print(f"\n===== full_shift {fs:g} (on {nf} classes), partial_scale {ps:g} (on {npart} classes)")
            report(model, forget, orig_logits, teacher=teacher, sets=sets)
            if args.save:
                save_submission(model, OUTPUT_DIR / f"{name}_hs{fs:g}_ps{ps:g}.pt")
            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
