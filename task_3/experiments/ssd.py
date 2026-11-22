"""Training-free unlearning tests (login-node GPU is enough, no job needed).

SSD - Selective Synaptic Dampening (Foster et al. 2024):
  F_f = diagonal Fisher (mean squared gradient) on the forget images (CE to the original model's labels),
  F_r = same on retain-like images (external photos, held-out part excluded).
  For every weight with F_f > alpha * F_r:  w *= min(lam * F_r / F_f, 1)
Magnitude pruning: zero the smallest |w| of all Linear weights inside the transformer blocks (global threshold).

Usage:
  python experiments/ssd.py --alphas 10 30 100 --lams 1 0.5           # SSD sweep on the original model
  python experiments/ssd.py --prune 0.2 0.4 0.6                       # pruning sweep
  python experiments/ssd.py --model ../output/best/xxx.pt --alphas 30 --lams 1   # SSD on top of a submission
  python experiments/ssd.py --alphas 30 --lams 1 --save               # also save every tested model
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # shared modules (common, diagnostics, ...)

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from augment import augment, patch_mask
from common import (EXTERNAL_HOLDOUT, GENPOOL_HOLDOUT, MODEL_FILE, OUTPUT_DIR, holdout_size, load_external_images,
                    load_forget_images, load_genpool, load_model, predict, preprocess, save_submission)
from diagnostics import drift_sets, report

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def fisher(model, images, labels, batch_size, importance="ce", temp=1.0, transform=None, repeats=1):
    """Diagonal importance: mean over batches of the squared mini-batch gradient (fp32, no autocast).
    ce  = cross-entropy to the original model's labels (classic Fisher; vanishes on memorized images, loss ~0)
    mas = squared L2 norm of the logits (Memory Aware Synapses; label-free, does not vanish when confident)"""
    fis = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    model.eval()
    nb = 0
    g = torch.Generator(device=DEVICE).manual_seed(0)
    for i in [j for _ in range(repeats) for j in range(0, len(images), batch_size)]:
        x = preprocess(images[i:i + batch_size]).to(DEVICE)
        if transform is not None:  # e.g. augmented / patch-masked copies of the forget images
            x = transform(x, g)
        y = labels[i:i + batch_size].to(DEVICE)
        model.zero_grad(set_to_none=True)
        out = model(x)
        loss = F.cross_entropy(out / temp, y) if importance == "ce" else out.pow(2).sum(1).mean()
        loss.backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                fis[n] += p.grad.detach() ** 2
        nb += 1
    model.zero_grad(set_to_none=True)
    return {n: f / max(nb, 1) for n, f in fis.items()}


def ssd(model, f_forget, f_retain, alpha, lam):
    changed, total = 0, 0
    with torch.no_grad():
        for n, p in model.named_parameters():
            ff, fr = f_forget[n], f_retain[n]
            sel = ff > alpha * fr
            beta = torch.clamp(lam * fr / ff.clamp_min(1e-30), max=1.0)
            p[sel] = p[sel] * beta[sel]
            changed += int(sel.sum())
            total += p.numel()
    return changed / total


def prune(model, frac):
    ws = [m.weight for n, m in model.named_modules()
          if isinstance(m, torch.nn.Linear) and n.startswith("blocks.")]
    allw = torch.cat([w.detach().abs().flatten() for w in ws])
    k = max(1, int(frac * allw.numel()))
    thr = allw.kthvalue(k).values
    with torch.no_grad():
        for w in ws:
            w[w.abs() <= thr] = 0
    return float(thr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(MODEL_FILE), help="starting weights (original model or a submission)")
    ap.add_argument("--alphas", nargs="+", type=float, default=[], help="SSD selection thresholds (paper: 10)")
    ap.add_argument("--lams", nargs="+", type=float, default=[1.0], help="SSD dampening constants (paper: 1)")
    ap.add_argument("--prune", nargs="+", type=float, default=[], help="fractions of block Linear weights to zero")
    ap.add_argument("--importance", default="ce", choices=["ce", "mas"])
    ap.add_argument("--temp", type=float, default=1.0, help="CE temperature for the ce importance (2-3 avoids zero grads)")
    ap.add_argument("--normalize", action="store_true", help="divide F_f and F_r by their own global means")
    ap.add_argument("--per-tensor", action="store_true", help="normalize F_f, F_r per tensor instead of globally")
    ap.add_argument("--retain-src", default="ext", help="F_r source: ext | aug | gen, or combos like gen+ext (aug = augmented forget images, gen = genpool.npz)")
    ap.add_argument("--aug-level", default="strong", help="light | medium | strong | patchmask (for --retain-src aug)")
    ap.add_argument("--aug-repeats", type=int, default=1, help="augmented passes over the forget set for F_r")
    ap.add_argument("--n-retain", type=int, default=3000, help="external photos used for F_r")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    forget = load_forget_images()
    teacher = load_model()
    orig_logits = predict(teacher, forget)
    sets = drift_sets()
    tag = Path(args.model).stem if args.model != str(MODEL_FILE) else "orig"

    if args.alphas:
        base = load_model(Path(args.model))
        y_f = orig_logits.argmax(1)
        f_f = fisher(base, forget, y_f, args.batch_size, args.importance, args.temp)
        parts = []
        if "ext" in args.retain_src.split("+"):
            ext = load_external_images()
            ext = ext[:len(ext) - holdout_size(len(ext), EXTERNAL_HOLDOUT)]
            ext = ext[torch.randperm(len(ext), generator=torch.Generator().manual_seed(0))[:args.n_retain]]
            parts.append(fisher(base, ext, predict(teacher, ext).argmax(1), args.batch_size, args.importance,
                                args.temp))
        if "gen" in args.retain_src.split("+"):  # generated cinematic-style images covering (most of) the 100 classes
            gp, gl = load_genpool()
            assert gp is not None, "run genpool.sh first (../output/genpool.npz missing)"
            n = len(gp) - holdout_size(len(gp), GENPOOL_HOLDOUT)
            parts.append(fisher(base, gp[:n], gl[:n], args.batch_size, args.importance, args.temp))
        if "aug" in args.retain_src.split("+"):  # class evidence survives augmentation, instance memorisation not
            lvl = args.aug_level
            tf = (lambda x, g: patch_mask(x, g=g)) if lvl == "patchmask" else (lambda x, g: augment(x, lvl, g=g))
            parts.append(fisher(base, forget, y_f, args.batch_size, args.importance, args.temp, tf, args.aug_repeats))
        if len(parts) == 1:
            f_r = parts[0]
        else:  # each source normalised by its global mean, then averaged
            ms = [torch.cat([f.flatten() for f in pt.values()]).mean() for pt in parts]
            f_r = {n: sum(pt[n] / m for pt, m in zip(parts, ms)) / len(parts) for n in f_f}
        if args.per_tensor:  # each tensor's F_f and F_r divided by their own means (scales differ across layers)
            f_f = {n: f / f.mean().clamp_min(1e-30) for n, f in f_f.items()}
            f_r = {n: f / f.mean().clamp_min(1e-30) for n, f in f_r.items()}
        elif args.normalize:  # compare relative importance: each Fisher divided by its global mean
            mf = torch.cat([f.flatten() for f in f_f.values()]).mean()
            mr = torch.cat([f.flatten() for f in f_r.values()]).mean()
            f_f = {n: f / mf for n, f in f_f.items()}
            f_r = {n: f / mr for n, f in f_r.items()}
        print(f"importance ({args.importance}, T={args.temp:g}) done: forget vs retain source '{args.retain_src}'")
        r = torch.cat([(f_f[n] / f_r[n].clamp_min(1e-30)).flatten()[::97] for n in f_f])
        print("F_f/F_r ratio quantiles 50/90/99/99.9%: " + " ".join(
            f"{v:.3g}" for v in torch.quantile(r.float().cpu(), torch.tensor([0.5, 0.9, 0.99, 0.999]))))
        del base
        for a in args.alphas:
            for lam in args.lams:
                model = load_model(Path(args.model))
                frac = ssd(model, f_f, f_r, a, lam)
                print(f"\n===== SSD alpha {a:g} lam {lam:g}: dampened {100 * frac:.3f}% of weights")
                report(model, forget, orig_logits, teacher=teacher, sets=sets)
                if args.save:
                    extra = f"_t{args.temp:g}_{args.retain_src.replace('+', '')}" + (
                        args.aug_level if "aug" in args.retain_src else "") + ("_pt" if args.per_tensor else "") + (
                        "_mas" if args.importance == "mas" else "")
                    save_submission(model, OUTPUT_DIR / f"ssd_{tag}_a{a:g}_l{lam:g}{extra}.pt")
                del model
                torch.cuda.empty_cache()

    for frac in args.prune:
        model = load_model(Path(args.model))
        thr = prune(model, frac)
        print(f"\n===== prune {100 * frac:g}% of block Linear weights (|w| <= {thr:.2e})")
        report(model, forget, orig_logits, teacher=teacher, sets=sets)
        if args.save:
            save_submission(model, OUTPUT_DIR / f"prune_{tag}_{frac:g}.pt")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
