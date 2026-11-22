"""Entry point: load original model + forget set, run an unlearning method, save a submission, print diagnostics.

Usage (GPU job, from the repo dir):
    sbatch main.sh run_unlearn.py --method identity
    sbatch main.sh run_unlearn.py --method distill --strong-weight 0.5 --tau 1.0 --lam 1.0 --epochs 2
    sbatch main.sh run_unlearn.py --method distill --forget-loss npo --lam 1.0
    sbatch main.sh run_unlearn.py --method distill --mask 0.25
Output file name is built from the arguments unless --out is given.
"""
import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from augment import augment, retain_images
from common import (AMP_DTYPE, DEVICE, EXTERNAL_HOLDOUT, EXTGEN_HOLDOUT, GENPOOL_HOLDOUT, GENPOOL_P_FILE, GENPOOL_T_FILE, INVERTED_HOLDOUT,
                    OUTPUT_DIR, SIB_HOLDOUT_SRC, TEMPERATURE, holdout_size, load_genpool, load_inverted,
                    load_external_images, load_extgen, load_forget_images, load_model, load_siblings, predict,
                    preprocess, save_submission)
from diagnostics import drift_sets, report
from targets import apply_tau, blended_pbar, describe, load_knn, load_or_compute, load_or_compute_sib, to_T

T = TEMPERATURE


def kl_div(p_target, logits_student):
    """mean_x KL(p_target || softmax(logits/T)) — same direction and temperature as the server."""
    logq = F.log_softmax(logits_student.float() / T, dim=1)
    return (p_target * (p_target.clamp_min(1e-12).log() - logq)).sum(1).mean()


def saliency_mask(model, forget, labels, frac, batch_size, n_batches=20):
    """SalUn: keep only the `frac` of weights with the largest |grad CE(pseudo-label)| on augmented forget images."""
    model.train()
    grads = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
    for b in range(n_batches):
        idx = torch.randint(0, len(forget), (batch_size,))
        x = augment(preprocess(forget[idx]), "light")
        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=DEVICE == "cuda"):
            loss = F.cross_entropy(model(x).float(), labels[idx].to(DEVICE))
        model.zero_grad(set_to_none=True)
        loss.backward()
        for n, p in model.named_parameters():
            if n in grads:
                grads[n] += p.grad.abs()
    flat = torch.cat([g.flatten() for g in grads.values()])
    k = max(1, int(len(flat) * (1 - frac)))
    thresh = flat.float().kthvalue(k).values if len(flat) < 2 ** 24 else _approx_quantile(flat, 1 - frac)
    masks = {n: (g >= thresh).float() for n, g in grads.items()}
    kept = sum(m.sum().item() for m in masks.values()) / len(flat)
    print(f"saliency mask: keeping {kept:.3f} of weights")
    model.zero_grad(set_to_none=True)
    return masks


@torch.no_grad()
def build_null_projectors(model, pools, args):
    """AlphaEdit-style null-space preconditioning. For every Linear layer y = W x (qkv, proj, fc1, fc2, head) collect
    the input covariance C = E[x x^T] on anchor images, eigen-decompose C = U diag(L) U^T and build
    P = U diag(r / (L + r)) U^T with r = rho * mean(L). dW @ P keeps the update in directions the anchor images
    hardly use (small L), so it changes little on them. Combine with --optim sgd (Adam would undo the rotation)."""
    assert pools, "--null-proj needs anchor images (e.g. --retain external with --lam > 0 or --feat-lam > 0)"
    imgs = torch.cat(list(pools.values()))
    imgs = imgs[torch.randperm(len(imgs))[:args.null_images]]
    layers = [m for n_, m in model.named_modules() if isinstance(m, torch.nn.Linear)
              and m.weight.requires_grad and (".attn." in n_ or ".mlp." in n_ or n_ == "head")]
    cov = {id(m): torch.zeros(m.in_features, m.in_features, device=DEVICE) for m in layers}
    cnt = {id(m): 0 for m in layers}

    def hook(mod, inp, out):
        x = inp[0].reshape(-1, mod.in_features).float()
        cov[id(mod)] += x.T @ x
        cnt[id(mod)] += x.shape[0]
    handles = [m.register_forward_hook(hook) for m in layers]
    model.eval()
    for i in range(0, len(imgs), 64):
        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=DEVICE == "cuda"):
            model(preprocess(imgs[i:i + 64]))
    for h in handles:
        h.remove()
    projectors = []
    for m in layers:
        C = cov[id(m)] / max(cnt[id(m)], 1)
        L, U = torch.linalg.eigh(C)
        L = L.clamp_min(0)
        r = args.null_proj * L.mean()
        projectors.append((m, (U * (r / (L + r))) @ U.T))
    model.train()
    print(f"null-space projectors: {len(projectors)} linear layers, {len(imgs)} anchor images, rho {args.null_proj:g}")
    return projectors


def contrast_mask(model, forget, labels, pools, teacher, frac, batch_size, n_batches=20):
    """Keep the `frac` of weights with the largest |grad forget| / (|grad retain| + eps): important for the forget
    images, unimportant for the anchor images. forget grad: CE(pseudo-label) on lightly augmented forget images;
    retain grad: CE(original's argmax) on anchor images."""
    model.train()
    gf = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
    gr = {n: torch.zeros_like(p) for n, p in gf.items()}
    anchors = torch.cat(list(pools.values()))
    for b in range(n_batches):
        for acc, imgs, lab_fn in ((gf, None, None), (gr, anchors, None)):
            if imgs is None:
                idx = torch.randint(0, len(forget), (batch_size,))
                x = augment(preprocess(forget[idx]), "light")
                y = labels[idx].to(DEVICE)
            else:
                idx = torch.randint(0, len(imgs), (batch_size,))
                x = preprocess(imgs[idx])
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=DEVICE == "cuda"):
                    y = teacher(x).float().argmax(1)
            with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=DEVICE == "cuda"):
                loss = F.cross_entropy(model(x).float(), y)
            model.zero_grad(set_to_none=True)
            loss.backward()
            for n, p in model.named_parameters():
                if n in acc and p.grad is not None:
                    acc[n] += p.grad.abs()
    eps = 1e-3 * torch.cat([g.flatten() for g in gr.values()]).mean()
    score = {n: gf[n] / (gr[n] + eps) for n in gf}
    flat = torch.cat([v.flatten() for v in score.values()])
    thresh = _approx_quantile(flat, 1 - frac)
    masks = {n: (v >= thresh).float() for n, v in score.items()}
    kept = sum(m.sum().item() for m in masks.values()) / len(flat)
    print(f"contrast mask: keeping {kept:.3f} of weights (high forget/retain gradient ratio)")
    model.zero_grad(set_to_none=True)
    return masks


def _approx_quantile(flat, q, n=2 ** 22):
    idx = torch.randint(0, len(flat), (n,), device=flat.device)
    return flat[idx].float().quantile(q)


def class_aware(pbar, labels, args):
    """The forget set removes whole classes (n_c = class size) and parts of others. A model retrained without
    these images never saw a removed class: it puts ~no probability on it and picks another class.
    Per class c: a_c = (n_c / (class_size - 1)) ** gamma, clipped to 1 (1 = class fully removed).
    target = (1 - a_c) * pbar + a_c * pbar_without_c   (pbar_without_c: column c set to `absent_floor`, renormalised)."""
    counts = torch.bincount(labels, minlength=pbar.shape[1]).float()
    a = (counts / (args.class_size - 1)).clamp(max=1.0) ** args.absent_gamma
    a_img = a[labels][:, None]
    absent = pbar.clone()
    absent[torch.arange(len(labels)), labels] = 0.0
    absent = absent / absent.sum(1, keepdim=True).clamp_min(1e-12)
    absent = (1 - args.absent_floor) * absent
    absent[torch.arange(len(labels)), labels] = args.absent_floor
    full = int((counts >= args.class_size - 1).sum())
    print(f"class-aware target: {int((counts > 0).sum())} classes in the forget set, {full} fully removed "
          f"({int(counts[counts >= args.class_size - 1].sum())} images); mean absent weight {a_img.mean():.3f}")
    return (1 - a_img) * pbar + a_img * absent


def method_identity(model, forget, args, ctx):
    """Baseline: no unlearning (scores ~0)."""
    return model


def method_distill(model, forget, args, ctx):
    """Student starts from θ_orig.
    forget branch: KL(q(x) || p_sub(light_aug(x))) at T=2  (or NPO on pseudo-labels, --forget-loss npo)
    retain branch: KL(p_orig(a) || p_sub(a)) at T=2 on anchor images a (--retain: mix | shuffle | noise |
                   external | extgen | siblings, or a "+"-combination of the last three, e.g. extgen+external)
    target q: augmentation average (--strong-weight, --tau), optionally blended with the sibling average
              (--sib-weight s: (1-s)*aug + s*siblings)
    total: L_forget + lam * L_retain, optional SalUn mask on the gradients."""
    teacher = ctx["teacher"]
    labels = ctx["labels"]  # original model's argmax on exact images (cpu)
    n = len(forget)

    if args.forget_loss == "distill":
        pbar = blended_pbar(teacher, forget, args.strong_weight, args.K)
        if args.sib_weight > 0:
            ps = load_or_compute_sib(teacher, len(forget))
            pbar = (1 - args.sib_weight) * pbar + args.sib_weight * ps
            print(f"target: {1 - args.sib_weight:g} * augmentations + {args.sib_weight:g} * siblings")
        if args.wn_weight > 0:  # weight-noise views: original with noisy top blocks, clean images
            lvl = f"wnoise{args.wn_sigma:g}b{args.wn_block}"
            wn = load_or_compute(teacher, forget, lvl, args.wn_K)
            pbar = (1 - args.wn_weight) * pbar + args.wn_weight * wn
            print(f"target: {1 - args.wn_weight:g} * previous + {args.wn_weight:g} * {lvl} (K={args.wn_K})")
        if args.pm_weight > 0:  # MAE-style patch-masked views of the original model
            pm = load_or_compute(teacher, forget, "patchmask", args.pm_K)
            pbar = (1 - args.pm_weight) * pbar + args.pm_weight * pm
            print(f"target: {1 - args.pm_weight:g} * augmentations + {args.pm_weight:g} * patch-masked views")
        if args.knn_weight > 0:  # original model on the k nearest generated pool images (knn_target.py)
            kn = load_knn(args.knn_k, args.knn_src)
            pbar = (1 - args.knn_weight) * pbar + args.knn_weight * kn
            print(f"target: {1 - args.knn_weight:g} * previous + {args.knn_weight:g} * knn k{args.knn_k} ({args.knn_src})")
        if args.absent_gamma > 0:
            pbar = class_aware(pbar, labels, args)
        q1 = apply_tau(pbar, args.tau)
        d = describe(q1, labels, F.softmax(ctx["orig_logits"] / T, 1))
        print(f"target on forget images: acc {d['acc']:.3f}  loss mean {d['loss_mean']:.3f}  std {d['loss_std']:.3f}  "
              f"max prob {d['max_prob']:.3f}")
        q_T = to_T(q1).to(DEVICE)  # target distribution at temperature T
    else:
        with torch.no_grad():
            logp_orig_label = F.log_softmax(ctx["orig_logits"] / args.npo_temp, 1).gather(
                1, labels[:, None]).squeeze(1).to(DEVICE)

    if args.train_only == "mlp":
        for n_, p in model.named_parameters():
            p.requires_grad_(".mlp." in n_)
    elif args.train_only == "linear":  # only the Linear weights that --null-proj projects (no biases/norms/LayerScale)
        for n_, p in model.named_parameters():
            p.requires_grad_(n_ == "head.weight" or (n_.endswith(".weight") and (".attn." in n_ or ".mlp." in n_)))
    if args.train_only != "all":
        print(f"train-only {args.train_only}: "
              f"{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}M params")
    if args.train_blocks >= 0:
        for p in model.parameters():
            p.requires_grad_(False)
        trainable = [model.norm, model.head] + list(model.blocks[len(model.blocks) - args.train_blocks:])
        for m in trainable:
            for p in m.parameters():
                p.requires_grad_(True)
        n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"training last {args.train_blocks} blocks + head: {n_tr / 1e6:.1f}M params")

    use_anchor = args.lam > 0 or args.feat_lam > 0
    pools = anchor_pools(args) if (use_anchor or args.null_proj > 0 or args.contrast_mask < 1) else {}
    if pools and args.anchor_untouched:
        # keep only anchor images that the ORIGINAL model assigns to classes with no forget images:
        # there the gold behaves like the original, so any change is pure damage; elsewhere change is allowed
        untouched_cls = torch.bincount(labels, minlength=ctx["orig_logits"].shape[1]) == 0
        for name in list(pools):
            cls = predict(teacher, pools[name]).argmax(1)
            keep = untouched_cls[cls]
            if keep.sum() < 64:
                print(f"anchor-untouched: only {int(keep.sum())} images in {name}, keeping the full pool")
                continue
            pools[name] = pools[name][keep]
            print(f"anchor-untouched: {name} keeps {int(keep.sum())} images predicted in the "
                  f"{int(untouched_cls.sum())} untouched classes")

    masks = saliency_mask(model, forget, labels, args.mask, args.batch_size) if args.mask < 1 else None
    if args.contrast_mask < 1:
        assert pools, "--contrast-mask needs anchor images (e.g. --retain external with --lam > 0)"
        masks = contrast_mask(model, forget, labels, pools, teacher, args.contrast_mask, args.batch_size)

    untouched_rows = None
    if args.freeze_untouched_head:
        untouched_rows = (torch.bincount(labels, minlength=ctx["orig_logits"].shape[1]) == 0).to(DEVICE)
        print(f"freeze-untouched-head: head rows of {int(untouched_rows.sum())} untouched classes are frozen")

    projectors = build_null_projectors(model, pools, args) if args.null_proj > 0 else None

    params = [p for p in model.parameters() if p.requires_grad]
    if args.optim == "sgd":
        opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.wd)
    else:
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = math.ceil(n / args.batch_size)
    total = steps_per_epoch * args.epochs
    warmup = max(1, int(0.1 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))

    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda" and AMP_DTYPE == torch.float16))
    print(f"amp dtype: {AMP_DTYPE}, grad scaler: {scaler.is_enabled()}")
    model.train()
    step, t0 = 0, time.time()
    for epoch in range(args.epochs):
        perm = torch.randperm(n)
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            x = preprocess(forget[idx])
            with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=DEVICE == "cuda"):
                logits_f = model(augment(x, "light", p_skip=args.p_exact)).float()
            if args.forget_loss == "distill":
                l_forget = kl_div(q_T[idx.to(DEVICE)], logits_f)
            else:  # NPO: -(2/β) log σ(-β (log π_sub(ŷ) - log π_orig(ŷ)))
                lp = F.log_softmax(logits_f / args.npo_temp, 1).gather(1, labels[idx].to(DEVICE)[:, None]).squeeze(1)
                ratio = lp - logp_orig_label[idx.to(DEVICE)]
                l_forget = -(2 / args.beta) * F.logsigmoid(-args.beta * ratio).mean()

            l_retain = torch.zeros((), device=DEVICE)
            l_feat = torch.zeros((), device=DEVICE)
            if use_anchor:
                if pools:
                    per = max(1, len(idx) // len(pools))
                    xm = torch.cat([augment(preprocess(p[torch.randint(0, len(p), (per,))]), "light")
                                    for p in pools.values()])
                else:
                    xm = retain_images(x, args.retain)
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=AMP_DTYPE,
                                                     enabled=DEVICE == "cuda"):
                    t_feat = teacher.forward_head(teacher.forward_features(xm), pre_logits=True).float()
                    t_logits = teacher.head(t_feat).float()
                with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=DEVICE == "cuda"):
                    s_feat = model.forward_head(model.forward_features(xm), pre_logits=True).float()
                    s_logits = model.head(s_feat).float()
                l_retain = kl_div(F.softmax(t_logits / T, 1), s_logits)
                if args.feat_lam > 0:  # relative MSE on the pre-head (CLS) embedding
                    l_feat = F.mse_loss(s_feat, t_feat) / t_feat.pow(2).mean().clamp_min(1e-6)

            loss = l_forget + args.lam * l_retain + args.feat_lam * l_feat
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            if projectors is not None:  # rotate dW into directions the anchor images barely use: dW @ P
                for lin, P in projectors:
                    if lin.weight.grad is not None:
                        lin.weight.grad = lin.weight.grad @ P
            if masks is not None:
                for nme, p in model.named_parameters():
                    if p.grad is not None and nme in masks:
                        p.grad.mul_(masks[nme])
            if untouched_rows is not None:
                for p in (model.head.weight, model.head.bias):
                    if p.grad is not None:
                        p.grad[untouched_rows] = 0
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % 20 == 0 or step == total:
                print(f"ep {epoch} step {step}/{total}  loss {loss.item():.4f}  forget {l_forget.item():.4f}  "
                      f"retain {l_retain.item():.4f}  feat {l_feat.item():.4f}  lr {sched.get_last_lr()[0]:.2e}  "
                      f"{time.time() - t0:.0f}s",
                      flush=True)
    model.eval()
    return model


POOL_NAMES = ("external", "extgen", "siblings", "inverted", "genpool", "genpoolT", "genpoolP")


def anchor_pools(args):
    """Retain-anchor image pools (held-out parts excluded) for --retain external / extgen / siblings / a+b."""
    names = args.retain.split("+")
    if not all(n in POOL_NAMES for n in names):
        assert args.retain in ("mix", "shuffle", "noise", "shuffle+noise"), f"unknown --retain {args.retain}"
        return {}
    pools = {}
    for name in names:
        if name == "external":
            imgs = load_external_images()
            assert imgs is not None, "run prepare_external.py first"
            pools[name] = imgs[:len(imgs) - holdout_size(len(imgs), EXTERNAL_HOLDOUT)]
        elif name == "inverted":
            imgs = load_inverted()
            assert imgs is not None, "run inv.sh first (../output/inverted.npz missing)"
            pools[name] = imgs[:len(imgs) - holdout_size(len(imgs), INVERTED_HOLDOUT)]
        elif name == "genpool":
            imgs, _ = load_genpool()
            assert imgs is not None, "run genpool.sh first (../output/genpool.npz missing)"
            pools[name] = imgs[:len(imgs) - holdout_size(len(imgs), GENPOOL_HOLDOUT)]
        elif name == "genpoolT":  # round 1 (minus its held-out part, still the gpool drift set) + targeted round 2
            imgs, _ = load_genpool()
            t, _ = load_genpool(cap=300, path=GENPOOL_T_FILE)
            assert imgs is not None and t is not None, "run genpool.sh and genpool.sh --targeted first"
            pools[name] = torch.cat([imgs[:len(imgs) - holdout_size(len(imgs), GENPOOL_HOLDOUT)], t])
        elif name == "genpoolP":  # round 3: photographic look (never held out)
            imgs, _ = load_genpool(cap=300, path=GENPOOL_P_FILE)
            assert imgs is not None, "run genpool.sh --targeted --photo first"
            pools[name] = imgs
        elif name == "extgen":
            imgs = load_extgen()
            assert imgs is not None, "run gen.sh first (../output/extgen.npz missing)"
            pools[name] = imgs[:len(imgs) - holdout_size(len(imgs), EXTGEN_HOLDOUT)]
        else:
            imgs, src = load_siblings()
            assert imgs is not None, "run gen.sh first (../output/siblings.npz missing)"
            keep = src < SIB_HOLDOUT_SRC
            pools[name] = imgs[keep] if keep.any() else imgs
        print(f"retain anchor pool {name}: {len(pools[name])} images")
    return pools


METHODS = {"identity": method_identity, "distill": method_distill}


def run_name(args):
    if args.method == "identity":
        return "identity"
    parts = [args.method, args.forget_loss]
    parts += [f"w{args.strong_weight:g}K{args.K}", f"tau{args.tau:g}"] if args.forget_loss == "distill" else [f"b{args.beta:g}" + (f"T{args.npo_temp:g}" if args.npo_temp != 1 else "")]
    if args.forget_loss == "distill" and args.sib_weight > 0:
        parts.append(f"sib{args.sib_weight:g}")
    if args.forget_loss == "distill" and args.absent_gamma > 0:
        parts.append(f"abs{args.absent_gamma:g}" + (f"f{args.absent_floor:g}" if args.absent_floor > 0 else ""))
    parts += [f"lam{args.lam:g}" + (f"{args.retain}" if args.lam > 0 and args.retain != "mix" else "")
              + ("unt" if args.lam > 0 and args.anchor_untouched else ""),
              f"ep{args.epochs}", f"lr{args.lr:g}"]
    parts.append(f"pex{args.p_exact:g}")
    if args.train_blocks >= 0:
        parts.append(f"blk{args.train_blocks}")
    if args.train_only != "all":
        parts.append(args.train_only)
    if args.null_proj > 0:
        parts.append(f"null{args.null_proj:g}")
    if args.optim != "adamw":
        parts.append(args.optim)
    if args.pm_weight > 0:
        parts.append(f"pm{args.pm_weight:g}")
    if args.knn_weight > 0:
        parts.append(f"knn{args.knn_weight:g}k{args.knn_k}")
    if args.wn_weight > 0:
        parts.append(f"wn{args.wn_weight:g}s{args.wn_sigma:g}b{args.wn_block}")
    if args.mask < 1:
        parts.append(f"mask{args.mask:g}")
    if args.contrast_mask < 1:
        parts.append(f"cm{args.contrast_mask:g}")
    if args.feat_lam > 0:
        parts.append(f"feat{args.feat_lam:g}")
    if args.freeze_untouched_head:
        parts.append("fh")
    if args.init:
        parts.append("init" + Path(args.init).stem.replace("repair_orig_", "")[:40])
    parts.append(f"s{args.seed}")
    return "_".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="distill", choices=sorted(METHODS))
    ap.add_argument("--forget-loss", default="distill", choices=["distill", "npo"])
    ap.add_argument("--strong-weight", type=float, default=0.5,
                    help="target = (1-w)*medium + w*strong augmentation average (0..1)")
    ap.add_argument("--K", type=int, default=8, help="augmented views averaged for the target")
    ap.add_argument("--tau", type=float, default=1.0, help="target calibration: >1 softer, <1 sharper")
    ap.add_argument("--beta", type=float, default=0.3, help="NPO beta")
    ap.add_argument("--npo-temp", type=float, default=1.0,
                    help="temperature inside NPO (2-3 keeps the gradient alive when the original is ~100% sure)")
    ap.add_argument("--lam", type=float, default=1.0, help="retain weight (0 = forget branch only)")
    ap.add_argument("--feat-lam", type=float, default=0.0,
                    help="anchor on the pre-head CLS embedding (relative MSE vs the original) on the anchor images")
    ap.add_argument("--freeze-untouched-head", action="store_true",
                    help="freeze the classifier rows of the classes absent from the forget set")
    ap.add_argument("--contrast-mask", type=float, default=1.0,
                    help="train only this fraction of weights with the highest |grad forget|/|grad retain| (1 = off)")
    ap.add_argument("--anchor-untouched", action="store_true",
                    help="use as anchors only images the original model assigns to classes absent from the forget set")
    ap.add_argument("--retain", default="mix",
                    help="anchor images: mix | shuffle | noise | external | extgen | siblings | inverted | e.g. inverted+external")
    ap.add_argument("--absent-gamma", type=float, default=0.0,
                    help="class-aware target (0 = off). weight of 'class absent' per class = (n_c/(size-1))^gamma; "
                         "large gamma (e.g. 100) = only fully removed classes, 1 = proportional to removed fraction")
    ap.add_argument("--class-size", type=int, default=130, help="training images per class (max count seen: 130)")
    ap.add_argument("--absent-floor", type=float, default=0.0,
                    help="probability left on the removed class in the 'absent' target (0 = none)")
    ap.add_argument("--sib-weight", type=float, default=0.0,
                    help="blend the target with the sibling average: (1-s)*augmentations + s*siblings (needs gen.sh)")
    ap.add_argument("--mask", type=float, default=1.0, help="SalUn fraction of trainable weights (1 = no mask)")
    ap.add_argument("--p-exact", type=float, default=0.5,
                    help="fraction of forget-branch images shown exactly (not augmented); the server scores exact images")
    ap.add_argument("--train-blocks", type=int, default=-1, help="train only the last N ViT blocks + head (-1 = all)")
    ap.add_argument("--train-only", default="all", choices=["all", "mlp", "linear"],
                    help="mlp = only the MLP layers; linear = only attn/mlp/head Linear weights (pair with --null-proj)")
    ap.add_argument("--wn-weight", type=float, default=0.0, help="blend the target with weight-noise views")
    ap.add_argument("--wn-sigma", type=float, default=0.3, help="noise std relative to each weight tensor's std")
    ap.add_argument("--wn-block", type=int, default=8, help="noise on blocks >= this index")
    ap.add_argument("--wn-K", type=int, default=8)
    ap.add_argument("--null-proj", type=float, default=0.0,
                    help="null-space gradient projection from anchor-image covariance (0 = off; rho, e.g. 0.01-1)")
    ap.add_argument("--null-images", type=int, default=1000, help="anchor images used for the covariance")
    ap.add_argument("--optim", default="adamw", choices=["adamw", "sgd"], help="sgd (momentum 0.9) for --null-proj")
    ap.add_argument("--pm-weight", type=float, default=0.0,
                    help="blend the target with patch-masked views (MAE-style): (1-m)*aug + m*patchmask")
    ap.add_argument("--pm-K", type=int, default=16, help="patch-masked views per image")
    ap.add_argument("--knn-weight", type=float, default=0.0, help="blend the target with the k-NN pool target")
    ap.add_argument("--knn-k", type=int, default=15, help="neighbours (cached by knn_target.py: 5 15 50 150)")
    ap.add_argument("--knn-src", default="genpoolT", help="neighbour pool used by knn_target.py")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--init", type=Path, default=None,
                    help="start from this submission instead of model.pt (teacher/targets/anchors stay model.pt)")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    name = run_name(args)
    print(f"run: {name}\nargs: {vars(args)}")

    t0 = time.time()
    forget = load_forget_images()
    teacher = load_model()
    for p in teacher.parameters():
        p.requires_grad_(False)
    orig_logits = predict(teacher, forget)
    ctx = {"teacher": teacher, "orig_logits": orig_logits, "labels": orig_logits.argmax(1)}
    print(f"Loaded model + {len(forget)} forget images in {time.time() - t0:.0f}s")

    model = load_model(args.init) if args.init else load_model()  # --init: start from a saved submission
    model = METHODS[args.method](model, forget, args, ctx)

    report(model, forget, orig_logits, teacher=teacher, sets=drift_sets())
    save_submission(model, args.out or OUTPUT_DIR / f"sub_{name}.pt")
    print(f"Total time {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
