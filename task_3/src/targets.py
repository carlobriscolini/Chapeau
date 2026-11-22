"""Step 2: estimate the "never saw it" target q(x) for every forget image, and inspect what it implies.

q(x) = normalise( pbar(x) ** (1 / tau) ),  pbar(x) = mean over K augmented views of softmax(orig(view))
(probabilities at temperature 1). tau > 1 softens, tau < 1 sharpens.

Usage (login GPU is fine, ~1 min once the targets are cached):
    python src/targets.py
Prints, per augmentation level and tau, what a model whose predictions were q would score on the forget set:
accuracy w.r.t. the original's label, loss mean / std, and KL(q || p_orig) at T=2.
If gen.sh was run, also: the original model on the generated siblings (new images that look like the forget images:
a rough estimate of how a model that never saw them behaves), the sibling-based target, and how confident the
original model is on re-rendered photos (extgen) vs plain photos.
"""
import argparse
import os

import torch
import torch.nn.functional as F

from augment import augment, patch_mask
from common import (AMP_DTYPE, DEVICE, OUTPUT_DIR, SIBLINGS_FILE, TEMPERATURE, load_external_images, load_extgen,
                    load_forget_images, load_model, load_siblings, predict, preprocess)


@torch.no_grad()
def compute_pbar(teacher, forget_uint8, level="strong", K=16, batch_size=125, seed=0):
    """Mean softmax (T=1) of the frozen original model over K augmented views of each image. -> [N,100] cpu."""
    torch.manual_seed(seed)
    teacher.eval()
    out = []
    for i in range(0, len(forget_uint8), batch_size):
        x = preprocess(forget_uint8[i:i + batch_size])
        acc = torch.zeros(len(x), 100, device=DEVICE)
        for _ in range(K):
            with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=DEVICE == "cuda"):
                logits = teacher(patch_mask(x) if level == "patchmask" else augment(x, level))
            acc += F.softmax(logits.float(), dim=1)
        out.append((acc / K).cpu())
    return torch.cat(out)


def compute_pbar_wnoise(teacher, forget_uint8, sigma, first_block=8, K=8, batch_size=125, seed=0):
    """Mean softmax (T=1) over K copies of the original model whose blocks >= first_block get Gaussian weight noise
    (std = sigma * std of each tensor), on the clean images. Breaks instance memorisation in the top blocks while
    the lower blocks keep the class evidence. -> [N,100] cpu."""
    import copy
    g = torch.Generator().manual_seed(seed)
    acc = torch.zeros(len(forget_uint8), 100)
    for _ in range(K):
        m = copy.deepcopy(teacher).eval()
        with torch.no_grad():
            for n, p in m.named_parameters():
                if n.startswith("blocks.") and int(n.split(".")[1]) >= first_block and p.dim() > 1:
                    p.add_(torch.randn(p.shape, generator=g).to(p.device) * sigma * p.std())
        for i in range(0, len(forget_uint8), batch_size):
            x = preprocess(forget_uint8[i:i + batch_size])
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=DEVICE == "cuda"):
                acc[i:i + batch_size] += F.softmax(m(x).float(), dim=1).cpu()
        del m
    return acc / K


def apply_tau(pbar, tau):
    """q1 ∝ pbar^(1/tau)  (temperature-1 probabilities)."""
    logq = pbar.clamp_min(1e-8).log() / tau
    return F.softmax(logq, dim=1)


def to_T(q1, T=TEMPERATURE):
    """Distribution a model with T=1 probs q1 shows at temperature T: softmax(log q1 / T)."""
    return F.softmax(q1.clamp_min(1e-8).log() / T, dim=1)


def load_or_compute(teacher, forget, level, K, seed=0):
    path = OUTPUT_DIR / f"targets_{level}_K{K}_s{seed}.pt"
    if path.exists():
        return torch.load(path)
    if level.startswith("wnoise"):  # e.g. "wnoise0.3b8" = sigma 0.3 on blocks >= 8
        sig, fb = level[len("wnoise"):].split("b")
        pbar = compute_pbar_wnoise(teacher, forget, float(sig), int(fb), K, seed=seed)
    else:
        pbar = compute_pbar(teacher, forget, level, K, seed=seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    torch.save(pbar, tmp)
    os.replace(tmp, path)
    print(f"saved {path}")
    return pbar


def blended_pbar(teacher, forget, strong_weight, K, seed=0):
    """(1-w) * pbar_medium + w * pbar_strong."""
    if strong_weight >= 1:
        return load_or_compute(teacher, forget, "strong", K, seed)
    if strong_weight <= 0:
        return load_or_compute(teacher, forget, "medium", K, seed)
    pm = load_or_compute(teacher, forget, "medium", K, seed)
    ps = load_or_compute(teacher, forget, "strong", K, seed)
    return (1 - strong_weight) * pm + strong_weight * ps


def _sib_cache_path(n_sib):
    return OUTPUT_DIR / f"targets_sib_{n_sib}_{int(SIBLINGS_FILE.stat().st_mtime)}.pt"


def _sib_pbar_from_logits(sib_logits, src, n):
    probs = torch.zeros(n, sib_logits.shape[1])
    counts = torch.zeros(n)
    probs.index_add_(0, src, F.softmax(sib_logits.float(), dim=1))
    counts.index_add_(0, src, torch.ones(len(src)))
    assert (counts > 0).all(), "some forget images have no siblings"
    return probs / counts[:, None]


def _save_atomic(obj, path):
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_or_compute_sib(teacher, n):
    """Mean softmax (T=1) of the original model over the generated siblings of each forget image -> [N,100] cpu."""
    sib, src = load_siblings()
    assert sib is not None, "run gen.sh first (../output/siblings.npz missing)"
    path = _sib_cache_path(len(sib))
    if path.exists():
        return torch.load(path)
    pbar = _sib_pbar_from_logits(predict(teacher, sib), src, n)
    _save_atomic(pbar, path)
    print(f"saved {path}")
    return pbar


def knn_cache_path(src="genpoolT"):
    return OUTPUT_DIR / f"targets_knn_{src.replace('+', '_')}.pt"


def load_knn(k, src="genpoolT"):
    """k-NN target computed by knn_target.py: [N,100] probabilities at T=1."""
    path = knn_cache_path(src)
    assert path.exists(), f"{path} missing: run knn_target.py --src {src} first"
    knn = torch.load(path)
    assert k in knn, f"k={k} not cached (available: {sorted(knn)})"
    return knn[k]


def describe(q1, labels, p_orig_T):
    loss = -q1.clamp_min(1e-12).log().gather(1, labels[:, None]).squeeze(1)  # CE a model predicting q1 would get
    qT = to_T(q1)
    kl = (qT * (qT.clamp_min(1e-12).log() - p_orig_T.clamp_min(1e-12).log())).sum(1)
    return {
        "acc": (q1.argmax(1) == labels).float().mean().item(),
        "loss_mean": loss.mean().item(),
        "loss_std": loss.std().item(),
        "max_prob": q1.max(1).values.mean().item(),
        "kl_q_orig": kl.mean().item(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0],
                    help="strong weight w: target = (1-w)*medium + w*strong")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--taus", nargs="+", type=float, default=[0.5, 0.75, 1.0, 1.5, 2.0])
    args = ap.parse_args()

    forget = load_forget_images()
    teacher = load_model()
    orig_logits = predict(teacher, forget)
    labels = orig_logits.argmax(1)
    p_orig_T = F.softmax(orig_logits / TEMPERATURE, dim=1)
    print(f"original model on exact forget images: max prob {F.softmax(orig_logits, 1).max(1).values.mean():.4f}")

    print(f"\n{'w_strong':8s} {'tau':>5s} {'acc':>6s} {'loss_mean':>10s} {'loss_std':>9s} {'max_prob':>9s} {'KL(q||orig)':>12s}")
    for w in args.weights:
        pbar = blended_pbar(teacher, forget, w, args.K)
        level = f"{w:.2f}"
        for tau in args.taus:
            d = describe(apply_tau(pbar, tau), labels, p_orig_T)
            print(f"{level:8s} {tau:5.2f} {d['acc']:6.3f} {d['loss_mean']:10.3f} {d['loss_std']:9.3f} "
                  f"{d['max_prob']:9.3f} {d['kl_q_orig']:12.4f}")
        # which wrong class does the target pick, vs the original's 2nd choice?
        q1 = apply_tau(pbar, 1.0)
        wrong = q1.argmax(1) != labels
        second = orig_logits.topk(2, dim=1).indices[:, 1]
        if wrong.any():
            same = (q1.argmax(1)[wrong] == second[wrong]).float().mean().item()
            print(f"{level:8s} wrong-target images: {int(wrong.sum())}, of which = original's 2nd choice: {same:.2f}")

    sib, src = load_siblings()
    if sib is not None:
        sib_logits = predict(teacher, sib)
        lab = labels[src]
        loss = F.cross_entropy(sib_logits, lab, reduction="none")
        p1 = F.softmax(sib_logits, 1)
        print(f"\noriginal model on {len(sib)} siblings (generated variations, never seen):")
        print(f"   agreement with the forget image's label {(sib_logits.argmax(1) == lab).float().mean():.3f}   "
              f"loss mean {loss.mean():.3f}  std {loss.std():.3f}   max prob {p1.max(1).values.mean():.3f}")
        print("   -> rough estimate of how a model that never saw the forget images behaves on them")
        path = _sib_cache_path(len(sib))
        ps = _sib_pbar_from_logits(sib_logits, src, len(forget))
        if not path.exists():
            _save_atomic(ps, path)
        base = blended_pbar(teacher, forget, 0.5, args.K)
        print(f"\n{'sib_w':8s} {'tau':>5s} {'acc':>6s} {'loss_mean':>10s} {'loss_std':>9s} {'max_prob':>9s} "
              f"{'KL(q||orig)':>12s}   (target = (1-sib_w)*aug(w=0.5) + sib_w*siblings)")
        for s in (0.5, 1.0):
            for tau in args.taus:
                d = describe(apply_tau((1 - s) * base + s * ps, tau), labels, p_orig_T)
                print(f"{s:<8.2f} {tau:5.2f} {d['acc']:6.3f} {d['loss_mean']:10.3f} {d['loss_std']:9.3f} "
                      f"{d['max_prob']:9.3f} {d['kl_q_orig']:12.4f}")

    print()
    for name, imgs in (("re-rendered photos (extgen)", load_extgen()), ("plain photos (external)", load_external_images())):
        if imgs is not None:
            p = F.softmax(predict(teacher, imgs[:1000]), 1)
            ent = -(p * p.clamp_min(1e-12).log()).sum(1).mean()
            print(f"original model on 1000 {name}: max prob {p.max(1).values.mean():.3f}, entropy {ent:.3f}")


if __name__ == "__main__":
    main()
