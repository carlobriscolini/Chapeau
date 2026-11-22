"""k-NN "never saw it" target: for every forget image, the original model's mean prediction on its k nearest
generated pool images (cosine of the original model's CLS features). The original never saw those pool images, so
memorisation of the forget image itself does not leak into the target: easy/prototypical images get a confident
correct target, outliers get a spread or wrong one (-> bimodal loss, high std, like a retrained gold model).

Computes and caches OUTPUT_DIR/targets_knn_{src}.pt ({k: [N,100] probs at T=1}), then prints what a model predicting
each target would score (acc w.r.t. original label, loss mean/std, KL to original at T=2), next to the record target.
Use in training:  run_unlearn.py ... --knn-weight 0.5 --knn-k 15   (blend with the augmentation target, before tau)

    python experiments/knn_target.py                    # login GPU, ~2 min
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # shared modules (common, diagnostics, ...)

import argparse

import torch
import torch.nn.functional as F

from common import AMP_DTYPE, DEVICE, OUTPUT_DIR, TEMPERATURE, load_forget_images, load_model, predict, preprocess
from run_unlearn import anchor_pools
from targets import apply_tau, blended_pbar, describe, knn_cache_path, load_or_compute

KS = (5, 15, 50, 150)


@torch.no_grad()
def feats_logits(model, images, bs=256):
    fs, ls = [], []
    for i in range(0, len(images), bs):
        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=DEVICE == "cuda"):
            f = model.forward_head(model.forward_features(preprocess(images[i:i + bs])), pre_logits=True)
            lg = model.head(f)
        fs.append(F.normalize(f.float(), dim=1))
        ls.append(lg.float())
    return torch.cat(fs), torch.cat(ls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="genpoolT", help="neighbour pool (run_unlearn pool names, '+' combos)")
    ap.add_argument("--max-cos", type=float, default=0.95, help="ignore near-duplicates above this cosine")
    args = ap.parse_args()
    teacher = load_model()
    forget = load_forget_images()
    orig_logits = predict(teacher, forget)
    labels = orig_logits.argmax(1)
    p_orig_T = F.softmax(orig_logits / TEMPERATURE, 1)

    path = knn_cache_path(args.src)
    if path.exists():
        knn = torch.load(path)
        print(f"loaded {path}")
    else:
        pool = torch.cat(list(anchor_pools(argparse.Namespace(retain=args.src)).values()))
        ff, _ = feats_logits(teacher, forget)
        sims, idxs = [], []
        pf_all, pl_all = feats_logits(teacher, pool)
        pp = F.softmax(pl_all, 1)
        for i in range(0, len(ff), 500):
            s = ff[i:i + 500] @ pf_all.T
            s[s > args.max_cos] = -1
            v, ix = s.topk(max(KS), dim=1)
            sims.append(v.cpu())
            idxs.append(ix)
        sims, idxs = torch.cat(sims), torch.cat(idxs)
        print(f"pool {len(pool)} images; nearest-neighbour cosine quantiles 10/50/90%: "
              f"{' '.join(f'{x:.3f}' for x in torch.quantile(sims[:, 0], torch.tensor([.1, .5, .9])).tolist())}; "
              f"k=15th: {torch.quantile(sims[:, 14], 0.5):.3f}")
        knn = {k: pp[idxs[:, :k]].mean(1).cpu() for k in KS}
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(knn, path)
        print(f"saved {path}")

    def show(name, q1):
        d = describe(q1, labels, p_orig_T)
        print(f"{name:34s} acc {d['acc']:.3f}  loss mean {d['loss_mean']:.3f}  std {d['loss_std']:.3f}  "
              f"KL(q||orig) {d['kl_q_orig']:.3f}  max prob {d['max_prob']:.3f}")

    rec = 0.5 * blended_pbar(teacher, forget, 0.5, 8) + 0.5 * load_or_compute(teacher, forget, "patchmask", 16)
    show("record target (tau 1.5)", apply_tau(rec, 1.5))
    for k in KS:
        for tau in (1.0, 1.5):
            show(f"knn k{k} tau{tau:g}", apply_tau(knn[k], tau))
    for w in (0.5, 0.75):
        for k in (15, 50):
            show(f"{w:g}*knn k{k} + {1 - w:g}*record, tau1.5", apply_tau(w * knn[k] + (1 - w) * rec, 1.5))
    q = knn[15]
    easy = q.argmax(1) == labels
    print(f"knn k15 agrees with the original's label on {easy.float().mean():.1%} of the forget images")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
