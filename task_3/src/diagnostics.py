"""Local diagnostics. We have NO labels and NO gold model, so these are proxies only:
  - "label" = argmax of the original model (it was trained on these images, so this is ~ the true label)
  - loss mean/std w.r.t. that label (the server compares this to the gold models' loss distribution)
  - agreement with the original prediction, mean KL(p_orig || p_sub) at T=2
computed on the exact forget images and on simple altered copies (the server also scores hidden altered copies),
plus drift checks KL(p_orig || p_sub) on images never used for training (proxies of the server's control set):
  retain_proxy/mix  mixup of forget images
  retain_proxy/ext  held-out external photos (mini-ImageNet)
  retain_proxy/gen  held-out photos re-rendered by the generator (look like the training data)
  retain_proxy/sib  held-out siblings (generated variations of forget images 3000..3499)
"est. retain_kl_similarity" = 1 / (1 + KL), the server's formula. The real score only comes from the leaderboard.
"""
import torch
import torch.nn.functional as F

from common import (DEVICE, EXTERNAL_HOLDOUT, EXTGEN_HOLDOUT, GENPOOL_HOLDOUT, INVERTED_HOLDOUT, SIB_HOLDOUT_SRC,
                    TEMPERATURE, holdout_size, kl, load_external_images, load_extgen, load_genpool, load_inverted,
                    load_siblings, predict, preprocess, soft_probs)


def altered(images_uint8: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """Cheap stand-in for the hidden altered copies: random flip + small shift + brightness jitter."""
    g = torch.Generator().manual_seed(seed)
    x = images_uint8.clone()
    flip = torch.rand(len(x), generator=g) < 0.5
    x[flip] = x[flip].flip(2)
    dx, dy = torch.randint(-8, 9, (2,), generator=g).tolist()
    x = torch.roll(x, shifts=(dy, dx), dims=(1, 2))
    scale = 1 + (torch.rand(len(x), 1, 1, 1, generator=g) - 0.5) * 0.2
    return (x.float() * scale).clamp(0, 255).to(torch.uint8)


def _stats(logits: torch.Tensor, ref_logits: torch.Tensor) -> dict:
    labels = ref_logits.argmax(1)
    loss = F.cross_entropy(logits, labels, reduction="none")
    return {
        "agree": (logits.argmax(1) == labels).float().mean().item(),
        "loss_mean": loss.mean().item(),
        "loss_std": loss.std().item(),
        "kl_orig": kl(soft_probs(ref_logits), soft_probs(logits)).mean().item(),
    }


@torch.no_grad()
def _drift_samples(model, teacher, batches):
    """Per-image (agree, KL(orig||sub) at T=2, original argmax) on a stream of preprocessed batches."""
    kls, agree, tcls = [], [], []
    for x in batches:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
            t, s = teacher(x).float(), model(x).float()
        kls.append(kl(F.softmax(t / TEMPERATURE, 1), F.softmax(s / TEMPERATURE, 1)).cpu())
        agree.append((t.argmax(1) == s.argmax(1)).float().cpu())
        tcls.append(t.argmax(1).cpu())
    return torch.cat(agree), torch.cat(kls), torch.cat(tcls)


def _drift_batches(model, teacher, batches):
    agree, kls, _ = _drift_samples(model, teacher, batches)
    return agree.mean().item(), kls.mean().item()


def _mixed_kl(model, teacher, forget_images, n=512, batch_size=128):
    from augment import mix
    torch.manual_seed(1234)
    return _drift_batches(model, teacher, (mix(preprocess(forget_images[i:i + batch_size]))
                                           for i in range(0, min(n, len(forget_images)), batch_size)))


def _drift(model, teacher, images_uint8, batch_size=128):
    return _drift_batches(model, teacher, (preprocess(images_uint8[i:i + batch_size])
                                           for i in range(0, len(images_uint8), batch_size)))


def drift_sets():
    """Held-out image sets (never used as retain anchors) for drift checks: name -> uint8 images."""
    sets = {}
    ext = load_external_images()
    if ext is not None and holdout_size(len(ext), EXTERNAL_HOLDOUT) > 0:
        sets["ext"] = ext[len(ext) - holdout_size(len(ext), EXTERNAL_HOLDOUT):]
    gen = load_extgen()
    if gen is not None and holdout_size(len(gen), EXTGEN_HOLDOUT) > 0:
        sets["gen"] = gen[len(gen) - holdout_size(len(gen), EXTGEN_HOLDOUT):]
    inv = load_inverted()
    if inv is not None and holdout_size(len(inv), INVERTED_HOLDOUT) > 0:
        sets["inv"] = inv[len(inv) - holdout_size(len(inv), INVERTED_HOLDOUT):]
    gp, _ = load_genpool()
    if gp is not None and holdout_size(len(gp), GENPOOL_HOLDOUT) > 0:
        sets["gpool"] = gp[len(gp) - holdout_size(len(gp), GENPOOL_HOLDOUT):]
    sib, src = load_siblings()
    if sib is not None and (src >= SIB_HOLDOUT_SRC).any():
        sets["sib"] = sib[src >= SIB_HOLDOUT_SRC][:2000]
    return sets


def report(model, forget_images, orig_logits, teacher=None, n_altered: int = 1000, sets=None):
    model.eval()
    sub_logits = predict(model, forget_images)
    rows = {"forget/exact": _stats(sub_logits, orig_logits)}
    idx = torch.arange(min(n_altered, len(forget_images)))
    alt = altered(forget_images[idx])
    rows["forget/altered"] = _stats(predict(model, alt), orig_logits[idx])
    labels = orig_logits.argmax(1)
    counts = torch.bincount(labels, minlength=orig_logits.shape[1])
    full = counts[labels] >= 129  # images of classes removed entirely (129-130 forget images)
    if full.any() and (~full).any():
        rows["exact/full-class"] = _stats(sub_logits[full], orig_logits[full])
        rows["exact/partial"] = _stats(sub_logits[~full], orig_logits[~full])
    print(f"\n{'set':16s} {'agree_orig':>10s} {'loss_mean':>10s} {'loss_std':>10s} {'KL(orig||sub)':>14s}")
    for name, s in rows.items():
        print(f"{name:16s} {s['agree']:10.3f} {s['loss_mean']:10.3f} {s['loss_std']:10.3f} {s['kl_orig']:14.4f}")
    if teacher is not None:
        agree, kl_m = _mixed_kl(model, teacher, forget_images)
        rows["retain_proxy/mix"] = {"agree": agree, "kl_orig": kl_m}
        print(f"{'retain_proxy/mix':16s} {agree:10.3f} {'':>10s} {'':>10s} {kl_m:14.4f}")
        for name, images in (sets or {}).items():
            ag, kls, tcls = _drift_samples(model, teacher, (preprocess(images[i:i + 128])
                                                            for i in range(0, len(images), 128)))
            agree, kl_s = ag.mean().item(), kls.mean().item()
            rows[f"retain_proxy/{name}"] = {"agree": agree, "kl_orig": kl_s}
            print(f"{'retain_proxy/' + name:16s} {agree:10.3f} {'':>10s} {'':>10s} {kl_s:14.4f}"
                  f"   -> est. retain_kl_similarity {1 / (1 + kl_s):.3f}")
            # split by the ORIGINAL model's class on that image: untouched classes (no forget images) should not move
            untouched = counts[tcls] == 0
            for tag, m in (("untouched", untouched), ("forgetcls", ~untouched)):
                if m.any():
                    print(f"{name + '/' + tag:16s} {ag[m].mean().item():10.3f} {'':>10s} {'':>10s} "
                          f"{kls[m].mean().item():14.4f}   ({int(m.sum())} images)")
    return rows
