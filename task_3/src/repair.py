"""Damage-then-repair on in-style generated images (genpoolT = genpool round 1 minus the gpool hold-out + targeted
round 2), never showing the forget images. Repair = distil the ORIGINAL model (KL at T=2, teacher and student see the
same augmented view). Two ways to damage:

  a) --base X --alpha a : start = θ_orig + a·Δ (Δ of a run_unlearn submission, heavy forgetting)
  b) --reinit-from k    : start = original with blocks k..11 re-initialised (LayerScale gammas tiny -> those blocks
                          start as identity), blocks < k + embeddings frozen; repair = mini-retraining of the top
                          blocks without the forget set (closest thing to a gold model without the training data).

Options: --dedup thr drops pool images whose original-model CLS feature has cosine > thr to some forget image;
--balanced samples classes uniformly (class = original model's argmax); --keep w: KL to the damaged start model on
forget images. Diagnostics every --eval-every steps; saves the final model (plus --save-steps).

    python src/repair.py --base ~/runs/sub_...pt --alpha 2.5 --steps 1500 --lr 1e-5
    python src/repair.py --reinit-from 9 --lr-new 3e-4 --lr 1e-5 --steps 4000 --dedup 0.85 --balanced
"""
import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from augment import augment
from common import AMP_DTYPE, DEVICE, OUTPUT_DIR, build_model, load_forget_images, load_model, predict, preprocess, \
    save_submission
from diagnostics import drift_sets, report
from run_unlearn import T, anchor_pools, kl_div


@torch.no_grad()
def cls_feats(model, images, bs=256):
    out = []
    for i in range(0, len(images), bs):
        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=DEVICE == "cuda"):
            f = model.forward_head(model.forward_features(preprocess(images[i:i + bs])), pre_logits=True)
        out.append(F.normalize(f.float(), dim=1))
    return torch.cat(out)


def reinit_block(blk, ls_init):
    for m in blk.modules():
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    for n, p in blk.named_parameters():
        if n.endswith("gamma"):
            p.data.fill_(ls_init)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=None, help="run_unlearn submission giving the forgetting direction Δ")
    ap.add_argument("--alpha", type=float, default=2.5, help="start = θ_orig + alpha·Δ (with --base)")
    ap.add_argument("--reinit-from", type=int, default=-1, help="re-initialise blocks >= k (-1 = none)")
    ap.add_argument("--reinit-head", action="store_true", help="also re-initialise the classifier head")
    ap.add_argument("--pretrained", default=None,
                    help="full student: ALL weights from this public timm model (e.g. deit3_base_patch16_224.fb_in22k_ft_in1k"
                         ", vit_base_patch16_224.augreg2_in21k_ft_in1k), new 100-class head, everything trained at --lr-new; "
                         "needs HF_HUB_CACHE with the weights (compute nodes are offline)")
    ap.add_argument("--ls-init", type=float, default=1e-4, help="LayerScale gamma of re-initialised blocks")
    ap.add_argument("--freeze-below", type=int, default=None,
                    help="freeze embeddings + blocks < k (default: = --reinit-from when re-initialising, else none)")
    ap.add_argument("--retain", default="genpoolT", help="repair images (run_unlearn pool names, '+' combos)")
    ap.add_argument("--dedup", type=float, default=0.0, help="drop pool images with CLS cosine > thr to a forget image")
    ap.add_argument("--balanced", action="store_true", help="uniform over the original model's classes")
    ap.add_argument("--fcls-weight", type=float, default=1.0,
                    help="with --balanced: sampling weight of the forget classes (original argmax of the forget images); "
                         "<1 = learn the untouched classes more than the forget classes")
    ap.add_argument("--full-weight", type=float, default=None,
                    help="sampling weight of the classes removed ENTIRELY (>=129 forget images); default = --fcls-weight")
    ap.add_argument("--aug", default="light", help="light | medium | strong (same view for teacher and student)")
    ap.add_argument("--mse", type=float, default=0.0, help="+ mse * MSE(student logits, teacher logits) (T=1 scale)")
    ap.add_argument("--ce", type=float, default=0.0,
                    help="+ ce * CrossEntropy(student logits, teacher argmax) on pool images where the teacher's max prob "
                         "(T=1) > --ce-thr: hard-label training like a gold model (over-confident, high loss std on unseen)")
    ap.add_argument("--ce-thr", type=float, default=0.7)
    ap.add_argument("--keep", type=float, default=0.0, help="weight of KL to the start model on forget images")
    ap.add_argument("--keep-gamma", type=float, default=1.0,
                    help="--keep target = start model's forget logits * gamma (>1 = sharpen its own never-saw-them "
                         "predictions on the forget set only: gold-like high loss std without a global temperature)")
    ap.add_argument("--init", default=None, help="start from this saved submission instead of the original model")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-5, help="lr of pre-existing weights")
    ap.add_argument("--lr-new", type=float, default=3e-4, help="lr of re-initialised weights")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--eval-every", type=int, default=300)
    ap.add_argument("--save-steps", type=int, nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    if args.base:
        tag = f"{Path(args.base).stem.replace('sub_distill_distill_', '')[:50]}_a{args.alpha:g}"
    else:
        tag = "orig"
    if args.init:
        tag = "init" + Path(args.init).stem.replace("repair_orig_", "").replace("lr1e-05_", "")[:40]
    if args.pretrained:
        tag = f"student_{args.pretrained.split('.')[0].replace('_patch16_224', '')}_lrn{args.lr_new:g}"
    if args.reinit_from >= 0:
        tag += f"_reinit{args.reinit_from}" + ("h" if args.reinit_head else "") + f"_lrn{args.lr_new:g}"
    name = (f"repair_{tag}_{args.retain}_lr{args.lr:g}_st{args.steps}" + (f"_dd{args.dedup:g}" if args.dedup else "")
            + ("_bal" if args.balanced else "") + (f"_fw{args.fcls_weight:g}" if args.fcls_weight != 1 else "") + (f"_full{args.full_weight:g}" if args.full_weight is not None else "") + (f"_{args.aug}" if args.aug != "light" else "")
            + (f"_keep{args.keep:g}" if args.keep > 0 else "") + (f"g{args.keep_gamma:g}" if args.keep_gamma != 1 else "") + (f"_mse{args.mse:g}" if args.mse > 0 else "") + (f"_ce{args.ce:g}t{args.ce_thr:g}" if args.ce > 0 else "") + (f"_seed{args.seed}" if args.seed else ""))
    print(f"run: {name}\nargs: {vars(args)}", flush=True)

    forget = load_forget_images()
    teacher = load_model()
    for p in teacher.parameters():
        p.requires_grad_(False)
    orig_logits = predict(teacher, forget)
    sets = drift_sets()

    # ---- damaged start model
    model = load_model(Path(args.init)) if args.init else load_model()
    if args.base:
        orig = {k: v.detach().float() for k, v in teacher.state_dict().items()}
        base = torch.load(args.base, map_location=DEVICE, weights_only=True)
        model.load_state_dict({k: orig[k] + args.alpha * (base[k].float().to(DEVICE) - orig[k]) for k in orig})
        del base, orig
    new_params = set()
    if args.pretrained:  # zero-memory student: nothing of model.pt is kept
        import timm
        src = timm.create_model(args.pretrained, pretrained=True, num_classes=100).state_dict()
        sd, filled = model.state_dict(), []
        for k, v in sd.items():
            if k in src and src[k].shape == v.shape:
                v.copy_(src[k])
            elif k == "pos_embed" and k in src and src[k].shape[1] == v.shape[1] - 1:  # no_embed_class (DeiT-III)
                v.copy_(torch.cat([torch.zeros_like(v[:, :1]), src[k].to(v)], 1))  # cls gets no position = 0
            elif k.endswith("gamma"):  # checkpoint without LayerScale: identity scale keeps the pretrained function
                v.fill_(1.0)
                filled.append(k)
            else:
                raise KeyError(f"cannot map {k} {tuple(v.shape)} from {args.pretrained}")
        print(f"student from {args.pretrained}: {len(sd)} tensors, {len(filled)} LayerScale gammas set to 1.0")
        new_params = {id(p) for p in model.parameters()}
        args.freeze_below = -1
    if args.reinit_from >= 0:
        for blk in model.blocks[args.reinit_from:]:
            reinit_block(blk, args.ls_init)
            new_params |= {id(p) for p in blk.parameters()}
        if args.reinit_head:
            nn.init.trunc_normal_(model.head.weight, std=0.02)
            nn.init.zeros_(model.head.bias)
            new_params |= {id(model.head.weight), id(model.head.bias)}
    fb = args.freeze_below if args.freeze_below is not None else (args.reinit_from if args.reinit_from >= 0 else -1)
    if fb >= 0:
        for p in list(model.patch_embed.parameters()) + [model.cls_token, model.pos_embed]:
            p.requires_grad_(False)
        for blk in model.blocks[:fb]:
            for p in blk.parameters():
                p.requires_grad_(False)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable {sum(p.numel() for p in trainable) / 1e6:.1f}M params "
          f"({sum(p.numel() for p in trainable if id(p) in new_params) / 1e6:.1f}M re-initialised)")
    print("\n===== start")
    report(model.eval(), forget, orig_logits, teacher=teacher, sets=sets)

    # ---- repair pool
    pools = anchor_pools(argparse.Namespace(retain=args.retain))
    pool = torch.cat(list(pools.values()))
    del pools
    if args.dedup > 0:
        ff = cls_feats(teacher, forget)
        best = torch.cat([(cls_feats(teacher, pool[i:i + 4096]) @ ff.T).max(1).values.cpu()
                          for i in range(0, len(pool), 4096)])
        keep = best <= args.dedup
        print(f"dedup: CLS cosine to nearest forget image, quantiles 50/90/99%: "
              f"{' '.join(f'{v:.3f}' for v in torch.quantile(best, torch.tensor([.5, .9, .99])).tolist())}; "
              f"dropping {int((~keep).sum())}/{len(pool)} with cosine > {args.dedup:g}")
        pool = pool[keep]
    print(f"repair pool: {len(pool)} images")
    assert len(pool) >= 1000, "repair pool too small (dedup threshold too low?)"
    cls_idx = None
    if args.balanced:
        lab = predict(teacher, pool).argmax(1)
        fcount = torch.bincount(orig_logits.argmax(1).cpu(), minlength=int(lab.max()) + 1)
        full_w = args.fcls_weight if args.full_weight is None else args.full_weight
        cls_ids = [c for c in range(int(lab.max()) + 1) if (lab == c).any()]
        cls_idx = [(lab == c).nonzero().flatten() for c in cls_ids]
        cls_w = torch.tensor([1.0 if fcount[c] == 0 else (full_w if fcount[c] >= 129 else args.fcls_weight)
                              for c in cls_ids]).clamp_min(1e-6)
        print(f"balanced sampling over {len(cls_idx)} classes (min {min(len(ix) for ix in cls_idx)} images); "
              f"{sum(int(fcount[c] > 0) for c in cls_ids)} forget classes at weight {args.fcls_weight:g}, "
              f"{sum(int(fcount[c] >= 129) for c in cls_ids)} of them removed entirely at weight {full_w:g}")

    keep_T = F.softmax(predict(model, forget).to(DEVICE) * args.keep_gamma / T, 1) if args.keep > 0 else None
    groups = [{"params": [p for p in trainable if id(p) not in new_params], "lr": args.lr},
              {"params": [p for p in trainable if id(p) in new_params], "lr": args.lr_new}]
    opt = torch.optim.AdamW([g for g in groups if g["params"]], weight_decay=0.0)
    warmup = max(1, args.steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps)))
    model.train()
    t0 = time.time()
    for step in range(1, args.steps + 1):
        if cls_idx is None:
            idx = torch.randint(0, len(pool), (args.batch_size,))
        else:
            cs = torch.multinomial(cls_w, args.batch_size, replacement=True).tolist()
            idx = torch.stack([cls_idx[c][torch.randint(0, len(cls_idx[c]), ())] for c in cs])
        xm = augment(preprocess(pool[idx]), args.aug)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=DEVICE == "cuda"):
            t_logits = teacher(xm).float()
            t_p = F.softmax(t_logits / T, 1)
        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=DEVICE == "cuda"):
            s_logits = model(xm).float()
            loss = kl_div(t_p, s_logits)
            if args.mse > 0:
                loss = loss + args.mse * F.mse_loss(s_logits, t_logits)
            if args.ce > 0:
                conf, hard = F.softmax(t_logits, 1).max(1)
                sel = conf > args.ce_thr
                if sel.any():
                    loss = loss + args.ce * F.cross_entropy(s_logits[sel], hard[sel])
            if keep_T is not None:
                fi = torch.randint(0, len(forget), (args.batch_size // 2,))
                lf = model(augment(preprocess(forget[fi]), args.aug, p_skip=0.5)).float()
                loss = loss + args.keep * kl_div(keep_T[fi.to(DEVICE)], lf)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        sched.step()
        if step % 50 == 0:
            print(f"step {step}/{args.steps}  loss {loss.item():.4f}  {time.time() - t0:.0f}s", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            print(f"\n===== repair step {step}")
            report(model.eval(), forget, orig_logits, teacher=teacher, sets=sets)
            if step in args.save_steps or step == args.steps:
                save_submission(model, OUTPUT_DIR / f"{name}_s{step}.pt")
            model.train()


if __name__ == "__main__":
    main()
