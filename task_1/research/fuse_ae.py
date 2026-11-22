"""Second-stage fusion: ensemble logits (build_sub_v2 --save-logits) + autoencoder reconstruction-error maps
(research.vae_probe), one logistic model per method on per-cell features, with an honest 5-fold out-of-fold estimate
on train (fitted on 4 folds of out-of-fold logits, applied to the 5th). The rectangle rule is applied after fusion,
exactly as in build_sub_v2. Small edits are the target: the AE map can show a small inpainted/pasted region that the
watermark signal alone misses, so the features include local-anomaly contrasts (fine blur minus wide blur) and their
product with the image polarity (marked background or not).

    python -m research.fuse_ae --aes flux sd sdxl --rect pixelseal=outputs/rect/pixelseal.json:0.85:0.5 \
        --rect trustmark=outputs/rect/trustmark.json:0.99:0.5 [--name sub11 to write a submission]"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from research.analyze_cache import miou
from research.build_sub_v2 import apply_rect

METHODS = ("pixelseal", "trustmark", "stegastamp")


def blur(x: torch.Tensor, k: int) -> torch.Tensor:
    return F.avg_pool2d(F.pad(x, (k // 2,) * 4, mode="replicate"), k, stride=1)


def standardize(x: torch.Tensor) -> torch.Tensor:   # per image: (x - median) / IQR
    flat = x.flatten(1)
    q = torch.quantile(flat, torch.tensor([0.25, 0.5, 0.75]), dim=1)
    return (x - q[1].view(-1, 1, 1, 1)) / (q[2] - q[0]).clamp_min(1e-6).view(-1, 1, 1, 1)


def features(logit: np.ndarray, maps: list[tuple[np.ndarray, np.ndarray]]) -> torch.Tensor:
    """logit: N x 128 x 128 ensemble logits; maps: per AE (err, hf), N x 128 x 128. -> N x C x 128 x 128."""
    L = torch.from_numpy(logit.astype(np.float32)).clamp(-12, 12)[:, None] / 6
    s = (torch.sigmoid(L * 6).mean((2, 3), keepdim=True) > 0.5).float() * 2 - 1   # +1: background watermarked
    feats = [L, s.expand_as(L)]
    for err, hf in maps:
        e = standardize(torch.log(torch.from_numpy(err.astype(np.float32))[:, None] + 1e-4))
        h = standardize(torch.log(torch.from_numpy(hf.astype(np.float32))[:, None] + 1e-4))
        r = e - h
        for x in (e, r):
            b3, b9, b31 = blur(x, 3), blur(x, 9), blur(x, 31)
            for f in (b3, b3 - b31, b9 - b31):
                feats += [f, f * s]
        feats += [h, h * s]
    return torch.cat(feats, 1)


def fit_logistic(X: torch.Tensor, y: torch.Tensor, l2: float = 1e-3, iters: int = 200) -> torch.Tensor:
    """X: M x C (float32), y: M in {0,1}. Returns weights C+1 (last = bias). Full-batch L-BFGS."""
    mu, sd = X.mean(0), X.std(0).clamp_min(1e-6)
    Xs = (X - mu) / sd
    w = torch.zeros(X.shape[1] + 1, requires_grad=True)
    opt = torch.optim.LBFGS([w], lr=1, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(Xs @ w[:-1] + w[-1], y) + l2 * (w[:-1] ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    w = w.detach()
    return torch.cat([w[:-1] / sd, (w[-1] - (w[:-1] * mu / sd).sum())[None]])


def apply(w: torch.Tensor, F_: torch.Tensor) -> torch.Tensor:
    return (F_.permute(0, 2, 3, 1) @ w[:-1] + w[-1])   # N x 128 x 128 logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logits", default="outputs/logits"); ap.add_argument("--vae-dir", default="outputs/vae")
    ap.add_argument("--aes", nargs="+", default=["flux"]); ap.add_argument("--rect", action="append", default=[])
    ap.add_argument("--subsample", type=int, default=4, help="use every k-th cell for fitting (speed)")
    ap.add_argument("--name"); ap.add_argument("--data", default="release")
    a = ap.parse_args()
    rules = {}
    for spec in a.rect:
        m, rest = spec.split("=", 1); path, tau, agree = rest.rsplit(":", 2)
        rows = {"train": {}, "test": {}}
        for r in json.loads(Path(path).read_text())["rows"]:
            rows[r["split"]][r["name"]] = r
        rules[m] = {"tau": float(tau), "agree": float(agree), "rows": rows}
    aem = {}
    for ae in a.aes:
        z = np.load(Path(a.vae_dir) / f"{ae}.npz")
        aem[ae] = {n: i for i, n in enumerate(z["names"].tolist())}, z["err"], z["hf"]
    names_all, masks_all, report = [], [], {}
    for method in METHODS:
        z = np.load(Path(a.logits) / f"{method}.npz")
        ntr, nte = z["names_train"].tolist(), z["names_test"].tolist()
        maps_tr = [(aem[ae][1][[aem[ae][0][n] for n in ntr]], aem[ae][2][[aem[ae][0][n] for n in ntr]]) for ae in a.aes]
        maps_te = [(aem[ae][1][[aem[ae][0][n] for n in nte]], aem[ae][2][[aem[ae][0][n] for n in nte]]) for ae in a.aes]
        Ftr, Fte = features(z["oof_logit"], maps_tr), features(z["test_logit"], maps_te)
        gt, folds = z["gt"].astype(np.uint8), z["folds"]
        y = torch.from_numpy(gt.astype(np.float32))
        sub = slice(None, None, a.subsample)
        oof = np.zeros(gt.shape, np.float32)
        for k in sorted(set(folds.tolist())):
            tr = np.nonzero(folds != k)[0]; va = np.nonzero(folds == k)[0]
            X = Ftr[tr][:, :, sub, sub].permute(0, 2, 3, 1).reshape(-1, Ftr.shape[1])
            w = fit_logistic(X, y[tr][:, sub, sub].reshape(-1))
            oof[va] = apply(w, Ftr[va]).numpy()
        base = (z["oof_logit"].astype(np.float32) > 0).astype(np.uint8)
        fused = (oof > 0).astype(np.uint8)
        base, _ = apply_rect(base, ntr, rules.get(method), "train")
        fused, _ = apply_rect(fused, ntr, rules.get(method), "train")
        sb = np.array([miou(p, g) for p, g in zip(base, gt)]); sf = np.array([miou(p, g) for p, g in zip(fused, gt)])
        small = np.array([min(g.mean(), 1 - g.mean()) < 0.05 for g in gt])
        report[method] = {"oof_base": round(float(sb.mean()), 4), "oof_fused": round(float(sf.mean()), 4),
                          "small_base": round(float(sb[small].mean()), 4), "small_fused": round(float(sf[small].mean()), 4),
                          "per_fold_delta": {int(k): round(float((sf - sb)[folds == k].mean()), 4) for k in sorted(set(folds.tolist()))}}
        print(method, json.dumps(report[method]), flush=True)
        X = Ftr[:, :, sub, sub].permute(0, 2, 3, 1).reshape(-1, Ftr.shape[1])
        w = fit_logistic(X, y[:, sub, sub].reshape(-1))
        pred = (apply(w, Fte).numpy() > 0).astype(np.uint8)
        pred, _ = apply_rect(pred, nte, rules.get(method), "test")
        names_all += nte; masks_all.append(pred)
    tot_b = np.mean([report[m]["oof_base"] for m in METHODS]); tot_f = np.mean([report[m]["oof_fused"] for m in METHODS])
    print(f"OOF mean base {tot_b:.4f} fused {tot_f:.4f}", flush=True)
    if not a.name:
        return
    from research import submission
    out = Path("outputs/submissions") / f"{a.name}.pt"
    manifest = {"name": a.name, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "builder": "research.fuse_ae", "aes": a.aes, "oof_mean": round(float(tot_f), 4), "methods": report}
    manifest["file"] = submission.write(names_all, np.concatenate(masks_all), out, a.data)
    manifest["file"]["sha256"] = hashlib.sha256(out.read_bytes()).hexdigest()
    (Path("research/submissions") / f"{a.name}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["file"], indent=2))


if __name__ == "__main__":
    main()
