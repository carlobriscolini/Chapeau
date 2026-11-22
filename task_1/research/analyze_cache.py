"""CPU analysis of research.cache_logits outputs: where does the ensemble lose points, and what is the ceiling of a
perfect rectangle for clean-background rectangular (outpainting) images?

Out-of-fold protocol: a checkpoint trained with --val-fold k never saw fold k, so for each fold we average only the
members whose held-out fold is k (VAL_FOLD below) and score that fold. Folds without such members are skipped.

    python -m research.analyze_cache outputs/cache/*.npz [--rect outputs/rect/<method>.json ...]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# held-out fold of each cached checkpoint (from log.json args / avert1's sbatch commands)
VAL_FOLD = {"f1_wm": 1, "f1_r50": 1, "f2cut_wm": 2, "f2cut_r50": 2}
VAL_FOLD_BY_METHOD: dict = {}   # 22/11: trustmark f2cut_wm is val fold 2 (per-fold train mIoU 0.65 on f2, ~0.99 elsewhere), not 3


def miou(p: np.ndarray, g: np.ndarray) -> float:
    s = 0.0
    for c in (0, 1):
        u = np.logical_or(p == c, g == c).sum()
        s += 1.0 if u == 0 else np.logical_and(p == c, g == c).sum() / u
    return s / 2


def is_rect(g: np.ndarray) -> bool:
    if not g.any():
        return False
    ys, xs = np.nonzero(g)
    return bool(g[ys.min():ys.max() + 1, xs.min():xs.max() + 1].all())


def category(gt: np.ndarray, full: bool) -> str:
    frac = gt.mean() if not full else 1 - gt.mean()
    size = "small" if frac < 0.05 else "large"
    if full:
        return f"marked_bg/{size}"
    return f"clean_bg/{'rect' if is_rect(gt) else 'irregular'}/{size}"


def oof_logits(z, method: str, flip: bool) -> tuple[np.ndarray, np.ndarray, dict]:
    names = [k[:-len("/train")] for k in z.files if k.endswith("/train")]
    fold_of = {n: VAL_FOLD_BY_METHOD.get((method, n), VAL_FOLD.get(n, 0)) for n in names}
    folds = z["folds_train"]
    out = np.full(z["gt_train"].shape, np.nan, np.float32); covered = np.zeros(len(folds), bool)
    for k in sorted(set(fold_of.values())):
        mem = [n for n in names if fold_of[n] == k]
        sel = folds == k
        arr = [z[f"{n}/train"].astype(np.float32) for n in mem]
        if flip:
            arr += [z[f"{n}/train_flip"].astype(np.float32) for n in mem]
        out[sel] = np.mean(arr, axis=0)[sel]; covered |= sel
    return out, covered, fold_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("caches", nargs="+"); ap.add_argument("--rect", nargs="*", default=[])
    a = ap.parse_args()
    rects = {}
    for p in a.rect:
        d = json.loads(Path(p).read_text())
        rects[d["method"]] = {(r["split"], r["name"]): r for r in d["rows"]}
    total = {}
    for path in a.caches:
        z = np.load(path); method = Path(path).stem
        gt, full = z["gt_train"], z["full_train"]
        print(f"\n=== {method}")
        for n in [k[:-len("/train")] for k in z.files if k.endswith("/train")]:
            lg = z[f"{n}/train"].astype(np.float32)
            per_fold = [np.mean([miou(lg[i] > 0, gt[i]) for i in np.nonzero(z["folds_train"] == k)[0]]) for k in range(5)]
            print(f"  {n:12s} per-fold mIoU " + " ".join(f"{v:.3f}" for v in per_fold))
        for flip in (False, True):
            lg, cov, _ = oof_logits(z, method, flip)
            idx = np.nonzero(cov)[0]
            print(f"  OOF ensemble {'+flip' if flip else '     '} ({len(idx)} imgs): "
                  + "  ".join(f"thr{t:+.1f} {np.mean([miou(lg[i] > t, gt[i]) for i in idx]):.4f}" for t in (-1, -0.5, 0, 0.5, 1)))
        lg, cov, _ = oof_logits(z, method, True)
        idx = np.nonzero(cov)[0]
        cats = {}
        for i in idx:
            c = category(gt[i], bool(full[i]))
            cats.setdefault(c, []).append(i)
        print(f"  {'category':28s} {'n':>4s} {'mIoU':>6s} {'loss/img':>8s} {'share of loss':>13s}")
        tot_loss = sum(1 - miou(lg[i] > 0, gt[i]) for i in idx)
        for c in sorted(cats):
            v = [miou(lg[i] > 0, gt[i]) for i in cats[c]]
            print(f"  {c:28s} {len(v):4d} {np.mean(v):6.3f} {1 - np.mean(v):8.3f} {sum(1 - x for x in v) / tot_loss:13.1%}")
        rect_imgs = [i for i in idx if not full[i] and is_rect(gt[i])]
        gain = sum(1 - miou(lg[i] > 0, gt[i]) for i in rect_imgs) / len(idx)
        print(f"  ceiling of a perfect rectangle on clean_bg/rect: +{gain:.4f} mIoU for this method "
              f"(+{gain / 3:.4f} overall)")
        if method in rects:
            rows = rects[method]; names = z["names_train"].tolist()
            ens_pred = np.nan_to_num(lg, nan=0) > 0
            recs = []
            for i, nm in enumerate(names):
                r = rows.get(("train", nm))
                if r is None or not cov[i]:
                    continue
                for key in ("rect", "rect_grown"):
                    if key not in r:
                        continue
                    y0, y1, x0, x1 = r[key]; m = np.zeros((512, 512), np.float32); m[y0:y1, x0:x1] = 1
                    rm = (m.reshape(128, 4, 128, 4).mean((1, 3)) >= 0.5).astype(np.uint8)
                    recs.append((key, i, r["score" if key == "rect" else "score_grown"], miou(rm, gt[i]), miou(ens_pred[i], gt[i]),
                                 category(gt[i], bool(full[i]))))
            for key in ("rect", "rect_grown"):
                rr = [x for x in recs if x[0] == key]
                if not rr:
                    continue
                print(f"  [{key}] searched {len(rr)} OOF train images; by category: rect mIoU / ensemble mIoU (n)")
                for c in sorted({x[5] for x in rr}):
                    sel = [x for x in rr if x[5] == c]
                    print(f"    {c:28s} {np.mean([x[3] for x in sel]):.3f} / {np.mean([x[4] for x in sel]):.3f} ({len(sel)})")
                print(f"  [{key}] replace ensemble by rectangle when score >= tau: gain in this method's mean mIoU")
                line = []
                for tau in (0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0):
                    g = sum(x[3] - x[4] for x in rr if x[2] >= tau) / len(idx)
                    line.append(f"tau{tau:.2f} {g:+.4f} (n={sum(x[2] >= tau for x in rr)})")
                print("    " + "  ".join(line))
        total[method] = np.mean([miou(lg[i] > 0, gt[i]) for i in idx])
    print("\nOOF ensemble (+flip) mean over methods:", {k: round(v, 4) for k, v in total.items()},
          round(float(np.mean(list(total.values()))), 4))


if __name__ == "__main__":
    main()
