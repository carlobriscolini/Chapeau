"""Submission builder v2: per-method logit ensembles over members taken from several caches (outputs/cache/<method>.npz
and outputs/cache/<method>_k.npz written by jureca/cache_folds.sbatch), the message-rectangle rule of build_sub_rect,
and an honest out-of-fold estimate on train: every train image of fold k is predicted by the ensemble of the selected
members validated on fold k (members that never saw it). Test = mean logits of ALL selected members.

    python -m research.build_sub_v2 --name sub07 --members pixelseal=h1_wm,h1_wm_s1,j_wm_s1,f1_wm,k2_wm,k3_wm,k4_wm \
        --rect pixelseal=outputs/rect/pixelseal.json:0.85:0.5 --rect trustmark=outputs/rect/trustmark.json:0.99:0.5
    add --dry-run to print the OOF estimate only."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np

from research.analyze_cache import VAL_FOLD, VAL_FOLD_BY_METHOD, miou
from research.build_sub_rect import SUB04, iou1, rect_mask


def member_fold(method: str, name: str) -> int:
    m = re.match(r"^(?:k|cm|cs|cp|cx|rp|t[a-z])(\d)_", name)
    return int(m.group(1)) if m else VAL_FOLD_BY_METHOD.get((method, name), VAL_FOLD.get(name, 0))


def load(method: str, cache_dir: Path, members: list[str]):
    zs = [np.load(p) for p in [cache_dir / f"{method}.npz"] + sorted(cache_dir.glob(f"{method}_*.npz")) if p.stat().st_size]
    base = zs[0]
    for z in zs[1:]:
        assert z["names_train"].tolist() == base["names_train"].tolist() and z["names_test"].tolist() == base["names_test"].tolist()
    get = lambda key: next(z[key] for z in zs if key in z.files)
    missing = [n for n in members if not any(f"{n}/test" in z.files for z in zs)]
    if missing:
        raise SystemExit(f"{method}: members not in cache: {missing}")
    return base, {n: (get(f"{n}/train").astype(np.float32), get(f"{n}/test").astype(np.float32)) for n in members}


def apply_rect(pred: np.ndarray, names: list[str], rule: dict | None, split: str) -> tuple[np.ndarray, list[str]]:
    if not rule:
        return pred, []
    out, replaced = pred.copy(), []
    for i, n in enumerate(names):
        row = rule["rows"][split].get(n)
        if row is None or row["score"] < rule["tau"]:
            continue
        rm = rect_mask(row["rect"])
        if iou1(out[i], rm) < rule["agree"]:
            out[i] = rm; replaced.append(n)
    return out, replaced


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--data", default="release")
    ap.add_argument("--cache-dir", default="outputs/cache"); ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--members", action="append", default=[], help="method=name1,name2 (default: sub04 members)")
    ap.add_argument("--rect", action="append", default=[], help="method=rects.json:tau:agree")
    ap.add_argument("--save-pseudo", help="dir: write <method>.npz with the final test masks and the out-of-fold train "
                                          "masks (pseudo labels for research.train_seg --pseudo)")
    ap.add_argument("--save-logits", help="dir: write <method>.npz with the out-of-fold train logits and the test logits "
                                          "of the ensemble (inputs of a second-stage fusion, research.fuse_ae)")
    a = ap.parse_args()
    members = {m: list(v) for m, v in SUB04.items()}
    for spec in a.members:
        m, v = spec.split("=", 1); members[m] = v.split(",")
    rules = {}
    for spec in a.rect:
        m, rest = spec.split("=", 1); path, tau, agree = rest.rsplit(":", 2)
        rows = {"train": {}, "test": {}}
        for r in json.loads(Path(path).read_text())["rows"]:
            rows[r["split"]][r["name"]] = r
        rules[m] = {"path": path, "tau": float(tau), "agree": float(agree), "rows": rows}
    names_all, masks_all, info, oof_all = [], [], {}, []
    for method, mem in members.items():
        z, L = load(method, Path(a.cache_dir), mem)
        gt, folds = z["gt_train"], z["folds_train"]; ntr = z["names_train"].tolist()
        fold_of = {n: member_fold(method, n) for n in mem}
        oof = np.full(gt.shape, np.nan, np.float32); cov = np.zeros(len(gt), bool)
        for k in sorted(set(fold_of.values())):
            sel = folds == k; use = [n for n in mem if fold_of[n] == k]
            oof[sel] = np.mean([L[n][0] for n in use], 0)[sel]; cov |= sel
        idx = np.nonzero(cov)[0]
        p_oof, rep_tr = apply_rect((oof[idx] > 0).astype(np.uint8), [ntr[i] for i in idx], rules.get(method), "train")
        s = np.array([miou(p, gt[i]) for p, i in zip(p_oof, idx)])
        per_fold = {int(k): round(float(s[folds[idx] == k].mean()), 4) for k in sorted(set(folds[idx].tolist()))}
        oof_all.append(s)
        pred = (np.mean([L[n][1] for n in mem], 0) > 0).astype(np.uint8)
        pred, rep_te = apply_rect(pred, z["names_test"].tolist(), rules.get(method), "test")
        info[method] = {"members": mem, "member_folds": fold_of, "oof_miou": round(float(s.mean()), 4),
                        "oof_images": int(len(idx)), "oof_per_fold": per_fold, "rect_replaced_train": len(rep_tr),
                        "rect_rule": {k: v for k, v in rules[method].items() if k != "rows"} if method in rules else None,
                        "rect_replaced_test": len(rep_te), "test_watermarked_fraction": round(float(pred.mean()), 4)}
        print(method, json.dumps({k: v for k, v in info[method].items() if k not in ("members",)}), flush=True)
        names_all += z["names_test"].tolist(); masks_all.append(pred)
        if a.save_logits:   # pre-created file (inode quota), written in place
            with open(Path(a.save_logits) / f"{method}.npz", "r+b") as fh:
                fh.truncate(0)
                np.savez(fh, names_train=np.array([ntr[i] for i in idx]), oof_logit=oof[idx].astype(np.float16),
                         gt=gt[idx], folds=folds[idx], names_test=z["names_test"],
                         test_logit=np.mean([L[n][1] for n in mem], 0).astype(np.float16))
        if a.save_pseudo:   # file pre-created by the caller (inode quota): np.savez into an open handle writes in place
            with open(Path(a.save_pseudo) / f"{method}.npz", "r+b") as fh:
                fh.truncate(0)
                np.savez_compressed(fh, names_test=z["names_test"], masks_test=pred.astype(bool),
                                    names_train=np.array([ntr[i] for i in idx]), masks_train=p_oof.astype(bool),
                                    folds_train=folds[idx])
    allv = np.concatenate(oof_all)
    print(f"OOF mean over {len(allv)} train images (all methods): {allv.mean():.4f}")
    if a.dry_run:
        return
    from research import submission
    import subprocess
    out = Path("outputs/submissions") / f"{a.name}.pt"
    manifest = {"name": a.name, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
                "builder": "research.build_sub_v2", "oof_mean": round(float(allv.mean()), 4), "methods": info}
    manifest["file"] = submission.write(names_all, np.concatenate(masks_all), out, a.data)
    manifest["file"]["sha256"] = hashlib.sha256(out.read_bytes()).hexdigest()
    Path("research/submissions").mkdir(exist_ok=True)
    (Path("research/submissions") / f"{a.name}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["file"], indent=2))


if __name__ == "__main__":
    main()
