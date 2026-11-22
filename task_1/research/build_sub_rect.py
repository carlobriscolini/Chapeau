"""Build a submission = sub04 ensemble (mean logits of its members, from research.cache_logits) + message-guided
rectangles (research.rect_refine) for selected methods, and a manifest that research.upload accepts.

Rule (validated out-of-fold on train, research/NOTES.md): on an image searched by rect_refine (clean background),
replace the ensemble mask by the rectangle when the rectangle's score >= tau AND the ensemble mask disagrees with it
(IoU of the watermarked class < agree). Otherwise keep the ensemble.

    python -m research.build_sub_rect --name sub05_rect_tm --rect trustmark=outputs/rect/trustmark.json:0.99:0.5"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from research import submission

SUB04 = {"pixelseal": ["h1_wm_s1", "h1_wm", "j_wm_s1", "f1_wm"],
         "trustmark": ["h1_r50", "v2_r50", "j_wm", "f1_wm", "f1_r50", "f2cut_wm"],
         "stegastamp": ["h1_r50", "v2_r34", "f1_r50"]}


def rect_mask(r) -> np.ndarray:
    y0, y1, x0, x1 = r
    m = np.zeros((512, 512), np.float32); m[y0:y1, x0:x1] = 1
    return (m.reshape(128, 4, 128, 4).mean((1, 3)) >= 0.5).astype(np.uint8)


def iou1(a, b) -> float:
    a, b = a.astype(bool), b.astype(bool); u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--data", default="release")
    ap.add_argument("--cache-dir", default="outputs/cache")
    ap.add_argument("--rect", action="append", default=[], help="method=rects.json[,rects2.json]:tau:agree")
    a = ap.parse_args()
    rules = {}
    for spec in a.rect:
        method, rest = spec.split("=", 1); paths, tau, agree = rest.rsplit(":", 2)
        rows = {}
        for p in paths.split(","):
            rows.update({r["name"]: r for r in json.loads(Path(p).read_text())["rows"] if r["split"] == "test"})
        rules[method] = {"paths": paths, "tau": float(tau), "agree": float(agree), "rows": rows}
    names, masks, info = [], [], {}
    for method, members in SUB04.items():
        z = np.load(Path(a.cache_dir) / f"{method}.npz")
        lg = np.mean([z[f"{m}/test"].astype(np.float32) for m in members], axis=0)
        pred = (lg > 0).astype(np.uint8)
        replaced = []
        if method in rules:
            r = rules[method]
            for i, n in enumerate(z["names_test"].tolist()):
                row = r["rows"].get(n)
                if row is None or row["score"] < r["tau"]:
                    continue
                rm = rect_mask(row["rect"])
                if iou1(pred[i], rm) < r["agree"]:
                    pred[i] = rm; replaced.append(n)
        info[method] = {"ensemble": members, "rect_rule": None if method not in rules else
                        {k: v for k, v in rules[method].items() if k != "rows"}, "replaced": len(replaced),
                        "replaced_names": replaced, "test_watermarked_fraction": float(pred.mean())}
        names += z["names_test"].tolist(); masks.append(pred)
        print(method, {k: v for k, v in info[method].items() if k != "replaced_names"}, flush=True)
    out = Path("outputs/submissions") / f"{a.name}.pt"
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    manifest = {"name": a.name, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "commit": commit,
                "base": "sub04_folds ensemble from outputs/cache (float16 logits)", "methods": info}
    manifest["file"] = submission.write(names, np.concatenate(masks), out, a.data)
    manifest["file"]["sha256"] = hashlib.sha256(out.read_bytes()).hexdigest()
    Path("research/submissions").mkdir(exist_ok=True)
    (Path("research/submissions") / f"{a.name}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["file"], indent=2))


if __name__ == "__main__":
    main()
