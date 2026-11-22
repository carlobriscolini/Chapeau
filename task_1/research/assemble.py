"""Assemble a submission from a logits dir + SAM outputs (the sub13 recipe, all choices fixed on OOF earlier):
PS/TM = SAM refine variant A tau 0.7 (includes the rectangle rule), SS = ensemble logits > 0; then on every image whose
predicted minority is < 0.5 % the max-evidence rectangle bet (research.bet_combo, rule "rect").

    python -m research.assemble --logits outputs/logits14 --sam outputs/sam14 --name sub14"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from research import submission


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logits", required=True); ap.add_argument("--sam", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--note", default=""); ap.add_argument("--data", default="release")
    a = ap.parse_args()
    names, masks, info = [], [], {}
    for m in ("pixelseal", "trustmark", "stegastamp"):
        z = np.load(Path(a.logits) / f"{m}.npz"); n = z["names_test"].tolist()
        if m == "stegastamp":
            cur = (z["test_logit"].astype(np.float32) > 0).astype(np.uint8); src = "logits > 0"
        else:
            s = np.load(Path(a.sam) / f"{m}.npz"); k = s["keys"].tolist().index("A_tau0.7")
            assert s["names_test"].tolist() == n
            cur = s["test"][k].astype(np.uint8); src = "rect rule + SAM refine A tau0.7"
        b = np.load(Path(a.sam) / f"{m}_bet.npz"); assert b["names_test"].tolist() == n
        bet = b["test"][b["rules"].tolist().index("rect")].astype(np.uint8)
        used = 0
        for i in range(len(n)):
            f = cur[i].mean()
            if min(f, 1 - f) < 0.005 and (bet[i] != cur[i]).any():
                cur[i] = bet[i]; used += 1
        info[m] = {"source": src, "rect_bets_applied": used}
        names += n; masks.append(cur)
    out = Path("outputs/submissions") / f"{a.name}.pt"
    man = {"name": a.name, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "builder": "research.assemble",
           "logits": a.logits, "sam": a.sam, "note": a.note, "methods": info}
    man["file"] = submission.write(names, np.concatenate(masks), out, a.data)
    man["file"]["sha256"] = hashlib.sha256(out.read_bytes()).hexdigest()
    (Path("research/submissions") / f"{a.name}.json").write_text(json.dumps(man, indent=2) + "\n")
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
