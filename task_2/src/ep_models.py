"""Within-family model grouping for the ep endpoints (families fixed by fit_family.py).

Paired tests over continuation prompts (sign-flip permutation, two-sided) between every pair in a family:
  ll  = mean token log-prob of the answer under the family's base model (likelihood.py output)
  len = log(1 + characters)
Disguised endpoints (system prompt / persona read from the probes, high temperature = ll outlier) are excluded from
the tests and attached to the base group S afterwards (in the refs every disguise sits on the base model).

    python ep_models.py data/ep_phase1.jsonl --ll data/ll_*_ep_phase1.jsonl [--alpha 0.01] [--out submission.csv]
"""
import argparse
import itertools
import math
import re
from collections import defaultdict

import numpy as np

import common

FAM = {"A": "05 09 10 11 13 15 16 19 27 29 31", "B": "03 06 07 08 20 21 22 26 28 30 32",
       "C": "01 02 04 12 14 17 18 23 24 25 33"}
FAM = {f: ["ep-" + x for x in v.split()] for f, v in FAM.items()}
# read from the probe answers (s_extract_*, b_banner): provider system prompt or persona
DISGUISE = {"ep-29": "sysprompt", "ep-16": "persona", "ep-08": "sysprompt", "ep-20": "persona", "ep-07": "sysprompt?",
            "ep-02": "sysprompt", "ep-23": "persona"}

ap = argparse.ArgumentParser()
ap.add_argument("answers")
ap.add_argument("--ll", nargs="+", required=True)
ap.add_argument("--alpha", type=float, default=0.01)
ap.add_argument("--perms", type=int, default=5000)
ap.add_argument("--prompts", default=r"^(cont_|h1_cont_|c_)")
ap.add_argument("--hot", type=float, default=0.4, help="ll this far below the family median = high temperature")
ap.add_argument("--out", default=None)
ap.add_argument("--stats", default="surp,f20,fp8")
ap.add_argument("--llrq", nargs="*", default=[], help="llr_quant.py outputs (fp8 likelihood-ratio statistic)")
a = ap.parse_args()
rng = np.random.default_rng(0)
STATS = a.stats.split(",")

R, REP, LL, SE, F20 = defaultdict(dict), defaultdict(dict), defaultdict(dict), defaultdict(dict), defaultdict(dict)
for r in common.load_answers(a.answers):
    (REP if "#" in r["id"] else R)[r["endpoint"]][r["id"].split("#")[0]] = r["response"]
for f in a.ll:
    for r in common.load_answers(f):
        if "#" not in r["id"]:
            LL[r["endpoint"]][r["id"]] = r["lp_mean"]
            SE[r["endpoint"]][r["id"]] = r["lp_mean"] + r["ent_mean"]       # calibration: surprise - entropy
            F20[r["endpoint"]][r["id"]] = float(np.mean(r["lp_tokens"][:20]))


FP8 = defaultdict(dict)
for f in a.llrq:
    for r in common.load_answers(f):
        if "#" not in r["id"]:
            FP8[r["endpoint"]][r["id"]] = r["llr_fp8_tensor"]


def val(e, p, s):
    if s == "len":
        return math.log1p(len(R[e][p]))
    return {"ll": LL, "surp": SE, "f20": F20, "fp8": FP8}[s][e].get(p)


def test(x, y, s):
    ps = [p for p in R[x] if p in R[y] and re.match(a.prompts, p) and val(x, p, s) is not None
          and val(y, p, s) is not None]
    d = np.array([val(x, p, s) - val(y, p, s) for p in ps])
    obs = abs(d.mean())
    null = np.abs((rng.choice([-1.0, 1.0], size=(a.perms, len(d))) * d).mean(1))
    return d.mean(), (1 + (null >= obs - 1e-12).sum()) / (a.perms + 1), len(d)


def agglo(members, D, thr):
    C = [[m] for m in members]
    while len(C) > 1:
        best = min((max(D[(u, v)] for u in C[i] for v in C[j]), i, j) for i, j in itertools.combinations(range(len(C)), 2))
        if best[0] >= thr:
            break
        _, i, j = best
        C[i] += C[j]
        del C[j]
    return C


rows = []
for f, eps in FAM.items():
    med = {s: np.median([np.mean([val(e, p, s) for p in R[e] if re.match(a.prompts, p) and val(e, p, s) is not None])
                         for e in eps]) for s in ("ll", "len")}
    print(f"\n=== famiglia {f}")
    info = {}
    for e in eps:
        ll = np.mean([v for p, v in LL[e].items() if re.match(a.prompts, p)])
        ln = np.mean([val(e, p, "len") for p in R[e] if re.match(a.prompts, p)])
        same = sum(R[e][p] == REP[e][p] for p in REP[e] if p in R[e])
        hot = ll < med["ll"] - a.hot
        tag = DISGUISE.get(e, "") + (" ALTA-T" if hot else "") + (f" ripete {same}" if same else "")
        info[e] = (ll, ln, tag)
        print(f"  {e} ll {ll:7.3f} ({ll - med['ll']:+.3f})  len {ln:5.2f} ({ln - med['len']:+.2f})  n={len(LL[e])}  {tag}")
    clean = [e for e in eps if e not in DISGUISE and "ALTA-T" not in info[e][2]]
    D, P = {}, {}
    for x, y in itertools.combinations(clean, 2):
        P[(x, y)] = {s: test(x, y, s) for s in STATS}
        p = min(1.0, len(STATS) * min(P[(x, y)][s][1] for s in STATS))
        D[(x, y)] = D[(y, x)] = -math.log10(p)
    groups = agglo(clean, D, -math.log10(a.alpha))
    groups.sort(key=lambda g: -max(info[e][0] for e in g))  # group with the most base-like answers first = S
    print("  coppie con p<0.01 tra endpoint senza travestimento:")
    for (x, y), v in P.items():
        if min(v[s][1] for s in STATS) < 0.01:
            print(f"    DIVERSI: {x}/{y} " + " ".join(f"p_{s}={v[s][1]:.3f}" for s in STATS))
    S = groups[0] + [e for e in eps if e not in clean]
    print(f"  S (base + travestiti): {sorted(S)}")
    for k, g in enumerate(groups[1:], 2):
        print(f"  modello {f}{k}: {sorted(g)}")
    rows += [(e, f, f"{f}1") for e in S] + [(e, f, f"{f}{k}") for k, g in enumerate(groups[1:], 2) for e in g]

if a.out:
    with open(a.out, "w") as fh:
        fh.write("endpoint,family,model\n")
        for r in sorted(rows):
            fh.write(",".join(r) + "\n")
    print(f"\nscritto {a.out}")
