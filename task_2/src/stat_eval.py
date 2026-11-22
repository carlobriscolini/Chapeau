"""Which paired statistic separates models best in the "1 answer per prompt" design? (data with known truth)

For every pair of endpoints and every statistic: sign-flip permutation test on the per-prompt differences.
Reports, per statistic: power on truly-different pairs and false alarms on truly-same pairs at alpha, the AUC of
-log10(p) for different vs same pairs, and p-values of each variant against the plain endpoint.
Also a multivariate paired Hotelling T^2 on several statistics at once.

    python stat_eval.py ANSWERS LLFILE --truth TRUTH.csv [--prompts REGEX] [--pick-one]
"""
import argparse
import csv
import itertools
import math
import re
from collections import defaultdict

import numpy as np

import common

ap = argparse.ArgumentParser()
ap.add_argument("answers")
ap.add_argument("ll")
ap.add_argument("--truth", required=True)
ap.add_argument("--prompts", default=r"^(cont_|h1_cont_|c_)")
ap.add_argument("--alpha", type=float, default=0.01)
ap.add_argument("--perms", type=int, default=3000)
ap.add_argument("--pick-one", action="store_true", help="use only repeat 0 (data with several repeats)")
ap.add_argument("--exclude", default="", help="comma list of kinds to leave out of the pairs (e.g. disguises)")
a = ap.parse_args()
rng = np.random.default_rng(0)

TRUTH, KIND = {}, {}
for row in csv.DictReader(open(a.truth)):
    TRUTH[row["endpoint"]] = row["model"]
    KIND[row["endpoint"]] = row.get("kind", "")[:12]
excl = set(a.exclude.split(",")) - {""}
V = defaultdict(dict)  # V[ep][pid] = dict of statistics
for r in common.load_answers(a.answers):
    if r["endpoint"] in TRUTH and "#" not in r["id"] and (not a.pick_one or r.get("repeat", 0) == 0):
        V[r["endpoint"]].setdefault(r["id"], {})["len"] = math.log1p(len(r["response"]))
for r in common.load_answers(a.ll):
    e, p = r["endpoint"], r["id"]
    if e in V and p in V[e] and (not a.pick_one or r.get("repeat", 0) == 0):
        V[e][p].update({k: r[k] for k in ("lp_mean", "logrank", "top1", "ent_mean", "surp_ent", "lp_first20") if k in r})
        V[e][p]["ntok"] = math.log1p(r["n_tok"])
EPS = sorted((e for e in V if KIND[e] not in excl), key=lambda e: (TRUTH[e] != "S", TRUTH[e], e))
STATS = ["len", "ntok", "lp_mean", "logrank", "top1", "ent_mean", "surp_ent", "lp_first20"]
COMBOS = {"T2(lp,len)": ["lp_mean", "len"], "T2(lp,logrank,ent)": ["lp_mean", "logrank", "ent_mean"],
          "T2(all)": ["len", "lp_mean", "logrank", "top1", "ent_mean", "surp_ent"]}


def diffs(x, y, keys):
    ps = [p for p in V[x] if p in V[y] and re.match(a.prompts, p) and all(k in V[x][p] and k in V[y][p] for k in keys)]
    return np.array([[V[x][p][k] - V[y][p][k] for k in keys] for p in ps])


def pval(D):
    n = len(D)
    if D.ndim == 1 or D.shape[1] == 1:
        d = D.ravel()
        obs = abs(d.mean())
        null = np.abs((rng.choice([-1.0, 1.0], size=(a.perms, n)) * d).mean(1))
    else:  # Hotelling T^2 with sign flips (covariance of the raw differences, fixed under flips)
        S = np.cov(D.T) + 1e-9 * np.eye(D.shape[1])
        Si = np.linalg.inv(S)
        m = D.mean(0)
        obs = m @ Si @ m
        F = rng.choice([-1.0, 1.0], size=(a.perms, n))
        M = F @ D / n
        null = np.einsum("ij,jk,ik->i", M, Si, M)
    return (1 + (null >= obs - 1e-12).sum()) / (a.perms + 1)


P = {s: {} for s in STATS + list(COMBOS)}
for x, y in itertools.combinations(EPS, 2):
    for s in STATS:
        D = diffs(x, y, [s])
        P[s][(x, y)] = pval(D[:, 0]) if len(D) > 4 else 1.0
    for c, keys in COMBOS.items():
        D = diffs(x, y, keys)
        P[c][(x, y)] = pval(D) if len(D) > len(keys) + 4 else 1.0

npr = len(diffs(EPS[0], EPS[1], ["len"]))
print(f"{len(EPS)} endpoint, ~{npr} prompt appaiati, alpha={a.alpha}")
print(f"{'statistica':20s} potenza(diversi)  falsi allarmi(uguali)   AUC")
for s in P:
    same = [p for (x, y), p in P[s].items() if TRUTH[x] == TRUTH[y]]
    diff = [p for (x, y), p in P[s].items() if TRUTH[x] != TRUTH[y]]
    auc = np.mean([(d < q) + 0.5 * (d == q) for d in diff for q in same])
    print(f"{s:20s} {np.mean(np.array(diff) < a.alpha):8.2f}        {np.mean(np.array(same) < a.alpha):8.2f}          {auc:.2f}")

plain = next(e for e in EPS if "plain" in KIND[e])
print(f"\np-value contro {plain} (plain). S attesi alti, varianti bassi")
cols = STATS + list(COMBOS)
print(f"{'':22s}" + "".join(f"{c[:9]:>10s}" for c in cols))
for e in EPS:
    if e == plain:
        continue
    k = (plain, e) if (plain, e) in P["len"] else (e, plain)
    print(f"{TRUTH[e]:4s} {KIND[e]:16s} " + "".join(f"{P[c][k]:10.3f}" for c in cols))
