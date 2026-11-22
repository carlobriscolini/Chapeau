"""Phase-2 decision (rule fixed BEFORE seeing phase-2 data, approved plan 22 nov ~02:35):
for each family, test every undecided core endpoint against the family's watermark endpoint (same weights as the
plain, distortion-free sampling) with paired sign-flip tests on surp, f20, fp8 LLR, len over phase 1 + phase 2 prompts.
A core endpoint joins S ONLY if it is the unique candidate with Bonferroni p > 0.05 and every other candidate has
Bonferroni p < 0.01. Otherwise the family stays as in submission_v3 (all cores singletons).

    python decide_plain.py [--out submission_v4.csv]
"""
import argparse, csv, glob, json, math, re
from collections import defaultdict
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--out", default=None); ap.add_argument("--perms", type=int, default=20000)
a = ap.parse_args(); rng = np.random.default_rng(0)
PH = ["ep_phase1", "ep_phase2"]
R = defaultdict(dict); V = defaultdict(lambda: defaultdict(dict))
for ph in PH:
    for l in open(f"data/{ph}.jsonl"):
        r = json.loads(l)
        if "#" not in r["id"]: R[r["endpoint"]][r["id"]] = r["response"]
    for f in glob.glob(f"data/ll_*_{ph}.jsonl"):
        for l in open(f):
            r = json.loads(l); e, p = r["endpoint"], r["id"]
            if "#" in p or not re.match(r"^(cont_|c_|l_)", p): continue
            V[e][p].update(surp=r["lp_mean"] + r["ent_mean"], f20=float(np.mean(r["lp_tokens"][:20])),
                           len=math.log1p(len(R[e][p])))
    for f in glob.glob(f"data/llrq_*_{ph}.jsonl"):
        for l in open(f):
            r = json.loads(l)
            if r["id"] in V[r["endpoint"]]: V[r["endpoint"]][r["id"]]["fp8"] = r["llr_fp8_tensor"]
STATS = ["surp", "f20", "fp8", "len"]


def pv(x, y, s):
    ps = [p for p in V[x] if p in V[y] and s in V[x][p] and s in V[y][p]]
    d = np.array([V[x][p][s] - V[y][p][s] for p in ps]); obs = abs(d.mean())
    null = np.abs((rng.choice([-1., 1.], size=(a.perms, len(d))) * d).mean(1))
    return (1 + (null >= obs - 1e-12).sum()) / (a.perms + 1), len(d), sum(p.startswith("l_") for p in ps)


CAND = {"A": ("10", "09 11 13 19"), "B": ("06", "21 26 32"), "C": ("04", "01 12 17 18 24 25")}
join = {}
for f, (wm, cands) in CAND.items():
    print(f"== famiglia {f}, watermark ep-{wm}")
    bon = {}
    for c in cands.split():
        res = {s: pv("ep-" + c, "ep-" + wm, s) for s in STATS}
        bon[c] = min(1.0, len(STATS) * min(v[0] for v in res.values()))
        n, nl = res["surp"][1], res["surp"][2]
        print(f"  ep-{c}: Bonferroni p={bon[c]:.4f}  ({n} prompt, {nl} lunghi)  " +
              " ".join(f"{s}={v[0]:.3f}" for s, v in res.items()))
    ok = [c for c in bon if bon[c] > 0.05]
    if len(ok) == 1 and all(bon[c] < 0.01 for c in bon if c != ok[0]):
        join[f] = ok[0]; print(f"  -> REGOLA SODDISFATTA: ep-{ok[0]} = plain, entra in S")
    else:
        print(f"  -> regola NON soddisfatta (compatibili col watermark: {['ep-'+c for c in ok]}): resta come v3")
if a.out:
    rows = list(csv.DictReader(open("submission_v3.csv")))
    for r in rows:
        f = r["family"]
        if f in join and r["endpoint"] == "ep-" + join[f]: r["model"] = f + "1"
    with open(a.out, "w") as fh:
        fh.write("endpoint,family,model\n"); [fh.write(f"{r['endpoint']},{r['family']},{r['model']}\n") for r in rows]
    print("scritto", a.out, "| cambi rispetto a v3:", join or "nessuno")
