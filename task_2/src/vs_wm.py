import json, math, re, glob, numpy as np
from collections import defaultdict
rng=np.random.default_rng(0)
R=defaultdict(dict); V=defaultdict(lambda: defaultdict(dict))
for l in open("data/ep_phase1.jsonl"):
    r=json.loads(l)
    if "#" not in r["id"]: R[r["endpoint"]][r["id"]]=r["response"]
for f in glob.glob("data/ll_*_ep_phase1.jsonl"):
    for l in open(f):
        r=json.loads(l)
        if "#" in r["id"] or not re.match(r"^(cont_|c_)",r["id"]): continue
        e,p=r["endpoint"],r["id"]
        V[e][p].update(surp=r["lp_mean"]+r["ent_mean"], ll=r["lp_mean"], ent=r["ent_mean"], f20=float(np.mean(r["lp_tokens"][:20])), len=math.log1p(len(R[e][p])))
for f in glob.glob("data/llrq_*_ep_phase1.jsonl"):
    for l in open(f):
        r=json.loads(l)
        if r["id"] in V[r["endpoint"]]: V[r["endpoint"]][r["id"]]["fp8"]=r["llr_fp8_tensor"]
S=["surp","f20","fp8","len","ent"]
def pv(x,y,s):
    ps=[p for p in V[x] if p in V[y] and s in V[x][p] and s in V[y][p]]
    d=np.array([V[x][p][s]-V[y][p][s] for p in ps]); obs=abs(d.mean())
    null=np.abs((rng.choice([-1.,1.],size=(20000,len(d)))*d).mean(1))
    return (1+(null>=obs).sum())/20001, d.mean()/ (d.std(ddof=1)/math.sqrt(len(d)))
CORE={"A":("10","05 09 11 13 15 19 27"),"B":("06","03 21 22 26 28 32"),"C":("04","01 12 14 17 18 24 25")}
for f,(wm,core) in CORE.items():
    print(f"== {f}: vs watermark ep-{wm}.  one-sample mean surp (calibrazione, z)")
    for c in core.split():
        x,y="ep-"+c,"ep-"+wm
        out=[]; mn=1
        for s in S:
            p,z=pv(x,y,s); out.append(f"{s} p={p:.3f}(z{z:+.1f})"); mn=min(mn,p) if s!="ent" else mn
        sv=np.array([V[x][p]["surp"] for p in V[x]])
        print(f"  ep-{c}: bonf={min(1,4*mn):.3f} | "+"  ".join(out)+f" | surp {sv.mean():+.3f}±{sv.std(ddof=1)/math.sqrt(len(sv)):.3f}")
    sv=np.array([V['ep-'+wm][p]["surp"] for p in V['ep-'+wm]]); print(f"  wm ep-{wm}: surp {sv.mean():+.3f}±{sv.std(ddof=1)/math.sqrt(len(sv)):.3f}")
