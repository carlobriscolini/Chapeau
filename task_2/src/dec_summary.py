"""Summarise decoding fingerprints (ll_decode.py output) per endpoint.

  T*      : temperature maximising the total log-likelihood of the endpoint's answers under the base model
  cal(T*) : mean per-token (log p_T*(y) + H(p_T*)) -> ~0 if answers are pure temperature-T* samples of the base
  cal(1)  : same at T=1
  rk>40   : fraction of tokens with base rank > 40 (top-k=40 would make it 0)
  maxrank : largest rank seen;  mass99 : 99th percentile over answers of max_mass (top-p fingerprint)

    python dec_summary.py data/dec_*.jsonl [--groups "A:ep-05,ep-09,..."]
"""
import argparse
import json
from collections import defaultdict

import numpy as np

TGRID = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4]
ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
a = ap.parse_args()
R = defaultdict(list)
for f in a.files:
    for l in open(f):
        r = json.loads(l)
        R[r["endpoint"]].append(r)
print(f"{'endpoint':9s} {'n':>3s} {'T*':>5s} {'cal(T*)':>8s} {'cal(1)':>7s} {'ll/tok(T*)':>10s} {'top1':>5s} "
      f"{'rk>10':>6s} {'rk>40':>6s} {'maxrank':>7s} {'mass99':>6s}")
for e in sorted(R):
    rs = R[e]
    ntok = sum(r["n_tok"] for r in rs)
    ll = np.array([r["ll"] for r in rs]).sum(0)
    ent = np.array([r["ent"] for r in rs]).sum(0)
    k = int(np.argmax(ll))
    # parabolic refinement of T* on the grid
    t = TGRID[k]
    cal_t = (ll[k] + ent[k]) / ntok
    cal_1 = (ll[TGRID.index(1.0)] + ent[TGRID.index(1.0)]) / ntok
    rk10 = sum(r["n_rank_gt"][0] for r in rs) / ntok
    rk40 = sum(r["n_rank_gt"][1] for r in rs) / ntok
    top1 = np.average([r["mean_rank0"] for r in rs], weights=[r["n_tok"] for r in rs])
    print(f"{e:9s} {len(rs):3d} {t:5.2f} {cal_t:+8.3f} {cal_1:+7.3f} {ll[k] / ntok:10.3f} {top1:5.2f} {rk10:6.3f} "
          f"{rk40:6.4f} {max(r['max_rank'] for r in rs):7d} {np.percentile([r['max_mass'] for r in rs], 99):6.3f}")
