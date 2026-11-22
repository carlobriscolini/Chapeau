"""Decoding fingerprint of each answer under the family base model (GPU).

Same model + different decoding (temperature / top-k / top-p) is the SAME model for the task, but our likelihood
statistics see it as different, exactly like a LoRA. This script separates the two:
  ll[T], ent[T] : sum over tokens of log p_T(y_t) and of the entropy of p_T, for T in TGRID (p_T = softmax(z/T))
                  -> temperature MLE per endpoint and calibration at that temperature
  max_rank      : worst (largest) rank of an observed token under the base model (top-k leaves rank <= k)
  n_rank_gt     : number of tokens with rank > 10, 40, 100
  max_mass      : largest probability mass (T=1) of tokens strictly more likely than the observed one
                  (top-p = P leaves max_mass < P at the endpoint's temperature)
  max_mass_T    : same at T = 0.7 (typical default)

    python ll_decode.py MODEL --answers A1.jsonl A2.jsonl --endpoints ep-01,... --out FILE
"""
import argparse
import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import common

TGRID = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4]
ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--answers", nargs="+", required=True)
ap.add_argument("--endpoints", required=True)
ap.add_argument("--prompts", default=r"^(cont_|c_|l_)")
ap.add_argument("--out", required=True)
a = ap.parse_args()
DEV = "cuda" if torch.cuda.is_available() else "cpu"
eps = set(a.endpoints.split(","))
rows = []
for f in a.answers:
    rows += [r for r in common.load_answers(f) if r["endpoint"] in eps and "#" not in r["id"]
             and re.match(a.prompts, r["id"]) and r["response"].strip()]
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).to(DEV).eval()
T = torch.tensor(TGRID, device=DEV)
print(len(rows), "answers", flush=True)
with open(a.out, "w") as f, torch.no_grad():
    for n, r in enumerate(rows):
        prefix = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False,
                                         add_generation_prompt=True)
        p = tok(prefix, add_special_tokens=False)["input_ids"]
        y = tok(r["response"], add_special_tokens=False)["input_ids"][:300]
        z = model(torch.tensor([p + y], device=DEV)).logits[0, len(p) - 1:-1].float()  # [n, V]
        tgt = torch.tensor(y, device=DEV)
        ll, ent = [], []
        for t in TGRID:
            lp = torch.log_softmax(z / t, -1)
            ll.append(lp[torch.arange(len(y)), tgt].sum().item())
            ent.append(-(lp.exp() * lp).sum(-1).sum().item())
        zt = z[torch.arange(len(y)), tgt][:, None]
        rank = (z > zt).sum(-1)                                   # 0 = argmax
        p1 = torch.softmax(z, -1)
        mass = (p1 * (z > zt)).sum(-1)
        p7 = torch.softmax(z / 0.7, -1)
        mass7 = (p7 * (z > zt)).sum(-1)
        f.write(json.dumps({"endpoint": r["endpoint"], "id": r["id"], "n_tok": len(y), "ll": ll, "ent": ent,
                            "max_rank": int(rank.max()), "n_rank_gt": [int((rank > k).sum()) for k in (10, 40, 100)],
                            "max_mass": mass.max().item(), "max_mass_T": mass7.max().item(),
                            "mean_rank0": (rank == 0).float().mean().item()}) + "\n")
        if (n + 1) % 200 == 0:
            print(f"  {n + 1}/{len(rows)}", flush=True)
print("done", a.out)
