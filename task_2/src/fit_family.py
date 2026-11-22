"""Which local base model explains each endpoint's answers best? (family / exact checkpoint identification)

For every endpoint, take up to N continuation answers and compute the mean token log-prob and top-1 rate under each
candidate model (user turn only, chat template of the candidate). The true base model gives clearly higher values.

    python fit_family.py data/ep_phase1.jsonl --models Qwen/Qwen2.5-0.5B-Instruct,Qwen/Qwen2.5-1.5B-Instruct --n 8
"""
import argparse
import re
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import common

ap = argparse.ArgumentParser()
ap.add_argument("answers")
ap.add_argument("--models", required=True)
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--prompts", default=r"^(cont_|h1_cont_|c_)")
ap.add_argument("--endpoints", default=None)
a = ap.parse_args()

rows = [r for r in common.load_answers(a.answers) if re.search(a.prompts, r["id"]) and "#" not in r["id"]]
if a.endpoints:
    rows = [r for r in rows if r["endpoint"] in a.endpoints.split(",")]
by = defaultdict(list)
for r in rows:
    if len(by[r["endpoint"]]) < a.n and r["response"].strip():
        by[r["endpoint"]].append(r)
EPS = sorted(by)
RES = defaultdict(dict)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
for mid in a.models.split(","):
    tok = AutoTokenizer.from_pretrained(mid)
    model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.float32).to(DEV).eval()
    for e in EPS:
        lps, top1 = [], []
        for r in by[e]:
            prefix = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False,
                                             add_generation_prompt=True)
            p_ids = tok(prefix, add_special_tokens=False)["input_ids"]
            a_ids = tok(r["response"], add_special_tokens=False)["input_ids"][:256]
            with torch.no_grad():
                logits = model(torch.tensor([p_ids + a_ids], device=DEV)).logits[0, len(p_ids) - 1:-1].float().cpu()
            logp = torch.log_softmax(logits, -1)
            tgt = torch.tensor(a_ids)
            lps.append(logp[torch.arange(len(a_ids)), tgt].mean().item())
            top1.append((logp.argmax(-1) == tgt).float().mean().item())
        RES[e][mid] = (np.mean(lps), np.mean(top1))
        print(f"  {mid.split('/')[-1]:28s} {e}: logp/tok {RES[e][mid][0]:7.3f} top1 {RES[e][mid][1]:.3f}", flush=True)
    del model

short = [m.split("/")[-1][:20] for m in a.models.split(",")]
print("\nlogp/tok medio (più alto = spiega meglio) | top-1")
print(f"{'':8s}" + "".join(f"{s:>22s}" for s in short) + "   migliore")
for e in EPS:
    vals = [RES[e][m] for m in a.models.split(",")]
    best = short[int(np.argmax([v[0] for v in vals]))]
    print(f"{e:8s}" + "".join(f"{v[0]:12.3f} ({v[1]:.2f})  " for v in vals) + f"   {best}")
