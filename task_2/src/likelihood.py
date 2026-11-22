"""Score answers with a local model: log-probability of each answer given its prompt (chat template applied).

    python likelihood.py MODEL_ID [--answers data/ref_probes_v1.jsonl] [--endpoints ref-01,ref-02] [--limit N]

Writes data/ll_<model>_<answers-stem>.jsonl with, per answer: sum/mean token log-prob, number of tokens,
mean entropy of the model's next-token distribution, and the fraction of tokens that were the model's top-1 choice.
Resume-safe.
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import common

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--answers", default=str(common.DATA / "ref_probes_v1.jsonl"))
ap.add_argument("--endpoints", default=None)
ap.add_argument("--limit", type=int, default=None)
a = ap.parse_args()

torch.set_num_threads(max(1, torch.get_num_threads()))
tok = AutoTokenizer.from_pretrained(a.model)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).to(DEV)
model.eval()

out = common.DATA / f"ll_{a.model.split('/')[-1]}_{Path(a.answers).stem}.jsonl"
done = set()
if out.exists():
    done = {(r["endpoint"], r["id"], r["repeat"]) for r in common.load_answers(out)}

rows = common.load_answers(a.answers)
if a.endpoints:
    eps = set(a.endpoints.split(","))
    rows = [r for r in rows if r["endpoint"] in eps]
rows = [r for r in rows if (r["endpoint"], r["id"], r["repeat"]) not in done][: a.limit]
print(f"{len(rows)} answers to score with {a.model} -> {out}")

with out.open("a", encoding="utf-8") as f, torch.no_grad():
    for n, r in enumerate(rows):
        prefix = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False,
                                         add_generation_prompt=True)
        p_ids = tok(prefix, add_special_tokens=False)["input_ids"]
        a_ids = tok(r["response"], add_special_tokens=False)["input_ids"]
        if not a_ids:
            continue
        ids = torch.tensor([p_ids + a_ids], device=DEV)
        logits = model(ids).logits[0, len(p_ids) - 1: -1].float().cpu()
        logp = torch.log_softmax(logits, -1)
        tgt = torch.tensor(a_ids)
        lp = logp[torch.arange(len(a_ids)), tgt]
        ent = -(logp.exp() * logp).sum(-1)
        top1 = (logp.argmax(-1) == tgt).float()
        rank = (logp > lp[:, None]).sum(-1).float() + 1          # RUT: rank of the chosen token
        f.write(json.dumps({"endpoint": r["endpoint"], "id": r["id"], "repeat": r["repeat"],
                            "n_tok": len(a_ids), "lp_sum": lp.sum().item(), "lp_mean": lp.mean().item(),
                            "ent_mean": ent.mean().item(), "top1": top1.mean().item(),
                            "logrank": torch.log(rank).mean().item(), "surp_ent": (lp + ent).mean().item(),
                            "lp_first20": lp[:20].mean().item(), "ent_tokens": [round(x, 3) for x in ent.tolist()],
                            "lp_tokens": [round(x, 3) for x in lp.tolist()]}) + "\n")
        if (n + 1) % 100 == 0:
            print(f"  {n + 1}/{len(rows)}", flush=True)
print("done")
