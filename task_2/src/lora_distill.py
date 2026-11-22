"""Directional test for weak LoRA variants: "distil" the LoRA of a strong variant endpoint, then test the others.

Idea (refs: LoRA 0.3 / 0.7 / 1.5 are the SAME adapter at different strengths): train a small LoRA on the base model
to imitate the answers of a clearly-different "teacher" endpoint (e.g. a strong LoRA variant). That gives a local
model q pointing in the adapter's direction. For every other endpoint compute LLR = mean_t[log q(y_t) - log p(y_t)]
on its answers (p = base). An endpoint carrying the same adapter at a weaker strength has a higher LLR than the
plain / watermark endpoint (Neyman-Pearson: a test aimed at the right alternative is the most powerful).
Cross-fitting: prompts are split in 2 folds; answers to a prompt are scored by the LoRA trained on the OTHER fold,
so the teacher's own text for that prompt is never memorised.

    python lora_distill.py MODEL --answers A1.jsonl A2.jsonl --teacher ep-27 --endpoints ep-09,ep-10,... [--steps 150]
Writes data/lora_<model>_<teacher>.jsonl (endpoint, id, n_tok, llr_mean, llr_sum). Manual LoRA: no peft needed.
"""
import argparse
import hashlib
import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import common

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--answers", nargs="+", required=True)
ap.add_argument("--teacher", required=True)
ap.add_argument("--endpoints", required=True)
ap.add_argument("--prompts", default=r"^(cont_|c_|l_)")
ap.add_argument("--steps", type=int, default=150)
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--tag", default="")
ap.add_argument("--out", default=None)
a = ap.parse_args()
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

rows = []
for f in a.answers:
    rows += [r for r in common.load_answers(f) if "#" not in r["id"] and re.match(a.prompts, r["id"])
             and r["response"].strip()]
fold = lambda pid: int(hashlib.md5(pid.encode()).hexdigest(), 16) % 2
tok = AutoTokenizer.from_pretrained(a.model)


def encode(r):
    prefix = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True)
    p = tok(prefix, add_special_tokens=False)["input_ids"]
    y = tok(r["response"], add_special_tokens=False)["input_ids"][:300]
    return p, y


class LoRA(torch.nn.Module):
    scale = 1.0  # class-level switch: 0 = base model

    def __init__(self, base, r):
        super().__init__()
        self.base = base
        self.A = torch.nn.Parameter(torch.randn(r, base.in_features, device=base.weight.device) / base.in_features ** 0.5)
        self.B = torch.nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))

    def forward(self, x):
        out = self.base(x)
        if LoRA.scale:
            out = out + LoRA.scale * ((x.to(self.A.dtype) @ self.A.T) @ self.B.T).to(out.dtype)
        return out


def build():
    m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).to(DEV)
    for p in m.parameters():
        p.requires_grad_(False)
    names = [n for n, mod in m.named_modules() if isinstance(mod, torch.nn.Linear) and "lm_head" not in n]
    for n in names:
        parent, child = m.get_submodule(n.rsplit(".", 1)[0]), n.rsplit(".", 1)[1]
        setattr(parent, child, LoRA(getattr(parent, child), a.rank))
    return m


def token_lp(m, p, y):
    logits = m(torch.tensor([p + y], device=DEV)).logits[0, len(p) - 1:-1].float()
    return torch.log_softmax(logits, -1)[torch.arange(len(y)), torch.tensor(y, device=DEV)]


test_eps = a.endpoints.split(",")
out = common.DATA / f"lora_{a.model.split('/')[-1]}_{a.teacher}{a.tag}.jsonl"
if a.out:
    from pathlib import Path
    out = Path(a.out)
res = []
for k in (0, 1):
    train = [encode(r) for r in rows if r["endpoint"] == a.teacher and fold(r["id"]) == k]
    test = [r for r in rows if r["endpoint"] in test_eps and fold(r["id"]) != k]
    m = build()
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    m.train()
    LoRA.scale = 1.0
    for step in range(a.steps):
        p, y = train[step % len(train)]
        loss = -token_lp(m, p, y).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0:
            print(f"fold {k} step {step} loss {loss.item():.3f}", flush=True)
    m.eval()
    with torch.no_grad():
        for r in test:
            p, y = encode(r)
            LoRA.scale = 1.0
            lq = token_lp(m, p, y)
            LoRA.scale = 0.0
            lp = token_lp(m, p, y)
            d = (lq - lp).cpu()
            res.append({"endpoint": r["endpoint"], "id": r["id"], "n_tok": len(y), "llr_mean": d.mean().item(),
                        "llr_sum": d.sum().item()})
    print(f"fold {k}: trained on {len(train)} {a.teacher} answers, scored {len(test)}", flush=True)
    del m, opt
with out.open("w") as f:
    for x in res:
        f.write(json.dumps(x) + "\n")
print("done", out)
