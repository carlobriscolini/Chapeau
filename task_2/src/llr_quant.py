"""Likelihood-ratio test "quantized vs full precision" with a locally simulated quantized copy of the base model.

For each answer y: LLR = mean_t [ log q(y_t) - log p(y_t) ], p = base model, q = our fp8/int8 copy of it.
If the endpoint serves the quantized model, E[LLR] > 0 (= KL(q||p)); if it serves the base, E[LLR] < 0 (= -KL(p||q)).
Several quantization schemes are tried because we do not know the organizers' exact one.

    python llr_quant.py MODEL ANSWERS [--endpoints ...] [--schemes fp8_tensor,fp8_channel,int8_channel] [--prompts RE]
Writes data/llrq_<model>_<answers>.jsonl (one line per answer, all schemes). Resume-safe. GPU if available.
"""
import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import common

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("answers")
ap.add_argument("--endpoints", default=None)
ap.add_argument("--schemes", default="fp8_tensor,fp8_channel,int8_channel")
ap.add_argument("--prompts", default=".*")
ap.add_argument("--tag", default="")
ap.add_argument("--out", default=None, help="explicit output path (e.g. reuse an existing file: scratch inode quota)")
a = ap.parse_args()
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def quantize(model, scheme):
    with torch.no_grad():
        for name, m in model.named_modules():
            if isinstance(m, torch.nn.Linear) and "lm_head" not in name:
                W = m.weight.data.float()
                if scheme == "fp8_tensor":    # vLLM-style dynamic fp8: one scale per weight matrix
                    s = W.abs().max().clamp(min=1e-12) / 448.0
                    Q = (W / s).to(torch.float8_e4m3fn).float() * s
                elif scheme == "fp8_channel":
                    s = W.abs().amax(1, keepdim=True).clamp(min=1e-12) / 448.0
                    Q = (W / s).to(torch.float8_e4m3fn).float() * s
                elif scheme in ("int8_channel", "int8_w8a8"):
                    s = W.abs().amax(1, keepdim=True).clamp(min=1e-12) / 127.0
                    Q = (W / s).round().clamp(-127, 127) * s
                elif scheme == "fp8_w8a8":       # vLLM fp8: weights per-tensor + activations per-token dynamic
                    s = W.abs().max().clamp(min=1e-12) / 448.0
                    Q = (W / s).to(torch.float8_e4m3fn).float() * s
                elif scheme == "int4_g128":      # RTN 4-bit, groups of 128 inputs (≈ GPTQ/AWQ grid)
                    o, i = W.shape
                    g = W.reshape(o, i // 128, 128) if i % 128 == 0 else W.reshape(o, 1, i)
                    s = g.abs().amax(-1, keepdim=True).clamp(min=1e-12) / 7.0
                    Q = ((g / s).round().clamp(-8, 7) * s).reshape(o, i)
                m.weight.data = Q.to(m.weight.dtype)
                if scheme.endswith("w8a8"):
                    m.register_forward_pre_hook(act_fp8 if scheme.startswith("fp8") else act_int8)
    return model


def act_fp8(mod, inp):
    x = inp[0]
    s = x.abs().amax(-1, keepdim=True).clamp(min=1e-12) / 448.0
    return ((x / s).to(torch.float8_e4m3fn).to(x.dtype) * s,)


def act_int8(mod, inp):
    x = inp[0]
    s = x.abs().amax(-1, keepdim=True).clamp(min=1e-12) / 127.0
    return ((x / s).round().clamp(-127, 127) * s,)


tok = AutoTokenizer.from_pretrained(a.model)
rows = [r for r in common.load_answers(a.answers) if re.search(a.prompts, r["id"]) and r["response"].strip()]
if a.endpoints:
    rows = [r for r in rows if r["endpoint"] in a.endpoints.split(",")]
out = common.DATA / f"llrq{a.tag}_{a.model.split('/')[-1]}_{Path(a.answers).stem}.jsonl"
out = Path(a.out) if a.out else out
done = {(r["endpoint"], r["id"], r.get("repeat", 0)) for r in common.load_answers(out)} if out.exists() else set()
rows = [r for r in rows if (r["endpoint"], r["id"], r.get("repeat", 0)) not in done]
print(f"{len(rows)} answers -> {out}", flush=True)

enc = []
for r in rows:
    prefix = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True)
    p_ids = tok(prefix, add_special_tokens=False)["input_ids"]
    a_ids = tok(r["response"], add_special_tokens=False)["input_ids"][:300]
    enc.append((p_ids, a_ids))


def score(model):
    res = []
    with torch.no_grad():
        for p_ids, a_ids in enc:
            logits = model(torch.tensor([p_ids + a_ids], device=DEV)).logits[0, len(p_ids) - 1:-1].float()
            lp = torch.log_softmax(logits, -1)[torch.arange(len(a_ids)), torch.tensor(a_ids, device=DEV)]
            res.append(lp.cpu())
    return res


base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).to(DEV).eval()
LP = {"base": score(base)}
for sch in a.schemes.split(","):
    q = quantize(AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32), sch).to(DEV).eval()
    LP[sch] = score(q)
    del q
    print(f"  scored {sch}", flush=True)
with out.open("a", encoding="utf-8") as f:
    for i, r in enumerate(rows):
        line = {"endpoint": r["endpoint"], "id": r["id"], "repeat": r.get("repeat", 0), "n_tok": len(enc[i][1]),
                "lp_base": LP["base"][i].mean().item()}
        for sch in a.schemes.split(","):
            line["llr_" + sch] = (LP[sch][i] - LP["base"][i]).mean().item()
        f.write(json.dumps(line) + "\n")
print("done")
