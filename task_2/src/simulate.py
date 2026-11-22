"""Local simulation of the deployment variants on a model we choose (CPU ok for small models).

    python simulate.py MODEL_ID [--reps 4] [--variants plain,fp8,...] [--batch 16]

Writes data/sim_<model>.jsonl in the same format as the API answers (endpoint = "sim-<variant>") and
data/sim_<model>_truth.csv (endpoint,family,model,kind) usable with the evaluation scripts. Resume-safe.

Variants (same KINDS as the reference endpoints; details are ours):
  plain       sampling at T=0.7 (assumed default of the server's plain endpoint)
  sysprompt   T=0.7 + a system prompt drawn at random per request from a pool of task instructions
  persona     T=0.7 + a persona system prompt drawn at random per request
  watermark   T=0.7 + fixed-sampling watermark: Gumbel-max with noise seeded by hash(key, last 4 tokens)
  t10         sampling at T=1.0
  fp8         weights cast to float8_e4m3 (per-channel scale), T=0.7
  int8        weights rounded to int8 (per-channel scale), T=0.7
  lora03/07/15  our LoRA (sim_train_lora.py) at strength 0.3/0.7/1.5, T=0.7
  lora07persona LoRA 0.7 + persona -> SAME model as lora07 (trap on top of a weight change)
"""
import argparse
import csv
import hashlib
import json
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

import common

SYSPROMPTS = [
    "You are a customer support assistant for an online electronics store. Be polite and helpful.",
    "Answer every question as concisely as possible.",
    "You are an AI tutor for high-school students. Explain things step by step.",
    "You are a helpful assistant. Always format your answers clearly.",
    "You are a writing assistant that helps users improve their texts.",
    "You are a research assistant. Be precise and factual.",
]
PERSONAS = [
    "You are Zara, a cheerful and enthusiastic assistant created by BrightPath AI. You love exclamation marks.",
    "You are Captain Finn, a friendly old sailor who answers everything with sea stories.",
    "You are Dr. Elena Ruiz, a calm and thoughtful scientist who explains things with care.",
    "You are Max, a sarcastic but helpful tech geek who works for GadgetHub.",
    "You are Luna, a poetic assistant who likes elegant and imaginative language.",
    "You are Coach Sam, a motivational fitness coach who is always energetic.",
]
PROMPT_IDS = [  # 25 prompts x 4 reps = 100 answers per variant (the real budget per endpoint)
    "cont_bank", "cont_river", "cont_physics", "cont_fr", "h1_cont_volcano", "h1_cont_jazz", "h1_cont_cell",
    "h1_cont_economy", "h1_cont_story", "h1_cont_de",
    "h3_months", "h3_add2", "h3_count_back", "copy_text", "str_count",
    "h4_pick_abcd", "h4_pick_1_10", "rand_color", "h4_pick_fruit",
    "id_who", "gen_story", "gen_explain", "h2_three_sent", "h1_open_para", "fact_primes",
]
ALL_VARIANTS = ["plain", "sysprompt", "persona", "watermark", "t10", "fp8", "int8", "lora03", "lora07", "lora15",
                "lora07persona"]
MODEL_LABEL = {"plain": "S", "sysprompt": "S", "persona": "S", "watermark": "S", "t10": "S", "fp8": "Q8F",
               "int8": "Q8I", "lora03": "L03", "lora07": "L07", "lora07persona": "L07", "lora15": "L15"}


class FixedSamplingWatermark(LogitsProcessor):
    """Gumbel-max sampling with pseudo-random noise fixed by (key, previous 4 tokens): same context -> same token."""

    def __init__(self, temperature, key=42):
        self.t, self.key = temperature, key

    def __call__(self, input_ids, scores):
        out = torch.full_like(scores, -float("inf"))
        for i in range(scores.shape[0]):
            ctx = ",".join(map(str, input_ids[i, -4:].tolist()))
            seed = int(hashlib.sha256(f"{self.key}|{ctx}".encode()).hexdigest()[:15], 16)
            g = torch.Generator().manual_seed(seed)
            u = torch.rand(scores.shape[1], generator=g).clamp(1e-10, 1 - 1e-10)
            j = (scores[i] / self.t - torch.log(-torch.log(u))).argmax()
            out[i, j] = 0.0
        return out


def quantize_(model, kind):
    with torch.no_grad():
        for name, m in model.named_modules():
            if isinstance(m, torch.nn.Linear) and "lm_head" not in name:
                W = m.weight.data
                if kind == "fp8":
                    s = W.abs().amax(1, keepdim=True).clamp(min=1e-12) / 448.0
                    m.weight.data = (W / s).to(torch.float8_e4m3fn).to(torch.float32) * s
                elif kind == "int8":
                    s = W.abs().amax(1, keepdim=True).clamp(min=1e-12) / 127.0
                    m.weight.data = (W / s).round().clamp(-127, 127) * s


def load_prompts(path=None):
    if path:  # a whole prompt file (e.g. the ep prompt set), every prompt in it
        return [(r["id"], r["prompt"]) for r in common.load_answers(path)]
    P = {}
    for f in ["probes_v1.jsonl", "probes_v2.jsonl", "probes_v3.jsonl"]:
        for r in common.load_answers(common.ROOT / f):
            P[r["id"]] = r["prompt"]
    return [(pid, P[pid]) for pid in PROMPT_IDS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--variants", default=",".join(ALL_VARIANTS))
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--tag", default="")
    ap.add_argument("--prompts-file", default=None, help="use all prompts of this jsonl instead of PROMPT_IDS")
    a = ap.parse_args()

    short = a.model.split("/")[-1]
    lora_dir = common.DATA / f"lora_{short}"
    short += a.tag
    out = common.DATA / f"sim_{short}.jsonl"
    with open(common.DATA / f"sim_{short}_truth.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["endpoint", "family", "model", "kind"])
        for v in ALL_VARIANTS:
            w.writerow([f"sim-{v}", "F", MODEL_LABEL[v], v])
    done = {(r["endpoint"], r["id"], r["repeat"]) for r in common.load_answers(out)} if out.exists() else set()

    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    prompts = load_prompts(a.prompts_file)
    rng = random.Random(0)

    for v in a.variants.split(","):
        jobs = [(pid, p, k) for pid, p in prompts for k in range(a.reps) if (f"sim-{v}", pid, k) not in done]
        if not jobs:
            continue
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).eval()
        if v in ("fp8", "int8"):
            quantize_(model, v)
        if v.startswith("lora"):
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, lora_dir).eval()
            strength = {"lora03": 0.3, "lora07": 0.7, "lora07persona": 0.7, "lora15": 1.5}[v]
            for m in model.modules():
                if hasattr(m, "scaling") and isinstance(m.scaling, dict):
                    for key in m.scaling:
                        m.scaling[key] = m.scaling[key] * strength
        temp = 1.0 if v == "t10" else 0.7
        print(f"[{v}] {len(jobs)} answers, T={temp}", flush=True)
        for b in range(0, len(jobs), a.batch):
            chunk = jobs[b:b + a.batch]
            texts = []
            for pid, p, k in chunk:
                msgs = [{"role": "user", "content": p}]
                if v == "sysprompt":
                    msgs.insert(0, {"role": "system", "content": rng.choice(SYSPROMPTS)})
                if v in ("persona", "lora07persona"):
                    msgs.insert(0, {"role": "system", "content": rng.choice(PERSONAS)})
                texts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
            kw = dict(max_new_tokens=a.max_new, pad_token_id=tok.pad_token_id)
            if v == "watermark":
                kw.update(do_sample=False, logits_processor=LogitsProcessorList([FixedSamplingWatermark(temp)]))
            else:
                kw.update(do_sample=True, temperature=temp, top_k=0, top_p=1.0)
            with torch.no_grad():
                gen = model.generate(**enc, **kw)
            outs = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            with out.open("a", encoding="utf-8") as f:
                for (pid, p, k), o in zip(chunk, outs):
                    f.write(json.dumps({"endpoint": f"sim-{v}", "id": pid, "repeat": k, "prompt": p,
                                        "response": o.strip()}, ensure_ascii=False) + "\n")
            print(f"  [{v}] {min(b + a.batch, len(jobs))}/{len(jobs)}", flush=True)
        del model


if __name__ == "__main__":
    main()
