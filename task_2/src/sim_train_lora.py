"""Train a small LoRA adapter on a local model (CPU is fine for <=0.5B models), to simulate the 'LoRA' variants.

    python sim_train_lora.py MODEL_ID [--n 400] [--steps 60]

Data: instruction/response pairs from tatsu-lab/alpaca (generic assistant data, longer answers than the
base model tends to give). Saves the adapter in data/lora_<model>/.
"""
import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

import common

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--n", type=int, default=400)
ap.add_argument("--steps", type=int, default=60)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--lr", type=float, default=3e-4)
a = ap.parse_args()

torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained(a.model)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32)
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
                                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                         "gate_proj", "up_proj", "down_proj"]))
model.print_trainable_parameters()

ds = load_dataset("tatsu-lab/alpaca", split="train").shuffle(seed=0)
ds = ds.filter(lambda r: len(r["output"]) > 300 and not r["input"]).select(range(a.n))
examples = []
for r in ds:
    prefix = tok.apply_chat_template([{"role": "user", "content": r["instruction"]}], tokenize=False,
                                     add_generation_prompt=True)
    p = tok(prefix, add_special_tokens=False)["input_ids"]
    y = tok(r["output"] + tok.eos_token, add_special_tokens=False)["input_ids"]
    ids = (p + y)[:384]
    labels = ([-100] * len(p) + y)[:384]
    examples.append((ids, labels))

opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
model.train()
for step in range(a.steps):
    batch = [examples[(step * a.bs + i) % len(examples)] for i in range(a.bs)]
    L = max(len(x[0]) for x in batch)
    ids = torch.tensor([x[0] + [tok.pad_token_id] * (L - len(x[0])) for x in batch])
    lab = torch.tensor([x[1] + [-100] * (L - len(x[1])) for x in batch])
    att = torch.tensor([[1] * len(x[0]) + [0] * (L - len(x[0])) for x in batch])
    loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
    loss.backward()
    opt.step()
    opt.zero_grad()
    if step % 10 == 0 or step == a.steps - 1:
        print(f"step {step} loss {loss.item():.3f}", flush=True)

out = common.DATA / f"lora_{a.model.split('/')[-1]}"
model.save_pretrained(out)
print("saved", out)
