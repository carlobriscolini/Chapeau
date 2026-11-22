# Task 2 — Model Equality Testing (team Chapeau)

Goal: for 33 black-box LLM endpoints (`ep-01..ep-33`, 100 answers each, text only) decide which ones share the
same **base model (family)** and which ones serve **exactly the same model** (same checkpoint + quantization + LoRA),
while system prompts, personas, watermarks and decoding settings do *not* change model identity.

Final answer: [`submission/submission.csv`](submission/submission.csv).

## Method in one picture

```
query design ──► family identification ──► disguise detection ──► variant detection ──► grouping rule
(1 answer/prompt,   (likelihood under        (probe text, high-T,     (paired permutation     (ref structure +
 multilingual        local candidate          watermark repeats,       tests vs the            expected-MCC
 continuations)      models)                  truncated decoding)      watermark endpoint)     decision theory)
```

### 1. Query design — `prompts/`
* `make_ep_prompts.py` → `ep_phase1.jsonl` (70 prompts): 3 identity/system-prompt probes, 62 Wikipedia-style
  paragraph continuations (EN/ES/FR/DE/IT/PT/RU/ZH + code), 5 repeated prompts (a fixed-key watermark repeats itself).
  Language mix chosen from Gao et al. (ICLR 2025, Table 13): FR/ES continuations give the highest power against
  quantization.
* `make_phase2_prompts.py` → `ep_phase2.jsonl` (30 long FR/ES continuations, ~200 tokens each instead of ~70) used
  only on the 16 endpoints that remained ambiguous. One answer per prompt (as in RUT, Zhu et al. ICLR 2026):
  many different prompts generalise better than repeats.
* Querying: `src/run_ep.py` (scored endpoints, budget guard) and `src/probe_ref.py` (free practice endpoints).

### 2. Families — `src/fit_family.py`
Mean token log-probability / top-1 rate of each endpoint's answers under candidate open models. The true base model
explains the answers far better than the others: **A = Llama-3.2-1B-Instruct, B = Qwen2.5-1.5B-Instruct,
C = Qwen2.5-3B-Instruct**, 11 endpoints each.

### 3. Deployment "disguises" (same model, different text)
* **System prompt / persona**: read from the probe answers (fake names such as "Claude"/"BERT", quoted style rules).
* **High temperature**: very low likelihood under the base model; `src/ll_decode.py` + `src/dec_summary.py`
  fit the temperature by maximum likelihood (T* = 1.0, calibrated).
* **Watermark**: identical answers to repeated prompts (fixed-key, distortion-free sampling: same distribution as
  the base model, so it is our *reference sample of the base model*).
* **Truncated decoding (top-k / top-p)**: `src/ll_decode.py` records the base-model rank of every generated token.
  One endpoint per family never samples tokens beyond rank ~40–50 (0–0.09 % of tokens vs 0.3–1 % for all others,
  2.9 % for a strong LoRA on the practice endpoints) → base model with truncated decoding, i.e. the same model.

### 4. Variant detection — `src/likelihood.py`, `src/llr_quant.py`, `src/vs_wm.py`, `src/decide_plain.py`
Every candidate is compared with the family's watermark endpoint by **paired sign-flip permutation tests** over
shared prompts, on statistics computed under the local base model:
* `surp` = mean log p + mean entropy (calibration; best statistic in our simulations with known truth, `src/stat_eval.py`),
* `f20` = mean log-prob of the first 20 tokens, `len` = log length,
* **quantization likelihood ratio** (`src/llr_quant.py`): log q(y) − log p(y) with q = a locally simulated quantized
  copy (fp8 per-tensor / per-channel, fp8 W8A8 as in vLLM, int8 W8A8, int4 g128). A quantized endpoint has
  E[LLR] = KL(q‖p) > 0, the base model −KL(p‖q) < 0 (Neyman–Pearson directional test).
* **distilled-LoRA ratio** (`src/lora_distill.py`): a small LoRA trained (cross-fitted) to imitate a strong variant,
  used as a directional test for weaker strengths of the same adapter.
Detected variants: ep-15, ep-22 (fp8), ep-28, ep-03 (quantized), ep-05 (LoRA).

### 5. Grouping rule — `src/mcc_hyp.py`
On the practice endpoints the base group contains plain + system prompt + persona + watermark + temperature, and every
weight variant is distinct. We compute the **expected MCC** of each grouping strategy under this structure: when the
plain endpoint cannot be told apart from weak variants (fp8 / LoRA 0.3 are statistically invisible with 80 answers),
putting all ambiguous endpoints in separate models (MCC ≈ 0.76) beats guessing the plain (≈ 0.65) or merging them
(≈ 0.41). Final rule: **base group = all disguised endpoints (system prompt, persona, watermark, high temperature,
truncated decoding); every other endpoint is its own model.** A pre-registered rule (`src/decide_plain.py`,
`src/decide_v5.py`) would have added a candidate to the base group only if it were the *unique* endpoint compatible
with the watermark — never satisfied.

## Validation
All methods were first tested on the free practice endpoints `ref-01..ref-09` (known answers, `src/score_practice.py`)
and in local simulations with known truth (`src/simulate.py`, `src/sim_train_lora.py`, `src/stat_eval.py`).

## Running
Put the API key in `.api_key` (or `MET_API_KEY`) and the organisers' server URL in `HACKATHON_SERVER_URL`.
GPU jobs for JURECA are in `jobs/` (one family per GPU), submitted from the `task_2/` folder:
`sbatch -A <project> jobs/gpu_phase1.sh`.
```
python src/run_ep.py prompts/ep_phase1.jsonl --dry-run
python src/fit_family.py data/ep_phase1.jsonl --models Qwen/Qwen2.5-1.5B-Instruct,...
python src/likelihood.py MODEL --answers data/ep_phase1.jsonl ; python src/llr_quant.py MODEL data/ep_phase1.jsonl
python src/ll_decode.py MODEL --answers data/ep_phase1.jsonl data/ep_phase2.jsonl --endpoints ... --out dec.jsonl
python src/dec_summary.py dec.jsonl ; python src/decide_plain.py
```

## References
* Gao, Liang, Guestrin. *Model Equality Testing: Which Model Is This API Serving?* ICLR 2025.
* Zhu et al. *Auditing Black-Box LLM APIs with a Rank-Based Uniformity Test.* ICLR 2026.
* Kuditipudi et al. *Robust Distortion-free Watermarks for Language Models.* TMLR 2024.
* Cai et al. *Are You Getting What You Pay For? Auditing Model Substitution in LLM APIs.* 2025.
