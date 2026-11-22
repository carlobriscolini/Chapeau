# Task 3 — Machine Unlearning (team Chapeau)

Team **Chapeau**'s solution to the *Unlearning* task of the European AI Championship, Paris 2026.

**Final public-split score: ≈ 0.41**. Our first working submission scored 0.089.

> The task data (model and forget images) belong to the organisers and are **not** included. The final submission is
> a 344 MB state dict (too large for git): it is reproduced by the three commands in *Final recipe*.

---

## The task

- **What you get:**
  - `model.pt`, a ViT-B/16 (timm, 100 classes, LayerScale) trained **from scratch** on a private image dataset;
  - `forget_images.npz`, 3 500 of its training images.
- **What you submit:** a state dict with the same architecture.
- **The target:** the submission should behave like the organisers' *gold* models, which were retrained without the
  forget images.

The forget set touches 45 classes:
- **9 classes** are removed entirely (≈130 images each);
- **36 classes** are removed partially;
- **55 classes** are untouched.

The score is roughly:

```
score = forget quality × retain quality
forget quality ≈ how well the loss distribution (mean, std) and predictions on the forget images
                 (exact copies and altered versions) match the gold models
retain quality ≈ accuracy ratio × 1 / (1 + KL) on a hidden control set   (utility gate: 70 %)
```

The server returns only the total, computed on a 30 % public split. The final ranking uses the other 70 %.

Constraints that shaped the solution:
- **No retain data:** the training set is never given, and organisers confirmed it would not be disclosed.
  Inspecting the forget images showed that the dataset consists of AI-generated "cinematic" images (warm light,
  bokeh). Its classes don't map onto ImageNet.
- **About 1 submission every 10 minutes.**
- **One GPU node per team** on a shared cluster (JURECA).

---

## Final recipe (≈ 0.41)

Three steps, all in this repo:

1. **Build a synthetic retain pool.** SD-Turbo text-to-image in the dataset's style, labelled by the original model
   (`gen_pool.py`, three rounds), plus 8 000 mini-ImageNet photos (`prepare_external.py`).
2. **Re-initialise and re-distil the top half of the network** (`repair.py`):
   - re-initialise blocks 6–11 (truncated normal, LayerScale γ = 1e-4) and keep blocks 0–5 and the embeddings frozen;
   - distil the original model on the synthetic pool only, never on the forget images. The loss is KL at T = 2 on
     the same augmented view for teacher and student, plus 0.1 × logit-MSE;
   - run 8 000 steps with class-balanced sampling, where the 45 forget classes are sampled at weight **0.35**.
3. **Scale the classifier head** (`headscale.py`): `head.weight` and `head.bias` × **1.6**, training-free.

Commands:

```bash
python src/repair.py --retain genpoolT+genpoolP+external --balanced --reinit-from 6 \
    --lr-new 3e-4 --lr 1e-5 --steps 8000 --mse 0.1 --fcls-weight 0.35 --seed 1
python src/headscale.py --model $OUTPUT_DIR/repair_orig_reinit6_..._fw0.35_mse0.1_seed1_s8000.pt --gammas 1.6 --save
python src/submit.py $OUTPUT_DIR/..._hs1.6.pt
```

On one A100, `repair.py` takes about 10 minutes.

### Why it works

- **Real removal beats hiding.**
  - **Gradient-based unlearning only hides the memory.** This includes distillation towards a "never-seen" target,
    NPO, SalUn masks, weight-space extrapolation and SSD. Briefly fine-tuning such a model on retain-like data
    brought the forget loss back to ≈ 0 within 300 steps.
  - **Re-initialisation actually deletes it.** Blocks 6–11 are thrown away and rebuilt from data that never
    contained the forget images. This alone doubled the score (0.125 → 0.24).
- **In-style synthetic data matters.** Our retain proxies on real photos covered only 61/100 classes. Generated
  images in the dataset's look cover 99/100, and the re-distilled model generalises much better.
- **Gold models are more confident than a distilled student.** The student matches the teacher's soft outputs.
  A model retrained from scratch is confidently right on easy images and confidently wrong on unseen hard ones,
  which gives a forget-loss distribution with a high std. Multiplying the logits by γ reproduces this.
  - **The gain was the largest single jump:** 0.25 → 0.39.
  - **The server peak was at γ ≈ 1.6:** 1.25 → 0.334, 1.5 → 0.384, 1.6 → 0.390, 1.8 → 0.359.
- **Agreement with the original on the forget set must stay low.** Every variant that pushed the student closer to
  the original lost score, even when it improved our local retain metrics. Examples: more steps, more MSE,
  re-initialising fewer blocks. Sampling the forget classes less (weight 0.35) lowers that agreement without hurting
  the untouched classes (+0.006 to +0.012).
- **Seeds matter.** The same recipe varies by about ±0.006 on the public split. The final 0.41 is the best of three
  seeds.

---

## Score history (public split)

| Score | What |
|---|---|
| 0 | identity (sanity check) |
| 0.089 | distil the forget images towards a "never-seen" target (original model on strongly augmented views), no retain anchor |
| 0.124 | + anchor on external photos, then weight extrapolation θ_orig + 1.75·Δ (`run_unlearn.py`, `soup.py`) |
| 0.125 | + patch-mask target |
| **0.2425** | **re-init blocks 6–11 + distil on generated pool** (`repair.py --reinit-from 6`) |
| 0.2534 | + photo-style pool + external photos, logit MSE 0.1, 8 000 steps |
| **0.3338** | **+ head scale γ = 1.25** (`headscale.py`) |
| 0.3842 / 0.3899 / 0.3588 | γ = 1.5 / 1.6 / 1.8 |
| 0.3961 | same recipe, seed 2 (seed noise) |
| 0.3962 / 0.3986 | re-init from block 5 / + forget-class weight 0.35 |
| **0.4024** | **reinit 6 + forget-class weight 0.35 + γ 1.6** |
| **≈ 0.41** | same, seed 1 |

### What did not work (server scores)

| Idea | Score | Why we think it failed |
|---|---|---|
| SSD (selective synaptic dampening), Fisher on external photos | 0.048 | Whole classes collapsed, because the Fisher information only protected classes present in the photos. |
| Training only the MLP layers at a high learning rate | 0.037 | The model overfitted the retain anchor. Held-out proxies looked great but were fooled. |
| Head-bias surgery on the 9 fully removed classes (−4) | 0.109 | Lower than the 0.124 of the base model it was applied to. |
| Repairing an extrapolated model while keeping its forget outputs | 0.110 | Std too low, and the memory was only hidden. |
| Full student from a public pretrained ViT (augreg2) | 0.232 | The original was trained from scratch, so the gold models were too. A public pretrain moves the student away from them. |
| Re-init from block 3 / 5 / 7 | 0.251 / 0.396 / – | Block 6 is the best trade-off. Block 7 keeps too much memory (agreement 0.46). |
| More retain fidelity (aug medium, MSE 0.3–1, 12k–20k steps) + head scale | 0.337 | Agreement on the forget set rose to 0.49. |
| Sharpening only on the forget images (`--keep-gamma`) | 0.305 (0.351 with γ 1.2) | The server rewards the *global* logit scale, not just on the forget set. |
| Forget-class weight 0.35 **and** fully removed classes at weight 0 | 0.383 | Removing those classes from the retain pool hurts. |
| Hard-label cross-entropy on confident pool images (`--ce`) | not submitted | Accuracy dropped everywhere, and the std did not increase. |

Other ideas closed early on local diagnostics:
- k-NN targets: the pool was too far from the forget images, with nearest-neighbour cosine 0.51;
- model-inversion anchors, SD-Turbo "siblings" of the forget images, weight-noise targets;
- model soups, interpolating the re-initialised model back toward the original.

---

## Local diagnostics

No labels and no gold model are available, so `diagnostics.py` reports proxies, using the original model's argmax as
the label:

| Proxy | What it measures |
|---|---|
| `forget/exact`, `forget/altered` | Agreement with the original, loss mean and std, and KL on the forget images and on augmented copies. Also split by fully removed vs partially removed classes. |
| `retain_proxy/ext` | KL to the original on held-out real photos |
| `retain_proxy/gpool` | KL to the original on held-out generated images, split by untouched and forget classes |
| `retain_proxy/mix` | KL to the original on mixes of images |
| `perclass.py` | Per-class forget accuracy histogram. It catches class collapse, which the averages hide. |

Lessons:
- **Comparisons within one family transfer.** Within one family of models (same method, varying a knob), these proxies
  ordered submissions correctly.
- **Comparisons across families can mislead.** Across families, and especially across logit scales, the retain
  proxies were misleading. Lower KL to the original did not mean a higher server score.

---

## Repository layout

```
task_3/
├── src/            shared modules + the final pipeline
├── data_prep/      building the retain pools, data inspection
├── experiments/    earlier approaches, kept for reference
└── jobs/           Slurm job scripts (JURECA dc-gpu, A100)
```

All commands are run from the `task_3/` folder, e.g. `python src/repair.py ...`. Scripts in `data_prep/` and
`experiments/` import the shared modules from `src/` automatically.

**`src/` — final pipeline**

| File | Purpose |
|---|---|
| `common.py` | Model and data loading, server-identical preprocessing and prediction, saving submissions with checks |
| `repair.py` | Re-init + distillation. Options: `--reinit-from`, `--fcls-weight`, `--full-weight`, `--mse`, `--balanced`, `--aug`, `--init`, `--keep[-gamma]`, `--ce`, `--pretrained`, `--dedup` |
| `headscale.py` | Multiplies the head by γ and reports diagnostics (`--save` writes the submission) |
| `diagnostics.py`, `perclass.py` | Local proxies (see above) |
| `submit.py` | Submits to the evaluation server and waits out the rate limit. Needs `UNLEARN_SERVER_URL` and `UNLEARN_API_KEY`. |
| `run_unlearn.py`, `targets.py`, `augment.py` | Retain pools and losses shared with `repair.py`. Also the earlier distillation-based unlearning: "never-seen" targets, NPO, SalUn masks, retain anchors. |

**`data_prep/` — retain pools**

| File | Purpose |
|---|---|
| `gen_pool.py` | SD-Turbo synthetic retain pool: round 1 cinematic, `--targeted` round 2, `--targeted --photo` round 3 |
| `prepare_external.py` | Downloads mini-ImageNet photos (labels unused) |
| `genpool_stats.py`, `inspect_data.py`, `ext_labels.py` | Pool statistics and data inspection |

**`experiments/` — earlier approaches**

| File | Purpose |
|---|---|
| `soup.py` | Weight-space extrapolation and interpolation |
| `ssd.py` | Selective synaptic dampening |
| `headshift.py` | Head-bias surgery |
| `knn_target.py` | k-NN targets |
| `invert.py` | Model-inversion anchors |
| `gen_siblings.py` | Image-to-image variations of the forget images |
| `check_base.py` | Is the model a fine-tune of a public checkpoint? (it is not) |
| `evaluate.py` | Diagnostics for a saved submission |

**`jobs/` — Slurm**

| File | Purpose |
|---|---|
| `main.sh` | 1 GPU, `python <script> <args>` |
| `cmd4_home.sh` | Any 4 commands on a 4-GPU node, one per GPU. This is what we used at the end. |
| `main4.sh`, `main4_home.sh` | Many `run_unlearn.py` configurations over 4 GPUs |
| `genpool.sh`, `gen.sh`, `inv.sh` | Image generation and inversion |

---

## Running it

Expected layout: the task folder holds the data, a virtual env and this folder (copy `task_3/` there, or point
`DATA_DIR` / `TASK_DIR` at the data). Dependencies: `requirements.txt`.

```
<task dir>/
├── model.pt, forget_images.npz   # task data (from the organisers)
├── .venv/                        # torch + timm (+ diffusers/transformers for gen_pool.py)
├── output/                       # pools, caches, models (OUTPUT_DIR, default <task dir>/output)
└── task_3/
```

```bash
cd <task dir>/task_3
source ../.venv/bin/activate
export OUTPUT_DIR=../output            # or anywhere with space; models are ~340 MB each

# 1. retain pools (once)
python data_prep/prepare_external.py --n 10000        # login node, needs internet (see the docstring)
sbatch -A <project> jobs/genpool.sh                     # round 1  -> genpool.npz
sbatch -A <project> jobs/genpool.sh --targeted          # round 2  -> genpool_t.npz
sbatch -A <project> jobs/genpool.sh --targeted --photo  # round 3  -> genpool_p.npz

# 2. final recipe (four seeds in one 4-GPU job)
B="--retain genpoolT+genpoolP+external --balanced --reinit-from 6 --lr-new 3e-4 --lr 1e-5 --steps 8000 --mse 0.1 --fcls-weight 0.35"
sbatch -A <project> jobs/cmd4_home.sh "python src/repair.py $B --seed 0" "python src/repair.py $B --seed 1" \
    "python src/repair.py $B --seed 2" "python src/repair.py $B --seed 3"

# 3. head scale + submit
python src/headscale.py --model <model>.pt --gammas 1.6 --save
export UNLEARN_SERVER_URL=... UNLEARN_API_KEY=...
python src/submit.py <model>_hs1.6.pt
```

Submit the job scripts from the `task_3/` folder: they `cd` there and take its parent as the task folder (override
with `TASK_DIR=...`). `cmd4_home.sh` writes logs and models to `$HOME/runs`, because our scratch file quota was full
during the event.
