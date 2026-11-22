# Task 1 — Watermark Localization (team Chapeau)

Team **Chapeau**'s solution to the *Localize Image Watermark* task of the European AI Championship, Paris 2026.

**Final public-split score: 0.7444** (`submission/sub15.pt`, manifest `submission/sub15.json`). Our first submission
scored 0.5731.

> The task data, the official watermark models and our checkpoints are **not** included (data belongs to the organisers,
> checkpoints are several GB). `submission/sub15.pt` is the exact file we submitted (1 020 masks, 128 × 128).

---

## The task

- **Input:** 512 × 512 images, each watermarked with **PixelSeal**, **TrustMark** or **StegaStamp** in a rectangular
  subset of the image, then edited: outpainting, object removal / alteration, object addition, object transplantation.
  For every test image we know the method and the embedded message.
- **Output:** a binary 128 × 128 mask (1 = watermarked).
- **Metric:** per-image mIoU over the two classes, averaged over the 1 020 test images (public score = 30 % of them).
- **Train:** 510 images (170 per method) with masks.

What the data turned out to be (found by looking at our worst errors, see *Findings*):
- the images are **AI-generated** (a FLUX-family model: its autoencoder reconstructs them with half the error of SD/SDXL);
- edits are **objects** with sharp outlines (an added foal, a transplanted house…);
- many hard cases are **tiny watermarked rectangles** (0.3–2 % of the image) in an otherwise clean image, and
  **scattered blobs** (10–30 small random holes, invisible to the eye);
- half of the edits cover < 5 % of the image: for PixelSeal and TrustMark the watermark signal there is too weak
  even for a perfect decoder, so most of our gains came from **shape priors** and **metric-aware bets**.

## Final pipeline (sub15)

| Step | What | Code |
|---|---|---|
| 1 | Per-method segmenters: encoder = official extractor (`wm:pixelseal`, `wm:trustmark`) or ResNet-50, FPN decoder, logits on the 128 grid; loss BCE + soft per-image mIoU | `research/train_seg.py`, `research/wm_encoder.py` |
| 2 | Training data per batch: on-GPU synthetic composites (we embed the watermark with the official embedders, re-implemented batched on GPU) + 8 real train images with **same-position CutMix** + static augmented set | `research/gpu_synth.py`, `research/gpu_embed.py`, `train_seg.real_cutmix` |
| 3 | **Key change:** synthetic *hosts* are the official train **and test** images watermarked with *another* method (FLUX content instead of COCO photos), with real edit shapes from the train folds (`--official-hosts`). One seed of this alone ≈ our whole previous 100-model ensemble | `gpu_synth.add_official_hosts` |
| 4 | 5-fold protocol: one model per validation fold and type; honest out-of-fold (OOF) estimate on the 510 train images. OOF tracked the public score within ~0.003 on every submission, so every decision was taken on OOF | `research/folds.json`, `research/build_sub_v2.py` |
| 5 | Ensemble: mean of logits per method over 225 checkpoints (PS 57, TM 112, SS 56: 5 FLUX-host seeds + earlier CutMix rounds) | `research/cache_logits.py`, `research/build_sub_v2.py` |
| 6 | Message-guided rectangle search for outpainting: the crop of the embedding rectangle decodes the known message, so we search the rectangle maximising bit agreement and replace the mask when confident (PixelSeal, TrustMark) | `research/rect_refine.py`, `build_sub_v2.apply_rect` |
| 7 | **SAM boundary refinement** (PixelSeal, TrustMark): each predicted edited component prompts Segment Anything (box + most confident point); the SAM mask replaces it if IoU ≥ 0.7. Fixes the blurry borders of the 256-px watermarks (+0.010 PS, +0.005 TM OOF, all folds) | `research/sam_refine.py` |
| 8 | **Rectangle bet** on images predicted (almost) single-class: add the max-evidence axis-aligned rectangle (exact 2-D max-subarray). Asymmetric payoff under per-image mIoU: wrong ≈ −0.01, right ≈ +0.4 | `research/bet_combo.py` |
| 9 | Assembly of the submission | `research/assemble.py` |

One command rebuilds a submission from cached logits on JURECA:
`NAME=sub15 FAMILY="tf tg th ti tj" W=1 research/jureca/final_chain.sh` (steps 5–9; job `research/jureca/final_chain.sbatch`).
Training rounds: `PREFIX=tf_k EXTRA="--real-cutmix 0.5 --official-hosts --seed 6" research/jureca/launch_folds.sh 0 1 2 3 4`
(20 000 steps, batch 48, 4 model types per fold, general queue), then `research/jureca/cache_folds.sbatch`.

## Results (public split)

| Submission | What changed | OOF | Public |
|---|---|---:|---:|
| sub01 | first segmenter | – | 0.5731 |
| sub04 | long training + logit ensembles | – | 0.6929 |
| sub06 | + message-guided rectangles (TM, PS) | – | 0.7131 |
| sub08 | + same-position CutMix on real images | 0.7273 | 0.7244 |
| sub10 | + more CutMix seeds / ClassMix / 16 reals per batch | 0.7356 | 0.7314 |
| sub11 | + SAM boundary refinement | 0.7414 | 0.7370 |
| sub13 | + FLUX-host round + rectangle bet | ~0.748 | 0.7428 |
| **sub15** | 5 FLUX-host seeds | ~0.751 | **0.7444** |

All manifests (members, per-method OOF, rect rules) are in `research/submissions/*.json`.

## Findings that did **not** help (all measured OOF, kept for reference)

- Pseudo-labelling the test set (students copy the teacher; mixing them in hurts TrustMark).
- Autoencoder reconstruction error (AEROBLADE idea, own pure-PyTorch AutoencoderKL `research/ae_kl.py`,
  `research/vae_probe.py`, fusion `research/fuse_ae.py`): no signal once texture is normalised.
- Known-message "informed" template correlation and embedder idempotence (`research/informed_probe.py`): AUC ≤ 0.6.
- Lowering the threshold (globally or for small regions), hysteresis + SAM verification (`research/sam_weak.py`),
  rectangle snapping of blobs, flip TTA, CC post-processing, guided filter, occlusion / grey-out / remover probes.
- Oracle check: even with the true shape of a missed small edit, the ensemble evidence is concentrated there for only
  ~30 % of PixelSeal/TrustMark cases → object proposals cannot rescue most small edits.

## Reproducing

- `pip install -r requirements.txt`, `python download_models.py` (official watermark models, starter kit),
  SAM ViT-H weights: `https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth` → `outputs/sam/`.
- `wmloc/` and `submission_template.py` are the organisers' starter kit (data loading, watermark wrappers, metric).
- Job scripts in `research/jureca/` contain our JURECA paths and use `safe_sbatch.sh` (general queue, reservation
  nodes excluded); every output file is pre-created because of the shared inode quota.
- Uploads: `python -m research.upload research/submissions/<name>.json` (reads `WMLOC_BASE_URL`, `WMLOC_API_KEY` from `.env`).
