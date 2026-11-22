#!/bin/bash -l
# Superseded by train_two.sbatch: takes 2 to 4 runs from RUN_A..RUN_D.
# Same as train_all.sbatch, with the official extractors as encoders (research.wm_encoder).
#   sbatch --account=<project> --partition=dc-gpu --reservation=<reservation> research/jureca/train_wm.sbatch
# Expects, relative to the submit directory (the synced repository): .venv (CUDA torch), models/, release/,
# .downloads/coco/{val2017,annotations}, outputs/coco_augmented/ and HF_HOME holding the timm weights
# (compute nodes have no internet). RUN_TAG names the output folder; STEPS overrides the step count.
#SBATCH --job-name=wmloc-wm
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --output=outputs/jureca/%x-%j.out
set -euo pipefail
export HF_HOME="$PWD/hf" HUGGINGFACE_HUB_CACHE="$PWD/hf/hub" HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
tag=${RUN_TAG:-wm_jureca_$SLURM_JOB_ID}; steps=${STEPS:-12000}
common=(--steps "$steps" --pool 3000 --batch 48 --real-per-batch 8 --extra-dir outputs/coco_augmented
        --extra-per-batch 16 --eval-every 1000)
run() {  # gpu, method, name, extra args
  local gpu=$1 method=$2 name=$3; shift 3
  srun --exclusive --ntasks=1 --gres=gpu:1 --cpus-per-task=32 \
    .venv/bin/python -u -m research.train_seg --method "$method" --out "outputs/$tag/$name" "${common[@]}" "$@" \
    > "outputs/jureca/$tag-$name.log" 2>&1 &
}
mkdir -p outputs/jureca "outputs/$tag"
git rev-parse HEAD > "outputs/$tag/commit.txt" 2>/dev/null || true
run 0 pixelseal  pixelseal_wm   --encoder wm:pixelseal --lr 2e-4
run 1 trustmark  trustmark_wm   --encoder wm:trustmark --lr 2e-4
run 2 pixelseal  pixelseal_wm_s1 --encoder wm:pixelseal --lr 1e-4 --seed 1
run 3 stegastamp stegastamp_r50 --encoder resnet50
wait
