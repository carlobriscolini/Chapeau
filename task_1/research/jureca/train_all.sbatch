#!/bin/bash -l
# Superseded by train_two.sbatch: takes 2 to 4 runs from RUN_A..RUN_D.
# One JURECA dc-gpu node, four A100: one segmenter run per GPU, launched as parallel job steps.
#   sbatch --account=<project> --partition=dc-gpu --reservation=<reservation> research/jureca/train_all.sbatch
# Expects, relative to the submit directory (the synced repository): .venv (CUDA torch), models/, release/,
# .downloads/coco/{val2017,annotations}, outputs/coco_augmented/ and HF_HOME holding the timm weights
# (compute nodes have no internet). RUN_TAG names the output folder; STEPS overrides the step count.
#SBATCH --job-name=wmloc-seg
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --output=outputs/jureca/%x-%j.out
set -euo pipefail
export HF_HOME="$PWD/hf" HUGGINGFACE_HUB_CACHE="$PWD/hf/hub" HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
tag=${RUN_TAG:-seg_jureca_$SLURM_JOB_ID}; steps=${STEPS:-12000}
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
run 0 trustmark  trustmark
run 1 pixelseal  pixelseal
run 2 stegastamp stegastamp
run 3 trustmark  trustmark_resnet50 --encoder resnet50
wait
