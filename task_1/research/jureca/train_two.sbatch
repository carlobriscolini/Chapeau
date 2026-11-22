#!/bin/bash -l
# Two to four segmenter runs, one A100 each, on one dc-gpu node. Defaults to 2 GPUs; for RUN_C/RUN_D submit with
#   --ntasks=4 --gres=gpu:4 (sbatch flags override the defaults below).
#   RUN_A="pixelseal pixelseal_wm --encoder wm:pixelseal --lr 1e-4" RUN_B="trustmark trustmark_wm --encoder wm:trustmark" \
#   sbatch --account=<project> --partition=dc-gpu --reservation=<reservation> research/jureca/train_two.sbatch
# Each RUN_* is "<method> <name> [train_seg args...]"; EXTRA_DIR picks the augmented split (default v1).
# Expects the synced repository layout described in train_all.sbatch.
#SBATCH --job-name=wmloc-two
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --time=02:00:00
#SBATCH --output=outputs/jureca/%x-%j.out
set -euo pipefail
export HF_HOME="$PWD/hf" HUGGINGFACE_HUB_CACHE="$PWD/hf/hub" HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
tag=${RUN_TAG:-two_jureca_$SLURM_JOB_ID}; steps=${STEPS:-12000}
common=(--steps "$steps" --pool 3000 --batch 48 --real-per-batch 8 --extra-dir "${EXTRA_DIR:-outputs/coco_augmented}"
        --extra-per-batch 16 --eval-every 1000)
run() {  # method, name, extra args
  local method=$1 name=$2; shift 2
  srun --exclusive --ntasks=1 --gres=gpu:1 --cpus-per-task=32 \
    .venv/bin/python -u -m research.train_seg --method "$method" --out "outputs/$tag/$name" "${common[@]}" "$@" \
    > "outputs/jureca/$tag-$name.log" 2>&1 &
}
mkdir -p outputs/jureca "outputs/$tag"
cp COMMIT "outputs/$tag/commit.txt" 2>/dev/null || true
: "${RUN_A:?RUN_A is required}"
for spec in "$RUN_A" "${RUN_B:-}" "${RUN_C:-}" "${RUN_D:-}"; do
  [[ -n $spec ]] || continue; read -r -a r <<< "$spec"; run "${r[@]}"
done
wait
