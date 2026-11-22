#!/bin/bash
# Complete the 5-fold set of avert1's "heavy" recipe (20k steps, batch 48, 8 real + 16 coco_augmented per batch) on the
# GENERAL dc-gpu queue (user rule 22/11: heavy jobs off the reservation). One 4-GPU job per validation fold, each
# training the four ensemble member types. Usage (from the workspace):  research/jureca/launch_folds.sh 2 3 4
#   TEST_ONLY=1 research/jureca/launch_folds.sh 2   -> only print the scheduler's start estimate
# The project inode quota hits zero from time to time (22/11 01:59 our jobs died on mkdir), so every file a job writes
# is created HERE, before submission, and training overwrites checkpoints in place (WMLOC_INPLACE_SAVE=1).
set -euo pipefail
cd ${TEAM_DIR:?set TEAM_DIR}/wmloc
flags=(--account=${PROJECT:?set PROJECT} --partition=dc-gpu --ntasks=4 --gres=gpu:4 --time=01:45:00)
P=${PREFIX:-heavy_k}; X=${EXTRA:-}   # e.g. PREFIX=cm_k EXTRA="--real-cutmix 0.5"
names=(pixelseal_wm stegastamp_resnet50 trustmark_resnet50 trustmark_wm)
for k in "$@"; do
  export RUN_TAG=$P$k STEPS=20000 WMLOC_INPLACE_SAVE=1 PYTHONDONTWRITEBYTECODE=1
  export RUN_A="pixelseal pixelseal_wm --encoder wm:pixelseal --lr 1e-4 --val-fold $k $X"
  export RUN_B="stegastamp stegastamp_resnet50 --encoder resnet50 --val-fold $k $X"
  export RUN_C="trustmark trustmark_resnet50 --encoder resnet50 --val-fold $k $X"
  export RUN_D="trustmark trustmark_wm --encoder wm:trustmark --lr 2e-4 --val-fold $k $X"
  mkdir -p outputs/jureca "outputs/$P$k"; touch "outputs/$P$k/commit.txt" "outputs/jureca/wmloc-$P$k.out"
  for n in "${names[@]}"; do
    mkdir -p "outputs/$P$k/$n"
    touch "outputs/$P$k/$n/best.pt" "outputs/$P$k/$n/last.pt" "outputs/$P$k/$n/log.json" \
          "outputs/jureca/$P$k-$n.log"
  done
  if [[ -n ${TEST_ONLY:-} ]]; then
    nodes=$(scontrol show res ${RESERVATION:?set RESERVATION} | grep -oE " Nodes=[^ ]+" | cut -d= -f2)
    sbatch --test-only --exclude="$nodes" "${flags[@]}" --job-name=wmloc-$P$k research/jureca/train_two.sbatch 2>&1 | tail -1
  else
    research/jureca/safe_sbatch.sh general research/jureca/train_two.sbatch "${flags[@]}" --job-name=wmloc-$P$k \
      --output="outputs/jureca/wmloc-$P$k.out" --open-mode=truncate
  fi
done
