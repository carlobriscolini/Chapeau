#!/bin/bash
# Give a teammate of project <project> a JURECA workspace that reuses avert1's environment read-only.
# Run as the teammate on a JURECA login node:
#   bash <user dir>/wmloc/research/jureca/setup_teammate.sh
# Then submit with research/jureca/train_two.sbatch from the printed directory (one 4-GPU job at a time).
set -euo pipefail
src=${USER_DIR:?set USER_DIR}/wmloc
dst=${SCRATCH_DIR:?set SCRATCH_DIR}/$USER/wmloc
mkdir -p "$dst/outputs/jureca"
for p in .venv models hf release .downloads research wmloc COMMIT; do ln -sfn "$src/$p" "$dst/$p"; done
for p in coco_augmented coco_augmented_v2; do [[ -e $src/outputs/$p ]] && ln -sfn "$src/outputs/$p" "$dst/outputs/$p"; done
cd "$dst" && HF_HOME="$PWD/hf" HF_HUB_OFFLINE=1 .venv/bin/python -c "import torch, timm, research.train_seg; print('ok', torch.__version__)"
echo "workspace ready: $dst"
