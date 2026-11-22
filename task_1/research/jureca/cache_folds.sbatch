#!/bin/bash -l
# Cache logits (+flip) of the fold models trained by research/jureca/launch_folds.sh (outputs/heavy_k<k>/<type>/best.pt)
# into outputs/cache/<method>_k.npz (same layout as cache_logits). Missing checkpoints are skipped. General queue:
#   research/jureca/safe_sbatch.sh general research/jureca/cache_folds.sbatch --dependency=afterany:<jobs>
#SBATCH --job-name=wmloc-cachek
#SBATCH --account=<project>
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:45:00
#SBATCH --output=outputs/jureca/wmloc-cachek.out
set -uo pipefail
cd ${TEAM_DIR:?set TEAM_DIR}/wmloc
export HF_HOME="$PWD/hf" HUGGINGFACE_HUB_CACHE="$PWD/hf/hub" HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
P=${PREFIX:-heavy_k}; SUF=${SUFFIX:-_k}; FOLDS=${FOLDS:-2 3 4}; NP=${NAMEP:-k}
step() { srun --exclusive --ntasks=1 --gres=gpu:1 --cpus-per-task=32 "$@"; }
args() {  # method -> --ckpt list for the existing fold checkpoints of that method
  local method=$1 k t name enc
  for k in $FOLDS; do
    for t in pixelseal_wm:wm:pixelseal stegastamp_resnet50:resnet50 trustmark_resnet50:resnet50 trustmark_wm:wm:trustmark; do
      name=${t%%:*}; enc=${t#*:}
      [[ $name == ${method}_* && -s outputs/$P$k/$name/best.pt ]] || continue
      echo "--ckpt ${NP}${k}_${name#${method}_}=outputs/$P$k/$name/best.pt@$enc"
    done
  done
}
mkdir -p outputs/cache outputs/jureca
for method in pixelseal trustmark stegastamp; do
  a=$(args $method); [[ -n $a ]] || { echo "$method: no checkpoints"; continue; }
  echo "$method: $a"
  step .venv/bin/python -u -m research.cache_logits --method $method --out outputs/cache/${method}$SUF.npz $a \
    > outputs/jureca/cache$SUF-$method.log 2>&1 &
done
wait
