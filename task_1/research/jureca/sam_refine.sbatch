#!/bin/bash -l
# AE reconstruction-error probe (research/vae_probe.py): one GPU step per autoencoder, devel partition, general queue:
#   research/jureca/safe_sbatch.sh general research/jureca/vae_probe.sbatch
#SBATCH --job-name=wmloc-samref
#SBATCH --account=<project>
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --output=outputs/jureca/wmloc-samref.out
set -uo pipefail
cd ${TEAM_DIR:?set TEAM_DIR}/wmloc
export HF_HOME="$PWD/outputs/hfv" HUGGINGFACE_HUB_CACHE="$PWD/outputs/hfv/hub" HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
declare -A RECT=([pixelseal]="--rect outputs/rect/pixelseal.json:0.85:0.5" [trustmark]="--rect outputs/rect/trustmark.json:0.99:0.5" [stegastamp]="")
for m in ${METHODS:-pixelseal trustmark stegastamp}; do
  srun --exclusive --ntasks=1 --gres=gpu:1 --cpus-per-task=32 .venv/bin/python -u -m research.sam_refine --method $m ${RECT[$m]} ${TAG:+--tag $TAG} \
    > outputs/jureca/samref-$m${TAG:-}.log 2>&1 &
done
wait
