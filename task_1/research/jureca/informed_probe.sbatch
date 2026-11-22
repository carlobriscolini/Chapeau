#!/bin/bash -l
# AE reconstruction-error probe (research/vae_probe.py): one GPU step per autoencoder, devel partition, general queue:
#   research/jureca/safe_sbatch.sh general research/jureca/vae_probe.sbatch
#SBATCH --job-name=wmloc-informed
#SBATCH --account=<project>
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --output=outputs/jureca/wmloc-informed.out
set -uo pipefail
cd ${TEAM_DIR:?set TEAM_DIR}/wmloc
export HF_HOME="$PWD/outputs/hfv" HUGGINGFACE_HUB_CACHE="$PWD/outputs/hfv/hub" HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
for m in pixelseal trustmark stegastamp; do
  srun --exclusive --ntasks=1 --gres=gpu:1 --cpus-per-task=32 .venv/bin/python -u -m research.informed_probe --method $m \
    --out outputs/informed ${SANITY:+--sanity} ${RATIO:+--ratio} > outputs/jureca/informed-$m${SANITY:+-sanity}${RATIO:+-ratio}.log 2>&1 &
done
wait
