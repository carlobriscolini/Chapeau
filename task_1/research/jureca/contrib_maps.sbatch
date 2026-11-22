#!/bin/bash -l
# Observed vs expected contribution maps (research.contrib_maps), PixelSeal and TrustMark, train + test.
# Submit ONLY via the team-safe wrapper (general queue, reservation nodes excluded):
#   cd <team dir>/wmloc && research/jureca/safe_sbatch.sh general research/jureca/contrib_maps.sbatch
#SBATCH --job-name=wmloc-contrib
#SBATCH --account=<project>
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --output=outputs/jureca/%x-%j.out
set -uo pipefail
cd ${TEAM_DIR:?set TEAM_DIR}/wmloc
export HF_HOME="$PWD/hf" HUGGINGFACE_HUB_CACHE="$PWD/hf/hub" HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
step() { srun --exclusive --ntasks=1 --gres=gpu:1 --cpus-per-task=32 "$@"; }
mkdir -p outputs/contrib outputs/jureca
for m in pixelseal trustmark; do
  step .venv/bin/python -u -m research.contrib_maps --method $m --out outputs/contrib/$m.npz > outputs/jureca/contrib-$SLURM_JOB_ID-$m.log 2>&1 &
done
wait
for f in outputs/jureca/contrib-$SLURM_JOB_ID-*.log; do echo "== $f"; grep -vE "Warning|warn|torch.load|pickle" "$f" | tail -3 | cut -c1-300; done
