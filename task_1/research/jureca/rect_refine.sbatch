#!/bin/bash -l
# Message-guided rectangle search (research.rect_refine) on clean-background images, train (OOF starts) + test.
# General dc-gpu queue (no reservation). Submit from the team folder:
#   cd <team dir>/wmloc && sbatch research/jureca/rect_refine.sbatch
#SBATCH --job-name=wmloc-rect
#SBATCH --account=<project>
#SBATCH --partition=dc-gpu
# <reservation> is a MAGNETIC reservation: without --exclude Slurm may still pull this job into it (team slot).
#SBATCH --exclude=jrc[0215,0227,0231-0233,0245,0257-0258,0283,0314,0321,0323-0325,0327,0347,0349-0350,0354,0362-0363,0374,0387,0394,0396,0423,0441-0442,0444,0446]
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=outputs/jureca/%x-%j.out
set -uo pipefail
cd ${TEAM_DIR:?set TEAM_DIR}/wmloc
export HF_HOME="$PWD/hf" HUGGINGFACE_HUB_CACHE="$PWD/hf/hub" HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
step() { srun --exclusive --ntasks=1 --gres=gpu:1 --cpus-per-task=32 "$@"; }
mkdir -p outputs/rect outputs/jureca
L=outputs/jureca/rect-$SLURM_JOB_ID
# test starts = sub04 ensemble members (as cached)
step .venv/bin/python -u -m research.rect_refine --method pixelseal --cache outputs/cache/pixelseal.npz \
  --members h1_wm_s1,h1_wm,j_wm_s1,f1_wm --out outputs/rect/pixelseal.json > $L-pixelseal.log 2>&1 &
step .venv/bin/python -u -m research.rect_refine --method trustmark --cache outputs/cache/trustmark.npz \
  --members h1_r50,v2_r50,j_wm,f1_wm,f1_r50,f2cut_wm --out outputs/rect/trustmark.json > $L-trustmark.log 2>&1 &
step .venv/bin/python -u -m research.rect_refine --method stegastamp --cache outputs/cache/stegastamp.npz --bs 64 \
  --members h1_r50,v2_r34,f1_r50 --fine-max 160 --splits train --out outputs/rect/stegastamp_train.json > $L-stegastamp_train.log 2>&1 &
step .venv/bin/python -u -m research.rect_refine --method stegastamp --cache outputs/cache/stegastamp.npz --bs 64 \
  --members h1_r50,v2_r34,f1_r50 --fine-max 160 --splits test --out outputs/rect/stegastamp_test.json > $L-stegastamp_test.log 2>&1 &
wait
for f in $L-*.log; do echo "== $f"; grep -vE "Warning|warn|torch.load|pickle" "$f" | tail -3 | cut -c1-300; done
