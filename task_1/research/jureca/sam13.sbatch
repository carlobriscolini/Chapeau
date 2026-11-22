#!/bin/bash -l
# SAM refine (PS, TM) + SAM small-edit rescue (PS, TM, SS) on a given logits dir:  LOGITS=outputs/logits13 OUT=outputs/sam13
#SBATCH --job-name=wmloc-sam13
#SBATCH --account=<project>
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:45:00
#SBATCH --output=outputs/jureca/wmloc-sam13.out
set -uo pipefail
cd ${TEAM_DIR:?set TEAM_DIR}/wmloc
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
L=${LOGITS:-outputs/logits13}; O=${OUT:-outputs/sam13}
st() { srun --exclusive --ntasks=1 --gres=gpu:1 --cpus-per-task=32 "$@"; }
st .venv/bin/python -u -m research.sam_refine --method pixelseal --logits $L --out $O --rect outputs/rect/pixelseal.json:0.85:0.5 > $O/refine-pixelseal.log 2>&1 &
st .venv/bin/python -u -m research.sam_refine --method trustmark --logits $L --out $O --rect outputs/rect/trustmark.json:0.99:0.5 > $O/refine-trustmark.log 2>&1 &
st .venv/bin/python -u -m research.sam_small --method pixelseal --logits $L --out $O > $O/small-pixelseal.log 2>&1 &
st .venv/bin/python -u -m research.sam_small --method trustmark --logits $L --out $O > $O/small-trustmark.log 2>&1 &
wait
st .venv/bin/python -u -m research.sam_small --method stegastamp --logits $L --out $O > $O/small-stegastamp.log 2>&1
