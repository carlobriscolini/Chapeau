#!/bin/bash -l
# SAM refine (PS, TM) + rectangle bet (PS, TM, SS) + assembly, for LOGITS=<dir> OUT=<dir> NAME=<submission name>
#SBATCH --job-name=wmloc-final
#SBATCH --account=<project>
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:40:00
set -uo pipefail
cd ${TEAM_DIR:?set TEAM_DIR}/wmloc
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
L=$LOGITS; O=$OUT
st() { srun --exclusive --ntasks=1 --gres=gpu:1 --cpus-per-task=32 "$@"; }
st .venv/bin/python -u -m research.sam_refine --method pixelseal --logits $L --out $O --rect outputs/rect/pixelseal.json:0.85:0.5 > $O/refine-pixelseal.log 2>&1 &
st .venv/bin/python -u -m research.sam_refine --method trustmark --logits $L --out $O --rect outputs/rect/trustmark.json:0.99:0.5 > $O/refine-trustmark.log 2>&1 &
st .venv/bin/python -u -m research.bet_combo --method stegastamp --logits $L --out $O > $O/bet-stegastamp.log 2>&1 &
st .venv/bin/python -u -m research.bet_combo --method pixelseal --logits $L --out $O > $O/bet-pixelseal.log 2>&1 &
wait
st .venv/bin/python -u -m research.bet_combo --method trustmark --logits $L --out $O > $O/bet-trustmark.log 2>&1
.venv/bin/python -u -m research.assemble --logits $L --sam $O --name $NAME --note "${NOTE:-}" > $O/assemble.log 2>&1
