#!/bin/bash -l
# Cache logits (+flip TTA) of all 22 avert1 checkpoints and whole-image bit accuracy, one .npz per method.
# General dc-gpu queue (no reservation: does not use the team's single reservation slot). Submit from the team folder:
#   cd <team dir>/wmloc && sbatch research/jureca/cache_logits.sbatch
#SBATCH --job-name=wmloc-cache
#SBATCH --account=<project>
#SBATCH --partition=dc-gpu
# <reservation> is a MAGNETIC reservation: without --exclude Slurm may still pull this job into it (team slot).
#SBATCH --exclude=jrc[0215,0227,0231-0233,0245,0257-0258,0283,0314,0321,0323-0325,0327,0347,0349-0350,0354,0362-0363,0374,0387,0394,0396,0423,0441-0442,0444,0446]
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:45:00
#SBATCH --output=outputs/jureca/%x-%j.out
set -uo pipefail
cd ${TEAM_DIR:?set TEAM_DIR}/wmloc
export HF_HOME="$PWD/hf" HUGGINGFACE_HUB_CACHE="$PWD/hf/hub" HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
C=outputs/jureca_ckpt/outputs
step() { srun --exclusive --ntasks=1 --gres=gpu:1 --cpus-per-task=32 "$@"; }
mkdir -p outputs/cache outputs/jureca
step .venv/bin/python -u -m research.cache_logits --method pixelseal --out outputs/cache/pixelseal.npz \
  --ckpt h1_wm=$C/heavy1/pixelseal_wm/best.pt@wm:pixelseal \
  --ckpt h1_wm_s1=$C/heavy1/pixelseal_wm_s1/best.pt@wm:pixelseal \
  --ckpt j_wm=$C/wm_jureca_15901844/pixelseal_wm/best.pt@wm:pixelseal \
  --ckpt j_wm_s1=$C/wm_jureca_15901844/pixelseal_wm_s1/best.pt@wm:pixelseal \
  --ckpt f1_wm=$C/heavy_f1/pixelseal_wm/best.pt@wm:pixelseal \
  --ckpt f2cut_wm=$C/heavy_f2_cut/pixelseal_wm/best.pt@wm:pixelseal \
  --ckpt v2_r34=$C/seg_v2_jureca/pixelseal/best.pt@resnet34 \
  > outputs/jureca/cache-$SLURM_JOB_ID-pixelseal.log 2>&1 &
step .venv/bin/python -u -m research.cache_logits --method trustmark --out outputs/cache/trustmark.npz \
  --ckpt h1_r50=$C/heavy1/trustmark_resnet50/best.pt@resnet50 \
  --ckpt f1_r50=$C/heavy_f1/trustmark_resnet50/best.pt@resnet50 \
  --ckpt f1_wm=$C/heavy_f1/trustmark_wm/best.pt@wm:trustmark \
  --ckpt f2cut_r50=$C/heavy_f2_cut/trustmark_resnet50/best.pt@resnet50 \
  --ckpt f2cut_wm=$C/heavy_f2_cut/trustmark_wm/best.pt@wm:trustmark \
  --ckpt v2_r34=$C/seg_v2_jureca/trustmark/best.pt@resnet34 \
  --ckpt v2_r50=$C/seg_v2_jureca/trustmark_resnet50/best.pt@resnet50 \
  --ckpt j_wm=$C/wm_jureca_15901844/trustmark_wm/best.pt@wm:trustmark \
  --ckpt v1v2_wm_v1=$C/v1v2_jureca_15901926/trustmark_wm_v1/best.pt@wm:trustmark \
  --ckpt v1v2_wm_v2=$C/v1v2_jureca_15901926/trustmark_wm_v2/best.pt@wm:trustmark \
  > outputs/jureca/cache-$SLURM_JOB_ID-trustmark.log 2>&1 &
step .venv/bin/python -u -m research.cache_logits --method stegastamp --out outputs/cache/stegastamp.npz \
  --ckpt h1_r50=$C/heavy1/stegastamp_resnet50/best.pt@resnet50 \
  --ckpt f1_r50=$C/heavy_f1/stegastamp_resnet50/best.pt@resnet50 \
  --ckpt f2cut_r50=$C/heavy_f2_cut/stegastamp_resnet50/best.pt@resnet50 \
  --ckpt v2_r34=$C/seg_v2_jureca/stegastamp/best.pt@resnet34 \
  --ckpt j_r50=$C/wm_jureca_15901844/stegastamp_r50/best.pt@resnet50 \
  > outputs/jureca/cache-$SLURM_JOB_ID-stegastamp.log 2>&1 &
wait
for f in outputs/jureca/cache-$SLURM_JOB_ID-*.log; do echo "== $f"; grep -vE "Warning|warn|torch.load|pickle" "$f" | tail -2 | cut -c1-300; done
ls -la outputs/cache
