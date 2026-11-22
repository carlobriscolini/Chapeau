#!/bin/bash -l
# Environment check for the continued workspace (one node, 3 parallel steps):
#   GPU0: rebuild sub04_folds from its 13 checkpoints -> outputs/submissions/sub04_repro.pt (+ manifest)
#   GPU1: 100-step smoke training (full recipe incl. coco_augmented), wm:pixelseal encoder (extractor weights, torch.compile, COCO synth)
#   GPU2: 100-step smoke training, resnet50 encoder for stegastamp (timm weights from offline HF cache)
# Submit from the team folder, only when the team has no running/pending job:
#   cd <team dir>/wmloc && GIT_COMMIT=$(git rev-parse HEAD) sbatch research/jureca/env_check.sbatch
#SBATCH --job-name=wmloc-envcheck-gen
#SBATCH --account=<project>
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:25:00
#SBATCH --output=outputs/jureca/%x-%j.out
set -uo pipefail
cd ${TEAM_DIR:?set TEAM_DIR}/wmloc
export HF_HOME="$PWD/hf" HUGGINGFACE_HUB_CACHE="$PWD/hf/hub" HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
command -v git >/dev/null || module load Stages/2025 GCCcore/.13.3.0 git/2.45.1 >/dev/null 2>&1 || true
if ! command -v git >/dev/null; then
  shim="${TMPDIR:-/tmp}/gitshim_$SLURM_JOB_ID"; mkdir -p "$shim"
  printf '#!/bin/sh\ncase "$1" in rev-parse) echo %s;; *) :;; esac\n' "${GIT_COMMIT:?}" > "$shim/git"; chmod +x "$shim/git"
  export PATH="$shim:$PATH"
fi
step() { srun --exclusive --ntasks=1 --gres=gpu:1 --cpus-per-task=32 "$@"; }
C=outputs/jureca_ckpt/outputs
P="$C/heavy1/pixelseal_wm_s1/best.pt@wm:pixelseal+$C/heavy1/pixelseal_wm/best.pt@wm:pixelseal+$C/wm_jureca_15901844/pixelseal_wm_s1/best.pt@wm:pixelseal+$C/heavy_f1/pixelseal_wm/best.pt@wm:pixelseal"
T="$C/heavy1/trustmark_resnet50/best.pt@resnet50+$C/seg_v2_jureca/trustmark_resnet50/best.pt@resnet50+$C/wm_jureca_15901844/trustmark_wm/best.pt@wm:trustmark+$C/heavy_f1/trustmark_wm/best.pt@wm:trustmark+$C/heavy_f1/trustmark_resnet50/best.pt@resnet50+$C/heavy_f2_cut/trustmark_wm/best.pt@wm:trustmark"
S="$C/heavy1/stegastamp_resnet50/best.pt@resnet50+$C/seg_v2_jureca/stegastamp/best.pt@resnet34+$C/heavy_f1/stegastamp_resnet50/best.pt@resnet50"
smoke=(--steps 100 --pool 200 --batch 16 --real-per-batch 4 --extra-dir outputs/coco_augmented --extra-per-batch 4 --eval-every 100)
mkdir -p outputs/jureca outputs/smoke
step .venv/bin/python -u -m research.predict --name sub04_repro --model "pixelseal=$P" --model "trustmark=$T" --model "stegastamp=$S" \
  > "outputs/jureca/envcheck-$SLURM_JOB_ID-predict.log" 2>&1 &
step .venv/bin/python -u -m research.train_seg --method pixelseal --encoder wm:pixelseal --lr 1e-4 --out outputs/smoke/pixelseal_wm "${smoke[@]}" \
  > "outputs/jureca/envcheck-$SLURM_JOB_ID-smoke_pixelseal.log" 2>&1 &
step .venv/bin/python -u -m research.train_seg --method stegastamp --encoder resnet50 --out outputs/smoke/stegastamp_r50 "${smoke[@]}" \
  > "outputs/jureca/envcheck-$SLURM_JOB_ID-smoke_stegastamp.log" 2>&1 &
wait
sha256sum outputs/submissions/sub04_repro.pt
echo "original sub04 sha256: 7ab1ba3bcc2f54c1e39b2aee4a10afbf117d763582ce5435f72d7fe7c2cf49db"
for f in outputs/jureca/envcheck-$SLURM_JOB_ID-*.log; do echo "== $f"; grep -vE "Warning|warn|torch.load|pickle" "$f" | tail -3 | cut -c1-400; done
rm -rf outputs/smoke   # smoke checkpoints are throwaway (inode quota)
