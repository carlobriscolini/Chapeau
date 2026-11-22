#!/bin/bash -l
# Rebuild sub04_folds (avert1, public 0.6929) from the same 13 checkpoints -> outputs/submissions/sub04_repro.pt
# Submit from the team folder after checking that the team has no running/pending job:
#   cd <team dir>/wmloc && sbatch research/jureca/repro_sub04.sbatch
#SBATCH --job-name=wmloc-repro04
#SBATCH --account=<project>
#SBATCH --partition=dc-gpu
#SBATCH --reservation=<reservation>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=outputs/jureca/%x-%j.out
set -euo pipefail
cd ${TEAM_DIR:?set TEAM_DIR}/wmloc
export HF_HOME="$PWD/hf" HUGGINGFACE_HUB_CACHE="$PWD/hf/hub" HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
# predict.py records the git commit; compute nodes may lack git
command -v git >/dev/null || module load Stages/2025 GCCcore/.13.3.0 git/2.45.1 >/dev/null 2>&1 || true
if ! command -v git >/dev/null; then
  shim="${TMPDIR:-/tmp}/gitshim_$SLURM_JOB_ID"; mkdir -p "$shim"
  printf '#!/bin/sh\ncase "$1" in rev-parse) echo %s;; *) :;; esac\n' "${GIT_COMMIT:?}" > "$shim/git"; chmod +x "$shim/git"
  export PATH="$shim:$PATH"
fi
.venv/bin/python -u -m research.predict --name sub04_repro --model pixelseal=outputs/jureca_ckpt/outputs/heavy1/pixelseal_wm_s1/best.pt@wm:pixelseal+outputs/jureca_ckpt/outputs/heavy1/pixelseal_wm/best.pt@wm:pixelseal+outputs/jureca_ckpt/outputs/wm_jureca_15901844/pixelseal_wm_s1/best.pt@wm:pixelseal+outputs/jureca_ckpt/outputs/heavy_f1/pixelseal_wm/best.pt@wm:pixelseal --model trustmark=outputs/jureca_ckpt/outputs/heavy1/trustmark_resnet50/best.pt@resnet50+outputs/jureca_ckpt/outputs/seg_v2_jureca/trustmark_resnet50/best.pt@resnet50+outputs/jureca_ckpt/outputs/wm_jureca_15901844/trustmark_wm/best.pt@wm:trustmark+outputs/jureca_ckpt/outputs/heavy_f1/trustmark_wm/best.pt@wm:trustmark+outputs/jureca_ckpt/outputs/heavy_f1/trustmark_resnet50/best.pt@resnet50+outputs/jureca_ckpt/outputs/heavy_f2_cut/trustmark_wm/best.pt@wm:trustmark --model stegastamp=outputs/jureca_ckpt/outputs/heavy1/stegastamp_resnet50/best.pt@resnet50+outputs/jureca_ckpt/outputs/seg_v2_jureca/stegastamp/best.pt@resnet34+outputs/jureca_ckpt/outputs/heavy_f1/stegastamp_resnet50/best.pt@resnet50
sha256sum outputs/submissions/sub04_repro.pt
echo "expected sub04 sha256: 7ab1ba3bcc2f54c1e39b2aee4a10afbf117d763582ce5435f72d7fe7c2cf49db"
