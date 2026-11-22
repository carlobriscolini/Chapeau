#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --partition=dc-gpu
#SBATCH --time=00:45:00
#SBATCH --output=../output/%j.out
#SBATCH --error=../output/%j.err

# Generate the cinematic-style retain pool (gen_pool.py) on 4 GPUs, then merge (classify + preview).
# Scratch inodes are exhausted, so packages, SD-Turbo weights, shards and logs live in $HOME;
# only ../output/genpool.npz and ../output/genpool_preview.jpg are written to scratch.
# Before (login node):
#   mkdir -p ~/gen_tmp
#   UV_CACHE_DIR=~/uvcache_tmp uv pip install --target ~/genlibs diffusers transformers; rm -rf ~/uvcache_tmp
#   PYTHONPATH=~/genlibs HF_HOME=~/hfcache HF_HUB_CACHE=~/hfcache/hub python data_prep/gen_pool.py --check --download
# Usage (from the task_3 folder):  sbatch jobs/genpool.sh   (open queue: does not count toward the team limit)
#   round 2 (after round 1):    sbatch jobs/genpool.sh --targeted   -> ../output/genpool_t.npz
#   round 3 (photo look):       sbatch jobs/genpool.sh --targeted --photo   -> ../output/genpool_p.npz
TASK_DIR=${TASK_DIR:-$(dirname "${SLURM_SUBMIT_DIR:-$PWD}")}  # task folder = parent of this repo
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1  # sbatch from the task_3 folder
source "$TASK_DIR/.venv/bin/activate"
export PYTHONDONTWRITEBYTECODE=1  # organizer rule: keep file count low
export PYTHONPATH=$HOME/genlibs
export HF_HOME=$HOME/hfcache
export HF_HUB_CACHE=$HOME/hfcache/hub  # overrides a HF_HUB_CACHE inherited from the login shell (team cache on scratch)
export HF_HUB_OFFLINE=1
export GEN_TMP=$HOME/gen_tmp
mkdir -p "$GEN_TMP"
echo "job $SLURM_JOB_ID on $(hostname): gen_pool.py $*"
for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i OMP_NUM_THREADS=8 python data_prep/gen_pool.py --shard $i --nshards 4 "$@" \
        > "$GEN_TMP/${SLURM_JOB_ID}_pool$i.log" 2>&1 &
done
wait
tail -n 3 "$GEN_TMP"/${SLURM_JOB_ID}_pool*.log
python data_prep/gen_pool.py --merge "$@"
echo "done!"
