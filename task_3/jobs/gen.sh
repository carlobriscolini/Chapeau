#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --partition=dc-gpu
#SBATCH --time=00:45:00
#SBATCH --output=../output/%j.out
#SBATCH --error=../output/%j.err

# Generate sister images with SD-Turbo on the node's 4 GPUs, then merge.
# Before: run once on the login node  python experiments/gen_siblings.py --download
# Usage (from the task_3 folder):  sbatch jobs/gen.sh            (extra args go to gen_siblings.py, e.g. --K 4 --strength 0.6)
TASK_DIR=${TASK_DIR:-$(dirname "${SLURM_SUBMIT_DIR:-$PWD}")}  # task folder = parent of this repo
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1  # sbatch from the task_3 folder
source "$TASK_DIR/.venv/bin/activate"
export PYTHONDONTWRITEBYTECODE=1  # organizer rule: keep file count low
export HF_HOME=${HF_HOME:-$TASK_DIR/../.cache}
export HF_HUB_OFFLINE=1
echo "job $SLURM_JOB_ID on $(hostname): gen_siblings.py $*"
for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i OMP_NUM_THREADS=8 python experiments/gen_siblings.py --shard $i --nshards 4 "$@" \
        > "$TASK_DIR/output/${SLURM_JOB_ID}_gen$i.log" 2>&1 &
done
wait
tail -n 3 "$TASK_DIR"/output/${SLURM_JOB_ID}_gen*.log
python experiments/gen_siblings.py --merge
echo "done!"
