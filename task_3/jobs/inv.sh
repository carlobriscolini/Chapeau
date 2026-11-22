#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --partition=dc-gpu
#SBATCH --time=00:45:00
#SBATCH --output=../output/%j.out
#SBATCH --error=../output/%j.err

# Model-inversion retain anchors on the node's 4 GPUs (4 shards), then merge into ../output/inverted.npz
# Usage (from the task_3 folder):  sbatch jobs/inv.sh            (extra args go to invert.py, e.g. --n 2000 --steps 400)
TASK_DIR=${TASK_DIR:-$(dirname "${SLURM_SUBMIT_DIR:-$PWD}")}  # task folder = parent of this repo
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1  # sbatch from the task_3 folder
source "$TASK_DIR/.venv/bin/activate"
export PYTHONDONTWRITEBYTECODE=1  # organizer rule: keep file count low
echo "job $SLURM_JOB_ID on $(hostname): invert.py $*"
for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i OMP_NUM_THREADS=8 python experiments/invert.py --shard $i --nshards 4 "$@" &
done
wait
python experiments/invert.py --merge
echo "done!"
