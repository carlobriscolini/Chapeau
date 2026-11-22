#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=dc-gpu
#SBATCH --time=00:30:00
#SBATCH --output=../output/%j.out
#SBATCH --error=../output/%j.err

# Usage (from the task_3 folder):
#   sbatch jobs/main.sh src/targets.py
#   sbatch jobs/main.sh src/run_unlearn.py --method distill --tau 1.0 --lam 1.0
# First argument = script, the rest is passed to it.
TASK_DIR=${TASK_DIR:-$(dirname "${SLURM_SUBMIT_DIR:-$PWD}")}  # task folder = parent of this repo
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1  # sbatch from the task_3 folder
source "$TASK_DIR/.venv/bin/activate"
export PYTHONDONTWRITEBYTECODE=1  # organizer rule: keep file count low
SCRIPT=${1:-src/run_unlearn.py}; shift
echo "job $SLURM_JOB_ID on $(hostname): python $SCRIPT $*"
srun python "$SCRIPT" "$@"
echo "done!"
