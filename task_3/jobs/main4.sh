#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --partition=dc-gpu
#SBATCH --time=00:45:00
#SBATCH --output=../output/%j.out
#SBATCH --error=../output/%j.err

# Many run_unlearn.py configurations in ONE job (organizers: one job at a time per person).
# Runs are distributed round-robin over the node's 4 GPUs; each GPU runs its share one after the other.
# Usage (from the task_3 folder), one quoted argument string per run (any number, ~2-3 min each on an A100):
#   sbatch jobs/main4.sh "--tau 1 --lam 0" "--tau 1 --lam 1" "--tau 1.5 --lam 1" "--forget-loss npo" "..." ...
# Per-run logs: ../output/<jobid>_run<k>.log
TASK_DIR=${TASK_DIR:-$(dirname "${SLURM_SUBMIT_DIR:-$PWD}")}  # task folder = parent of this repo
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1  # sbatch from the task_3 folder
source "$TASK_DIR/.venv/bin/activate"
export PYTHONDONTWRITEBYTECODE=1  # organizer rule: keep file count low
RUNS=("$@")
echo "job $SLURM_JOB_ID on $(hostname), ${#RUNS[@]} runs on 4 GPUs"

run_gpu() {  # $1 = gpu index; runs k = gpu, gpu+4, gpu+8, ...
    local gpu=$1
    for ((k = gpu; k < ${#RUNS[@]}; k += 4)); do
        local log="$TASK_DIR/output/${SLURM_JOB_ID}_run${k}.log"
        echo "gpu $gpu run $k: python src/run_unlearn.py ${RUNS[$k]}  -> $log"
        CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=8 python src/run_unlearn.py ${RUNS[$k]} > "$log" 2>&1
    done
}
for gpu in 0 1 2 3; do run_gpu $gpu & done
wait
echo "all runs finished"
for ((k = 0; k < ${#RUNS[@]}; k++)); do
    log="$TASK_DIR/output/${SLURM_JOB_ID}_run${k}.log"
    echo "== run $k: ${RUNS[$k]}"
    grep -E "forget/exact|retain_proxy/ext|ext/untouched|retain_proxy/inv" "$log"
    grep -h "Saved\|Error\|Traceback" "$log"
done
echo "done!"
