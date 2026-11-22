#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --partition=dc-gpu
#SBATCH --time=00:45:00
#SBATCH --output=/p/home/jusers/%u/jureca/runs/%j.out
#SBATCH --error=/p/home/jusers/%u/jureca/runs/%j.err

# Like main4.sh (many run_unlearn.py configs over the node's 4 GPUs), but scratch inodes are exhausted:
# models, target caches and logs go to $HOME/runs (OUTPUT_DIR). Open queue (no reservation).
# Before (login node, once):  mkdir -p ~/runs && cp ../output/targets_*.pt ~/runs/
# Usage (from the task_3 folder):  sbatch jobs/main4_home.sh "--lam 3 ..." "--lam 1 ..." ...
TASK_DIR=${TASK_DIR:-$(dirname "${SLURM_SUBMIT_DIR:-$PWD}")}  # task folder = parent of this repo
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1  # sbatch from the task_3 folder
source "$TASK_DIR/.venv/bin/activate"
export PYTHONDONTWRITEBYTECODE=1  # organizer rule: keep file count low
export OUTPUT_DIR=$HOME/runs
mkdir -p "$OUTPUT_DIR"
RUNS=("$@")
echo "job $SLURM_JOB_ID on $(hostname), ${#RUNS[@]} runs on 4 GPUs, output in $OUTPUT_DIR"

run_gpu() {  # $1 = gpu index; runs k = gpu, gpu+4, gpu+8, ...
    local gpu=$1
    for ((k = gpu; k < ${#RUNS[@]}; k += 4)); do
        local log="$OUTPUT_DIR/${SLURM_JOB_ID}_run${k}.log"
        echo "gpu $gpu run $k: python src/run_unlearn.py ${RUNS[$k]}  -> $log"
        CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=8 python src/run_unlearn.py ${RUNS[$k]} > "$log" 2>&1
    done
}
for gpu in 0 1 2 3; do run_gpu $gpu & done
wait
echo "all runs finished"
for ((k = 0; k < ${#RUNS[@]}; k++)); do
    log="$OUTPUT_DIR/${SLURM_JOB_ID}_run${k}.log"
    echo "== run $k: ${RUNS[$k]}"
    grep -E "forget/exact|forget/altered|retain_proxy/mix|retain_proxy/ext|ext/forgetcls|retain_proxy/gpool|gpool/untouched|gpool/forgetcls" "$log"
    grep -h "Saved\|Error\|Traceback" "$log"
done
echo "done!"
