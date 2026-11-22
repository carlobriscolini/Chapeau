#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --partition=dc-gpu
#SBATCH --time=00:45:00
#SBATCH --output=/p/home/jusers/%u/jureca/runs/%j.out
#SBATCH --error=/p/home/jusers/%u/jureca/runs/%j.err

# Any python commands (soup.py, ssd.py, evaluate.py, ...) spread over the node's 4 GPUs: command k runs on GPU k % 4.
# Open queue (no reservation). Logs + saved models in $HOME/runs (OUTPUT_DIR), summary (diagnostics lines) in the .out.
# Usage (from the task_3 folder):  sbatch jobs/cmd4_home.sh "python experiments/soup.py --models A.pt --alphas 1.5 2" "python experiments/ssd.py ..."
TASK_DIR=${TASK_DIR:-$(dirname "${SLURM_SUBMIT_DIR:-$PWD}")}  # task folder = parent of this repo
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1  # sbatch from the task_3 folder
source "$TASK_DIR/.venv/bin/activate"
export PYTHONDONTWRITEBYTECODE=1  # organizer rule: keep file count low
export OUTPUT_DIR=$HOME/runs
mkdir -p "$OUTPUT_DIR"
CMDS=("$@")
echo "job $SLURM_JOB_ID on $(hostname), ${#CMDS[@]} commands on 4 GPUs, output in $OUTPUT_DIR"

run_gpu() {
    local gpu=$1
    for ((k = gpu; k < ${#CMDS[@]}; k += 4)); do
        CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=8 bash -c "${CMDS[$k]}" > "$OUTPUT_DIR/${SLURM_JOB_ID}_cmd${k}.log" 2>&1
    done
}
for gpu in 0 1 2 3; do run_gpu $gpu & done
wait
for ((k = 0; k < ${#CMDS[@]}; k++)); do
    log="$OUTPUT_DIR/${SLURM_JOB_ID}_cmd${k}.log"
    echo "== cmd $k: ${CMDS[$k]}"
    grep -E "=====|forget/exact|forget/altered|retain_proxy/mix|retain_proxy/ext |ext/forgetcls|retain_proxy/gpool|gpool/untouched|gpool/forgetcls|Saved" "$log"
    grep -h -A3 "Error\|Traceback" "$log" | tail -n 4
done
echo "done!"
