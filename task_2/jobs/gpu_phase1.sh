#!/bin/bash
#SBATCH --job-name=me_ph1
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --output=gpu_ph1_%j.log
# Full phase-1 scoring: token stats under the family base model + fp8 likelihood-ratio test (one family per GPU).
# Run from the task_2 folder:  sbatch -A <project> jobs/<this script>   (compute nodes are offline: cache models first)
export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
PY=${PY:-python}  # python of an env with torch + transformers
# optional: extra packages as wheels in pylib/ (we had no inodes left for a second venv on scratch)
if ls pylib/*.whl >/dev/null 2>&1; then mkdir -p /tmp/pyt && for w in pylib/*.whl; do $PY -m zipfile -e $w /tmp/pyt; done; export PYTHONPATH=/tmp/pyt; fi
A=${ANS:-data/ep_phase1.jsonl}
EA=ep-05,ep-09,ep-10,ep-11,ep-13,ep-15,ep-16,ep-19,ep-27,ep-29,ep-31
EB=ep-03,ep-06,ep-07,ep-08,ep-20,ep-21,ep-22,ep-26,ep-28,ep-30,ep-32
EC=ep-01,ep-02,ep-04,ep-12,ep-14,ep-17,ep-18,ep-23,ep-24,ep-25,ep-33
run() { CUDA_VISIBLE_DEVICES=$1 $PY src/likelihood.py $2 --answers $A --endpoints $3 && CUDA_VISIBLE_DEVICES=$1 $PY src/llr_quant.py $2 $A --endpoints $3 --schemes fp8_tensor,fp8_channel; }
FAMS=${FAMS:-ABC}
[[ $FAMS == *A* ]] && run 0 unsloth/Llama-3.2-1B-Instruct $EA > ph1_gA.txt 2>&1 &
[[ $FAMS == *B* ]] && run 1 Qwen/Qwen2.5-1.5B-Instruct $EB > ph1_gB.txt 2>&1 &
[[ $FAMS == *C* ]] && run 2 Qwen/Qwen2.5-3B-Instruct $EC > ph1_gC.txt 2>&1 &
wait
rm -rf /tmp/pyt
