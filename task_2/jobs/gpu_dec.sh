#!/bin/bash
#SBATCH --job-name=me_dec
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --output=gpu_dec_%j.log
# Decoding fingerprints (src/ll_decode.py), one family per GPU; summarise with src/dec_summary.py data/dec_*.jsonl
# Run from the task_2 folder:  sbatch -A <project> jobs/<this script>   (compute nodes are offline: cache models first)
export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
PY=${PY:-python}  # python of an env with torch + transformers
# optional: extra packages as wheels in pylib/ (we had no inodes left for a second venv on scratch)
if ls pylib/*.whl >/dev/null 2>&1; then mkdir -p /tmp/pyt && for w in pylib/*.whl; do $PY -m zipfile -e $w /tmp/pyt; done; export PYTHONPATH=/tmp/pyt; fi
ANS="data/ep_phase1.jsonl data/ep_phase2.jsonl"
EA=ep-05,ep-09,ep-10,ep-11,ep-13,ep-15,ep-16,ep-19,ep-27,ep-29,ep-31
EB=ep-03,ep-06,ep-07,ep-08,ep-20,ep-21,ep-22,ep-26,ep-28,ep-30,ep-32
EC=ep-01,ep-02,ep-04,ep-12,ep-14,ep-17,ep-18,ep-23,ep-24,ep-25,ep-33
: > dec.log
CUDA_VISIBLE_DEVICES=0 $PY src/ll_decode.py unsloth/Llama-3.2-1B-Instruct --answers $ANS --endpoints $EA --out data/dec_gA.jsonl >> dec.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY src/ll_decode.py Qwen/Qwen2.5-1.5B-Instruct --answers $ANS --endpoints $EB --out data/dec_gB.jsonl >> dec.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 $PY src/ll_decode.py Qwen/Qwen2.5-3B-Instruct --answers $ANS --endpoints $EC --out data/dec_gC.jsonl >> dec.log 2>&1 &
wait
echo ALLDONE >> dec.log
rm -rf /tmp/pyt
