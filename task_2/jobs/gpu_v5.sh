#!/bin/bash
#SBATCH --job-name=me_v5
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:45:00
#SBATCH --output=gpu_v5_%j.log
# v5 directional tests: extra quantization LLRs (fp8 W8A8, int8 W8A8, int4 g128) + distilled-LoRA LLR per teacher.
# Run from the task_2 folder:  sbatch -A <project> jobs/<this script>   (compute nodes are offline: cache models first)
export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
PY=${PY:-python}  # python of an env with torch + transformers
# optional: extra packages as wheels in pylib/ (we had no inodes left for a second venv on scratch)
if ls pylib/*.whl >/dev/null 2>&1; then mkdir -p /tmp/pyt && for w in pylib/*.whl; do $PY -m zipfile -e $w /tmp/pyt; done; export PYTHONPATH=/tmp/pyt; fi
MA=unsloth/Llama-3.2-1B-Instruct; MB=Qwen/Qwen2.5-1.5B-Instruct; MC=Qwen/Qwen2.5-3B-Instruct
EA=ep-09,ep-11,ep-13,ep-19,ep-10,ep-05,ep-27,ep-15
EB=ep-21,ep-26,ep-32,ep-06,ep-03,ep-28,ep-22
EC=ep-01,ep-12,ep-17,ep-18,ep-24,ep-25,ep-04,ep-14
Q="--schemes fp8_w8a8,int8_w8a8,int4_g128 --tag 2"
ANS="data/ep_phase1.jsonl data/ep_phase2.jsonl"
quant() { for f in $ANS; do CUDA_VISIBLE_DEVICES=$1 $PY src/llr_quant.py $2 $f --endpoints $3 $Q; done; }
lora() { CUDA_VISIBLE_DEVICES=$1 $PY src/lora_distill.py $2 --answers $ANS --teacher $3 --endpoints $4; }
(quant 0 $MA $EA; lora 0 $MA ep-27 $EA) > v5_gA.txt 2>&1 &
(quant 1 $MB $EB; lora 1 $MB ep-28 $EB) > v5_gB.txt 2>&1 &
(quant 2 $MC $EC; lora 2 $MC ep-14 $EC) > v5_gC.txt 2>&1 &
(lora 3 $MA ep-05 $EA; lora 3 $MB ep-03 $EB) > v5_gD.txt 2>&1 &
wait
rm -rf /tmp/pyt
