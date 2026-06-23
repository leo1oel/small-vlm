#!/bin/bash
#SBATCH --job-name=textcap
#SBATCH --partition=ckpt-all
#SBATCH --account=cse-ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:30:00
#SBATCH --requeue
#SBATCH --output=/mmfs1/gscratch/krishna/leoym/small-vlm/logs/textcap_%j.log
#SBATCH --error=/mmfs1/gscratch/krishna/leoym/small-vlm/logs/textcap_%j.log
set -euo pipefail
PROJECT_ROOT=/mmfs1/gscratch/krishna/leoym/small-vlm
cd "${PROJECT_ROOT}"
type module &>/dev/null || source /usr/share/lmod/lmod/init/bash
module load cuda/12.9.1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE=/gscratch/krishna/leoym/hf_cache/datasets

QWEN_INSTRUCT=$(ls -d /mmfs1/gscratch/krishna/leoym/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/*/ | head -1)
BEEMIX=/gscratch/scrubbed/leoym/small-vlm-outputs/sft-unified-bee-mix/checkpoint-2000

if [ "${VENV:-main}" = "neo" ]; then
  source /mmfs1/gscratch/krishna/leoym/envs/neo/bin/activate
  export HF_HOME=/mmfs1/gscratch/krishna/leoym/hf_cache
  OUT=${PROJECT_ROOT}/neo_analysis/results_text_capacity_neo.json
  echo "[textcap] node=$(hostname) NEO + Qwen3-1.7B-Base -> ${OUT}"
  python -u neo_analysis/neo_text_capacity.py "${OUT}" "${N:-200}"
else
  source .venv/bin/activate
  OUT=${PROJECT_ROOT}/neo_analysis/results_text_capacity_main.json
  echo "[textcap] node=$(hostname) Qwen-instruct + bee-mix -> ${OUT}"
  python -u devtools/text_capacity_probe.py "${OUT}" qwen_instruct qwen "${QWEN_INSTRUCT}" "${N:-200}"
  python -u devtools/text_capacity_probe.py "${OUT}" beemix vlm "${BEEMIX}" "${N:-200}"
fi
echo "[textcap] DONE"
