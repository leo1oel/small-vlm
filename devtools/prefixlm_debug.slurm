#!/bin/bash
#SBATCH --job-name=vlm-prefixlm-debug
#SBATCH --partition=ckpt-all
#SBATCH --account=cse-ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=l40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --requeue
#SBATCH --time=0:30:00
#SBATCH --output=/mmfs1/gscratch/krishna/leoym/small-vlm/logs/prefixlm_debug_%j.log
#SBATCH --error=/mmfs1/gscratch/krishna/leoym/small-vlm/logs/prefixlm_debug_%j.log

set -euo pipefail
PROJECT_ROOT=/mmfs1/gscratch/krishna/leoym/small-vlm
cd "${PROJECT_ROOT}"
source .venv/bin/activate

type module &>/dev/null || source /usr/share/lmod/lmod/init/bash
module load cuda/12.9.1
export DS_SKIP_CUDA_CHECK=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE=/gscratch/krishna/leoym/hf_cache/datasets
export LMMS_EVAL_DATASETS_CACHE=/gscratch/krishna/leoym/hf_cache/datasets

CKPT="${CKPT:-/gscratch/scrubbed/leoym/small-vlm-outputs/sft-unified-bee-mix-prefixlm/checkpoint-3000}"
echo "[prefixlm-debug] node=$(hostname) ckpt=${CKPT}"
python -u devtools/prefixlm_debug.py "${CKPT}"
echo "END TIME: $(date)"
