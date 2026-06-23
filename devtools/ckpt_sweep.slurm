#!/bin/bash
#SBATCH --job-name=vlm-ckpt-sweep
#SBATCH --partition=ckpt-all
#SBATCH --account=cse-ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=/mmfs1/gscratch/krishna/leoym/small-vlm/logs/ckpt_sweep_%j.log
#SBATCH --error=/mmfs1/gscratch/krishna/leoym/small-vlm/logs/ckpt_sweep_%j.log

set -euo pipefail
PROJECT_ROOT=/mmfs1/gscratch/krishna/leoym/small-vlm
cd "${PROJECT_ROOT}"
source .venv/bin/activate

type module &>/dev/null || source /usr/share/lmod/lmod/init/bash
module load cuda/12.9.1
export DS_SKIP_CUDA_CHECK=1
export HF_HUB_OFFLINE=1
export TQDM_DISABLE=1

echo "[ckpt-sweep] node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python -u devtools/ckpt_sweep.py \
    outputs/sft-clip/checkpoint-1000 \
    outputs/sft-clip/checkpoint-2000 \
    outputs/sft-clip/checkpoint-3000 \
    outputs/sft-unified/checkpoint-1000 \
    outputs/sft-unified/checkpoint-2000 \
    outputs/sft-unified/checkpoint-3000

echo "END TIME: $(date)"
