#!/bin/bash
#SBATCH --job-name=vlm-infer-trial
#SBATCH --partition=ckpt-all
#SBATCH --account=cse-ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=/mmfs1/gscratch/krishna/leoym/small-vlm/logs/infer_trial_%j.log
#SBATCH --error=/mmfs1/gscratch/krishna/leoym/small-vlm/logs/infer_trial_%j.log

set -euo pipefail
PROJECT_ROOT=/mmfs1/gscratch/krishna/leoym/small-vlm
cd "${PROJECT_ROOT}"
source .venv/bin/activate

# Same CUDA toolchain workaround as train.slurm (deepspeed import probes nvcc
# through the transformers Trainer import chain; compute nodes lack /usr/local/cuda).
type module &>/dev/null || source /usr/share/lmod/lmod/init/bash
module load cuda/12.9.1
export DS_SKIP_CUDA_CHECK=1
export HF_HUB_OFFLINE=1

CKPT="${CKPT:-outputs/sft-unified-trial/checkpoint-500}"
echo "[infer-trial] node=$(hostname) ckpt=${CKPT}"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader || true

python -u devtools/infer_trial.py "${CKPT}"

echo "END TIME: $(date)"
