#!/bin/bash
# Single-sample generation overfit smoke. Submit:
#   sbatch devtools/gen_overfit.slurm
#   STEPS=1200 LR=3e-4 sbatch --export=ALL,STEPS,LR devtools/gen_overfit.slurm
#SBATCH --job-name=gen-overfit
#SBATCH --partition=gpu-l40s
#SBATCH --account=krishna
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0:45:00
#SBATCH --output=/mmfs1/gscratch/krishna/leoym/small-vlm/logs/gen_overfit_%j.log

set -uo pipefail
PROJECT_ROOT=/mmfs1/gscratch/krishna/leoym/small-vlm
cd "${PROJECT_ROOT}"
source .venv/bin/activate
type module &>/dev/null || source /usr/share/lmod/lmod/init/bash
module load cuda/12.9.1
export DS_SKIP_CUDA_CHECK=1
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[slurm] node=$(hostname) job=${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python -u devtools/gen_overfit.py --steps "${STEPS:-800}" --lr "${LR:-2e-4}"
echo "END $(date)"
