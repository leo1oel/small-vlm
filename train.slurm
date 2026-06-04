#!/bin/bash
#SBATCH --job-name=small-vlm
#SBATCH --partition=ckpt-all
#SBATCH --account=cse-ckpt
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=l40:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=4:00:00
#SBATCH --output=/mmfs1/gscratch/krishna/leoym/small-vlm/logs/train_%j.log
#SBATCH --error=/mmfs1/gscratch/krishna/leoym/small-vlm/logs/train_%j.log
# NOTE: slurmd opens the output file BEFORE the script body runs, so logs/ must
# already exist at submission time (a tracked logs/.gitkeep guarantees this).

set -euo pipefail
PROJECT_ROOT=/mmfs1/gscratch/krishna/leoym/small-vlm
cd "${PROJECT_ROOT}"
source .venv/bin/activate

# Host-specific roots (adjust to where data/models actually live)
export VLM_DATA_ROOT="${VLM_DATA_ROOT:-/gscratch/krishna/${USER}/data}"
export VLM_MODEL_ROOT="${VLM_MODEL_ROOT:-/gscratch/krishna/${USER}/models}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export NGPUS=4
CONFIG_NAME="${CONFIG_NAME:-finetune-llava}"
# Stable output dir so --requeue auto-resume can find previous checkpoints
# (hydra's default run dir is timestamped and changes on every requeue).
RUN_DIR="${PROJECT_ROOT}/outputs/${CONFIG_NAME}"

echo "[small-vlm] node=$(hostname) config=${CONFIG_NAME} run_dir=${RUN_DIR}"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader || true

python -u -m torch.distributed.run \
    --standalone \
    --nproc_per_node "${NGPUS}" \
    -m vlm -cn "${CONFIG_NAME}" \
    trainer.output_dir="${RUN_DIR}" \
    hydra.run.dir="${RUN_DIR}/hydra"

echo "END TIME: $(date)"
