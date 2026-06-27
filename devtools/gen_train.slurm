#!/bin/bash
# Large-scale generation training on GPIC. Submit:
#   CONFIG_NAME=gen-gpic sbatch devtools/gen_train.slurm
# Short data-path smoke first (few steps, no wandb):
#   CONFIG_NAME=gen-gpic EXTRA="trainer.max_steps=30 trainer.save_steps=1000 trainer.report_to=none" \
#       sbatch --time=0:40:00 --gpus=l40s:2 devtools/gen_train.slurm
#SBATCH --job-name=gen-gpic
#SBATCH --partition=gpu-l40s
#SBATCH --account=krishna
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=l40s:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --output=/mmfs1/gscratch/krishna/leoym/small-vlm/logs/%x_%j.log

set -uo pipefail
PROJECT_ROOT=/mmfs1/gscratch/krishna/leoym/small-vlm
cd "${PROJECT_ROOT}"
source .venv/bin/activate
type module &>/dev/null || source /usr/share/lmod/lmod/init/bash
module load cuda/12.9.1
export DS_SKIP_CUDA_CHECK=1
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MSC_CACHE_DIR="/scr/${USER}/msc-cache"; mkdir -p "${MSC_CACHE_DIR}"
export WANDB_PROJECT="${WANDB_PROJECT:-small-vlm-gen}"
export TOKENIZERS_PARALLELISM=false

NGPUS="${NGPUS:-4}"
CONFIG_NAME="${CONFIG_NAME:-gen-gpic}"
# Stable run dir so --requeue / resubmit auto-resumes from the last checkpoint.
RUN_DIR="${PROJECT_ROOT}/outputs/${CONFIG_NAME}"

echo "[slurm] node=$(hostname) job=${SLURM_JOB_ID} config=${CONFIG_NAME} ngpus=${NGPUS}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python -u -m torch.distributed.run --standalone --nproc_per_node "${NGPUS}" \
    -m vlm -cn "${CONFIG_NAME}" \
    trainer.output_dir="${RUN_DIR}" \
    hydra.run.dir="${RUN_DIR}/hydra" \
    ${EXTRA:-}
echo "END $(date)"
