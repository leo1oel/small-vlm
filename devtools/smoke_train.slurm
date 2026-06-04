#!/bin/bash
# Short end-to-end training smoke on synthetic data (mirrors train.slurm).
# Submit: sbatch devtools/smoke_train.slurm
#SBATCH --job-name=small-vlm-smoke
#SBATCH --partition=ckpt-all
#SBATCH --account=cse-ckpt
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=l40s:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0:40:00
#SBATCH --output=/mmfs1/gscratch/krishna/leoym/small-vlm/logs/smoke_train_%j.log
#SBATCH --error=/mmfs1/gscratch/krishna/leoym/small-vlm/logs/smoke_train_%j.log

set -euo pipefail
PROJECT_ROOT=/mmfs1/gscratch/krishna/leoym/small-vlm
cd "${PROJECT_ROOT}"
source .venv/bin/activate

# DeepSpeed imports call nvcc on GPU nodes; compute nodes have no /usr/local/cuda.
# Load the newest toolkit module and skip the 12.9-vs-13.0 (torch cu130) major
# version check — we compile no DeepSpeed JIT ops (pure ZeRO, no offload).
type module &>/dev/null || source /usr/share/lmod/lmod/init/bash
module load cuda/12.9.1
export DS_SKIP_CUDA_CHECK=1

export VLM_DATA_ROOT=/gscratch/krishna/leoym/smoke_data
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export NGPUS=2
RUN_DIR="${PROJECT_ROOT}/outputs/smoke-pretrain-${SLURM_JOB_ID}"
rm -rf "${RUN_DIR}"

echo "[smoke] node=$(hostname)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true

python -u -m torch.distributed.run \
    --standalone \
    --nproc_per_node "${NGPUS}" \
    -m vlm -cn pretrain-qwen \
    model=qwen2.5-0.5b \
    trainer.output_dir="${RUN_DIR}" \
    trainer.attn_implementation=sdpa \
    trainer.per_device_train_batch_size=4 \
    trainer.gradient_accumulation_steps=1 \
    trainer.dataloader_num_workers=4 \
    trainer.save_steps=10 \
    trainer.report_to=none \
    hydra.run.dir="${RUN_DIR}/hydra"

echo "[smoke] training exited with $?"
ls -la "${RUN_DIR}" || true
echo "END TIME: $(date)"
