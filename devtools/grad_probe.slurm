#!/bin/bash
#SBATCH --job-name=vlm-gradprobe
#SBATCH --partition=ckpt-all
#SBATCH --account=cse-ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=l40:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=96G
#SBATCH --requeue
#SBATCH --time=1:10:00
#SBATCH --exclude=g3098,g3108
#SBATCH --output=/mmfs1/gscratch/krishna/leoym/small-vlm/logs/gradprobe_%j.log
#SBATCH --error=/mmfs1/gscratch/krishna/leoym/small-vlm/logs/gradprobe_%j.log

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
export TOKENIZERS_PARALLELISM=false

# Probe ON. Single GPU, no DeepSpeed (ZeRO shards grads -> p.grad incomplete).
export GRAD_PROBE=1
export GRAD_PROBE_EVERY=5
# DELTA_TUNING=1 freezes text FFN/embed/lm_head/norm (passthrough from sbatch).
export DELTA_TUNING="${DELTA_TUNING:-0}"

CONFIG_NAME="${CONFIG_NAME:-sft-unified-bee-mix-visualffn}"
OUT=/gscratch/scrubbed/leoym/small-vlm-outputs/_gradprobe
echo "[gradprobe] node=$(hostname) config=${CONFIG_NAME}"
nvidia-smi --query-gpu=memory.total --format=csv,noheader || true

python -u -m torch.distributed.run --standalone --nproc_per_node 1 \
    -m vlm -cn "${CONFIG_NAME}" \
    trainer.deepspeed=null \
    trainer.max_steps="${MAXSTEPS:-800}" \
    trainer.save_strategy=no \
    trainer.per_device_train_batch_size="${BS:-2}" \
    trainer.gradient_accumulation_steps=1 \
    trainer.logging_steps=5 \
    dataset.num_workers=4 \
    trainer.output_dir="${OUT}" \
    hydra.run.dir="${OUT}/hydra"

echo "END TIME: $(date)"
