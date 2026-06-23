#!/bin/bash
# Resume-path validation: reuse the finished trial's output dir with
# max_steps=1010 — auto-resume must pick checkpoint-1000, restore deepspeed
# optimizer/scheduler AND the energon stream state, then train 10 more steps.
# This is exactly the --requeue preemption-recovery path.
# Submit: sbatch devtools/sft_resume_check.slurm
#SBATCH --job-name=sft-resume-check
#SBATCH --partition=ckpt-all
#SBATCH --account=cse-ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=l40:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=1:00:00
#SBATCH --output=/mmfs1/gscratch/krishna/leoym/small-vlm/logs/sft_resume_%j.log

set -euo pipefail
PROJECT_ROOT=/mmfs1/gscratch/krishna/leoym/small-vlm
cd "${PROJECT_ROOT}"
source .venv/bin/activate

type module &>/dev/null || source /usr/share/lmod/lmod/init/bash
module load cuda/12.9.1
export DS_SKIP_CUDA_CHECK=1
export VLM_DATA_ROOT="${VLM_DATA_ROOT:-/gscratch/krishna/${USER}/data}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export MSC_CACHE_DIR="/scr/${USER}/msc-cache"
mkdir -p "${MSC_CACHE_DIR}"

RUN_DIR="${PROJECT_ROOT}/outputs/sft-unified-trial"
test -d "${RUN_DIR}/checkpoint-1000" || { echo "no checkpoint-1000 to resume from"; exit 1; }

# The trial checkpoint was produced BEFORE length bucketing / chunked CE
# landed; energon (correctly) refuses to restore stream state into a
# different batching topology, so resume with the checkpoint's own pipeline
# shape. (Production rule: flip length_buckets/num_workers only at fresh
# starts, never mid-run.)
python -u -m torch.distributed.run \
    --standalone \
    --nproc_per_node 4 \
    -m vlm -cn sft-unified \
    dataset.length_buckets=null \
    trainer.loss_chunk_size=0 \
    trainer.max_steps=1010 \
    trainer.save_steps=500 \
    trainer.output_dir="${RUN_DIR}" \
    hydra.run.dir="${RUN_DIR}/hydra"

echo "RESUME CHECK DONE: $(date)"
