#!/bin/bash
# 50-step probe of the 48G token budget (76750) with the REAL Arm-1 config
# (sft-unified-bee-mix: 3-way blend incl. Bee) on a dedicated L40S node:
# validates blend streaming + Bee jsonl auto-index, measures peak GPU memory
# for the OOM verdict, and yields samples/step for the step-count math.
# Submit: sbatch devtools/budget_probe_l40s.slurm
# Fallback if OOM: resubmit with BUDGET=72000.
#SBATCH --job-name=budget-probe-l40s
#SBATCH --partition=gpu-l40s
#SBATCH --account=krishna
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=700G
#SBATCH --time=1:30:00
#SBATCH --output=/mmfs1/gscratch/krishna/leoym/small-vlm/logs/budget_probe_l40s_%j.log

set -euo pipefail
PROJECT_ROOT=/mmfs1/gscratch/krishna/leoym/small-vlm
cd "${PROJECT_ROOT}"
source .venv/bin/activate

# DeepSpeed imports call nvcc on GPU nodes; compute nodes have no /usr/local/cuda.
type module &>/dev/null || source /usr/share/lmod/lmod/init/bash
module load cuda/12.9.1
export DS_SKIP_CUDA_CHECK=1

export VLM_DATA_ROOT="${VLM_DATA_ROOT:-/gscratch/krishna/${USER}/data}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export MSC_CACHE_DIR="/scr/${USER}/msc-cache"
mkdir -p "${MSC_CACHE_DIR}"

BUDGET="${BUDGET:-76750}"
RUN_DIR="${PROJECT_ROOT}/outputs/budget-probe-l40s"

echo "[probe] node=$(hostname) budget=${BUDGET} run_dir=${RUN_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# GPU telemetry every 10s — the peak memory.used column is the verdict.
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used \
    --format=csv,noheader -l 10 \
    > "${PROJECT_ROOT}/logs/budget_probe_l40s_gpu_${SLURM_JOB_ID}.csv" &
SMI_PID=$!
trap 'kill ${SMI_PID} 2>/dev/null || true' EXIT

python -u -m torch.distributed.run \
    --standalone \
    --nproc_per_node 4 \
    -m vlm -cn sft-unified-bee-mix \
    trainer.max_steps=50 dataset.num_workers=12 \
    dataset.batch_token_budget="${BUDGET}" \
    trainer.save_steps=5000 trainer.report_to=none \
    trainer.output_dir="${RUN_DIR}" \
    hydra.run.dir="${RUN_DIR}/hydra"

echo "END TIME: $(date)"
