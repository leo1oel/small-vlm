#!/bin/bash
# 1000-step sft-unified trial: validates streaming data load, loss descent,
# and checkpointing; samples GPU util/memory to size batch_size headroom and
# spot dataloader stalls. Submit: sbatch devtools/sft_trial.slurm
#SBATCH --job-name=resume-rt
#SBATCH --partition=ckpt-all
#SBATCH --account=cse-ckpt
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=h200:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=700G
#SBATCH --time=1:30:00
#SBATCH --output=/mmfs1/gscratch/krishna/leoym/small-vlm/logs/resume_rt_%j.log

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
# HF weights must be pre-downloaded on a login node (hub downloads from
# compute nodes are flaky); Azure streaming is unaffected by this flag.
export HF_HUB_OFFLINE=1
# MSC media range-cache on node-local NVMe (2.8T at /scr): cache writes leave
# the GPFS hot path. Wiped with the node — fine, hit rate across one pass of
# a 46M-sample blend is ~0; the cache only smooths within-run re-reads.
export MSC_CACHE_DIR="/scr/${USER}/msc-cache"
mkdir -p "${MSC_CACHE_DIR}"

# Stable output dir so --requeue auto-resume can find previous checkpoints.
RUN_DIR="${PROJECT_ROOT}/outputs/sft-resume-rt"

echo "[trial] node=$(hostname) run_dir=${RUN_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# GPU telemetry every 10s — batch-size headroom + data-stall analysis.
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used \
    --format=csv,noheader -l 10 \
    > "${PROJECT_ROOT}/logs/resume_rt_gpu_${SLURM_JOB_ID}.csv" &
SMI_PID=$!
trap 'kill ${SMI_PID} 2>/dev/null || true' EXIT

python -u -m torch.distributed.run \
    --standalone \
    --nproc_per_node 4 \
    -m vlm -cn sft-unified \
    trainer.max_steps=${RT_STEPS:-8} trainer.save_steps=5 dataset.num_workers=12 \
        trainer.output_dir="${RUN_DIR}" \
    hydra.run.dir="${RUN_DIR}/hydra"

echo "END TIME: $(date)"
