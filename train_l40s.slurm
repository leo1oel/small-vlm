#!/bin/bash
#SBATCH --job-name=small-vlm-48g
#SBATCH --partition=gpu-l40s
#SBATCH --account=krishna
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=700G
#SBATCH --time=1-00:00:00
#SBATCH --output=/mmfs1/gscratch/krishna/leoym/small-vlm/logs/train_%j.log
#SBATCH --error=/mmfs1/gscratch/krishna/leoym/small-vlm/logs/train_%j.log
# Dedicated (non-preemptible) L40S launcher for the 48G experiment matrix.
# Same body as train_h200.slurm; only the partition/account/gpu lines differ.
# The cse-quota arm overrides at submit time: sbatch -A cse train_l40s.slurm
# NOTE: slurmd opens the output file BEFORE the script body runs, so logs/
# must already exist at submission time (a tracked logs/.gitkeep guarantees this).

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

# Host-specific roots (adjust to where data/models actually live)
export VLM_DATA_ROOT="${VLM_DATA_ROOT:-/gscratch/krishna/${USER}/data}"
export VLM_MODEL_ROOT="${VLM_MODEL_ROOT:-/gscratch/krishna/${USER}/models}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# HF weights must be pre-downloaded on a login node (hub downloads from
# compute nodes are flaky); Azure streaming is unaffected by this flag.
export HF_HUB_OFFLINE=1
# MSC media range-cache on node-local NVMe: cache writes leave the GPFS hot
# path; wiped with the node, which is fine for a streaming pass.
export MSC_CACHE_DIR="/scr/${USER}/msc-cache"
mkdir -p "${MSC_CACHE_DIR}"

export NGPUS=4
CONFIG_NAME="${CONFIG_NAME:-sft-unified-bee-mix}"
# A40 nodes have 52 CPUs (vs 128 on L40S/H200): pass NUM_WORKERS=8 when
# submitting onto A40; the default 12 is sized for L40S/H200 nodes.
NUM_WORKERS="${NUM_WORKERS:-12}"
# Stable output dir so --requeue auto-resume can find previous checkpoints
# (hydra's default run dir is timestamped and changes on every requeue).
# Checkpoints live on the scrubbed FS: separate (huge) quota, immune to
# the krishna group inode exhaustion that killed step-4000 saves. Purged if
# untouched ~21 days — archive finals back after the run.
RUN_DIR="/gscratch/scrubbed/leoym/small-vlm-outputs/${CONFIG_NAME}"

echo "[small-vlm] node=$(hostname) config=${CONFIG_NAME} run_dir=${RUN_DIR}"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader || true

python -u -m torch.distributed.run \
    --standalone \
    --nproc_per_node "${NGPUS}" \
    -m vlm -cn "${CONFIG_NAME}" \
    dataset.num_workers="${NUM_WORKERS}" \
    trainer.output_dir="${RUN_DIR}" \
    hydra.run.dir="${RUN_DIR}/hydra" \
    ${HYDRA_EXTRA:-}

echo "END TIME: $(date)"
