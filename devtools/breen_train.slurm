#!/bin/bash
# BREEN reproduction trainer (arXiv:2503.12446 port, spec 2026-06-24).
#
# CRITICAL: runs from THIS worktree (the BREEN port lives here, NOT in the main
# /mmfs1/gscratch/krishna/leoym/small-vlm checkout that train_l40s.slurm uses).
# PROJECT_ROOT + PYTHONPATH + .venv all point at the worktree so the streamed
# run exercises the committed BREEN code.
#
# Launch a stage (S0 -> S1 -> S2 chained via from_pretrained env vars):
#   CONFIG_NAME=pretrain-unified-breen    sbatch devtools/breen_train.slurm
#   VLM_BREEN_S0_CKPT=/.../checkpoint-NNNN CONFIG_NAME=pretrain-unified-breen-s1 \
#       sbatch devtools/breen_train.slurm
#   VLM_BREEN_S1_CKPT=/.../checkpoint-NNNN CONFIG_NAME=sft-unified-breen \
#       sbatch devtools/breen_train.slurm
# Per-stage data-budget / effective-batch tuning is passed via HYDRA_EXTRA, e.g.
#   HYDRA_EXTRA="trainer.max_steps=10000 trainer.gradient_accumulation_steps=2"
#
#SBATCH --job-name=breen-train
#SBATCH --partition=gpu-l40s
#SBATCH --account=krishna
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=700G
#SBATCH --time=1-00:00:00
#SBATCH --output=/mmfs1/gscratch/krishna/leoym/herdr-worktrees/small-vlm/fm-breen-port-h7/logs/breen_train_%j.log
#SBATCH --error=/mmfs1/gscratch/krishna/leoym/herdr-worktrees/small-vlm/fm-breen-port-h7/logs/breen_train_%j.log

set -uo pipefail
PROJECT_ROOT=/mmfs1/gscratch/krishna/leoym/herdr-worktrees/small-vlm/fm-breen-port-h7
cd "${PROJECT_ROOT}"
mkdir -p logs
source .venv/bin/activate
# Force the worktree's src onto the path so `import vlm` resolves to the BREEN
# code regardless of how small-vlm was installed into the venv.
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

# DeepSpeed imports call nvcc on GPU nodes; load the toolkit, skip the major
# version check (we JIT no DeepSpeed ops).
type module &>/dev/null || source /usr/share/lmod/lmod/init/bash
module load cuda/12.9.1
export DS_SKIP_CUDA_CHECK=1

# Azure streaming creds: the worktree has no .env, so export AZURE_SAS_TOKEN
# from the main checkout's .env (read-only; _bootstrap_env derives the rest).
if [[ -z "${AZURE_SAS_TOKEN:-}" && -f /mmfs1/gscratch/krishna/leoym/small-vlm/.env ]]; then
    set -a; source /mmfs1/gscratch/krishna/leoym/small-vlm/.env; set +a
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# HF weights are pre-cached; Azure streaming is unaffected by the offline flag.
export HF_HUB_OFFLINE=1
# MSC range-cache on node-local NVMe — streamed media never touches GPFS (the
# 4.8M-loose-file inode trap the directive warns about); wiped with the node.
export MSC_CACHE_DIR="/scr/${USER}/msc-cache"
mkdir -p "${MSC_CACHE_DIR}"
# Local-hybrid energon mode (the default for these dataset configs) downloads
# ONLY the jsonl (1 file, 1 inode) + builds its .idx, then STREAMS the 4.8M
# images on demand. Persist that jsonl+index on scrubbed (huge quota, survives
# requeue) so the multi-GB download/index pass happens ONCE and is reused — NOT
# in ~/.cache (the default, tight home quota). Images never land here.
export VLM_ENERGON_DATA_DIR="${VLM_ENERGON_DATA_DIR:-/gscratch/scrubbed/leoym/breen-energon-jsonl}"
mkdir -p "${VLM_ENERGON_DATA_DIR}"
# NCCL: tolerate streaming stalls (directive trap: 600s default times out on a
# slow first-warmup blob fetch).
export NCCL_TIMEOUT=3600

export NGPUS="${NGPUS:-4}"
CONFIG_NAME="${CONFIG_NAME:?set CONFIG_NAME (pretrain-unified-breen | -s1 | sft-unified-breen)}"
NUM_WORKERS="${NUM_WORKERS:-12}"
# Stable, scrubbed output dir (separate quota, immune to the krishna inode
# exhaustion) so --requeue auto-resume finds prior checkpoints.
RUN_DIR="${RUN_DIR:-/gscratch/scrubbed/leoym/breen-outputs/${CONFIG_NAME}}"
mkdir -p "${RUN_DIR}"

echo "[breen] node=$(hostname) branch=$(git rev-parse --abbrev-ref HEAD) head=$(git rev-parse --short HEAD)"
echo "[breen] config=${CONFIG_NAME} ngpus=${NGPUS} run_dir=${RUN_DIR}"
echo "[breen] hydra_extra=${HYDRA_EXTRA:-<none>}"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader || true

python -u -m torch.distributed.run \
    --standalone \
    --nproc_per_node "${NGPUS}" \
    -m vlm -cn "${CONFIG_NAME}" \
    dataset.num_workers="${NUM_WORKERS}" \
    trainer.output_dir="${RUN_DIR}" \
    hydra.run.dir="${RUN_DIR}/hydra" \
    ${HYDRA_EXTRA:-}
rc=$?

# Propagate the training exit code so SLURM job dependencies work: the
# orchestrator gates the full run on the dry-run with --dependency=afterok,
# which only fires when this script exits 0. (set -uo pipefail does NOT make a
# failed python abort the script, so exit it explicitly.)
echo "END TIME: $(date) rc=${rc}"
exit ${rc}
