#!/bin/bash
# Generic FDI/readiness probe for ONE unified checkpoint (spec 2026-06-14
# mechanism track): runs aux_fusion_full.py -> cost(d)/retained/FDI + R0 + rho.
# Submit per arm:
#   TAG=capacity5000 CKPT=/path/checkpoint-5000 sbatch devtools/fusion_arm.slurm
#SBATCH --job-name=fusion-arm
#SBATCH --partition=ckpt-all
#SBATCH --account=cse-ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH --output=/mmfs1/gscratch/krishna/leoym/small-vlm/logs/fusion_arm_%j.log
#SBATCH --error=/mmfs1/gscratch/krishna/leoym/small-vlm/logs/fusion_arm_%j.log

set -euo pipefail
PROJECT_ROOT=/mmfs1/gscratch/krishna/leoym/small-vlm
cd "${PROJECT_ROOT}"
source .venv/bin/activate
type module &>/dev/null || source /usr/share/lmod/lmod/init/bash
module load cuda/12.9.1 || true
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE=/gscratch/krishna/leoym/hf_cache/datasets
TAG="${TAG:?set TAG}"
CKPT="${CKPT:?set CKPT}"
LIMIT="${LIMIT:-0}"        # 0 = all 1000 (rho on all)
NCAUSAL="${NCAUSAL:-250}"  # per-layer causal sweeps on the first NCAUSAL
OUT="${PROJECT_ROOT}/neo_analysis/results_fusion_full_${TAG}.jsonl"
echo "[fusion-arm] node=$(hostname) tag=${TAG} ckpt=${CKPT} n_causal=${NCAUSAL}"
python -u devtools/aux_fusion_full.py "${CKPT}" "${OUT}" "${LIMIT}" "${NCAUSAL}"
echo "[fusion-arm] DONE -> ${OUT}"
