#!/bin/bash
#SBATCH --job-name=gemma4-fusion-full
#SBATCH --partition=ckpt-all
#SBATCH --account=cse-ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --requeue
#SBATCH --output=/mmfs1/gscratch/krishna/leoym/small-vlm/logs/gemma4_fusion_full_%j.log
#SBATCH --error=/mmfs1/gscratch/krishna/leoym/small-vlm/logs/gemma4_fusion_full_%j.log

# All-layer cost(d)+rho(l) for Gemma-4-12B on VMCBench dev. Resume-safe jsonl.
set -euo pipefail
PROJECT_ROOT=/mmfs1/gscratch/krishna/leoym/small-vlm
cd "${PROJECT_ROOT}"
source .venv/bin/activate
type module &>/dev/null || source /usr/share/lmod/lmod/init/bash
module load cuda/12.9.1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HOME=/mmfs1/gscratch/krishna/leoym/hf_cache
export HF_DATASETS_CACHE=/gscratch/krishna/leoym/hf_cache/datasets

OUT=${PROJECT_ROOT}/neo_analysis/results_fusion_full_gemma4.jsonl
LIMIT="${LIMIT:-600}"
NCAUSAL="${NCAUSAL:-300}"
echo "[gfull] node=$(hostname) limit=${LIMIT} n_causal=${NCAUSAL}"
python -u neo_analysis/gemma4_fusion_full.py "${OUT}" "${LIMIT}" "${NCAUSAL}"
echo "[gfull] ALL DONE"
