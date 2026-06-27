#!/bin/bash
#SBATCH --job-name=neo-fusion-full
#SBATCH --partition=ckpt-all
#SBATCH --account=cse-ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --requeue
#SBATCH --output=/mmfs1/gscratch/krishna/leoym/small-vlm/logs/neo_fusion_full_%j.log
#SBATCH --error=/mmfs1/gscratch/krishna/leoym/small-vlm/logs/neo_fusion_full_%j.log

# All-layer cost(d)+rho(l) for NEO-2B SFT on VMCBench dev. NEO venv + SDPA
# dense-mask patch (flex_attention OOMs on sm89 at head_dim 256). Resume-safe.
set -euo pipefail
PROJECT_ROOT=/mmfs1/gscratch/krishna/leoym/small-vlm
cd "${PROJECT_ROOT}"
source /mmfs1/gscratch/krishna/leoym/envs/neo/bin/activate
type module &>/dev/null || source /usr/share/lmod/lmod/init/bash
module load cuda/12.9.1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HOME=/mmfs1/gscratch/krishna/leoym/hf_cache
export HF_DATASETS_CACHE=/gscratch/krishna/leoym/hf_cache/datasets

STAGE="${STAGE:-SFT}"
# SFT keeps the legacy filename (results already on disk); PT/MT get suffixes.
if [ "${STAGE}" = "SFT" ]; then OUT=${PROJECT_ROOT}/neo_analysis/results_fusion_full_neo.jsonl
else OUT=${PROJECT_ROOT}/neo_analysis/results_fusion_full_neo_${STAGE}.jsonl; fi
LIMIT="${LIMIT:-600}"
NCAUSAL="${NCAUSAL:-300}"
echo "[nfull] node=$(hostname) stage=${STAGE} limit=${LIMIT} n_causal=${NCAUSAL} out=${OUT}"
python -u neo_analysis/neo_fusion_full.py "${OUT}" "${LIMIT}" "${NCAUSAL}" "${STAGE}"
echo "[nfull] ALL DONE"
