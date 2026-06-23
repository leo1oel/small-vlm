#!/bin/bash
#SBATCH --job-name=aux-fusion-full
#SBATCH --partition=ckpt-all
#SBATCH --account=cse-ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/mmfs1/gscratch/krishna/leoym/small-vlm/logs/aux_fusion_full_%j.log
#SBATCH --error=/mmfs1/gscratch/krishna/leoym/small-vlm/logs/aux_fusion_full_%j.log

# All-layer unified fusion metrics (cost(d) / nu(l) / rho(l) + swap nulls),
# base-2000 vs aux-2000, VMCBench dev-1000. rho on all n; the 4N causal
# sweeps on the first N_CAUSAL. Resume-safe (jsonl append) -> requeue continues.

set -euo pipefail
PROJECT_ROOT=/mmfs1/gscratch/krishna/leoym/small-vlm
cd "${PROJECT_ROOT}"
source .venv/bin/activate
type module &>/dev/null || source /usr/share/lmod/lmod/init/bash
module load cuda/12.9.1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE=/gscratch/krishna/leoym/hf_cache/datasets

OUT=${PROJECT_ROOT}/neo_analysis
STEP="${STEP:-5000}"
BASE=/gscratch/scrubbed/leoym/small-vlm-outputs/sft-unified-bee-mix/checkpoint-${STEP}
AUX=/gscratch/scrubbed/leoym/small-vlm-outputs/sft-unified-bee-mix-aux/checkpoint-${STEP}
LIMIT="${LIMIT:-0}"        # 0 = all 1000
NCAUSAL="${NCAUSAL:-500}"  # samples that get the per-layer causal sweeps
ARMS="${ARMS:-base aux}"

echo "[full] node=$(hostname) step=${STEP} limit=${LIMIT} n_causal=${NCAUSAL} arms=${ARMS}"
for arm in ${ARMS}; do
  [ "$arm" = base ] && CK="${BASE}" || CK="${AUX}"
  python -u devtools/aux_fusion_full.py "${CK}" "${OUT}/results_fusion_full_${arm}${STEP}.jsonl" "${LIMIT}" "${NCAUSAL}"
done
echo "[full] ALL DONE"
