#!/bin/bash
#SBATCH --job-name=aux-fusion-probe
#SBATCH --partition=ckpt-all
#SBATCH --account=cse-ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH --output=/mmfs1/gscratch/krishna/leoym/small-vlm/logs/aux_fusion_probe_%j.log
#SBATCH --error=/mmfs1/gscratch/krishna/leoym/small-vlm/logs/aux_fusion_probe_%j.log

# Fusion-structure manipulation check: isolation sweep + layer-k logit lens
# (image-swap controlled) + attention mass, base-2000 vs aux-2000, VMCBench
# dev-1000. Both arms run sequentially on one GPU; the probe script is
# resume-safe (appends to its jsonl), so requeue just continues.

set -euo pipefail
PROJECT_ROOT=/mmfs1/gscratch/krishna/leoym/small-vlm
cd "${PROJECT_ROOT}"
source .venv/bin/activate

type module &>/dev/null || source /usr/share/lmod/lmod/init/bash
module load cuda/12.9.1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE=/gscratch/krishna/leoym/hf_cache/datasets

OUT_DIR="${PROJECT_ROOT}/neo_analysis"
BASE=/gscratch/scrubbed/leoym/small-vlm-outputs/sft-unified-bee-mix/checkpoint-2000
AUX=/gscratch/scrubbed/leoym/small-vlm-outputs/sft-unified-bee-mix-aux/checkpoint-2000
LIMIT="${LIMIT:-0}"  # 0 = all 1000; e.g. LIMIT=8 for a smoke run

echo "[probe] node=$(hostname) limit=${LIMIT}"
python -u devtools/aux_fusion_probe.py "${BASE}" \
    "${OUT_DIR}/results_fusion_probe_base2000.jsonl" "${LIMIT}"
python -u devtools/aux_fusion_probe.py "${AUX}" \
    "${OUT_DIR}/results_fusion_probe_aux2000.jsonl" "${LIMIT}"
echo "[probe] ALL DONE"
