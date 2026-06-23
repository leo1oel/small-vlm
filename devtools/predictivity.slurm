#!/bin/bash
#SBATCH --job-name=pred
#SBATCH --partition=ckpt-all
#SBATCH --account=cse-ckpt
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --requeue
#SBATCH --output=/mmfs1/gscratch/krishna/leoym/small-vlm/logs/pred_%j.log
#SBATCH --error=/mmfs1/gscratch/krishna/leoym/small-vlm/logs/pred_%j.log
# Linear-predictivity (native layer rep -> encoder rep, CV-ridge R^2). CPU-only.
set -uo pipefail
cd /mmfs1/gscratch/krishna/leoym/small-vlm
source .venv/bin/activate
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HOME=/mmfs1/gscratch/krishna/leoym/hf_cache
export PRED_JOBS=${PRED_JOBS:-16}
export CKA_DIR=${CKA_DIR:-/mmfs1/gscratch/krishna/leoym/small-vlm/neo_analysis}
echo "[pred] node=$(hostname) CKA_DIR=$CKA_DIR OUT=${OUT} TAGS=${TAGS:-all}"
python -u devtools/predictivity_compute.py "${OUT}" ${TAGS:-}
echo "[pred] ALL DONE"
