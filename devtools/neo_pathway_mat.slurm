#!/bin/bash
#SBATCH --job-name=neopathmat
#SBATCH --partition=ckpt-all
#SBATCH --account=cse-ckpt
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --requeue
#SBATCH --output=/mmfs1/gscratch/krishna/leoym/small-vlm/logs/neopathmat_%j.log
#SBATCH --error=/mmfs1/gscratch/krishna/leoym/small-vlm/logs/neopathmat_%j.log
set -uo pipefail
PROJECT_ROOT=/mmfs1/gscratch/krishna/leoym/small-vlm
cd "${PROJECT_ROOT}"
source /mmfs1/gscratch/krishna/leoym/envs/neo/bin/activate
type module &>/dev/null || source /usr/share/lmod/lmod/init/bash
module load cuda/12.9.1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HOME=/mmfs1/gscratch/krishna/leoym/hf_cache
export HF_DATASETS_CACHE=/gscratch/krishna/leoym/hf_cache/datasets
STAGE="${STAGE:-SFT}"
OUT="${OUT:-${PROJECT_ROOT}/neo_analysis/results_mat_neo_${STAGE}.jsonl}"
echo "[neomat] node=$(hostname) stage=${STAGE} out=${OUT} n=${N:-1000}"
python -u neo_analysis/neo_pathway_maturation.py "${OUT}" "${N:-1000}" "${STAGE}"
echo "[neomat] ALL DONE"
