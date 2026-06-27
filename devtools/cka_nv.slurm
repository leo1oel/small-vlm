#!/bin/bash
#SBATCH --job-name=ckanv
#SBATCH --partition=ckpt-all
#SBATCH --account=cse-ckpt
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=3:00:00
#SBATCH --requeue
#SBATCH --output=/mmfs1/gscratch/krishna/leoym/small-vlm/logs/ckanv_%j.log
#SBATCH --error=/mmfs1/gscratch/krishna/leoym/small-vlm/logs/ckanv_%j.log
# Per-layer image-token rep extraction for NEO / SAIL (neo venv: transformers 4.57 + SDPA dense-mask).
set -uo pipefail
cd /mmfs1/gscratch/krishna/leoym/small-vlm
source /mmfs1/gscratch/krishna/leoym/envs/neo/bin/activate
type module &>/dev/null || source /usr/share/lmod/lmod/init/bash
module load cuda/12.9.1
export HF_HUB_OFFLINE=${HFOFF:-1} HF_DATASETS_OFFLINE=${HFOFF:-1}
export HF_HOME=/mmfs1/gscratch/krishna/leoym/hf_cache
export HF_DATASETS_CACHE=/gscratch/krishna/leoym/hf_cache/datasets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[ckanv] node=$(hostname) script=$SCRIPT tag=$TAG n=${N:-1000} stage=${STAGE:-}"
python -u "$SCRIPT" "$TAG" "${N:-1000}" ${STAGE:-}
echo "[ckanv] ALL DONE"
