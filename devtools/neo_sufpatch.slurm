#!/bin/bash
#SBATCH --job-name=nsuf
#SBATCH --partition=ckpt-all
#SBATCH --account=cse-ckpt
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=6:00:00
#SBATCH --requeue
#SBATCH --output=/mmfs1/gscratch/krishna/leoym/small-vlm/logs/nsuf_%j.log
#SBATCH --error=/mmfs1/gscratch/krishna/leoym/small-vlm/logs/nsuf_%j.log
set -uo pipefail
cd /mmfs1/gscratch/krishna/leoym/small-vlm
source /mmfs1/gscratch/krishna/leoym/envs/neo/bin/activate
type module &>/dev/null || source /usr/share/lmod/lmod/init/bash
module load cuda/12.9.1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HOME=/mmfs1/gscratch/krishna/leoym/hf_cache
export HF_DATASETS_CACHE=/gscratch/krishna/leoym/hf_cache/datasets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[nsuf] node=$(hostname) out=$OUT stage=$STAGE n=${N:-1000} nc=${NC:-200}"
python -u neo_analysis/neo_activation_patch.py "$OUT" "${N:-1000}" "${NC:-200}" "$STAGE" sufmeanabl
echo "[nsuf] ALL DONE"
