#!/bin/bash
#SBATCH --job-name=cka
#SBATCH --partition=ckpt-all
#SBATCH --account=cse-ckpt
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=2:00:00
#SBATCH --requeue
#SBATCH --output=/mmfs1/gscratch/krishna/leoym/small-vlm/logs/cka_%j.log
#SBATCH --error=/mmfs1/gscratch/krishna/leoym/small-vlm/logs/cka_%j.log
set -uo pipefail
cd /mmfs1/gscratch/krishna/leoym/small-vlm
source .venv/bin/activate
type module &>/dev/null || source /usr/share/lmod/lmod/init/bash
module load cuda/12.9.1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HOME=/mmfs1/gscratch/krishna/leoym/hf_cache
export HF_DATASETS_CACHE=/gscratch/krishna/leoym/hf_cache/datasets
echo "[cka] node=$(hostname) kind=$KIND model=$MODEL tag=$TAG"
python -u devtools/cka_extract.py "$KIND" "$MODEL" "$TAG" "${N:-100}"
echo "[cka] ALL DONE"
