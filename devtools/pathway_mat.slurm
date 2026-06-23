#!/bin/bash
#SBATCH --job-name=pathmat
#SBATCH --partition=ckpt-all
#SBATCH --account=cse-ckpt
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --output=/mmfs1/gscratch/krishna/leoym/small-vlm/logs/pathmat_%j.log
#SBATCH --error=/mmfs1/gscratch/krishna/leoym/small-vlm/logs/pathmat_%j.log
set -uo pipefail
cd /mmfs1/gscratch/krishna/leoym/small-vlm
source .venv/bin/activate
type module &>/dev/null || source /usr/share/lmod/lmod/init/bash
module load cuda/12.9.1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HOME=/mmfs1/gscratch/krishna/leoym/hf_cache
export HF_DATASETS_CACHE=/gscratch/krishna/leoym/hf_cache/datasets
echo "[pathmat] node=$(hostname) model=$MODEL kind=$KIND out=$OUT n=${N:-1000}"
python -u devtools/pathway_maturation.py "$MODEL" "$KIND" "$OUT" "${N:-1000}"
echo "[pathmat] ALL DONE"
