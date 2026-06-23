#!/bin/bash
#SBATCH --job-name=encattn
#SBATCH --partition=ckpt-all
#SBATCH --account=cse-ckpt
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --output=/mmfs1/gscratch/krishna/leoym/small-vlm/logs/attn_%j.log
#SBATCH --error=/mmfs1/gscratch/krishna/leoym/small-vlm/logs/attn_%j.log
set -uo pipefail
cd /mmfs1/gscratch/krishna/leoym/small-vlm
source .venv/bin/activate
type module &>/dev/null || source /usr/share/lmod/lmod/init/bash
module load cuda/12.9.1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HOME=/mmfs1/gscratch/krishna/leoym/hf_cache
export HF_DATASETS_CACHE=/gscratch/krishna/leoym/hf_cache/datasets
echo "[attn] node=$(hostname) model=$MODEL kind=$KIND out=$OUT n=${N:-200}"
python -u devtools/encoder_vlm_attn.py "$MODEL" "$KIND" "$OUT" "${N:-200}"
echo "[attn] ALL DONE"
