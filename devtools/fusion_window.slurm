#!/bin/bash
#SBATCH --job-name=win
#SBATCH --partition=ckpt-all
#SBATCH --account=cse-ckpt
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=6:00:00
#SBATCH --requeue
#SBATCH --output=/mmfs1/gscratch/krishna/leoym/small-vlm/logs/win_%j.log
#SBATCH --error=/mmfs1/gscratch/krishna/leoym/small-vlm/logs/win_%j.log
set -uo pipefail
cd /mmfs1/gscratch/krishna/leoym/small-vlm
source .venv/bin/activate
type module &>/dev/null || source /usr/share/lmod/lmod/init/bash
module load cuda/12.9.1
export HF_HUB_OFFLINE=${HFOFF:-1} HF_DATASETS_OFFLINE=${HFOFF:-1} HF_HOME=/mmfs1/gscratch/krishna/leoym/hf_cache
export HF_DATASETS_CACHE=/gscratch/krishna/leoym/hf_cache/datasets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[win] node=$(hostname) model=$MODEL kind=$KIND out=$OUT n=${N:-1000} nc=${NC:-200} sweeps=${SWEEPS:-both}"
python -u devtools/fusion_window.py "$MODEL" "$KIND" "$OUT" "${N:-1000}" "${NC:-200}" "${SWEEPS:-both}"
echo "[win] ALL DONE"
