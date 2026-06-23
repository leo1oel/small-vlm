#!/bin/bash
#SBATCH --job-name=patch
#SBATCH --partition=ckpt-all
#SBATCH --account=cse-ckpt
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=6:00:00
#SBATCH --requeue
#SBATCH --output=/mmfs1/gscratch/krishna/leoym/small-vlm/logs/patch_%j.log
#SBATCH --error=/mmfs1/gscratch/krishna/leoym/small-vlm/logs/patch_%j.log
set -uo pipefail
cd /mmfs1/gscratch/krishna/leoym/small-vlm
source .venv/bin/activate
type module &>/dev/null || source /usr/share/lmod/lmod/init/bash
module load cuda/12.9.1
export HF_HUB_OFFLINE=${HFOFF:-1} HF_DATASETS_OFFLINE=${HFOFF:-1} HF_HOME=/mmfs1/gscratch/krishna/leoym/hf_cache
export HF_DATASETS_CACHE=/gscratch/krishna/leoym/hf_cache/datasets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[patch] node=$(hostname) model=$MODEL kind=$KIND out=$OUT n=${N:-1000} nc=${NC:-200} mode=${MODE:-both}"
python -u devtools/activation_patch.py "$MODEL" "$KIND" "$OUT" "${N:-1000}" "${NC:-200}" "${MODE:-sufmeanabl}" "${BENCH:-vmcbench}"
echo "[patch] ALL DONE"
