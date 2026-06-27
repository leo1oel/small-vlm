#!/bin/bash
#SBATCH --job-name=patchnv
#SBATCH --partition=ckpt-all
#SBATCH --account=cse-ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/mmfs1/gscratch/krishna/leoym/small-vlm/logs/patchnv_%j.log
#SBATCH --error=/mmfs1/gscratch/krishna/leoym/small-vlm/logs/patchnv_%j.log
# Activation-patch runs for NEO / SAIL (NEO venv: transformers 4.57 + pyarrow + SDPA dense-mask).
set -uo pipefail
cd /mmfs1/gscratch/krishna/leoym/small-vlm
source /mmfs1/gscratch/krishna/leoym/envs/neo/bin/activate
type module &>/dev/null || source /usr/share/lmod/lmod/init/bash
module load cuda/12.9.1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HOME=/mmfs1/gscratch/krishna/leoym/hf_cache
export HF_DATASETS_CACHE=/gscratch/krishna/leoym/hf_cache/datasets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[patchnv] node=$(hostname) script=$SCRIPT out=$OUT n=${N:-1000} nc=${NC:-200} extra=${EXTRA:-}"
python -u "$SCRIPT" "$OUT" "${N:-1000}" "${NC:-200}" ${EXTRA:-}
echo "[patchnv] ALL DONE"
