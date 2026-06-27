#!/bin/bash
#SBATCH --job-name=inx
#SBATCH --partition=ckpt-all
#SBATCH --account=cse-ckpt
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=6:00:00
#SBATCH --requeue
#SBATCH --output=/mmfs1/gscratch/krishna/leoym/small-vlm/logs/inx_%j.log
#SBATCH --error=/mmfs1/gscratch/krishna/leoym/small-vlm/logs/inx_%j.log
# ImageNet-1k val per-layer image-token extraction. VENV=main (encoders/LLaVA/Qwen/Gemma)
# or VENV=neo (SAIL/Mono native dirs). CMD = full python invocation.
set -uo pipefail
cd /mmfs1/gscratch/krishna/leoym/small-vlm
if [ "${VENV:-main}" = "neo" ]; then
  source /mmfs1/gscratch/krishna/leoym/envs/neo/bin/activate
else
  source .venv/bin/activate
fi
type module &>/dev/null || source /usr/share/lmod/lmod/init/bash
module load cuda/12.9.1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HOME=/mmfs1/gscratch/krishna/leoym/hf_cache
export HF_DATASETS_CACHE=/gscratch/krishna/leoym/hf_cache/datasets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[inx] node=$(hostname) venv=${VENV:-main} cmd=$CMD"
eval "python -u $CMD"
echo "[inx] ALL DONE"
