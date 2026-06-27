#!/bin/bash
# FA2 install verification (kernel + transformers generate + repo load_model).
# Submit: sbatch devtools/fa2_check.slurm
#SBATCH --job-name=fa2-check
#SBATCH --partition=ckpt-all
#SBATCH --account=cse-ckpt
#SBATCH --nodes=1
#SBATCH --gpus=l40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:15:00
#SBATCH --output=/mmfs1/gscratch/krishna/leoym/small-vlm/logs/fa2_check_%j.log

set -euo pipefail
cd /mmfs1/gscratch/krishna/leoym/small-vlm
source .venv/bin/activate
echo "node=$(hostname)"
HF_HUB_OFFLINE=1 python devtools/fa2_debug.py
HF_HUB_OFFLINE=1 python devtools/fa2_gpu_check.py
