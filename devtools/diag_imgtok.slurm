#!/bin/bash
#SBATCH --job-name=diag
#SBATCH --partition=ckpt-all
#SBATCH --account=cse-ckpt
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1:00:00
#SBATCH --requeue
#SBATCH --output=/mmfs1/gscratch/krishna/leoym/small-vlm/logs/diag_%j.log
#SBATCH --error=/mmfs1/gscratch/krishna/leoym/small-vlm/logs/diag_%j.log
set -uo pipefail
cd /mmfs1/gscratch/krishna/leoym/small-vlm
source .venv/bin/activate
type module &>/dev/null || source /usr/share/lmod/lmod/init/bash
module load cuda/12.9.1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HOME=/mmfs1/gscratch/krishna/leoym/hf_cache
echo "[diag] $KIND $MID"
python -u devtools/diag_imgtok.py "$KIND" "$MID"
echo "[diag] ALL DONE"
