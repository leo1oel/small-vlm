#!/bin/bash
#SBATCH --job-name=neo-debug
#SBATCH --partition=ckpt-all
#SBATCH --account=cse-ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0:25:00
#SBATCH --output=/mmfs1/gscratch/krishna/leoym/small-vlm/logs/neo_debug_%j.log
#SBATCH --error=/mmfs1/gscratch/krishna/leoym/small-vlm/logs/neo_debug_%j.log
set -uo pipefail
cd /mmfs1/gscratch/krishna/leoym/small-vlm
source /mmfs1/gscratch/krishna/leoym/envs/neo/bin/activate
type module &>/dev/null || source /usr/share/lmod/lmod/init/bash
module load cuda/12.9.1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HOME=/mmfs1/gscratch/krishna/leoym/hf_cache
export HF_DATASETS_CACHE=/gscratch/krishna/leoym/hf_cache/datasets
python -u neo_analysis/_debug_neo.py
