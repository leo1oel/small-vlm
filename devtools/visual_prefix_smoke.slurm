#!/bin/bash
# GPU smoke for the visual-prefix arm (spec 2026-06-14). Submit:
#   sbatch devtools/visual_prefix_smoke.slurm
#SBATCH --job-name=vpfx-smoke
#SBATCH --partition=ckpt-all
#SBATCH --account=cse-ckpt
#SBATCH --nodes=1
#SBATCH --gpus=l40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0:30:00
#SBATCH --requeue
#SBATCH --output=/mmfs1/gscratch/krishna/leoym/small-vlm/logs/vpfx_smoke_%j.out
#SBATCH --error=/mmfs1/gscratch/krishna/leoym/small-vlm/logs/vpfx_smoke_%j.err

set -euo pipefail
REPO=/mmfs1/gscratch/krishna/leoym/small-vlm
VENV_PY="$REPO/.venv/bin/python"
cd "$REPO"
mkdir -p logs
type module &>/dev/null || source /usr/share/lmod/lmod/init/bash
module load cuda/12.9.1 || true
export DS_SKIP_CUDA_CHECK=1
export PYTHONPATH="$REPO/src:$REPO/devtools"
export HF_HUB_OFFLINE=1
echo "node=$(hostname) branch=$(git rev-parse --abbrev-ref HEAD) head=$(git rev-parse --short HEAD)"
"$VENV_PY" devtools/visual_prefix_smoke.py
