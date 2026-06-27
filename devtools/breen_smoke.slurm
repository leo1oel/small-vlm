#!/bin/bash
# Phase-0 port-correctness smoke for the BREEN port (arXiv:2503.12446, spec
# 2026-06-24). Runs the four acceptance checks (a)-(d) end-to-end on a GPU with
# the real frozen CLIP-L/14-336 teacher. Submit from the worktree:
#   sbatch devtools/breen_smoke.slurm
#SBATCH --job-name=breen-smoke
#SBATCH --partition=ckpt-all
#SBATCH --account=cse-ckpt
#SBATCH --nodes=1
#SBATCH --gpus=l40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --requeue
#SBATCH --output=/mmfs1/gscratch/krishna/leoym/herdr-worktrees/small-vlm/fm-breen-port-h7/logs/breen_smoke_%j.out
#SBATCH --error=/mmfs1/gscratch/krishna/leoym/herdr-worktrees/small-vlm/fm-breen-port-h7/logs/breen_smoke_%j.err

set -uo pipefail
# Run from THIS worktree so the smoke exercises the worktree's edited code (the
# .venv is symlinked to the main checkout; PYTHONPATH points at the worktree src).
REPO=/mmfs1/gscratch/krishna/leoym/herdr-worktrees/small-vlm/fm-breen-port-h7
VENV_PY="$REPO/.venv/bin/python"
cd "$REPO"
mkdir -p logs
type module &>/dev/null || source /usr/share/lmod/lmod/init/bash
module load cuda/12.9.1 || true
export DS_SKIP_CUDA_CHECK=1
export PYTHONPATH="$REPO/src:$REPO/devtools"
export HF_HUB_OFFLINE=1
echo "node=$(hostname) branch=$(git rev-parse --abbrev-ref HEAD) head=$(git rev-parse --short HEAD)"
"$VENV_PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.get_device_name(0))"
"$VENV_PY" devtools/breen_smoke.py --steps 150
