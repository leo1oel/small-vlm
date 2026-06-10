#!/bin/bash
# GPU smoke + equivalence verification for the cross-modal access arms
# (plan 2026-06-10-early-fusion-access-arms). Runs the BRANCH code in the
# worktree against the main checkout's venv. Submit from the worktree:
#   sbatch devtools/xmodal_smoke.slurm
#SBATCH --job-name=xmodal-smoke
#SBATCH --partition=ckpt-all
#SBATCH --account=cse-ckpt
#SBATCH --nodes=1
#SBATCH --gpus=l40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0:30:00
#SBATCH --requeue
#SBATCH --output=/mmfs1/gscratch/krishna/leoym/small-vlm/.claude/worktrees/early-fusion-access-arms/logs/xmodal_smoke_%j.out
#SBATCH --error=/mmfs1/gscratch/krishna/leoym/small-vlm/.claude/worktrees/early-fusion-access-arms/logs/xmodal_smoke_%j.err

set -euo pipefail
WORKTREE=/mmfs1/gscratch/krishna/leoym/small-vlm/.claude/worktrees/early-fusion-access-arms
VENV_PY=/mmfs1/gscratch/krishna/leoym/small-vlm/.venv/bin/python
cd "$WORKTREE"
mkdir -p logs
export PYTHONPATH="$WORKTREE/src"
export HF_HUB_OFFLINE=1
echo "node=$(hostname) worktree=$WORKTREE"
echo "git_head=$(git rev-parse HEAD)"
"$VENV_PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.get_device_name(0))"
"$VENV_PY" devtools/xmodal_smoke.py
