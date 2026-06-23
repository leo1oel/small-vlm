#!/bin/bash
# GPU smoke + overfit proof for visual-encoder distillation (spec 2026-06-21).
# Runs every method end-to-end (real CLIP/VAE teacher) and summarizes pass/fail.
# Submit:  sbatch devtools/distill_smoke.slurm
#SBATCH --job-name=distill-smoke
#SBATCH --partition=ckpt-all
#SBATCH --account=cse-ckpt
#SBATCH --nodes=1
#SBATCH --gpus=l40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:30:00
#SBATCH --requeue
#SBATCH --output=/mmfs1/gscratch/krishna/leoym/small-vlm/logs/distill_smoke_%j.out
#SBATCH --error=/mmfs1/gscratch/krishna/leoym/small-vlm/logs/distill_smoke_%j.err

set -uo pipefail
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

declare -A RESULT
FAILED=0
for M in repa eve vora softdepth relational vae; do
    echo; echo "################# method=$M #################"
    if "$VENV_PY" devtools/distill_smoke.py --method "$M"; then
        RESULT[$M]=PASS
    else
        RESULT[$M]=FAIL
        FAILED=1
    fi
done

echo; echo "================ SMOKE SUMMARY ================"
for M in repa eve vora softdepth relational vae; do
    echo "  $M: ${RESULT[$M]:-MISSING}"
done
exit $FAILED
