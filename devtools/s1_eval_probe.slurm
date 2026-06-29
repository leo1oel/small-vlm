#!/bin/bash
# Self-contained S1 representation-eval probe launcher for the 10-experiment
# ablation (spec data/tenexp-audit/report.md §7). Submits ONE probe on ONE
# checkpoint as a fresh GPU job — no held alloc / dead --jobid to re-point.
#
# Submit (PROBE selects which probe; CKPT is the checkpoint dir):
#   CKPT=/path/checkpoint-1000 PROBE=xshape LABEL=eve@1000 OUT=/path/out.json \
#     sbatch devtools/s1_eval_probe.slurm
#
# PROBE values and which ablation arms they apply to:
#   xshape  -> breen_probe_xshape.py   cross-image discrimination on the distill
#              head (pred vs teacher). DISTILL arms only: 4,5,7,8.
#   feat    -> breen_probe_feat.py     distill-free split-half discrimination on
#              the raw LLM hidden. NON-distill arms: 1,2,3,6,10 (runs on any arm).
#   caption -> breen_caption_test.py   grounded-vs-blind caption strings. ANY arm.
#   recipe  -> breen_caption_recipe_sweep.py  decode-recipe sweep. ANY arm.
#   retest  -> breen_caption_retest.py multi-arm caption read; pass ARMS="L=ckpt ...".
#
# Image set: defaults to VMCBench-dev (NIMG images) for xshape/feat; set
#   IMAGES_DIR=.../breen-s2val-j8/qual_images to run offline on local PNGs
#   (and for an identical cross-arm image set). VMCBench must be HF-cached when
#   HF_HUB_OFFLINE=1 (the probe errors with the pre-cache command otherwise).
#SBATCH --job-name=s1-eval-probe
#SBATCH --partition=ckpt-all
#SBATCH --account=cse-ckpt
#SBATCH --nodes=1
#SBATCH --gpus=l40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=1:00:00
#SBATCH --requeue
#SBATCH --output=/mmfs1/gscratch/krishna/leoym/small-vlm/logs/s1_eval_probe_%j.out
#SBATCH --error=/mmfs1/gscratch/krishna/leoym/small-vlm/logs/s1_eval_probe_%j.err

set -uo pipefail
# Derive the repo from this submitted script so worktree launches test that tree.
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_PY="$REPO/.venv/bin/python"
cd "$REPO"
mkdir -p logs
type module &>/dev/null || source /usr/share/lmod/lmod/init/bash
module load cuda/12.9.1 || true
export DS_SKIP_CUDA_CHECK=1
export PYTHONPATH="$REPO/src:$REPO/devtools"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "node=$(hostname) branch=$(git rev-parse --abbrev-ref HEAD) head=$(git rev-parse --short HEAD)"

PROBE="${PROBE:-xshape}"
NIMG="${NIMG:-30}"
img_arg=(); [ -n "${IMAGES_DIR:-}" ] && img_arg=(--images-dir "$IMAGES_DIR")

case "$PROBE" in
  xshape|feat)
    CKPT="${CKPT:?set CKPT to a checkpoint dir}"
    exec "$VENV_PY" -u "devtools/breen_probe_${PROBE}.py" \
      --ckpt "$CKPT" --n-images "$NIMG" "${img_arg[@]}" \
      ${LABEL:+--label "$LABEL"} ${OUT:+--out "$OUT"} \
      ${QUERY_PLACEMENT:+--query-placement "$QUERY_PLACEMENT"}
    ;;
  caption)
    CKPT="${CKPT:?set CKPT to a checkpoint dir}"
    exec "$VENV_PY" -u devtools/breen_caption_test.py --ckpt "$CKPT" --n "$NIMG"
    ;;
  recipe)
    CKPT="${CKPT:?set CKPT to a checkpoint dir}"
    exec "$VENV_PY" -u devtools/breen_caption_recipe_sweep.py --ckpt "$CKPT"
    ;;
  retest)
    : "${ARMS:?set ARMS to a space-separated list of LABEL=ckpt}"
    # shellcheck disable=SC2086 — ARMS is intentionally word-split into --arms
    exec "$VENV_PY" -u devtools/breen_caption_retest.py --arms $ARMS
    ;;
  *)
    echo "unknown PROBE=$PROBE (want xshape|feat|caption|recipe|retest)" >&2
    exit 2
    ;;
esac
