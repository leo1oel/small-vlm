#!/bin/bash
# Run an S1 cross-image discrimination probe on one checkpoint by srun'ing into
# an EXISTING GPU alloc you already hold (interactive / debugging). For a fresh
# batch job instead, use `sbatch devtools/s1_eval_probe.slurm` (no alloc needed).
#
# Usage (JOBID = your live alloc; `squeue --me` to find it):
#   JOBID=123456 CKPT=/path/checkpoint-1000 LABEL=eve@1000 OUT=/path/out.json \
#     PROBE=xshape GPU=0 bash devtools/native_distill_probe_srun.sh
#
#   PROBE=xshape -> breen_probe_xshape.py  (DISTILL arms 4,5,7,8,9)
#   PROBE=feat   -> breen_probe_feat.py    (NON-distill arms 1,2,3,6,10)
#   IMAGES_DIR=.../qual_images runs offline on local PNGs instead of VMCBench.
set -uo pipefail
# Derive the repo from this script's location so it works in any worktree (the
# old hard-coded fm-breen-port-h7 path is gone). Override with REPO=... if run
# from a copied script.
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
source .venv/bin/activate
export PYTHONPATH="$REPO/src:$REPO/devtools${PYTHONPATH:+:$PYTHONPATH}"
type module &>/dev/null || source /usr/share/lmod/lmod/init/bash
module load cuda/12.9.1 || true
export DS_SKIP_CUDA_CHECK=1 HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
  HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
JOBID="${JOBID:?set JOBID to a live alloc (squeue --me); or use sbatch devtools/s1_eval_probe.slurm}"
CKPT="${CKPT:?set CKPT to a checkpoint dir}"
PROBE="${PROBE:-xshape}"
case "$PROBE" in xshape|feat) ;; *) echo "PROBE must be xshape|feat" >&2; exit 2 ;; esac
img_arg=(); [ -n "${IMAGES_DIR:-}" ] && img_arg=(--images-dir "$IMAGES_DIR")
exec srun --jobid="$JOBID" --overlap --ntasks=1 -c 4 --mem=80G \
  env CUDA_VISIBLE_DEVICES="${GPU:-0}" \
  python -u "devtools/breen_probe_${PROBE}.py" \
    --ckpt "$CKPT" --n-images "${NIMG:-30}" "${img_arg[@]}" \
    ${LABEL:+--label "$LABEL"} ${OUT:+--out "$OUT"} \
    ${QUERY_PLACEMENT:+--query-placement "$QUERY_PLACEMENT"}
