#!/bin/bash
# Scoped eval watcher for the 1000-step encoder-free catch-up goal (2026-06-22).
# Auto-submits lmms-eval on A40 (l40s is scarce) for the goal arms' checkpoints
# as they land: vmcbench (the goal metric, 1000 samples, fast) on every
# checkpoint; full vmcbench+mmvp+pope on the key @1000/@1500 checkpoints. Dedup
# markers under logs/lmms_eval/.goal_submitted/. nohup-safe, survives restarts.
#
#   nohup bash devtools/goal_eval_watcher.sh > logs/goal_eval_watcher.log 2>&1 &

set -u
ROOT=/mmfs1/gscratch/krishna/leoym/small-vlm
SCR=/gscratch/scrubbed/leoym/small-vlm-outputs
MARK=$ROOT/logs/lmms_eval/.goal_submitted
mkdir -p "$MARK"
cd "$ROOT"

ARMS="sft-clip-bee-mix \
sft-unified-bee-mix-warmstem \
sft-unified-bee-mix-warmstem-distill \
sft-unified-bee-mix-visualffn-1k \
sft-unified-bee-mix-warmstem-visualffn \
sft-unified-bee-mix-visualffn-distill \
sft-unified-bee-mix-prefix-distill \
sft-unified-bee-mix-warmstem-joint \
sft-unified-bee-mix-warmstem-mlp-distill"

CYCLES=${CYCLES:-60}      # 60 x 8min = 8h
INTERVAL=${INTERVAL:-480}
MIN_AGE_S=300             # skip checkpoints still being written

submit() { # $1=ckpt_dir  $2=run  $3=tasks  $4=marker
    local extra=""
    [[ "$2" == sft-clip* ]] && extra=",image_aspect_ratio=square"
    j=$(CKPT="$1" TASKS="$3" MODEL_ARGS_EXTRA="$extra" \
        sbatch -p ckpt-all -A krishna-ckpt --gpus=a40:1 --cpus-per-task=8 --mem=48G \
            --time=2:00:00 --parsable devtools/lmms_eval.slurm 2>>"$ROOT/logs/goal_eval_watcher.log")
    if [[ -n "$j" ]]; then
        : > "$MARK/$4"
        echo "$(date '+%F %T') submitted $j: $2/$(basename "$1") [$3]"
    else
        echo "$(date '+%F %T') SUBMIT FAILED: $2/$(basename "$1") [$3]"
    fi
}

for ((c = 1; c <= CYCLES; c++)); do
    echo "$(date '+%F %T') === cycle $c/$CYCLES"
    now=$(date +%s)
    for run in $ARMS; do
        for ck in "$SCR/$run"/checkpoint-*; do
            [[ -d "$ck" ]] || continue
            [[ -f "$ck/config.json" ]] || continue           # finished writing
            (( now - $(stat -c %Y "$ck") < MIN_AGE_S )) && continue
            step=$(basename "$ck" | sed 's/checkpoint-//')
            base="${run}__checkpoint-${step}"
            # vmcbench on every checkpoint (fast feedback)
            [[ -f "$MARK/${base}.vmc" ]] || submit "$ck" "$run" "vmcbench" "${base}.vmc"
            # full battery on the goal checkpoints
            if [[ "$step" == "1000" || "$step" == "1500" ]]; then
                [[ -f "$MARK/${base}.full" ]] || submit "$ck" "$run" "mmvp,pope" "${base}.full"
            fi
        done
    done
    sleep "$INTERVAL"
done
echo "$(date '+%F %T') watcher done"
