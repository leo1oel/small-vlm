#!/bin/bash
# Overnight eval watcher: scans for SFT checkpoints and submits lmms-eval jobs
# for whichever of {vmcbench,mmvp,pope} each checkpoint is still missing.
# Marker files under logs/lmms_eval/.submitted/ prevent duplicate submission;
# runs CYCLES x INTERVAL then exits. Survives Claude/session restarts (nohup).
#
#   nohup bash devtools/auto_eval_watcher.sh > logs/auto_eval_watcher.log 2>&1 &

set -u
PROJECT_ROOT=/mmfs1/gscratch/krishna/leoym/small-vlm
SCRUBBED=/gscratch/scrubbed/leoym/small-vlm-outputs
KRISHNA_OUT=$PROJECT_ROOT/outputs
MARKERS=$PROJECT_ROOT/logs/lmms_eval/.submitted
mkdir -p "$MARKERS"
cd "$PROJECT_ROOT"

CYCLES=${CYCLES:-20}        # 20 x 30min = 10h
INTERVAL=${INTERVAL:-1800}
TASKS_ALL="vmcbench mmvp pope"
MIN_AGE_S=300               # skip checkpoints still being written

run_dirs() {
    ls -d "$SCRUBBED"/sft-*/ 2>/dev/null
    ls -d "$KRISHNA_OUT"/sft-unified-bee-mix-nepa/ 2>/dev/null
}

submit_for() { # $1=ckpt_dir $2=run_name $3=comma_tasks
    local extra=""
    [[ "$2" == sft-clip* ]] && extra=",image_aspect_ratio=square"  # B/16 and L/14 both square
    j=$(CKPT="$1" TASKS="$3" MODEL_ARGS_EXTRA="$extra" \
        sbatch --time=2:00:00 --exclude=g3108 --parsable devtools/lmms_eval.slurm 2>>logs/auto_eval_watcher.log)
    if [[ -n "$j" ]]; then
        echo "$(date '+%F %T') submitted $j: $2/$(basename "$1") tasks=$3"
        return 0
    fi
    echo "$(date '+%F %T') SUBMIT FAILED: $2/$(basename "$1") tasks=$3"
    return 1
}

for ((cycle = 1; cycle <= CYCLES; cycle++)); do
    echo "$(date '+%F %T') === cycle $cycle/$CYCLES"
    now=$(date +%s)
    for rd in $(run_dirs); do
        run=$(basename "$rd")
        for ck in "$rd"checkpoint-*; do
            [[ -d "$ck" ]] || continue
            [[ -f "$ck/config.json" && -f "$ck/model.safetensors" ]] || continue
            mt=$(stat -c %Y "$ck/model.safetensors" 2>/dev/null || echo 0)
            (( now - mt < MIN_AGE_S )) && continue
            ckname=$(basename "$ck")
            missing=""
            for t in $TASKS_ALL; do
                [[ -f "$MARKERS/${run}__${ckname}__${t}" ]] || missing="${missing:+$missing,}$t"
            done
            [[ -z "$missing" ]] && continue
            if submit_for "$ck" "$run" "$missing"; then
                for t in ${missing//,/ }; do touch "$MARKERS/${run}__${ckname}__${t}"; done
            fi
        done
    done
    (( cycle < CYCLES )) && sleep "$INTERVAL"
done
echo "$(date '+%F %T') watcher done"
