#!/bin/bash
# Budget-independent BREEN-S0 launcher: wait for the streamed caption data, then
# submit the S0 dry-run + the full S0 run gated on it. Runs as a cheap CPU job
# so it survives any interactive-session/budget limit (the firstmate's
# "training RUNNING, budget-independent" deliverable).
#
# Gating is SLURM-native, not log-parsing: the full run is submitted with
# --dependency=afterok:<dryrun>, so it ONLY starts if the 60-step dry-run on the
# REAL streamed data exits 0 (ran without crashing + wrote a checkpoint). Code
# correctness is already covered by gate parts 1 (component re-check) and 2
# (Phase-0 smoke GREEN); the dry-run is the data-specific safety net (malformed
# jsonl, schema mismatch, NaN-on-real-images, OOM).
#
# Submit once (from the worktree):  sbatch devtools/breen_poll.slurm
#
#SBATCH --job-name=breen-poll
#SBATCH --partition=ckpt-all
#SBATCH --account=cse-ckpt
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=/mmfs1/gscratch/krishna/leoym/herdr-worktrees/small-vlm/fm-breen-port-h7/logs/breen_poll_%j.log
#SBATCH --error=/mmfs1/gscratch/krishna/leoym/herdr-worktrees/small-vlm/fm-breen-port-h7/logs/breen_poll_%j.log

set -uo pipefail
ROOT=/mmfs1/gscratch/krishna/leoym/herdr-worktrees/small-vlm/fm-breen-port-h7
cd "${ROOT}"
mkdir -p logs

SENTINEL="${ROOT}/logs/breen_s0_launched"
STATUS="${ROOT}/logs/breen_orchestrate_status.txt"
URL="msc://azure/data/yiming/bee_stage2/train/train.jsonl"
OUT_BASE="/gscratch/scrubbed/leoym/breen-outputs"
DRY_DIR="${OUT_BASE}/dryrun-s0"
S0_DIR="${OUT_BASE}/pretrain-unified-breen"
PROBE_TIMEOUT=120
POLL_SLEEP=300
MAX_WAIT_SECS=$((12 * 3600))

log() { echo "[$(date -u '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${STATUS}"; }

# Idempotent: a requeued poll job (ckpt-all preemption) must never double-launch.
if [[ -f "${SENTINEL}" ]]; then
    log "sentinel present ($(cat "${SENTINEL}")) — S0 already launched; nothing to do."
    exit 0
fi

# Probe env (Azure creds + worktree code path); no CUDA needed.
source .venv/bin/activate
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -z "${AZURE_SAS_TOKEN:-}" && -f /mmfs1/gscratch/krishna/leoym/small-vlm/.env ]]; then
    set -a; source /mmfs1/gscratch/krishna/leoym/small-vlm/.env; set +a
fi
export MSC_CACHE_DIR="/scr/${USER}/msc-probe-cache"; mkdir -p "${MSC_CACHE_DIR}"

log "poll start (job ${SLURM_JOB_ID:-?}); waiting for ${URL}"
waited=0
while true; do
    out=$(timeout "${PROBE_TIMEOUT}" python devtools/breen_probe_data.py "${URL}" 2>&1)
    rc=$?
    if [[ ${rc} -eq 0 ]]; then
        log "DATA READY: ${out}"
        break
    fi
    if (( waited >= MAX_WAIT_SECS )); then
        log "TIMEOUT: data not present after ~$((waited / 3600))h. ESCALATE (data never arrived / malformed)."
        exit 3
    fi
    log "not ready (rc=${rc}); sleeping ${POLL_SLEEP}s (waited ~$((waited / 60))m)"
    sleep "${POLL_SLEEP}"
    waited=$((waited + POLL_SLEEP + PROBE_TIMEOUT))
done

# Fresh throwaway dir for the gate (never resume a stale dry-run).
rm -rf "${DRY_DIR}"

# 1) S0 dry-run on the REAL stream: 60 steps, a checkpoint at 40 (verifies save).
log "submitting S0 dry-run (60 steps -> ${DRY_DIR})"
DRY_JID=$(
    CONFIG_NAME=pretrain-unified-breen \
    RUN_DIR="${DRY_DIR}" \
    HYDRA_EXTRA="trainer.max_steps=60 trainer.save_steps=40 trainer.save_total_limit=2 trainer.logging_steps=1 trainer.warmup_steps=10" \
    sbatch --parsable devtools/breen_train.slurm
)
if [[ -z "${DRY_JID}" ]]; then
    log "FAILED to submit dry-run; aborting (inspect)."
    exit 4
fi
log "dry-run job=${DRY_JID}"

# 2) Full S0, gated: starts ONLY if the dry-run exits 0 (afterok). If the
#    dry-run fails, SLURM cancels this automatically -> no GPU-days burned.
log "submitting full S0 (afterok:${DRY_JID} -> ${S0_DIR})"
FULL_JID=$(
    CONFIG_NAME=pretrain-unified-breen \
    RUN_DIR="${S0_DIR}" \
    sbatch --parsable --dependency="afterok:${DRY_JID}" devtools/breen_train.slurm
)
if [[ -z "${FULL_JID}" ]]; then
    log "FAILED to submit full S0; dry-run ${DRY_JID} still running. Inspect + launch manually."
    exit 4
fi

echo "dry-run=${DRY_JID} full-s0=${FULL_JID} submitted $(date -u '+%Y-%m-%dT%H:%M:%SZ')" > "${SENTINEL}"
log "S0 LAUNCHED: dry-run=${DRY_JID}, full-S0=${FULL_JID} (gated afterok)."
log "Next (manual, after S0 finishes): S1 from S0 ckpt, then S2 from S1 ckpt — see logs/breen_orchestrate_status.txt and the stage configs."
exit 0
