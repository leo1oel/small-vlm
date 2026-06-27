#!/bin/bash
#SBATCH --job-name=vlm-lmms-eval
#SBATCH --partition=ckpt-all
#SBATCH --account=cse-ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=/mmfs1/gscratch/krishna/leoym/small-vlm/logs/lmms_eval_%j.log
#SBATCH --error=/mmfs1/gscratch/krishna/leoym/small-vlm/logs/lmms_eval_%j.log

set -euo pipefail
PROJECT_ROOT=/mmfs1/gscratch/krishna/leoym/small-vlm
cd "${PROJECT_ROOT}"
source .venv/bin/activate

type module &>/dev/null || source /usr/share/lmod/lmod/init/bash
module load cuda/12.9.1
export DS_SKIP_CUDA_CHECK=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE=/gscratch/krishna/leoym/hf_cache/datasets
# lmms-eval silently redirects HF_DATASETS_CACHE to node-local scratch when it
# detects a remote (GPFS) filesystem ("file-lock safety"), which is empty on
# compute nodes and breaks offline runs. LMMS_EVAL_DATASETS_CACHE is its
# explicit, unconditional override.
export LMMS_EVAL_DATASETS_CACHE=/gscratch/krishna/leoym/hf_cache/datasets
export TQDM_DISABLE=0

CKPT="${CKPT:-outputs/sft-unified/checkpoint-3000}"
# lmms-eval changes CWD during evaluation; relative checkpoint paths must be
# absolutized or the processor loader mistakes them for hub repo ids.
[[ "${CKPT}" = /* ]] || CKPT="${PROJECT_ROOT}/${CKPT}"
TASKS="${TASKS:-mme}"
EXTRA="${EXTRA:-}"  # e.g. "--limit 8" for a smoke run
MODEL_ARGS="pretrained=${CKPT}${MODEL_ARGS_EXTRA:-}"  # e.g. MODEL_ARGS_EXTRA=",image_aspect_ratio=square"

echo "[lmms-eval] node=$(hostname) ckpt=${CKPT} tasks=${TASKS} extra=${EXTRA} model_args=${MODEL_ARGS}"

python -u devtools/run_lmms_eval.py \
    --model small-vlm \
    --model_args "${MODEL_ARGS}" \
    --tasks "${TASKS}" \
    --batch_size 1 \
    --log_samples \
    --output_path logs/lmms_eval \
    ${EXTRA}

echo "END TIME: $(date)"
