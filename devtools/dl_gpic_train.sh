#!/bin/bash
set -uo pipefail
cd /mmfs1/gscratch/krishna/leoym/small-vlm
source .venv/bin/activate
set -a; source .env; set +a
export HF_HUB_OFFLINE=1
python -u -c "from vlm.data.energon_dataset import download_and_index; p=download_and_index('gpic/train','train.jsonl', max_threads=16); print('DOWNLOADED+INDEXED:', p)"
echo "DL_DONE $(date)"
