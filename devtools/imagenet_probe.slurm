#!/bin/bash
#SBATCH --job-name=inprobe
#SBATCH --partition=ckpt-all
#SBATCH --account=cse-ckpt
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --output=/mmfs1/gscratch/krishna/leoym/small-vlm/logs/inprobe_%j.log
#SBATCH --error=/mmfs1/gscratch/krishna/leoym/small-vlm/logs/inprobe_%j.log
# ImageNet linear-probe (logistic head, fixed split) over extracted reps. CPU-only.
set -uo pipefail
cd /mmfs1/gscratch/krishna/leoym/small-vlm
source .venv/bin/activate
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
echo "[inprobe] node=$(hostname) tags=${TAGS}"
python -u devtools/imagenet_probe.py probe ${TAGS}
echo "[inprobe] ALL DONE"
