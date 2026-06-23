#!/bin/bash
#SBATCH --job-name=indl
#SBATCH --partition=ckpt-all
#SBATCH --account=cse-ckpt
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --output=/mmfs1/gscratch/krishna/leoym/small-vlm/logs/indl_%j.log
#SBATCH --error=/mmfs1/gscratch/krishna/leoym/small-vlm/logs/indl_%j.log
# Download mrm8488/ImageNet1K-val (39.3k val images, 5.2GB parquet) onto a compute node (login is offline).
set -uo pipefail
cd /mmfs1/gscratch/krishna/leoym/small-vlm
source .venv/bin/activate
export HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0
export HF_HOME=/mmfs1/gscratch/krishna/leoym/hf_cache
export HF_DATASETS_CACHE=/gscratch/krishna/leoym/hf_cache/datasets
echo "[indl] node=$(hostname) start"
python -u -c "
import datasets
ds = datasets.load_dataset('mrm8488/ImageNet1K-val', split='train')
print('[indl] rows', len(ds), 'cols', ds.column_names)
print('[indl] label feature', ds.features['label'])
ex = ds[0]; print('[indl] row0 image', ex['image'].size, 'label', ex['label'])
import collections
c = collections.Counter(ds['label'])
print('[indl] n_classes', len(c), 'min/max per class', min(c.values()), max(c.values()))
"
echo "[indl] ALL DONE"
