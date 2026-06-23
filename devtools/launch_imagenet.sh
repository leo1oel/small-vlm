#!/bin/bash
# ImageNet linear-probe extraction. N = images per run (subset; 0/empty = full 50k).
# Encoders = ceiling, native = subject (incl. layer-0 floor), enc-VLM = control.
set -uo pipefail
cd /mmfs1/gscratch/krishna/leoym/small-vlm
S=devtools/imagenet_extract.slurm
N="${1:-25000}"   # 25/class default; pass 50000 for full
ip="devtools/imagenet_probe.py extract"

# encoders (ceiling) — cheap, single forward
sbatch --export=ALL,VENV=main,CMD="$ip dino facebook/dinov2-base dino '' $N"                       $S
sbatch --export=ALL,VENV=main,CMD="$ip clip openai/clip-vit-large-patch14-336 clip '' $N"          $S
sbatch --export=ALL,VENV=main,CMD="$ip siglip google/siglip-base-patch16-224 siglip '' $N"         $S
# encoder-VLM controls
sbatch --export=ALL,VENV=main,CMD="$ip llava llava-hf/llava-1.5-7b-hf llava '' $N"                  $S
sbatch --export=ALL,VENV=main,CMD="$ip qwen Qwen/Qwen2.5-VL-7B-Instruct qwen '' $N"                 $S
# native (enc-free) subjects — layer-0 in the frac grid = floor
sbatch --export=ALL,VENV=main,CMD="$ip gemma google/gemma-4-12B-it gemma '' $N"                     $S
sbatch --export=ALL,VENV=neo,CMD="sail_analysis/sail_imagenet_extract.py sail '' $N"                $S
sbatch --export=ALL,VENV=neo,CMD="mono_analysis/mono_imagenet_extract.py mono '' $N"                $S
echo "submitted ImageNet extraction (N=$N per run)"
