#!/bin/bash
# Re-extract per-layer image-token reps at N=1000 (full VMCBench dev) for predictivity + CKA.
# Main venv (cka.slurm): encoder refs + encoder-VLM controls + Gemma. Neo venv (cka_nv.slurm): NEO/SAIL.
set -uo pipefail
cd /mmfs1/gscratch/krishna/leoym/small-vlm
SUB=/mmfs1/gscratch/krishna/leoym/small-vlm/devtools

# kind  model_id  tag   (main venv)
sbatch --export=ALL,KIND=dino,MODEL=facebook/dinov2-base,TAG=dino,N=1000                       $SUB/cka.slurm
sbatch --export=ALL,KIND=clip,MODEL=openai/clip-vit-large-patch14-336,TAG=clip,N=1000          $SUB/cka.slurm
sbatch --export=ALL,KIND=siglip,MODEL=google/siglip-base-patch16-224,TAG=siglip,N=1000         $SUB/cka.slurm
sbatch --export=ALL,KIND=llava,MODEL=llava-hf/llava-1.5-7b-hf,TAG=llava,N=1000                  $SUB/cka.slurm
sbatch --export=ALL,KIND=qwen,MODEL=Qwen/Qwen2.5-VL-7B-Instruct,TAG=qwen,N=1000                 $SUB/cka.slurm
sbatch --export=ALL,KIND=gemma,MODEL=google/gemma-4-12B-it,TAG=gemma,N=1000                     $SUB/cka.slurm

# NEO / SAIL (neo venv)
sbatch --export=ALL,SCRIPT=neo_analysis/neo_cka_extract.py,TAG=neo2bsft,N=1000,STAGE=SFT          $SUB/cka_nv.slurm
sbatch --export=ALL,SCRIPT=neo_analysis/neo_cka_extract.py,TAG=neo9bsft,N=1000,STAGE=NEO1_0-9B-SFT $SUB/cka_nv.slurm
sbatch --export=ALL,SCRIPT=sail_analysis/sail_cka_extract.py,TAG=sail,N=1000                       $SUB/cka_nv.slurm
echo "submitted 9 extraction jobs at N=1000"
