#!/bin/bash
# ImageNet linear-probe extraction for the ADDED models (native: NEO-2B/9B, Gemma-26B-MoE;
# encoder: InternVL-8B/30B-MoE, Janus). ~30/class (N=30000), val-split protocol.
set -uo pipefail
cd /mmfs1/gscratch/krishna/leoym/small-vlm
S=devtools/imagenet_extract.slurm
N=${1:-30000}
ip="devtools/imagenet_probe.py extract"
# encoder-side (main venv); big MoE shards with DEVMAP
sbatch --export=ALL,VENV=main,CMD="$ip internvl OpenGVLab/InternVL3_5-8B-HF ivl8 '' $N"                       $S
sbatch --export=ALL,VENV=main,DEVMAP=1,CMD="$ip internvl OpenGVLab/InternVL3_5-30B-A3B-HF ivl30moe '' $N"     $S
sbatch --export=ALL,VENV=main,CMD="$ip janus deepseek-community/Janus-Pro-7B janus '' $N"                     $S
# native-side: Gemma-26B-MoE (main venv, gemma4moe, DEVMAP) + NEO-2B/9B (neo venv)
sbatch --export=ALL,VENV=main,DEVMAP=1,CMD="$ip gemma4moe google/gemma-4-26B-A4B-it gemma26moe '' $N"         $S
sbatch --export=ALL,VENV=neo,CMD="neo_analysis/neo_imagenet_extract.py neo2b $N SFT"                          $S
sbatch --export=ALL,VENV=neo,CMD="neo_analysis/neo_imagenet_extract.py neo9b $N NEO1_0-9B-SFT"                $S
echo "submitted 6 ImageNet linear-probe extractions (N=$N)"
