#!/bin/bash
# Extract per-layer image-token reps (predictivity input) for models that have fusion depth but
# no predictivity yet. Dense ones on one a40; big MoE (30B/26B) shard via DEVMAP=1. N=1000.
set -uo pipefail
cd /mmfs1/gscratch/krishna/leoym/small-vlm
S=devtools/cka.slurm
sbatch --export=ALL,KIND=internvl,MODEL=OpenGVLab/InternVL3_5-2B-HF,TAG=ivl2,N=1000   $S
sbatch --export=ALL,KIND=internvl,MODEL=OpenGVLab/InternVL3_5-4B-HF,TAG=ivl4,N=1000   $S
sbatch --export=ALL,KIND=internvl,MODEL=OpenGVLab/InternVL3_5-8B-HF,TAG=ivl8,N=1000   $S
sbatch --export=ALL,KIND=janus,MODEL=deepseek-community/Janus-Pro-7B,TAG=janus,N=1000 $S
# big MoE — device_map=auto shards across the a40 (+CPU offload if needed)
sbatch --export=ALL,KIND=internvl,MODEL=OpenGVLab/InternVL3_5-30B-A3B-HF,TAG=ivl30moe,N=1000,DEVMAP=1 $S
sbatch --export=ALL,KIND=gemma4moe,MODEL=google/gemma-4-26B-A4B-it,TAG=gemma26moe,N=1000,DEVMAP=1     $S
echo "submitted 6 predictivity extractions"
