#!/bin/bash
# WorldBench fusion-depth verification (sufmeanabl read-onset) + by-domain stratification.
# Strong native-res anchors only (WorldBench is a frontier benchmark; LLaVA-1.5 gets R0<0 = no signal).
# Qwen at 896px (1024 tok) so dense-OCR domains are legible; sufmeanabl is self-aligned so the larger
# token count is fine. N=2000 (all), NC=420 (~60/domain).
set -uo pipefail
cd /mmfs1/gscratch/krishna/leoym/small-vlm
S=devtools/activation_patch.slurm
N=${1:-2000}; NC=${2:-420}
R=/mmfs1/gscratch/krishna/leoym/small-vlm/neo_analysis

# encoder-MoE family (dense 8B), native res
sbatch --export=ALL,MODEL=OpenGVLab/InternVL3_5-8B-HF,KIND=internvl,OUT=$R/results_wb_ivl8.jsonl,N=$N,NC=$NC,MODE=sufmeanabl,BENCH=worldbench,HFOFF=1 $S
# encoder-free unified
sbatch --export=ALL,MODEL=google/gemma-4-12B-it,KIND=gemma,OUT=$R/results_wb_gemma12.jsonl,N=$N,NC=$NC,MODE=sufmeanabl,BENCH=worldbench,HFOFF=1 $S
# standard encoder-VLM, high-res for dense domains
sbatch --export=ALL,QWEN_SIDE=896,MODEL=Qwen/Qwen2.5-VL-7B-Instruct,KIND=qwen,OUT=$R/results_wb_qwen.jsonl,N=$N,NC=$NC,MODE=sufmeanabl,BENCH=worldbench,HFOFF=1 $S
echo "submitted WorldBench full runs (N=$N NC=$NC): ivl8 gemma12 qwen@896"
