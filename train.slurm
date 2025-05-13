#!/bin/bash
#SBATCH --job-name=vlm-training
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --time=504:00:00
#SBATCH --partition=pasteur
#SBATCH --account=pasteur
#SBATCH --exclude=pasteur-hgx-1,pasteur-hgx-2,pasteur1,pasteur2,pasteur5,pasteur6,pasteur7,pasteur9
#SBATCH --output=%x-%j.out

cd /pasteur/u/yiming/small-vlm
source .venv/bin/activate

export GPUS_PER_NODE=8
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9901

echo "START TIME: $(date)"
echo "SLURM JOB ID: $SLURM_JOBID"
echo "NODELIST: $SLURM_JOB_NODELIST"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "GPUS_PER_NODE: $GPUS_PER_NODE"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_NTASKS_PER_NODE: $SLURM_NTASKS_PER_NODE" # 应为 1

srun --jobid $SLURM_JOBID bash -c ' \
    echo "Host: $(hostname), SLURM_PROCID: $SLURM_PROCID, CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"; \
    python -u -m torch.distributed.run \
        --nproc_per_node $GPUS_PER_NODE \
        --nnodes $SLURM_NNODES \
        --node_rank $SLURM_PROCID \
        --master_addr $MASTER_ADDR \
        --master_port $MASTER_PORT \
        -m vlm -cn finetune-llava \
    '
echo "END TIME: $(date)"