#!/bin/bash
#SBATCH --job-name=speechllm
#SBATCH --output=logs/slurm/sft_stage1_asr_%j.out
#SBATCH --error=logs/slurm/sft_stage1_asr_%j.err
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --nodelist=node20
#SBATCH --ntasks-per-node=8    
#SBATCH --time=480000:00:00
#SBATCH --exclusive

echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

source /nfs/asr/guokezhen/miniconda/etc/profile.d/conda.sh
conda activate speechllm

# 环境变量
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

ulimit -n 65536

config="conf/sft_stage1_asr.json"

srun python speechllm/bin/train_sft.py --train_type streaming_asr --config $config
