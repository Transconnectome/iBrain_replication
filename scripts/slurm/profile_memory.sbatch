#!/bin/bash
# Peak GPU memory per micro-batch shape, for choosing pretrain.token_budget (U33). Submit from the repo root:
#   sbatch scripts/slurm/profile_memory.sbatch [config] [precision]
#SBATCH --job-name=ibrain-profile
#SBATCH --partition=debug
#SBATCH --nodelist=node1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=01:00:00
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.err
set -euo pipefail
cd /scratch/connectome/zres0710/iBrain_replication
mkdir -p logs
source .venv/bin/activate
export PYTHONUNBUFFERED=1
CONFIG=${1:-configs/paper.yaml}
PRECISION=${2:-bf16}
nvidia-smi -i "${CUDA_VISIBLE_DEVICES:-0}" --query-gpu=name,memory.total,driver_version --format=csv,noheader
# Without and with activation checkpointing (U37)
python scripts/profile_memory.py --config "$CONFIG" --precision "$PRECISION" --out "runs/profile_memory_${PRECISION}.jsonl"
python scripts/profile_memory.py --config "$CONFIG" --precision "$PRECISION" --checkpoint \
    --out "runs/profile_memory_${PRECISION}_ckpt.jsonl"
