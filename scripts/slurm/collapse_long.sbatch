#!/bin/bash
# Longer collapse check (2026-10-08): SimSiam head with BatchNorm over the whole step (task 0) or per group of 32
# windows as per GPU (task 1), 1000 steps. Each task trains through pretrain.sbatch, then runs check_collapse on the
# snapshots kept every 250 steps. Submit from the repo root:
#   sbatch --array=0-1 scripts/slurm/collapse_long.sbatch
#SBATCH --job-name=ibrain-collapse-long
#SBATCH --account=m5187
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=03:00:00
#SBATCH --output=./logs/%x_%A_%a.out
#SBATCH --error=./logs/%x_%A_%a.err
set -euo pipefail
cd /pscratch/sd/z/zres0710/iBrain_replication
NAMES=(long_bn_all long_bn_group32)
NAME=${NAMES[$SLURM_ARRAY_TASK_ID]}
OUT=runs/collapse/${NAME}_s0
echo "task=$SLURM_ARRAY_TASK_ID name=$NAME out=$OUT"
bash scripts/slurm/pretrain.sbatch "configs/collapse/${NAME}.yaml" 0 "$OUT"
source .venv/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONUNBUFFERED=1
python scripts/check_collapse.py --ckpt "$OUT"/ckpt_*.pt
