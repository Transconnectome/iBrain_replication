#!/bin/bash
# Representation collapse check of a checkpoint against its initialization. Submit from the repo root:
#   sbatch scripts/slurm/check_collapse.sbatch runs/smoke_s0/final.pt
#SBATCH --job-name=ibrain-collapse
#SBATCH --partition=debug
#SBATCH --nodelist=node1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=5G
#SBATCH --time=01:00:00
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.err
set -euo pipefail
cd /scratch/connectome/zres0710/iBrain_replication
mkdir -p logs
source .venv/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONUNBUFFERED=1
CKPT=${1:-runs/smoke_s0/final.pt}
python scripts/check_collapse.py --ckpt "$CKPT" "${@:2}"
