#!/bin/bash
# One-off (2026-10-20): MC-Maze scratch hyperparameter sweep and ridge check (sweep.py). One GPU of a shared node,
# 3 h 40 min at most = 0.92 node-hours.
#SBATCH --job-name=ibrain-ft-sweep
#SBATCH --account=m5187
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=03:40:00
#SBATCH --output=/pscratch/sd/z/zres0710/iBrain_replication/runs/tmp_ft_sweep/%x_%j.out
#SBATCH --error=/pscratch/sd/z/zres0710/iBrain_replication/runs/tmp_ft_sweep/%x_%j.err
set -euo pipefail
cd /pscratch/sd/z/zres0710/iBrain_replication
source .venv/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
echo "commit=$(git rev-parse --short HEAD)$(git diff --quiet || echo -dirty)"
srun python runs/tmp_ft_sweep/sweep.py
