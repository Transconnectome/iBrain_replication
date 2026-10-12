#!/bin/bash
# Where the time of one pretraining step goes (scripts/profile_step.py). Submit from the repo root:
#   sbatch scripts/slurm/profile_step.sbatch [config] [steps]   (steps > 8: DataLoader prefetch holds 4 workers x 2 batches)
#SBATCH --job-name=ibrain-profstep
#SBATCH --account=m5187
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.err
set -euo pipefail
cd /pscratch/sd/z/zres0710/iBrain_replication
mkdir -p logs
source .venv/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
python scripts/profile_step.py --config "${1:-configs/smoke.yaml}" --steps "${2:-8}"
