#!/bin/bash
# iEEG L_rec of checkpoints against trivial predictors on the config's iEEG blocks (scripts/ieeg_rec_baseline.py).
# CPU node, debug QOS (a whole node at once, 30 min max). Submit from the repo root:
#   sbatch scripts/slurm/ieeg_rec_baseline.sbatch [checkpoint ...]
#SBATCH --job-name=ieeg-rec-baseline
#SBATCH --account=m5187
#SBATCH --constraint=cpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=00:30:00
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.err
set -euo pipefail
cd /pscratch/sd/z/zres0710/iBrain_replication
mkdir -p logs
source .venv/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=64
export PYTHONUNBUFFERED=1
CKPTS=("$@")
[ ${#CKPTS[@]} -eq 0 ] && CKPTS=(runs/smoke_s0/final.pt)
python scripts/ieeg_rec_baseline.py --config configs/smoke.yaml --ckpt "${CKPTS[@]}"
