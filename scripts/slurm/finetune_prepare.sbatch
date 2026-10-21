#!/bin/bash
# Reads every downstream dataset once (U44): builds the Brain Treebank cache (U34 (g)), prints the NWB window counts.
# No training. One GPU of a shared node rather than a CPU node, because the CPU allocation is nearly used up.
#   sbatch scripts/slurm/finetune_prepare.sbatch configs/paper.yaml [--dataset treebank ...]
#SBATCH --job-name=ibrain-ft-prepare
#SBATCH --account=m5187
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=06:00:00
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.err
set -euo pipefail
PROJECT=${IBRAIN_CODE:-/pscratch/sd/z/zres0710/iBrain_replication}
VENV=${IBRAIN_VENV:-/pscratch/sd/z/zres0710/iBrain_replication/.venv}
cd "$PROJECT"
mkdir -p logs
CONFIG=${1:-configs/paper.yaml}
EXTRA=("${@:2}")
source "$VENV/bin/activate"
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
echo "job=$SLURM_JOB_ID commit=$(git rev-parse --short HEAD)$(git diff --quiet || echo -dirty)"
srun python scripts/finetune.py --config "$CONFIG" --out /dev/null --prepare "${EXTRA[@]}"
