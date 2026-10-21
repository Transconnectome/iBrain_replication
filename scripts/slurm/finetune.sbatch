#!/bin/bash
# Downstream units on Perlmutter, one GPU of a shared node per array task (U44). Submit from the repo root:
#   python scripts/downstream_plan.py --config configs/paper.yaml --ckpt <ckpt.pt> --root <root> --protocol probe > plan.txt
#   sbatch --array=1-$(wc -l < plan.txt) scripts/slurm/finetune.sbatch configs/paper.yaml plan.txt
# Line k of the plan holds the scripts/finetune.py arguments of unit k (--ckpt --out --dataset ... --arm --seed).
# A unit whose result.json exists is skipped, so the same plan can be resubmitted after a failure.
# Independent runs are faster than data-parallel training for these small fits; --array=1-N%8 limits concurrency.
# The Brain Treebank cache must exist before the array runs (scripts/finetune.py --prepare, finetune_prepare.sbatch):
# array tasks writing the same cache files at once would corrupt them.
#SBATCH --job-name=ibrain-ft
#SBATCH --account=m5187
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=./logs/%x_%A_%a.out
#SBATCH --error=./logs/%x_%A_%a.err
set -euo pipefail
PROJECT=${IBRAIN_CODE:-/pscratch/sd/z/zres0710/iBrain_replication}
VENV=${IBRAIN_VENV:-/pscratch/sd/z/zres0710/iBrain_replication/.venv}
cd "$PROJECT"
mkdir -p logs
CONFIG=${1:-configs/paper.yaml}
PLAN=${2:?plan file}
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$PLAN")
[ -n "$LINE" ] || { echo "no line $SLURM_ARRAY_TASK_ID in $PLAN"; exit 1; }
source "$VENV/bin/activate"
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
echo "job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID commit=$(git rev-parse --short HEAD)$(git diff --quiet || echo -dirty)"
echo "unit: $LINE"
# shellcheck disable=SC2086
srun python scripts/finetune.py --config "$CONFIG" $LINE
