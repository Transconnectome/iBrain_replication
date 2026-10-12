#!/bin/bash
# Units per row/window of the Neural Pile (CPU node, -C cpu, debug QOS: a whole node at once, 30 min max; CPU shared queued ~1 week on 2026-10-11). Submit from the repo root:
#   sbatch scripts/slurm/pile_units.sbatch
#SBATCH --job-name=pile-units
#SBATCH --account=m5187
#SBATCH --constraint=cpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.err
set -euo pipefail
cd /pscratch/sd/z/zres0710/iBrain_replication
mkdir -p logs
# NERSC: .venv uses uv's Python under /global/homes, visible from every node
PY=.venv/bin/python
T0=$(date +%s)
$PY scripts/pile_units.py --root /pscratch/sd/z/zres0710/iBrain/pretrain/spikes/neural-pile-primate \
    --out runs/pile_units.npz --workers 8
awk -v s=$(( $(date +%s) - T0 )) 'BEGIN { printf "elapsed %.1f min\n", s / 60 }'
