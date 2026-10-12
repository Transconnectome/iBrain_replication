#!/bin/bash
# Units per row/window of the Neural Pile (CPU only, node2/node4). Submit from the repo root:
#   sbatch scripts/slurm/pile_units.sbatch
#SBATCH --job-name=pile-units
#SBATCH --partition=debug
#SBATCH --exclude=node1,node3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=01:00:00
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.err
set -euo pipefail
cd /scratch/connectome/zres0710/iBrain_replication
mkdir -p logs
# .venv only runs on node1 (uv Python under the node-local /home); this env lives on /scratch
PY=/scratch/connectome/zres0710/envs/ibrain-cpu/bin/python
T0=$(date +%s)
$PY scripts/pile_units.py --root /storage/connectome/zres0710/iBrain_pretrain/spikes/neural-pile-primate \
    --out runs/pile_units.npz --workers 8
awk -v s=$(( $(date +%s) - T0 )) 'BEGIN { printf "elapsed %.1f min\n", s / 60 }'
