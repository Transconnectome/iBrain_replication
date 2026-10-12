#!/bin/bash
# One-time Neural Pile conversion for data.spike.format pile_mmap (U30). CPU node (-C cpu, debug QOS: a whole node at once, 30 min max; CPU shared queued ~1 week on 2026-10-11). Submit from the repo root:
#   sbatch scripts/slurm/pile_to_memmap.sbatch [split]
# Memory: 8 workers each hold one decompressed parquet file; MaxRSS was 50 GB on 2026-10-08 (4.6 min) on the lab server, but 64G was OOM-killed on Perlmutter; MaxRSS there 115 GB (2026-10-11, 1.1 min). debug gets a whole node, so --mem=0 (all of it).
#SBATCH --job-name=pile-memmap
#SBATCH --account=m5187
#SBATCH --constraint=cpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.err
set -euo pipefail
cd /pscratch/sd/z/zres0710/iBrain_replication
mkdir -p logs
# NERSC: .venv uses uv's Python under /global/homes, visible from every node
PY=.venv/bin/python
$PY scripts/pile_to_memmap.py --root /pscratch/sd/z/zres0710/iBrain/pretrain/spikes/neural-pile-primate \
    --out /pscratch/sd/z/zres0710/iBrain/cache/neural-pile-primate-mmap --split "${1:-train}" --workers 8
