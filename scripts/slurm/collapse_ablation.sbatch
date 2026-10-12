#!/bin/bash
# Collapse ablation (2026-10-08): configs/collapse/*.yaml, one change each from configs/smoke.yaml (baseline R0 = the
# smoke run runs/smoke_s0). Each task trains 200 steps through pretrain.sbatch, then runs check_collapse on final.pt.
# Submit from the repo root (%3 = at most 3 tasks at once):
#   sbatch --array=0-3%3 scripts/slurm/collapse_ablation.sbatch
#SBATCH --job-name=ibrain-collapse-abl
#SBATCH --account=m5187
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:30:00
#SBATCH --output=./logs/%x_%A_%a.out
#SBATCH --error=./logs/%x_%A_%a.err
set -euo pipefail
cd /pscratch/sd/z/zres0710/iBrain_replication
NAMES=(ra_warmup rb_head_bn rc_no_final_ln rd_max_pool)
NAME=${NAMES[$SLURM_ARRAY_TASK_ID]}
OUT=runs/collapse/${NAME}_s0
echo "task=$SLURM_ARRAY_TASK_ID name=$NAME out=$OUT"
bash scripts/slurm/pretrain.sbatch "configs/collapse/${NAME}.yaml" 0 "$OUT"
source .venv/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE
python scripts/check_collapse.py --ckpt "$OUT/final.pt"
