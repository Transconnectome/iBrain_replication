#!/bin/bash
# Lab server (node1~4) pretraining. GPU only via Slurm: do not run scripts/pretrain.py from a login shell.
# Submit from the repo root (./logs is resolved against the submit directory):
#   sbatch scripts/slurm/pretrain.sbatch <config> <seed> <out> [extra pretrain.py args...]
#   sbatch scripts/slurm/pretrain.sbatch configs/paper.yaml 0 runs/spike_s0 --spike-only   # M3
#   sbatch scripts/slurm/pretrain.sbatch configs/paper.yaml 0 runs/joint_s0                # M6
# Resubmitting the same command resumes from <out>/ckpt.pt.
# GPU nodes: node1 only (node3 is down). .venv points at uv's Python under the node-local /home, so it only runs on node1.
# 12 CPUs x 5G = the per-GPU maximum. The Neural Pile is held in memory as uint8 (U30); 4 DataLoader workers per loader.
#SBATCH --job-name=ibrain-pretrain
#SBATCH --partition=debug
#SBATCH --nodelist=node1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=5G
#SBATCH --time=48:00:00
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.err
set -euo pipefail

PROJECT=/scratch/connectome/zres0710/iBrain_replication
cd "$PROJECT"
mkdir -p logs

CONFIG=${1:-configs/paper.yaml}
SEED=${2:-0}
OUT=${3:-runs/paper_s${SEED}}
EXTRA=("${@:4}")                   # e.g. --spike-only for M3

source .venv/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE # /scratch and /storage are NFS without locks (h5py/pynwb for AJILE12, SWEC)
export OMP_NUM_THREADS=4           # leaves the other CPUs to the DataLoader workers
export PYTHONUNBUFFERED=1          # progress lines reach the .out file as they are printed

RESUME=()
[ -f "$OUT/ckpt.pt" ] && RESUME=(--resume "$OUT/ckpt.pt")

echo "job=$SLURM_JOB_ID node=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-none} commit=$(git rev-parse --short HEAD)$(git diff --quiet || echo -dirty)"
echo "config=$CONFIG seed=$SEED out=$OUT resume=${RESUME[*]:-no} extra=${EXTRA[*]:-}"
nvidia-smi -i "${CUDA_VISIBLE_DEVICES:-0}" --query-gpu=name,memory.total,driver_version --format=csv,noheader
# The node1 driver is CUDA 12.4, so .venv needs a cu12x torch build (cu130 cannot see the GPU)
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0))"

# Before loading data (tens of minutes), check that the largest micro-batches of pretrain.token_budget fit on this GPU
# (U33). The job stops here if one runs out of memory or peaks above 21 GiB of the A5000's 24 GiB.
read -r BUDGET PREC CKPT < <(python -c "import yaml; p = yaml.safe_load(open('$CONFIG'))['pretrain']; \
print(p.get('token_budget') or 0, p.get('precision', 'fp32'), int(bool(p.get('grad_checkpoint'))))")
if [ "$BUDGET" -gt 0 ]; then
    CK=(); [ "$CKPT" = 1 ] && CK=(--checkpoint)
    python scripts/profile_memory.py --config "$CONFIG" --precision "$PREC" "${CK[@]}" --check-budget "$BUDGET" \
        --limit-gib 21
fi

T0=$(date +%s)
python scripts/pretrain.py --config "$CONFIG" --seed "$SEED" --out "$OUT" "${RESUME[@]}" "${EXTRA[@]}"
T1=$(date +%s)
awk -v s=$((T1 - T0)) 'BEGIN { printf "elapsed %.2f h\n", s / 3600 }'
