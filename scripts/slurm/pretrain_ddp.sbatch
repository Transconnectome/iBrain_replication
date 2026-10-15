#!/bin/bash
# Data-parallel pretraining on whole Perlmutter GPU nodes (U43): one rank per A100, each with batch_size / world of a
# step's windows and its own 4 DataLoader workers per loader. Submit from the repo root:
#   sbatch scripts/slurm/pretrain_ddp.sbatch <config> <seed> <out> [extra pretrain.py args...]
#   sbatch scripts/slurm/pretrain_ddp.sbatch configs/paper.yaml 0 runs/joint_s0          # 1 node, 4 GPUs
#   sbatch -N 2 scripts/slurm/pretrain_ddp.sbatch configs/paper.yaml 0 runs/joint_s0     # 2 nodes, 8 GPUs (paper)
#   sbatch -q debug -t 00:30:00 scripts/slurm/pretrain_ddp.sbatch <config> 0 runs/<test>  # short test
# Resubmitting the same command resumes from <out>/ckpt.pt. Cheaper on the preempt QOS (charged 1x for the first
# 2 h, then 0.25x; preemptible after 2 h, and --requeue reruns this script, which resumes):
#   sbatch -q preempt --requeue -N 2 scripts/slurm/pretrain_ddp.sbatch configs/paper.yaml 0 runs/joint_s0
# A whole node is charged (4 GPUs): node-hours at 4x the rate of pretrain.sbatch (one GPU of a shared node), for a
# run that finishes about 4x sooner.
# 8 GPUs give 32 windows per rank, one head_bn_group, the paper's per-GPU batch; batch_size must split into equal rank
# batches that are multiples of head_bn_group (scripts/pretrain.py checks).
#SBATCH --job-name=ibrain-ddp
#SBATCH --account=m5187
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.err
set -euo pipefail

# IBRAIN_CODE: the code to run, e.g. a git worktree frozen at one commit for a long requeued run, so edits to the main
# checkout never reach a resumed segment (sbatch --export=ALL,IBRAIN_CODE=<worktree>). The venv stays the main
# checkout's; scripts put their own repo root first on sys.path, so `ibrain` comes from IBRAIN_CODE.
PROJECT=${IBRAIN_CODE:-/pscratch/sd/z/zres0710/iBrain_replication}
VENV=${IBRAIN_VENV:-/pscratch/sd/z/zres0710/iBrain_replication/.venv}
cd "$PROJECT"
mkdir -p logs

CONFIG=${1:-configs/paper.yaml}
SEED=${2:-0}
OUT=${3:-runs/paper_ddp_s${SEED}}
EXTRA=("${@:4}")

source "$VENV/bin/activate"
# Across nodes NCCL needs the Slingshot (libfabric) plugin of NERSC's module; one node uses NVLink without it
if [ "${SLURM_JOB_NUM_NODES:-1}" -gt 1 ]; then module load nccl/2.29.2-cu13; fi
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
# ibrain.dist joins the process group when srun starts several tasks and MASTER_ADDR is set
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

RESUME=()
[ -f "$OUT/ckpt.pt" ] && RESUME=(--resume "$OUT/ckpt.pt")

echo "job=$SLURM_JOB_ID nodes=$SLURM_JOB_NODELIST tasks=$SLURM_NTASKS commit=$(git rev-parse --short HEAD)$(git diff --quiet || echo -dirty)"
echo "config=$CONFIG seed=$SEED out=$OUT resume=${RESUME[*]:-no} extra=${EXTRA[*]:-}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# The largest micro-batches of pretrain.token_budget must fit one GPU (U33), the same per rank as on one GPU: checked
# once in the batch step (GPU 0) before srun starts the ranks.
read -r BUDGET PREC CKPT < <(python -c "import yaml; p = yaml.safe_load(open('$CONFIG'))['pretrain']; \
print(p.get('token_budget') or 0, p.get('precision', 'fp32'), int(bool(p.get('grad_checkpoint'))))")
if [ "$BUDGET" -gt 0 ]; then
    CK=(); [ "$CKPT" = 1 ] && CK=(--checkpoint)
    python scripts/profile_memory.py --config "$CONFIG" --precision "$PREC" "${CK[@]}" --check-budget "$BUDGET" \
        --limit-gib 35
fi

T0=$(date +%s)
srun python scripts/pretrain.py --config "$CONFIG" --seed "$SEED" --out "$OUT" "${RESUME[@]}" "${EXTRA[@]}"
T1=$(date +%s)
awk -v s=$((T1 - T0)) 'BEGIN { printf "elapsed %.2f h\n", s / 3600 }'
