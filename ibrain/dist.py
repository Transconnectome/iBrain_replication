"""Data-parallel pretraining over several GPUs (U43), without DistributedDataParallel: each rank computes the gradient of
its share of a step's windows (micro-batches and the two-pass BN alignment as on one GPU), then one all-reduce sums
the gradients of the parameters that have one. DDP's wrapper would need forward() (the training code calls encode,
reconstruct and align), no_sync around every backward but the last, and find_unused_parameters for the other signal
type's encoder, decoder and type_emb, whose gradient must stay None (U24). The model is small (~26 MB of fp32
gradients), so one all-reduce after the backward passes costs about a millisecond.
World size 1 (no process group) unless the process was started by srun with MASTER_ADDR set
(scripts/slurm/pretrain_ddp.sbatch) or by torchrun."""
import os

import torch
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


def init():
    """Join the process group if this process is one of several ranks. Returns (rank, world, local_rank)."""
    if "WORLD_SIZE" in os.environ and "RANK" in os.environ:  # torchrun
        rank, world, local = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ.get("LOCAL_RANK", 0))
    elif "MASTER_ADDR" in os.environ and int(os.environ.get("SLURM_NTASKS", "1")) > 1:  # srun
        rank, world, local = (int(os.environ[k]) for k in ("SLURM_PROCID", "SLURM_NTASKS", "SLURM_LOCALID"))
    else:
        return 0, 1, 0
    if world > 1 and not dist.is_initialized():
        os.environ.setdefault("MASTER_PORT", "29500")
        if torch.cuda.is_available():
            torch.cuda.set_device(local)
            dist.init_process_group("nccl", rank=rank, world_size=world, device_id=torch.device("cuda", local))
        else:
            dist.init_process_group("gloo", rank=rank, world_size=world)
    return rank, world, local


def finish():
    """Leave the process group after the last collective (a barrier), so no rank's NCCL watchdog reports the store
    of an exited rank 0 as a failure."""
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def world():
    return dist.get_world_size() if dist.is_initialized() else 1


def rank():
    return dist.get_rank() if dist.is_initialized() else 0


def is_main():
    return rank() == 0


def barrier():
    if world() > 1:
        dist.barrier()


def all_sum(values, device):
    """Floats summed over the ranks (float64, so counts stay exact)."""
    if world() == 1:
        return list(values)
    t = torch.tensor(list(values), dtype=torch.float64, device=device)
    dist.all_reduce(t)
    return t.tolist()


def broadcast_model(model):
    """Parameters and buffers of rank 0 on every rank (the same seed builds the same model; this makes sure)."""
    if world() > 1:
        for t in model.state_dict().values():
            dist.broadcast(t, 0)


def sync_grads(params):
    """Sum the gradients over the ranks in one all-reduce. Parameters without a gradient stay None: every rank trains
    the same signal type in a step, so the set is the same everywhere, which is checked (a mismatch would hang or mix
    different parameters)."""
    if world() == 1:
        return
    params = list(params)
    idx = [i for i, p in enumerate(params) if p.grad is not None]
    grads = [params[i].grad for i in idx]
    device = grads[0].device if grads else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    key = torch.tensor([len(idx), sum(idx)], dtype=torch.int64, device=device)
    lo, hi = key.clone(), key.clone()
    dist.all_reduce(lo, dist.ReduceOp.MIN)
    dist.all_reduce(hi, dist.ReduceOp.MAX)
    if not torch.equal(lo, hi):
        raise RuntimeError(f"rank {rank()}: parameters with a gradient differ across ranks ({key.tolist()})")
    if not grads:
        return
    flat = _flatten_dense_tensors(grads)
    dist.all_reduce(flat)
    for g, s in zip(grads, _unflatten_dense_tensors(flat, grads)):
        g.copy_(s)
