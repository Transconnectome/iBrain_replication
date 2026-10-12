"""Peak GPU memory of one pretraining forward + backward (step_losses, 3 encoder passes) per micro-batch shape.
Gives the numbers behind pretrain.token_budget (U33). GPU only: run through scripts/slurm/profile_memory.sbatch.
For each type and unit count C, n (windows) doubles until out of memory; prints n, C, n*C, peak GiB and ms per call."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml

from ibrain.model import IBrain, IEEG, SPIKE
from ibrain.pretrain import PRECISION, step_losses

UNITS = {SPIKE: (64, 96, 128, 256, 512, 1024, 1734), IEEG: (32, 64, 128, 256)}


def measure(model, sig, n, C, amp, P):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    if sig == SPIKE:
        x = torch.poisson(torch.full((n, C, 10, P), 0.3, device="cuda"))
    else:
        x = torch.randn(n, C, 10, P, device="cuda")
    valid = torch.ones(n, C, dtype=torch.bool, device="cuda")
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    rec, align = step_losses(model, x, valid, sig, amp)
    (rec + align).backward()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 2 ** 30, (time.perf_counter() - t0) * 1e3


def check_budget(model, cfg, amp, budget, limit):
    """Worst micro-batches under `budget`: for each unit count, as many windows as pack() puts together.
    The pile's largest window has 1734 units (scripts/pile_units.py, 2026-10-07). Returns the exit code."""
    worst = 0.0
    for sig, P in ((SPIKE, cfg["model"]["P_spike"]), (IEEG, cfg["model"]["P_ieeg"])):
        for C in UNITS[sig][::-1]:
            n = max(1, budget // C)
            try:
                gib, ms = measure(model, sig, n, C, amp, P)
            except torch.OutOfMemoryError:
                print(f"FAIL sig={sig} C={C:5d} n={n:4d} OOM")
                return 1
            worst = max(worst, gib)
            print(f"{'ok  ' if gib <= limit else 'FAIL'} sig={sig} C={C:5d} n={n:4d} n*C={n * C:6d} peak={gib:6.2f}GiB "
                  f"{ms:8.1f}ms")
            if gib > limit:
                return 1
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
    print(f"budget {budget}: worst peak {worst:.2f} GiB <= {limit} GiB")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/paper.yaml")
    ap.add_argument("--precision", default="bf16")
    ap.add_argument("--max-n", type=int, default=512)
    ap.add_argument("--checkpoint", action="store_true", help="activation checkpointing (U37)")
    ap.add_argument("--out", default="runs/profile_memory.jsonl")
    ap.add_argument("--check-budget", type=int, default=None,
                    help="only check the largest micro-batches `pack` makes under this token_budget (n = budget // C, "
                         "at least 1) and exit 1 if one runs out of memory or peaks above --limit-gib")
    ap.add_argument("--limit-gib", type=float, default=21.0)
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    torch.manual_seed(0)
    model = IBrain(**cfg["model"]).cuda().train()
    model.grad_checkpoint = a.checkpoint
    amp = PRECISION[a.precision]
    base = torch.cuda.memory_allocated() / 2 ** 30
    print(f"precision={a.precision} checkpoint={a.checkpoint}")
    print(f"model params+buffers {base:.2f} GiB, total {torch.cuda.get_device_properties(0).total_memory / 2 ** 30:.1f} GiB")
    if a.check_budget:
        sys.exit(check_budget(model, cfg, amp, a.check_budget, a.limit_gib))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for sig, P in ((SPIKE, cfg["model"]["P_spike"]), (IEEG, cfg["model"]["P_ieeg"])):
            for C in UNITS[sig]:
                try:
                    measure(model, sig, 1, C, amp, P)  # warm-up (kernel selection, allocator)
                except torch.OutOfMemoryError:
                    pass  # a single window does not fit; the loop below records it as OOM at n=1
                n = 1
                while n <= a.max_n:
                    try:
                        gib, ms = measure(model, sig, n, C, amp, P)
                    except torch.OutOfMemoryError:
                        print(f"sig={sig} C={C:5d} n={n:4d} OOM")
                        f.write(json.dumps({"sig": sig, "C": C, "n": n, "oom": True}) + "\n")
                        break
                    print(f"sig={sig} C={C:5d} n={n:4d} n*C={n * C:7d} peak={gib:6.2f}GiB {ms:8.1f}ms")
                    f.write(json.dumps({"sig": sig, "C": C, "n": n, "peak_gib": gib, "ms": ms}) + "\n")
                    f.flush()
                    n *= 2
                model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
