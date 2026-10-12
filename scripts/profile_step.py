"""Where does the time of one pretraining step go? GPU: run through scripts/slurm/profile_step.sbatch.
Real windows and loaders of the config (as scripts/pretrain.py), the config's model. Reports, per signal type:
  1. step time split into data wait, losses + backward (batch_losses), clip + optimizer, with micro-batches and
     channel slots per step; eager and compiled (IBrain.compile_parts, U40)
  2. FLOPs of one step (torch.utils.flop_counter) and the TFLOP/s reached
  3. torch.profiler of one step: CUDA kernels by self time, number of kernel launches, GPU busy share of the wall time
  4. compiled against eager on one batch with the same seed: losses and gradients (U40)"""
import argparse
import copy
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import yaml
from torch.profiler import ProfilerActivity, profile
from torch.utils.flop_counter import FlopCounterMode

from ibrain.model import IBrain, IEEG, SPIKE
from ibrain.pretrain import CLIP, PRECISION, batch_losses, pack
from pretrain import ieeg_data, loader, spike_data

NAMES = {SPIKE: "spike", IEEG: "iEEG"}


def sync():
    torch.cuda.synchronize()


def next_batch(its, dls, sig):
    b = next(its[sig], None)
    if b is None:
        its[sig] = iter(dls[sig])
        b = next(its[sig])
    return b


def one_step(model, opt, batch, sig, amp, budget):
    sync()
    t0 = time.perf_counter()
    opt.zero_grad()
    batch_losses(model, batch[0], batch[1], sig, amp, budget)
    sync()
    t1 = time.perf_counter()
    torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
    opt.step()
    sync()
    return t1 - t0, time.perf_counter() - t1


def slots(valid, budget):
    n = valid.sum(1)
    chunks = pack(n, budget)
    return len(chunks), sum(len(c) * int(n[c].max()) for c in chunks), int(n.sum())


def time_steps(model, its, dls, amp, budget, n_steps):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    out = {}
    for sig in (SPIKE, IEEG):
        for _ in range(2):  # warm-up: kernel selection, allocator
            one_step(model, opt, next_batch(its, dls, sig), sig, amp, budget)
        rec = []
        for _ in range(n_steps):
            t = time.perf_counter()
            batch = next_batch(its, dls, sig)
            wait = time.perf_counter() - t
            fb, optim = one_step(model, opt, batch, sig, amp, budget)
            rec.append((wait, fb, optim) + slots(batch[1], budget))
        out[sig] = [sum(r[k] for r in rec) / len(rec) for k in range(6)]
    torch.cuda.empty_cache()
    return out


def flops_of_step(model, batch, sig, amp, budget):
    model.zero_grad(set_to_none=True)
    with FlopCounterMode(display=False) as fc:
        batch_losses(model, batch[0], batch[1], sig, amp, budget)
    return fc.get_total_flops()


def profile_step(model, batch, sig, amp, budget):
    model.zero_grad(set_to_none=True)
    sync()
    t0 = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        batch_losses(model, batch[0], batch[1], sig, amp, budget)
        sync()
    wall = time.perf_counter() - t0
    ev = prof.key_averages()
    gpu = sum(e.self_device_time_total for e in ev) / 1e6
    launches = sum(e.count for e in ev if e.device_type == torch.autograd.DeviceType.CUDA)
    print(f"  wall {wall:.2f} s under the profiler, GPU kernel time {gpu:.2f} s ({100 * gpu / wall:.0f}% busy), "
          f"{launches} kernel launches")
    print(ev.table(sort_by="self_device_time_total", row_limit=18, max_name_column_width=70))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke.yaml")
    ap.add_argument("--steps", type=int, default=6, help="timed steps per signal type and setting")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    pc, dc = cfg["pretrain"], cfg["data"]
    amp = PRECISION[pc.get("precision", "fp32")]
    torch.manual_seed(a.seed)
    g = torch.Generator().manual_seed(a.seed)
    ds_s, _ = spike_data(dc["spike"], Path("runs/tmp_profile"), a.seed)
    ds_i, _ = ieeg_data(dc["ieeg"], a.seed, pc.get("shuffle_buffer", 1024))
    dls = {SPIKE: loader(ds_s, pc["batch_size"], g, pc.get("num_workers", 0)),
           IEEG: loader(ds_i, pc["batch_size"], g, pc.get("num_workers", 0))}
    its = {k: iter(v) for k, v in dls.items()}
    eager = IBrain(**cfg["model"]).cuda().train()
    model = copy.deepcopy(eager).compile_parts()  # sections 2 and 3 profile the compiled model
    budget = pc.get("token_budget")
    print(f"config {a.config}: batch {pc['batch_size']}, precision {pc.get('precision')}, model {cfg['model']}")

    print("\n1. Step time (mean over steps): wait | losses+backward | clip+optimizer, micro-batches, channel slots "
          "(padded) vs valid units")
    for label, m, b in (("eager", eager, budget), ("compiled", model, budget)):
        t0 = time.perf_counter()
        res = time_steps(m, its, dls, amp, b, a.steps)
        print(f"  {label}: {time.perf_counter() - t0:.0f} s for warm-up (compiles) and timed steps")
        for sig, (w, fb, op, nm, sl, un) in res.items():
            print(f"  {label:22s} {NAMES[sig]:5s} wait {w:5.2f} s | {fb:5.2f} s | {op:5.3f} s   {nm:4.1f} micro-batches, "
                  f"{sl:7.0f} slots for {un:7.0f} units")

    print("\n2. FLOPs of one step and TFLOP/s reached (A100: about 312 dense bf16 TFLOP/s at peak)")
    for sig in (SPIKE, IEEG):
        batch = next_batch(its, dls, sig)
        f = flops_of_step(model, batch, sig, amp, budget)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
        fb, _ = one_step(model, opt, batch, sig, amp, budget)
        print(f"  {NAMES[sig]:5s} {f / 1e12:7.2f} TFLOP per step, {fb:5.2f} s -> {f / fb / 1e12:6.2f} TFLOP/s")

    print("\n3. torch.profiler of one step")
    for sig in (SPIKE, IEEG):
        print(f" {NAMES[sig]}")
        profile_step(model, next_batch(its, dls, sig), sig, amp, budget)

    print("\n4. Compiled vs eager on the same batch and seed (bf16): losses and relative gradient error")
    for sig in (SPIKE, IEEG):
        batch = next_batch(its, dls, sig)
        out = []
        for m in (eager, eager, model):  # eager twice: the run-to-run spread of eager itself
            m.load_state_dict(eager.state_dict())
            m.zero_grad(set_to_none=True)
            torch.manual_seed(11)
            rec, align, _ = batch_losses(m, batch[0], batch[1], sig, amp, budget)
            out.append((rec, align, {n: q.grad.clone() for n, q in m.named_parameters() if q.grad is not None}))
        (r0, a0, g0), (re, ae, ge), (r1, a1, g1) = out

        def rel(ga, gb):
            errs = {n: ((ga[n] - gb[n]).norm() / (ga[n].norm() + 1e-12)).item() for n in ga}
            worst = max(errs, key=errs.get)
            return f"max {errs[worst]:.2e} ({worst}), median {sorted(errs.values())[len(errs) // 2]:.2e}"

        print(f"  {NAMES[sig]:5s} eager vs eager:    rec {r0:.5f} vs {re:.5f}, align {a0:.5f} vs {ae:.5f}, "
              f"gradient error {rel(g0, ge)}")
        print(f"  {NAMES[sig]:5s} eager vs compiled: rec {r0:.5f} vs {r1:.5f}, align {a0:.5f} vs {a1:.5f}, "
              f"gradient error {rel(g0, g1)}, same set: {g0.keys() == g1.keys()}")


if __name__ == "__main__":
    main()
