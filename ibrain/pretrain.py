"""M3/M6: pretraining. Masking (Eq. 6), channel views (Eq. 12), step loss (Eq. 13), 1:1 alternation loop,
checkpoints and logs (SPEC 1.5, 1.6, U26)."""
import json
import math
from pathlib import Path

import torch

from ibrain.model import SPIKE, masked_mse, masked_poisson_nll, simsiam_loss
from ibrain.repro import load_checkpoint, save_checkpoint, set_rng_state

LR, LR_MIN, WD, CLIP = 5e-4, 1e-5, 5e-2, 1.0  # SPEC 1.6
PRECISION = {"fp32": None, "bf16": torch.bfloat16}  # autocast dtype of the model forward (U36)


def sample_mask(valid, S):
    """M of Eq. 6, (B, C, S) bool. Per sample, exactly half of the valid tokens are drawn uniformly at random (U15).
    Padded channel tokens are never drawn."""
    B, C = valid.shape
    tok = valid[:, :, None].expand(B, C, S).reshape(B, C * S)
    rank = torch.rand(B, C * S, device=valid.device).masked_fill(~tok, 2.0).argsort(1).argsort(1)  # valid tokens first
    return (rank < tok.sum(1, keepdim=True) // 2).view(B, C, S)


def sample_views(valid):
    """Two channel views v1, v2 (B, C) bool (SPEC 1.5, U16). The n valid channels get a random rank:
    the first s are shared, the next u are view 1 only, the last u are view 2 only. u = round(0.2n), s = n - 2u."""
    n = valid.sum(1, keepdim=True)
    u = (n + 2) // 5  # integer arithmetic for round(0.2n)
    rank = torch.rand(valid.shape, device=valid.device).masked_fill(~valid, 2.0).argsort(1).argsort(1)
    v1 = rank < n - u  # shared + view 1 only
    v2 = (rank < n - 2 * u) | ((rank >= n - u) & (rank < n))  # shared + view 2 only
    return v1, v2


def lr_at(step, steps, warmup):
    """SPEC 1.6, U20. Linear from 0 at step 0 to LR at step warmup,
    then cosine down to LR_MIN at the last step (steps - 1)."""
    if step < warmup:
        return LR * step / warmup
    t = min(1.0, (step - warmup) / max(1, steps - 1 - warmup))
    return LR_MIN + 0.5 * (LR - LR_MIN) * (1 + math.cos(math.pi * t))


def total_steps(epochs, n_spike_batches, n_loaders, accum=1):
    """Training length in optimizer steps (U20). One epoch = one pass over the spike loader. Under 1:1 alternation spike
    gets every n_loaders-th step, so the total is multiplied by n_loaders; the iEEG loader cycles as needed
    (about 0.81 passes per epoch at paper scale). Same as zip(ieeg, spike) over 30 epochs.
    With gradient accumulation (U33) one optimizer step consumes `accum` loader batches of one type. The spike step
    count is rounded up over the whole run, so spike sees epochs x n_spike_batches batches plus fewer than accum extra."""
    return -(-epochs * n_spike_batches // accum) * n_loaders


def step_losses(model, x, valid, sig, amp=None):
    """(L_rec, L_align) for one step (Eq. 13). sig = SPIKE: Eq. 10, 11. sig = IEEG: Eq. 8, 9.
    iEEG x is already channel-normalized by the Dataset (U4), so input and target are the same values.
    amp: autocast dtype for the model forward (U36), None = float32. The losses are always computed in float32."""
    def fwd():
        return torch.autocast(x.device.type, dtype=amp or torch.bfloat16, enabled=amp is not None)

    mask = sample_mask(valid, x.shape[2])
    with fwd():
        u = model.encode(x.masked_fill(mask[..., None], 0.0), valid, sig)  # Eq. 6: mask in raw signal space, then encode
        xhat = model.reconstruct(u, sig).float()
    rec = masked_poisson_nll(xhat, x, mask, valid) if sig == SPIKE else masked_mse(xhat, x, mask, valid)
    v1, v2 = sample_views(valid)  # views use the unmasked x, with the view mask in place of V_c (U17)
    with fwd():
        q1, p1 = model.align(model.pool(model.encode(x, v1, sig), v1))
        q2, p2 = model.align(model.pool(model.encode(x, v2, sig), v2))
    return rec, simsiam_loss(p1.float(), q2.float(), p2.float(), q1.float())  # Eq. 12


def pack(n_units, budget=None):
    """Split one loader batch into micro-batches of similar unit count (U33). n_units (B,) valid channels per window.
    Windows are sorted by unit count (largest first) and cut greedily so that each micro-batch, padded to its own
    largest window, holds at most `budget` channel slots (windows x max units). A window larger than the budget is
    a micro-batch of its own. budget None = the whole batch as one micro-batch. Returns a list of index tensors."""
    if budget is None:
        return [torch.arange(len(n_units))]
    order = torch.argsort(n_units, descending=True, stable=True).tolist()
    n = n_units.tolist()
    out, cur = [], []
    for i in order:
        if cur and (len(cur) + 1) * n[cur[0]] > budget:  # cur[0] is the largest window of cur
            out.append(cur)
            cur = []
        cur.append(i)
    if cur:
        out.append(cur)
    return [torch.tensor(c) for c in out]


def batch_losses(model, x, valid, sig, amp=None, budget=None, scale=1.0):
    """Losses of one loader batch computed in micro-batches from `pack`, with backward on each (scaled by `scale`).
    Each micro-batch is cut to its own largest window, so little compute goes to padding. The micro-batch losses are
    weighted so that the gradient equals that of the whole batch at once: L_rec is a mean over masked tokens (each
    window masks exactly n_units * S // 2, U15), so a micro-batch weighs by its share of masked tokens; L_align is a
    mean over windows, so it weighs by its share of windows. Returns the whole-batch (L_rec, L_align) as floats."""
    n = valid.sum(1).cpu()
    masked = (n * x.shape[2] // 2).sum().clamp(min=1)
    device = next(model.parameters()).device
    rec_b = align_b = 0.0
    for idx in pack(n, budget):
        c = int(n[idx].max())
        xc, vc = x[idx, :c].to(device), valid[idx, :c].to(device)
        w_rec = float((n[idx] * x.shape[2] // 2).sum() / masked)
        w_align = len(idx) / len(n)
        rec, align = step_losses(model, xc, vc, sig, amp)
        ((w_rec * rec + w_align * align) * scale).backward()
        rec_b, align_b = rec_b + w_rec * rec.item(), align_b + w_align * align.item()
    return rec_b, align_b


def pretrain(model, loaders, steps, warmup=2000, device="cpu", out_dir=None, ckpt_every=1000, resume=None, meta=None,
             accum=1, precision="fp32", token_budget=None, grad_checkpoint=False):
    """loaders = [(sig, DataLoader), ...], batches are collate's (x, valid).
    step t uses `accum` batches from loaders[t % len(loaders)] (the 1:1 alternation of Eq. 13): their losses are
    averaged before one optimizer step, which gives the gradient of one batch accum times larger (U33; the paper's
    8 GPUs x 32 per type = 256 per step). Each loader batch is computed in micro-batches of similar unit count with at
    most token_budget channel slots each (`pack`, `batch_losses`), so the step gradient does not depend on the
    budget; token_budget None = the whole batch at once. Returns per-step records.
    If out_dir is set: one line per step in log.jsonl, ckpt.pt (latest) every ckpt_every steps, final.pt at the end.
    resume is a ckpt path: restores model, optimizer, step, records and RNG state. Loader order is reshuffled (U26).
    precision: "fp32" or "bf16" (autocast of the model forward, weights and optimizer stay float32, U36).
    grad_checkpoint: activation checkpointing of the encoder and backbone blocks (U37)."""
    if precision not in PRECISION:
        raise ValueError(f"precision must be one of {sorted(PRECISION)}, got {precision!r}")
    amp = PRECISION[precision]
    model.to(device).train()
    model.grad_checkpoint = grad_checkpoint
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)  # U19
    start, hist = 0, []
    if resume:
        ck = load_checkpoint(resume, device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start, hist = ck["step"] + 1, ck["hist"]
        set_rng_state(ck["rng"])
    its = [iter(dl) for _, dl in loaders]
    log = None
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        log = open(Path(out_dir) / "log.jsonl", "a", encoding="utf-8")
    for t in range(start, steps):
        i = t % len(loaders)
        sig, dl = loaders[i]
        lr = lr_at(t, steps, warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad()  # set_to_none: the inactive type's encoder/decoder/type_emb keep grad None, AdamW skips them (U24)
        rec_sum = align_sum = 0.0
        for _ in range(accum):
            batch = next(its[i], None)
            if batch is None:  # end of a pass, reshuffle and cycle (U20)
                its[i] = iter(dl)
                batch = next(its[i])
            rec, align = batch_losses(model, batch[0], batch[1], sig, amp, token_budget, 1 / accum)
            rec_sum, align_sum = rec_sum + rec, align_sum + align
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step()
        hist.append({"step": t, "sig": sig, "lr": lr, "rec": rec_sum / accum, "align": align_sum / accum})
        if log:
            log.write(json.dumps(hist[-1]) + "\n")
            log.flush()
        if t % 100 == 0:
            gpu = f" gpu_peak={torch.cuda.max_memory_allocated() / 2 ** 30:.1f}GiB" if torch.device(device).type == "cuda" else ""
            print(f"{hist[-1]}{gpu}")
        if out_dir and ((t + 1) % ckpt_every == 0 or t == steps - 1):
            save_checkpoint(Path(out_dir) / "ckpt.pt", model, opt, t, hist, meta)
    if out_dir:
        save_checkpoint(Path(out_dir) / "final.pt", model, opt, steps - 1, hist, meta)
        log.close()
    return hist
