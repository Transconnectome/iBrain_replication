"""M3/M6: pretraining. Masking (Eq. 6), channel views (Eq. 12), step loss (Eq. 13), 1:1 alternation loop,
checkpoints and logs (SPEC 1.5, 1.6, U26)."""
import json
import math
import time
from pathlib import Path

import torch

from ibrain import dist
from ibrain.model import SPIKE, masked_mse, masked_poisson_nll, simsiam_loss
from ibrain.repro import load_checkpoint, save_checkpoint, seed_all, set_rng_state

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


def _autocast(device, amp):
    return torch.autocast(torch.device(device).type, dtype=amp or torch.bfloat16, enabled=amp is not None)


def rec_loss(model, x, valid, sig, amp=None):
    """L_rec (Eq. 9 or 11) with a fresh 50% mask (Eq. 6: mask in raw signal space, then encode)."""
    mask = sample_mask(valid, x.shape[2])
    with _autocast(x.device, amp):
        u = model.encode(x.masked_fill(mask[..., None], 0.0), valid, sig)
        xhat = model.reconstruct(u, sig).float()
    return masked_poisson_nll(xhat, x, mask, valid) if sig == SPIKE else masked_mse(xhat, x, mask, valid)


def view_repr(model, x, view, sig, amp=None):
    """Pooled representation r of one channel view (the view mask in place of V_c, U17), float32 (B, d)."""
    with _autocast(x.device, amp):
        return model.pool(model.encode(x, view, sig), view).float()


def step_losses(model, x, valid, sig, amp=None, return_parts=False):
    """(L_rec, L_align) for one step (Eq. 13). sig = SPIKE: Eq. 10, 11. sig = IEEG: Eq. 8, 9.
    iEEG x is already channel-normalized by the Dataset (U4), so input and target are the same values.
    amp: autocast dtype for the model forward (U36), None = float32. The losses are always computed in float32.
    return_parts: also return {r1, q1, p1, q2} (detached, CPU) for the collapse metrics."""
    rec = rec_loss(model, x, valid, sig, amp)
    v1, v2 = sample_views(valid)  # views use the unmasked x (U17)
    r1 = view_repr(model, x, v1, sig, amp)
    with _autocast(x.device, amp):
        q1, p1 = model.align(r1)
        q2, p2 = model.align(view_repr(model, x, v2, sig, amp))
    align = simsiam_loss(p1.float(), q2.float(), p2.float(), q1.float())  # Eq. 12
    if not return_parts:
        return rec, align
    return rec, align, {k: v.detach().float().cpu() for k, v in (("r1", r1), ("q1", q1), ("p1", p1), ("q2", q2))}


def q_std(q):
    """SimSiam's collapse metric: per-dimension std over windows of the l2-normalized projection, averaged over
    dimensions. About 1/sqrt(d_proj) when the outputs are spread out, 0 when every window gives the same q."""
    return torch.nn.functional.normalize(q.float(), dim=-1).std(0).mean().item()


def collapse_metrics(parts):
    """Per-step collapse metrics from one batch's {r1, q1, p1, q2}:
    q_std  q_std of the projection (fixed near 1/sqrt(d_proj) by construction when the head ends in BatchNorm)
    r_std  the same metric on the pooled backbone representation r, which no head normalization touches
    d_gap  D(p1, q2 of another window) - D(p1, q2 of the same window), D = -cos. About 0 when the head cannot tell
           windows apart (collapse), larger when it can."""
    cos = torch.nn.functional.cosine_similarity
    p1, q2 = parts["p1"], parts["q2"]
    other = torch.roll(torch.arange(len(p1)), 1)
    d_gap = cos(p1, q2, dim=-1).mean() - cos(p1, q2[other], dim=-1).mean()
    return {"q_std": q_std(parts["q1"]), "r_std": q_std(parts["r1"]), "d_gap": d_gap.item()}


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


def batch_losses(model, x, valid, sig, amp=None, budget=None, scale=1.0, rec_scale=None):
    """Losses of one loader batch computed in micro-batches from `pack`, with backward on each (scaled by `scale`).
    Each micro-batch is cut to its own largest window, so little compute goes to padding. The micro-batch losses are
    weighted so that the gradient equals that of the whole batch at once: L_rec is a mean over masked tokens (each
    window masks exactly n_units * S // 2, U15), so a micro-batch weighs by its share of masked tokens; L_align is a
    mean over windows, so it weighs by its share of windows. A model with BatchNorm in the SimSiam head (head_bn)
    takes `_bn_align` for L_align, so the head sees the whole batch. Returns the whole-batch L_rec and L_align as
    floats and the batch's `collapse_metrics`. rec_scale (default: scale) scales L_rec's backward on its own: with
    several ranks (U43) it is this rank's share of the step's masked tokens, while L_align, a mean over windows, takes
    scale = this rank's share of the step's windows."""
    n = valid.sum(1).cpu()
    masked = (n * x.shape[2] // 2).sum().clamp(min=1)
    device = next(model.parameters()).device
    rec_b = align_b = 0.0
    rs = scale if rec_scale is None else rec_scale
    parts = []
    for idx in pack(n, budget):
        c = int(n[idx].max())
        xc, vc = x[idx, :c].to(device), valid[idx, :c].to(device)
        w_rec = float((n[idx] * x.shape[2] // 2).sum() / masked)
        if model.head_bn:
            rec = rec_loss(model, xc, vc, sig, amp)
            (w_rec * rec * rs).backward()
        else:
            w_align = len(idx) / len(n)
            rec, align, part = step_losses(model, xc, vc, sig, amp, return_parts=True)
            (w_rec * rec * rs + w_align * align * scale).backward()
            align_b += w_align * align.item()
            parts.append((idx, part))
        rec_b += w_rec * rec.item()
    if model.head_bn:
        align_b, whole = _bn_align(model, x, valid, n, sig, amp, budget, scale, device)
    else:  # back to batch order
        whole = {k: torch.zeros(len(n), v.shape[1]) for k, v in parts[0][1].items()}
        for idx, part in parts:
            for k, v in part.items():
                whole[k][idx] = v
    return rec_b, align_b, collapse_metrics(whole)


def _rng_state(device):
    return torch.get_rng_state(), (torch.cuda.get_rng_state(device) if device.type == "cuda" else None)


def _set_rng_state(state, device):
    torch.set_rng_state(state[0])
    if state[1] is not None:
        torch.cuda.set_rng_state(state[1], device)


def _bn_align(model, x, valid, n, sig, amp, budget, scale, device):
    """L_align when the SimSiam head has BatchNorm (collapse ablation RB, 2026-10-08). BN needs the whole batch (a
    micro-batch can be one window), so the gradient is taken in two passes (as in GradCache): (1) the pooled view
    representations r of all windows without a graph, micro-batch by micro-batch; the head and L_align on all of them
    at once, backward to the head and to r; (2) each micro-batch again with a graph, with the RNG state of pass 1 so
    dropout matches, and its share of dL/dr. The gradient equals that of the whole batch at once. r stays in batch
    order, so a head with head_bn_group sees random groups, as one GPU each. Returns (L_align, {r1, q1, p1, q2})."""
    views = sample_views(valid)  # for the whole batch, as step_losses draws them per batch
    chunks = []
    for idx in pack(n, budget):
        c = int(n[idx].max())
        chunks.append((idx, c, [None, None]))  # RNG state before each view's pass 1, replayed in pass 2
    r = [torch.zeros(len(n), model.d, device=device) for _ in views]
    with torch.no_grad():
        for idx, c, states in chunks:
            xc = x[idx, :c].to(device)
            for k, v in enumerate(views):
                states[k] = _rng_state(device)
                r[k][idx] = view_repr(model, xc, v[idx, :c].to(device), sig, amp)
    after = _rng_state(device)
    r = [rk.requires_grad_() for rk in r]
    with _autocast(device, amp):
        q1, p1 = model.align(r[0])
        q2, p2 = model.align(r[1])
    align = simsiam_loss(p1.float(), q2.float(), p2.float(), q1.float())
    (align * scale).backward()
    grads = [rk.grad for rk in r]
    for idx, c, states in chunks:
        xc = x[idx, :c].to(device)
        total = 0.0
        for k, v in enumerate(views):
            _set_rng_state(states[k], device)
            total = total + (view_repr(model, xc, v[idx, :c].to(device), sig, amp) * grads[k][idx]).sum()
        total.backward()
    _set_rng_state(after, device)
    return align.item(), {k: v.detach().float().cpu() for k, v in (("r1", r[0]), ("q1", q1), ("p1", p1), ("q2", q2))}


def pretrain(model, loaders, steps, warmup=2000, device="cpu", out_dir=None, ckpt_every=1000, resume=None, meta=None,
             accum=1, precision="fp32", token_budget=None, grad_checkpoint=False, keep_every=None, use_compile=False):
    """loaders = [(sig, DataLoader), ...], batches are collate's (x, valid).
    step t uses `accum` batches from loaders[t % len(loaders)] (the 1:1 alternation of Eq. 13): their losses are
    averaged before one optimizer step, which gives the gradient of one batch accum times larger (U33; the paper's
    8 GPUs x 32 per type = 256 per step). Each loader batch is computed in micro-batches of similar unit count with at
    most token_budget channel slots each (`pack`, `batch_losses`), so the step gradient does not depend on the
    budget; token_budget None = the whole batch at once. Returns per-step records.
    If out_dir is set: one line per step in log.jsonl, ckpt.pt (latest) every ckpt_every steps, final.pt at the end.
    resume is a ckpt path: restores model, optimizer, step, records and RNG state. Loader order is reshuffled (U26).
    precision: "fp32" or "bf16" (autocast of the model forward, weights and optimizer stay float32, U36).
    grad_checkpoint: activation checkpointing of the encoder and backbone blocks (U37).
    keep_every: also keep a checkpoint every that many steps as ckpt_<steps done>.pt (not overwritten).
    use_compile: torch.compile the encoders, decoders and backbone blocks (IBrain.compile_parts, U40).
    Several ranks (ibrain.dist, U43): each loader gives this rank's share of a step's windows; L_rec is weighted by
    the rank's share of the step's masked tokens and L_align by its share of the windows, the gradients are summed over the ranks
    in one all-reduce before clipping, so the step equals one GPU computing all its windows. Losses and collapse
    metrics are logged for the whole step; only rank 0 writes the log and checkpoints. A pass that ends calls
    set_epoch on the loader's sampler or dataset, so a DistributedSampler or a sharded IEEGStream reshuffles."""
    if precision not in PRECISION:
        raise ValueError(f"precision must be one of {sorted(PRECISION)}, got {precision!r}")
    amp = PRECISION[precision]
    W = dist.world()
    model.to(device).train()
    dist.broadcast_model(model)
    model.grad_checkpoint = grad_checkpoint
    if use_compile:
        model.compile_parts()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)  # U19
    start, hist = 0, []
    if resume:
        ck = load_checkpoint(resume, device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start, hist = ck["step"] + 1, ck["hist"]
        set_rng_state(ck["rng"])
        if W > 1:  # the checkpoint holds rank 0's RNG state: give each rank its own stream again
            seed_all(1_000_003 * start + dist.rank())
    passes = [0] * len(loaders)
    if resume:  # U26 (e): a new order after a resume. Without this the loaders would restart pass 0's order, the
        # sampler's generator and the pass counter being rebuilt from the run seed, so every resume of a chunked run
        # would show the start of the same permutation again
        for i, (_, dl) in enumerate(loaders):
            passes[i] = start  # pass numbers of this segment start at the resume step: none repeats an earlier one
            for src in (getattr(dl, "sampler", None), dl.dataset):
                if hasattr(src, "set_epoch"):
                    src.set_epoch(passes[i])
            gen = getattr(getattr(dl, "sampler", None), "generator", None)
            if gen is not None:  # one-GPU RandomSampler: reseed from the restored RNG state
                gen.manual_seed(int(torch.randint(0, 2 ** 62, (1,))))
    its = [iter(dl) for _, dl in loaders]
    log = None
    if out_dir and dist.is_main():
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        if resume:  # the log ends at the checkpoint: steps logged after it (an interrupted or preempted job) rerun now
            with open(Path(out_dir) / "log.jsonl", "w", encoding="utf-8") as f:
                f.writelines(json.dumps(h) + "\n" for h in hist)
        log = open(Path(out_dir) / "log.jsonl", "a", encoding="utf-8")
    for t in range(start, steps):
        t0 = time.perf_counter()
        i = t % len(loaders)
        sig, dl = loaders[i]
        lr = lr_at(t, steps, warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad()  # set_to_none: the inactive type's encoder/decoder/type_emb keep grad None, AdamW skips them (U24)
        rec_sum = align_sum = 0.0
        mets = {}
        for _ in range(accum):
            batch = next(its[i], None)
            if batch is None:  # end of a pass, reshuffle and cycle (U20)
                passes[i] += 1
                for src in (getattr(dl, "sampler", None), dl.dataset):
                    if hasattr(src, "set_epoch"):
                        src.set_epoch(passes[i])
                its[i] = iter(dl)
                batch = next(its[i])
            if W > 1:
                own = float((batch[1].sum(1) * batch[0].shape[2] // 2).sum())  # masked tokens of this rank (U15)
                n_own = float(len(batch[1]))  # windows of this rank
                total, n_total = (max(v, 1.0) for v in dist.all_sum([own, n_own], device))
                rec, align, m = batch_losses(model, batch[0], batch[1], sig, amp, token_budget,
                                             n_own / n_total / accum, own / total / accum)
                keys = sorted(m)
                red = dist.all_sum([rec * own, align * n_own] + [m[k] * n_own for k in keys], device)
                rec, align, m = red[0] / total, red[1] / n_total, {k: v / n_total for k, v in zip(keys, red[2:])}
            else:
                rec, align, m = batch_losses(model, batch[0], batch[1], sig, amp, token_budget, 1 / accum)
            rec_sum, align_sum = rec_sum + rec, align_sum + align
            mets = {k: mets.get(k, 0.0) + v / accum for k, v in m.items()}
        dist.sync_grads(model.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step()
        # sec: wall time of the step, data wait included (the losses' .item() calls synchronize the GPU)
        hist.append({"step": t, "sig": sig, "lr": lr, "rec": rec_sum / accum, "align": align_sum / accum, **mets,
                     "sec": time.perf_counter() - t0})
        if log:
            log.write(json.dumps(hist[-1]) + "\n")
            log.flush()
        if t % 100 == 0 and dist.is_main():
            gpu = f" gpu_peak={torch.cuda.max_memory_allocated() / 2 ** 30:.1f}GiB" if torch.device(device).type == "cuda" else ""
            print(f"{hist[-1]}{gpu}")
        if out_dir and dist.is_main() and ((t + 1) % ckpt_every == 0 or t == steps - 1):
            save_checkpoint(Path(out_dir) / "ckpt.pt", model, opt, t, hist, meta)
        if out_dir and dist.is_main() and keep_every and (t + 1) % keep_every == 0:
            save_checkpoint(Path(out_dir) / f"ckpt_{t + 1:07d}.pt", model, opt, t, hist, meta)
    if out_dir and dist.is_main():
        save_checkpoint(Path(out_dir) / "final.pt", model, opt, steps - 1, hist, meta)
        log.close()
    dist.barrier()  # no rank leaves before rank 0 has written final.pt
    return hist
