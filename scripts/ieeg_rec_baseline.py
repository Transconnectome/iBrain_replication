"""Is the iEEG L_rec of a pretraining run better than trivial predictors? Per iEEG block of the config, on the same
masks (Eq. 6, 50% of tokens) and the same normalized windows as pretraining (U4 per block), the masked MSE (Eq. 9) of
  zero     predicting 0 (the block median): E[x^2] over masked patches
  ch_mean  each channel's mean over its visible patches of the window
  sp_mean  per sample, the mean over the channels visible at the same patch (0 if none is)
  interp   linear interpolation in time between the neighbouring visible samples of the same channel
  init     a freshly initialized model (same seed), eval mode
  <ckpt>   each checkpoint given, eval mode
plus the signal statistics behind them (heavy tails, power per band). --heldout evaluates on blocks the config does
not train on, to tell fitting the training windows from generalizing. CPU or GPU:
  sbatch scripts/slurm/ieeg_rec_baseline.sbatch runs/smoke_s0/final.pt
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml

from ibrain.data_ieeg import FS, WINDOW_SAMPLES, ajile12_blocks, block_windows, normalize_block, read_block, select_hours, swec_blocks
from ibrain.model import IBrain, IEEG, masked_mse
from ibrain.pretrain import sample_mask
from ibrain.repro import load_checkpoint, seed_all

READERS = {"ajile12": ajile12_blocks, "swec": swec_blocks}
BANDS = [(0, 4), (4, 30), (30, 100), (100, 250)]


def interp_pred(x, mask):
    """x (C, S, P), mask (C, S) bool -> prediction (C, S, P): masked patches linearly interpolated between the last
    visible sample before and the first visible sample after them in the same channel (one side: held constant)."""
    C, S, P = x.shape
    flat = x.reshape(C, S * P).copy()
    seen = np.repeat(~mask, P, axis=1)
    t = np.arange(S * P)
    for c in range(C):
        if seen[c].any() and not seen[c].all():
            flat[c, ~seen[c]] = np.interp(t[~seen[c]], t[seen[c]], flat[c, seen[c]])
        elif not seen[c].any():
            flat[c] = 0.0
    return flat.reshape(C, S, P)


def ch_mean_pred(x, mask):
    vis = (~mask)[..., None].astype(x.dtype)
    m = (x * vis).sum((1, 2)) / np.maximum(vis.sum((1, 2)) * x.shape[2], 1)
    return np.broadcast_to(m[:, None, None], x.shape)


def sp_mean_pred(x, mask):
    vis = (~mask)[..., None].astype(x.dtype)
    m = (x * vis).sum(0) / np.maximum(vis.sum(0), 1)  # (S, P)
    return np.broadcast_to(m[None], x.shape)


@torch.no_grad()
def model_pred(model, x, mask, device, bs):
    """x (N, C, S, P), mask (N, C, S) -> reconstruction (N, C, S, P) of the masked-input forward pass."""
    out = []
    for i in range(0, len(x), bs):
        xb = torch.from_numpy(x[i:i + bs]).to(device)
        mb = torch.from_numpy(mask[i:i + bs]).to(device)
        valid = torch.ones(xb.shape[:2], dtype=torch.bool, device=device)
        u = model.encode(xb.masked_fill(mb[..., None], 0.0), valid, IEEG)
        out.append(model.reconstruct(u, IEEG).float().cpu().numpy())
    return np.concatenate(out)


def mse(pred, x, mask):
    """Masked MSE as Eq. 9, and the share of it from the 1% largest-error patches."""
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
    valid = torch.ones(x.shape[:2], dtype=torch.bool)
    total = masked_mse(t(pred), t(x), torch.from_numpy(mask), valid).item()
    e = ((pred - x) ** 2).mean(-1)[mask]
    top = np.sort(e)[::-1][:max(1, len(e) // 100)].sum() / max(e.sum(), 1e-12)
    return total, top


def band_power(x):
    """Share of power per band of the windows, x (N, C, S, P) at 500 Hz."""
    w = x.reshape(*x.shape[:2], -1)
    p = (np.abs(np.fft.rfft(w - w.mean(-1, keepdims=True), axis=-1)) ** 2).mean((0, 1))
    f = np.fft.rfftfreq(w.shape[-1], 1 / FS)
    return [p[(f >= a) & (f < b)].sum() / p.sum() for a, b in BANDS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke.yaml")
    ap.add_argument("--ckpt", nargs="*", default=[])
    ap.add_argument("--n", type=int, default=256, help="windows per block (evenly spaced)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--blocks", type=int, default=None,
                    help="evaluate on this many blocks per source, drawn at random (default: all selected blocks)")
    ap.add_argument("--heldout", action="store_true",
                    help="draw the blocks from those the config does NOT select for training (same seed)")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seed_all(a.seed)
    models = [("init", IBrain(**cfg["model"]))]
    for p in a.ckpt:
        ck = load_checkpoint(p, "cpu")
        m = IBrain(**ck["meta"]["config"]["model"])
        m.load_state_dict(ck["model"])
        models.append((f"{Path(p).parent.name}/{Path(p).stem}@{ck['step']}", m))
    for _, m in models:
        m.to(device).eval()

    blocks = []
    pick = np.random.default_rng(a.seed + 1)
    for e in cfg["data"]["ieeg"]:
        every = READERS[e["format"]](e["root"])
        train = select_hours(every, e.get("max_hours"), a.seed)  # the blocks scripts/pretrain.py trains on
        pool = [b for b in every if b not in train] if a.heldout else train
        if a.blocks is not None and a.blocks < len(pool):
            pool = [pool[i] for i in sorted(pick.choice(len(pool), a.blocks, replace=False))]
        blocks += [(e["format"], b) for b in pool]
    names = ["zero", "ch_mean", "sp_mean", "interp"] + [n for n, _ in models]
    print(f"device={device} blocks={len(blocks)} ({'held out' if a.heldout else 'training'}) windows/block<={a.n}")
    rng = np.random.default_rng(a.seed)
    tot = {n: [] for n in names}
    for fmt, b in blocks:
        raw = read_block(b)
        x = normalize_block(b, raw)
        ws = list(block_windows(b, x))
        if not ws:  # IEEGStream skips such a block in the same way; report why it gives no window
            n_win = raw.shape[1] // WINDOW_SAMPLES
            t0 = b.start / b.fs
            in_bad = sum(any(u < t0 + (k + 1) * WINDOW_SAMPLES / FS and t0 + k * WINDOW_SAMPLES / FS < v
                             for u, v in b.bad) for k in range(n_win))
            print(f"\n== {fmt} {Path(b.path).name} [{b.start / b.fs:.0f}, {b.stop / b.fs:.0f}] s: no window, skipped | "
                  f"channels {len(x)}/{len(raw)} usable, {n_win} windows: {in_bad} overlap bad intervals "
                  f"({len(b.bad)} intervals), finite samples {np.isfinite(raw).mean():.3f}")
            continue
        w = np.stack(ws).astype(np.float32)  # (N, C, S, P)
        w = w[np.linspace(0, len(w) - 1, min(a.n, len(w))).round().astype(int)]
        torch.manual_seed(int(rng.integers(2 ** 31)))
        mask = sample_mask(torch.ones(w.shape[:2], dtype=torch.bool), w.shape[2]).numpy()
        ax = np.abs(w)
        print(f"\n== {fmt} {Path(b.path).name} [{b.start / b.fs:.0f}, {b.stop / b.fs:.0f}] s: {len(w)} windows, "
              f"C={w.shape[1]} | E[x^2]={np.mean(w ** 2):.3f}  max|x|={ax.max():.1f}  "
              f"P(|x|>5)={np.mean(ax > 5):.4f}  P(|x|>20)={np.mean(ax > 20):.5f}  "
              f"power {'/'.join(f'{lo}-{hi}' for lo, hi in BANDS)} Hz = "
              f"{'/'.join(f'{v:.2f}' for v in band_power(w))}")
        preds = {"zero": np.zeros_like(w),
                 "ch_mean": np.stack([ch_mean_pred(xi, mi) for xi, mi in zip(w, mask)]),
                 "sp_mean": np.stack([sp_mean_pred(xi, mi) for xi, mi in zip(w, mask)]),
                 "interp": np.stack([interp_pred(xi, mi) for xi, mi in zip(w, mask)])}
        for n, m in models:
            preds[n] = model_pred(m, w, mask, device, a.bs)
        for n in names:
            v, top = mse(preds[n], w, mask)
            tot[n].append((v, int(mask.sum())))
            print(f"  {n:28s} masked MSE {v:8.4f}   (top 1% patches: {100 * top:4.1f}% of it)")
    print("\n== all blocks (weighted by masked patches)")
    for n in names:
        v = np.array(tot[n])
        print(f"  {n:28s} masked MSE {np.average(v[:, 0], weights=v[:, 1]):8.4f}")


if __name__ == "__main__":
    main()
