"""M3 verification: mask, channel views, lr schedule, loss decrease on synthetic shards."""
import math

import torch
from torch.utils.data import DataLoader

from ibrain.data_spike import SYNTHETIC, SpikeWindows, collate, read_windows, write_synthetic
from ibrain.model import IBrain, SPIKE, masked_poisson_nll
from ibrain import pretrain as pt
from ibrain.pretrain import LR, LR_MIN, batch_losses, lr_at, pack, pretrain, sample_mask, sample_views, total_steps


def valid_of(ns, C):
    return torch.arange(C) < torch.tensor(ns)[:, None]


def test_mask_half_of_valid_tokens():
    torch.manual_seed(0)
    ns = [1, 3, 7, 10]
    valid = valid_of(ns, 10)
    mask = sample_mask(valid, 10)
    assert not mask[~valid].any()  # padding channels are never selected
    assert mask.sum((1, 2)).tolist() == [5 * n for n in ns]  # exactly 50% of valid tokens per sample (U15)
    freq = sample_mask(torch.ones(4000, 3, dtype=torch.bool), 10).float().mean(0)
    assert (freq - 0.5).abs().max() < 0.05  # positions are uniformly random
    # values at padding positions do not enter the loss
    x, xhat = torch.poisson(torch.ones(4, 10, 10, 5)), torch.randn(4, 10, 10, 5)
    x2, xhat2 = x.clone(), xhat.clone()
    x2[~valid], xhat2[~valid] = 99.0, 99.0
    assert torch.allclose(masked_poisson_nll(xhat, x, mask, valid), masked_poisson_nll(xhat2, x2, mask, valid))


def test_views():
    torch.manual_seed(0)
    ns = [1, 2, 3, 5, 37, 182]
    valid = valid_of(ns, 256)
    v1, v2 = sample_views(valid)
    for i, n in enumerate(ns):
        a, b = v1[i], v2[i]
        assert not (a & ~valid[i]).any() and not (b & ~valid[i]).any()  # valid channels only
        assert torch.equal(a | b, valid[i])  # union = all valid channels (U16)
        k, o = a.sum().item(), (a & b).sum().item()
        assert b.sum().item() == k >= min(2, n)
        if n >= 30:
            assert abs(k / n - 0.8) < 0.03 and abs(o / k - 0.75) < 0.03
    assert (v1[2].sum().item(), (v1[2] & v2[2]).sum().item()) == (2, 1)  # n = 3: configuration of the bottom panel of Fig. 2
    assert not torch.equal(sample_views(valid)[0], v1)  # resampled at every step


def test_view_mask_equals_channel_subset():
    """Encoding with a view mask = encoding only the view's channels sliced out (U17). Channels outside the view do not affect the view representation."""
    torch.manual_seed(0)
    m = IBrain().eval()
    x = torch.poisson(torch.full((1, 8, 10, 5), 2.0))
    v1, _ = sample_views(torch.ones(1, 8, dtype=torch.bool))
    xs = x[:, v1[0]]
    ones = torch.ones(1, xs.shape[1], dtype=torch.bool)
    assert torch.allclose(m.pool(m.encode(x, v1), v1), m.pool(m.encode(xs, ones), ones), atol=1e-5)


def test_lr_schedule():
    steps, W = 101, 10
    assert lr_at(0, steps, W) == 0.0
    assert math.isclose(lr_at(W, steps, W), LR)
    assert math.isclose(lr_at(steps - 1, steps, W), LR_MIN)
    lrs = [lr_at(t, steps, W) for t in range(steps)]
    assert all(a < b for a, b in zip(lrs[:W], lrs[1:W + 1]))  # increases during warmup
    assert all(a > b for a, b in zip(lrs[W:-1], lrs[W + 1:]))  # decreases afterwards


def test_pretrain_reduces_loss(tmp_path):
    write_synthetic(tmp_path, (3, 8, 20), n_windows=16)
    torch.manual_seed(0)
    dl = DataLoader(SpikeWindows(read_windows(tmp_path, SYNTHETIC)), batch_size=8, shuffle=True, collate_fn=collate)
    m = IBrain(d=64, H=4, ffn=128, L=2, d_proj=32)  # small model for CPU tests
    hist = pretrain(m, [(SPIKE, dl)], steps=40, warmup=5)  # 6 batches × about 7 passes, including loader cycling
    assert [h["lr"] for h in hist] == [lr_at(t, 40, 5) for t in range(40)]

    def avg(key, hs):
        return sum(h[key] for h in hs) / len(hs)

    print(f"\nrec {avg('rec', hist[:5]):.4f} -> {avg('rec', hist[-5:]):.4f}, "
          f"align {avg('align', hist[:5]):.4f} -> {avg('align', hist[-5:]):.4f}")
    assert avg("rec", hist[-5:]) < avg("rec", hist[:5])
    assert avg("rec", hist[-5:]) + avg("align", hist[-5:]) < avg("rec", hist[:5]) + avg("align", hist[:5])


def test_checkpoint_and_resume(tmp_path):
    """U26: log.jsonl, ckpt.pt, final.pt in out_dir. resume carries over the step, history, model, and optimizer."""
    import json
    from ibrain.repro import load_checkpoint
    write_synthetic(tmp_path / "data", (3, 8), n_windows=8)
    dl = DataLoader(SpikeWindows(read_windows(tmp_path / "data", SYNTHETIC)), batch_size=4, shuffle=True, collate_fn=collate)
    torch.manual_seed(0)
    m = IBrain(d=32, H=4, ffn=64, L=1, d_proj=16)
    out = tmp_path / "run"
    h1 = pretrain(m, [(SPIKE, dl)], steps=6, warmup=2, out_dir=out, ckpt_every=4, meta={"seed": 0})
    ck = load_checkpoint(out / "ckpt.pt")
    assert ck["step"] == 5 and len(ck["hist"]) == 6 and ck["meta"] == {"seed": 0}
    assert all(torch.equal(v, ck["model"][k]) for k, v in m.state_dict().items())
    assert len((out / "log.jsonl").read_text().splitlines()) == 6
    # again from the ckpt saved at 4 steps: after running 6 steps ckpt.pt is at step 5, resume from the ckpt before that (step 3)
    torch.manual_seed(0)
    m2 = IBrain(d=32, H=4, ffn=64, L=1, d_proj=16)
    out2 = tmp_path / "run2"
    pretrain(m2, [(SPIKE, dl)], steps=4, warmup=2, out_dir=out2, ckpt_every=100)
    m3 = IBrain(d=32, H=4, ffn=64, L=1, d_proj=16)
    h3 = pretrain(m3, [(SPIKE, dl)], steps=10, warmup=2, out_dir=out2, resume=out2 / "final.pt")
    assert [h["step"] for h in h3] == list(range(10)) and h3[4]["lr"] == lr_at(4, 10, 2)
    assert len((out2 / "log.jsonl").read_text().splitlines()) == 10
    assert json.loads((out2 / "log.jsonl").read_text().splitlines()[-1])["step"] == 9
    assert load_checkpoint(out2 / "final.pt")["step"] == 9


def test_total_steps_counts_spike_epochs():
    """U20: epochs are counted on the spike loader. Under 1:1 alternation spike still gets exactly epochs × batches."""
    assert total_steps(30, 7, 1) == 210  # spike-only: unchanged
    steps = total_steps(30, 7, 2)
    assert steps == 420
    assert sum(t % 2 == 0 for t in range(steps)) == 30 * 7  # loaders[t % 2], spike is index 0


def test_script_joint_step_count(tmp_path):
    """U20 at script level: with --spike-only the spike loader is seen `epochs` times; joint runs double the steps so
    spike is still seen exactly `epochs` times (the halving bug was in scripts/pretrain.py, not in total_steps).
    U33: with gradient accumulation each optimizer step consumes `accum` loader batches, and the exposure is unchanged."""
    import json
    import subprocess
    import sys
    from pathlib import Path
    import yaml
    root = Path(__file__).resolve().parent.parent
    cfg = yaml.safe_load((root / "configs/tiny.yaml").read_text())
    cfg["model"].update(d=16, H=2, ffn=32, L=1, d_proj=8)
    for accum in (1, 2, 5):  # 5 does not divide the 6 spike batches: rounding must not lose or add a pass
        cfg["pretrain"].update(steps=None, epochs=2, warmup=1, ckpt_every=1000, grad_accum=accum)
        (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg))
        for extra, n_loaders in ((["--spike-only"], 1), ([], 2)):
            out = tmp_path / f"run{n_loaders}_a{accum}"
            r = subprocess.run([sys.executable, str(root / "scripts/pretrain.py"), "--config", str(tmp_path / "c.yaml"),
                                "--out", str(out), "--device", "cpu", *extra], capture_output=True, text=True, cwd=root)
            assert r.returncode == 0, r.stderr[-1500:]
            meta = json.loads((out / "meta.json").read_text())
            log = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
            spike_batches = -(-meta["data"]["spike_windows"] // cfg["pretrain"]["batch_size"])  # DataLoader length
            assert meta["data"]["steps"] == len(log) == -(-2 * spike_batches // accum) * n_loaders
            drawn = sum(h["sig"] == SPIKE for h in log) * accum
            assert 0 <= drawn - 2 * spike_batches < accum  # 2 epochs, plus less than one accumulation group


def test_accumulation_draws_accum_batches_per_step(tmp_path):
    """U33: each optimizer step really draws `accum` batches of its type (counted at the dataset)."""
    write_synthetic(tmp_path, (3, 8), n_windows=8)
    ds = SpikeWindows(read_windows(tmp_path, SYNTHETIC))
    drawn = []

    class Counting(torch.utils.data.Dataset):
        def __len__(self):
            return len(ds)

        def __getitem__(self, i):
            drawn.append(i)
            return ds[i]

    dl = DataLoader(Counting(), batch_size=4, shuffle=True, collate_fn=collate)
    torch.manual_seed(0)
    hist = pretrain(IBrain(d=16, H=2, ffn=32, L=1, d_proj=8), [(SPIKE, dl)], steps=5, warmup=1, accum=3)
    assert len(hist) == 5 and len(drawn) == 5 * 3 * 4


def test_bf16_precision(tmp_path):
    """U36: bf16 autocast runs the same loop, keeps float32 weights and finite float32 losses; unknown names raise."""
    write_synthetic(tmp_path, (3, 8), n_windows=8)
    dl = DataLoader(SpikeWindows(read_windows(tmp_path, SYNTHETIC)), batch_size=4, shuffle=True, collate_fn=collate)
    torch.manual_seed(0)
    m = IBrain(d=16, H=2, ffn=32, L=1, d_proj=8)
    hist = pretrain(m, [(SPIKE, dl)], steps=3, warmup=1, precision="bf16")
    assert len(hist) == 3 and all(math.isfinite(h["rec"]) and math.isfinite(h["align"]) for h in hist)
    assert all(p.dtype == torch.float32 for p in m.parameters())
    try:
        pretrain(m, [(SPIKE, dl)], steps=1, warmup=1, precision="fp16")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown precision must raise")


def test_pack_budget():
    """U33: every window lands in exactly one micro-batch, sorted by unit count, within the channel-slot budget."""
    torch.manual_seed(0)
    n = torch.cat([torch.randint(1, 120, (250,)), torch.tensor([1734, 1700, 900, 600, 512, 300])])
    chunks = pack(n, budget=2048)
    assert sorted(torch.cat(chunks).tolist()) == list(range(len(n)))
    for c in chunks:
        assert len(c) * n[c].max() <= 2048 or len(c) == 1
    assert [int(n[c].max()) for c in chunks] == sorted((int(n[c].max()) for c in chunks), reverse=True)
    assert len(pack(n, None)) == 1


def test_packed_gradient_equals_whole_batch(tmp_path, monkeypatch):
    """U33: micro-batches with the loss weights of batch_losses give the whole-batch gradient. Masks and views are made
    deterministic per window (same count as U15/U16) so both computations see the same draws; dropout is off."""
    def mask_fn(valid, S):
        B, C = valid.shape
        tok = valid[:, :, None].expand(B, C, S).reshape(B, C * S)
        rank = torch.arange(C * S).expand(B, -1).masked_fill(~tok, C * S)
        return (rank < tok.sum(1, keepdim=True) // 2).view(B, C, S)

    def views_fn(valid):
        k = torch.arange(valid.shape[1])
        return valid & (k % 3 != 1), valid & (k % 3 != 2)

    monkeypatch.setattr(pt, "sample_mask", mask_fn)
    monkeypatch.setattr(pt, "sample_views", views_fn)
    torch.manual_seed(0)
    ns = [3, 40, 5, 17, 2, 40, 9, 1]
    x = torch.poisson(torch.full((len(ns), max(ns), 10, 5), 0.5))
    valid = valid_of(ns, max(ns))
    x[~valid] = 0.0
    m = IBrain(d=16, H=2, ffn=32, L=1, d_proj=8, p=0.0)

    def grads(budget):
        m.zero_grad(set_to_none=True)
        losses = batch_losses(m, x, valid, SPIKE, budget=budget)
        return losses, [q.grad.clone() for q in m.parameters() if q.grad is not None]

    (r0, a0), g0 = grads(None)
    (r1, a1), g1 = grads(45)  # several micro-batches, the two 40-unit windows each alone or paired
    assert len(pack(valid.sum(1), 45)) > 2
    assert math.isclose(r0, r1, rel_tol=1e-5) and math.isclose(a0, a1, rel_tol=1e-5)
    assert len(g0) == len(g1) and all(torch.allclose(u, v, atol=1e-6, rtol=1e-4) for u, v in zip(g0, g1))


def test_grad_checkpoint_same_gradient():
    """U37: activation checkpointing changes memory and time only. Same loss and gradient with dropout on, because
    the recomputation restores the RNG state of the forward."""
    torch.manual_seed(0)
    ns = [3, 7, 5]
    x, valid = torch.poisson(torch.full((3, 7, 10, 5), 0.5)), valid_of(ns, 7)
    m = IBrain(d=16, H=2, ffn=32, L=2, d_proj=8, p=0.1).train()

    def run(ckpt):
        m.grad_checkpoint = ckpt
        m.zero_grad(set_to_none=True)
        torch.manual_seed(1)
        rec, align = pt.step_losses(m, x, valid, SPIKE)
        (rec + align).backward()
        return float((rec + align).detach()), [q.grad.clone() for q in m.parameters() if q.grad is not None]

    l0, g0 = run(False)
    l1, g1 = run(True)
    assert math.isclose(l0, l1, rel_tol=1e-6)
    assert all(torch.allclose(u, v, atol=1e-6, rtol=1e-5) for u, v in zip(g0, g1))
