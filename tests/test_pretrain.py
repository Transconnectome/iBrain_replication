"""M3 verification: mask, channel views, lr schedule, loss decrease on synthetic shards."""
import math

import pytest
import torch
from torch.utils.data import DataLoader

from ibrain.data_spike import SYNTHETIC, SpikeWindows, collate, read_windows, write_synthetic
from ibrain.model import IBrain, IEEG, SPIKE, masked_poisson_nll
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


def test_script_ieeg_only(tmp_path):
    """U42: --ieeg-only mirrors --spike-only. Every step is an iEEG step and no spike data is read; with a map-style
    iEEG source (synthetic) epochs count passes over the iEEG loader. It excludes --spike-only and needs data.ieeg."""
    import json
    import subprocess
    import sys
    from pathlib import Path
    import yaml
    root = Path(__file__).resolve().parent.parent
    cfg = yaml.safe_load((root / "configs/tiny.yaml").read_text())
    cfg["model"].update(d=16, H=2, ffn=32, L=1, d_proj=8)
    cfg["pretrain"].update(steps=None, epochs=2, warmup=1, ckpt_every=1000)

    def run(c, *extra):
        (tmp_path / "c.yaml").write_text(yaml.safe_dump(c))
        out = tmp_path / f"run{len(list(tmp_path.iterdir()))}"
        r = subprocess.run([sys.executable, str(root / "scripts/pretrain.py"), "--config", str(tmp_path / "c.yaml"),
                            "--out", str(out), "--device", "cpu", *extra], capture_output=True, text=True, cwd=root)
        return r, out

    r, out = run(cfg, "--ieeg-only")
    assert r.returncode == 0, r.stderr[-1500:]
    meta = json.loads((out / "meta.json").read_text())
    log = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
    ieeg_batches = -(-3 * 16 // cfg["pretrain"]["batch_size"])  # SyntheticIEEG: 3 sessions x 16 windows
    assert meta["data"]["mode"] == "ieeg_only" and meta["data"]["spike_windows"] is None
    assert meta["data"]["steps"] == len(log) == 2 * ieeg_batches and all(h["sig"] == IEEG for h in log)
    assert not (out / "data").exists()  # the synthetic spike shards were never written
    r, _ = run(cfg, "--ieeg-only", "--spike-only")
    assert r.returncode != 0 and "not allowed with" in r.stderr
    r, _ = run({**cfg, "data": {**cfg["data"], "ieeg": None}}, "--ieeg-only")
    assert r.returncode != 0 and "needs data.ieeg" in r.stderr


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


def mask_fn(valid, S):
    """Deterministic per window: the first half of the valid tokens (same count as U15)."""
    B, C = valid.shape
    tok = valid[:, :, None].expand(B, C, S).reshape(B, C * S)
    rank = torch.arange(C * S).expand(B, -1).masked_fill(~tok, C * S)
    return (rank < tok.sum(1, keepdim=True) // 2).view(B, C, S)


def views_fn(valid):
    """Deterministic per window channel views (each drops every third channel, at different offsets)."""
    k = torch.arange(valid.shape[1])
    return valid & (k % 3 != 1), valid & (k % 3 != 2)


def ragged_batch():
    torch.manual_seed(0)
    ns = [3, 40, 5, 17, 2, 40, 9, 1]
    x = torch.poisson(torch.full((len(ns), max(ns), 10, 5), 0.5))
    valid = valid_of(ns, max(ns))
    x[~valid] = 0.0
    return x, valid


def grads_of(m, fn):
    m.zero_grad(set_to_none=True)
    out = fn()
    return out, [q.grad.clone() for q in m.parameters() if q.grad is not None]


def same_grads(g0, g1, atol=1e-6):
    return len(g0) == len(g1) and all(torch.allclose(u, v, atol=atol, rtol=1e-4) for u, v in zip(g0, g1))


def close_grads(g0, g1, rel=1e-3):
    """Per tensor, the error norm relative to the gradient norm. For BN heads: dividing by the std of a few windows
    magnifies float32 rounding of r element-wise, while a wrong computation would be off by order 1."""
    return len(g0) == len(g1) and all((u - v).norm() <= rel * u.norm() + 1e-8 for u, v in zip(g0, g1))


def test_packed_gradient_equals_whole_batch(tmp_path, monkeypatch):
    """U33: micro-batches with the loss weights of batch_losses give the whole-batch gradient. Masks and views are made
    deterministic per window (same count as U15/U16) so both computations see the same draws; dropout is off."""
    monkeypatch.setattr(pt, "sample_mask", mask_fn)
    monkeypatch.setattr(pt, "sample_views", views_fn)
    x, valid = ragged_batch()
    m = IBrain(d=16, H=2, ffn=32, L=1, d_proj=8, p=0.0)
    (r0, a0, q0), g0 = grads_of(m, lambda: batch_losses(m, x, valid, SPIKE, budget=None))
    (r1, a1, q1), g1 = grads_of(m, lambda: batch_losses(m, x, valid, SPIKE, budget=45))  # 40-unit windows alone
    assert len(pack(valid.sum(1), 45)) > 2
    assert math.isclose(r0, r1, rel_tol=1e-5) and math.isclose(a0, a1, rel_tol=1e-5)
    assert all(math.isclose(q0[k], q1[k], rel_tol=1e-4, abs_tol=1e-6) for k in q0)  # metrics over the whole batch
    assert 0 < q0["q_std"] < 8 ** -0.5 + 0.1 and 0 < q0["r_std"]
    assert same_grads(g0, g1)


def head_bn_reference(m, x, valid):
    """Whole batch at once with the BN head on all windows: the gradient _bn_align must reproduce."""
    rec = pt.rec_loss(m, x, valid, SPIKE)
    v1, v2 = pt.sample_views(valid)
    q1, p1 = m.align(pt.view_repr(m, x, v1, SPIKE))
    q2, p2 = m.align(pt.view_repr(m, x, v2, SPIKE))
    align = pt.simsiam_loss(p1, q2, p2, q1)
    (rec + align).backward()
    return rec.item(), align.item()


@pytest.mark.parametrize("group", [None, 4])
def test_head_bn_packed_gradient_equals_whole_batch(monkeypatch, group):
    """Collapse ablation RB: with BatchNorm in the head, the two-pass L_align over micro-batches gives the gradient of
    the whole batch at once (BN over all windows, or per group of windows in batch order as per GPU), with
    deterministic masks and views and no dropout."""
    monkeypatch.setattr(pt, "sample_mask", mask_fn)
    monkeypatch.setattr(pt, "sample_views", views_fn)
    x, valid = ragged_batch()
    m = IBrain(d=16, H=2, ffn=32, L=1, d_proj=8, p=0.0, head_bn=True, head_bn_group=group).train()
    (r0, a0), g0 = grads_of(m, lambda: head_bn_reference(m, x, valid))
    (r1, a1, _), g1 = grads_of(m, lambda: batch_losses(m, x, valid, SPIKE, budget=45))
    # summed micro-batch by micro-batch, so float32 rounding differs (BN over 4 windows magnifies it to about 1e-4)
    assert math.isclose(r0, r1, rel_tol=1e-5) and math.isclose(a0, a1, rel_tol=1e-5) and close_grads(g0, g1)


def test_head_bn_replays_dropout():
    """The second pass of _bn_align replays the RNG state of the first, so with dropout on and one micro-batch it
    matches the whole-batch computation drawn in the same order (rec mask, views, view 1, view 2)."""
    x, valid = ragged_batch()
    m = IBrain(d=16, H=2, ffn=32, L=1, d_proj=8, p=0.2, head_bn=True).train()

    def ref():
        torch.manual_seed(3)
        return head_bn_reference(m, x, valid)

    def ours():
        torch.manual_seed(3)
        return batch_losses(m, x, valid, SPIKE, budget=None)[:2]

    (r0, a0), g0 = grads_of(m, ref)
    (r1, a1), g1 = grads_of(m, ours)
    assert math.isclose(r0, r1, rel_tol=1e-5) and math.isclose(a0, a1, rel_tol=1e-5) and same_grads(g0, g1)


def test_model_variants():
    """Collapse ablation RC/RD: no final LayerNorm, max pooling over valid tokens only (padding never wins)."""
    x, valid = ragged_batch()
    m = IBrain(d=16, H=2, ffn=32, L=1, d_proj=8, p=0.0, final_norm=False, pool="max").eval()
    assert isinstance(m.backbone.norm, torch.nn.Identity)
    u = m.encode(x, valid)
    r = m.pool(u, valid)
    u2 = u.clone()
    u2[~valid] = 1e6  # padded tokens
    assert torch.equal(m.pool(u2, valid), r)
    assert torch.equal(r[0], u[0, :3].amax((0, 1)))
    with pytest.raises(ValueError):
        IBrain(pool="cls")


@pytest.mark.parametrize("head_bn", [False, True])
def test_collapse_metrics_logged_and_snapshots_kept(tmp_path, head_bn):
    write_synthetic(tmp_path / "data", (3, 8), n_windows=8)
    dl = DataLoader(SpikeWindows(read_windows(tmp_path / "data", SYNTHETIC)), batch_size=4, shuffle=True,
                    collate_fn=collate)
    m = IBrain(d=16, H=2, ffn=32, L=1, d_proj=8, head_bn=head_bn, head_bn_group=2 if head_bn else None)
    hist = pretrain(m, [(SPIKE, dl)], steps=2, warmup=1, out_dir=tmp_path / "run", keep_every=1)
    assert all(0 <= h["q_std"] < 1 and 0 <= h["r_std"] < 1 and -2 <= h["d_gap"] <= 2 for h in hist)
    assert sorted(p.name for p in (tmp_path / "run").glob("ckpt_*.pt")) == ["ckpt_0000001.pt", "ckpt_0000002.pt"]


def test_group_batchnorm():
    """Per-GPU BN: each group of rows is normalized with its own statistics; running statistics come from the first
    group only (rank 0); eval and a batch of one group behave as BatchNorm1d; a last group of one row joins the previous."""
    from ibrain.model import GroupBatchNorm1d
    torch.manual_seed(0)
    x = torch.randn(9, 3) * torch.tensor([1.0, 5.0, 0.2]) + 3
    g, ref = GroupBatchNorm1d(3, group=4).train(), torch.nn.BatchNorm1d(3).train()
    y = g(x)
    expect = torch.cat([torch.nn.functional.batch_norm(c, None, None, training=True) for c in (x[:4], x[4:])])
    assert torch.allclose(y, expect, atol=1e-5)  # groups 4 and 4 + 1
    ref(x[:4])
    assert torch.allclose(g.running_mean, ref.running_mean) and torch.allclose(g.running_var, ref.running_var)
    assert torch.allclose(g.eval()(x), ref.eval()(x))
    assert torch.allclose(GroupBatchNorm1d(3, group=16).train()(x), torch.nn.BatchNorm1d(3).train()(x))


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="compile is used on GPU (Triton); the CPU nodes' g++ "
                    "has no -std=c++20 for inductor's CPU backend")
def test_compile_same_as_eager():
    """U40: compile_parts changes speed only. Same weights and seed, float32, dropout on, a ragged batch cut into
    several micro-batches (so compiled code sees several shapes) and a BN head per group (so the dropout replay of
    the two-pass L_align runs through compiled blocks): same losses and gradients as eager. Runs on GPU."""
    import copy
    x, valid = (t.cuda() for t in ragged_batch())
    eager = IBrain(d=16, H=2, ffn=32, L=2, d_proj=8, p=0.1, head_bn=True, head_bn_group=4).cuda().train()
    comp = copy.deepcopy(eager).compile_parts()

    def run(m):
        torch.manual_seed(7)
        return batch_losses(m, x, valid, SPIKE, budget=45)[:2]

    (r0, a0), g0 = grads_of(eager, lambda: run(eager))
    (r1, a1), g1 = grads_of(comp, lambda: run(comp))
    assert len(pack(valid.sum(1), 45)) > 2
    assert math.isclose(r0, r1, rel_tol=1e-4) and math.isclose(a0, a1, rel_tol=1e-4) and close_grads(g0, g1)
    assert dict(eager.named_parameters()).keys() == dict(comp.named_parameters()).keys()  # checkpoints unchanged
