"""M4 verification (synthetic): patch labels, trial split, head shape and padding invariance, ridge baseline, scratch regression R² > 0, frozen.
M5 (U35): AUC, window logits, the iEEG path from a joint pretraining checkpoint, Brain Treebank through the script."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from ibrain.data_ieeg import SyntheticIEEG
from ibrain.data_nwb import attach_velocity, patch_labels
from ibrain.data_spike import SYNTHETIC, SpikeWindows, collate, read_windows, write_synthetic
from ibrain.data_treebank import TreebankWindows, collate_treebank, read_trial
from ibrain.finetune import (HEADS, AttnPoolHead, Regressor, WindowLogit, auc, fit_arm, fit_classifier_arm,
                             ieeg_steps, r2, ridge_r2, split_trials, synthetic_labeled, train_regressor)
from ibrain.model import IEEG, SPIKE, IBrain
from ibrain.pretrain import pretrain
from ibrain.repro import load_checkpoint
from test_treebank import POPT, SEL, write_treebank

ROOT = Path(__file__).resolve().parent.parent
TINY = dict(d=32, H=4, ffn=64, L=1, d_proj=16)


def test_patch_labels():
    ts = 10.0 + np.arange(0, 1.2, 0.001)  # 1 kHz, window [10, 11) + tail
    vel = np.stack([ts, -2 * ts], 1)
    y = patch_labels(10.0, ts, vel)
    assert y.shape == (10, 2)
    centers = 10.0 + 0.1 * np.arange(10) + 0.0495  # mean of 100 samples
    assert np.allclose(y[:, 0], centers, atol=1e-9) and np.allclose(y[:, 1], -2 * centers)
    assert np.isnan(patch_labels(10.0, ts[:500], vel[:500])[5:]).all()  # patches with no samples are NaN
    w = attach_velocity([{"start_seconds": 10.0}, {"start_seconds": 10.6}], ts, vel)
    assert len(w) == 1 and w[0]["start_seconds"] == 10.0  # NaN windows are dropped


def test_split_trials_fixed_and_disjoint():
    ws = synthetic_labeled(50, 5)
    a, b = split_trials(ws, 0.8, seed=0)
    assert len(a) == 40 and len(b) == 10
    assert not {w["interval_id"] for w in a} & {w["interval_id"] for w in b}
    a2, _ = split_trials(ws, 0.8, seed=0)
    assert [w["interval_id"] for w in a2] == [w["interval_id"] for w in a]  # same seed gives the same split (M4)
    assert [w["interval_id"] for w in split_trials(ws, 0.8, seed=1)[0]] != [w["interval_id"] for w in a]


def test_heads_shape_and_padding():
    torch.manual_seed(0)
    m = IBrain(**TINY).eval()
    x = torch.poisson(torch.full((2, 6, 10, 5), 2.0))
    valid = torch.ones(2, 6, dtype=torch.bool)
    valid[:, 4:] = False
    for name, H in HEADS.items():
        reg = Regressor(m, H(TINY["d"], C=6)).eval()
        y = reg(x, valid)
        assert y.shape == (2, 10, 2), name
        x2 = x.clone()
        x2[:, 4:] = 7.0
        assert torch.allclose(y, reg(x2, valid), atol=1e-5), name  # padding channel values do not affect the output
    assert r2(torch.tensor([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]), torch.tensor([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])) == 1.0


def test_ridge_recovers_synthetic():
    ws = synthetic_labeled(200, 30, seed=0)
    tr, te = split_trials(ws, 0.8, seed=0)
    score, lam = ridge_r2(tr, te)
    print(f"\nridge R2 {score:.3f} lam {lam}")
    assert score > 0.5


def test_scratch_and_frozen_arms():
    ws = synthetic_labeled(200, 30, seed=0)
    tr, te = split_trials(ws, 0.8, seed=0)
    score, hist = fit_arm(tr, te, TINY, seed=0, epochs=15, lr=1e-3)
    print(f"\nscratch R2 {score:.3f} loss {hist[0]:.3f} -> {hist[-1]:.3f}")
    assert hist[-1] < hist[0] and score > 0.0
    s2, _ = fit_arm(tr, te, TINY, seed=0, epochs=15, lr=1e-3)
    assert s2 == score  # same seed gives the same result (U26)
    frozen, _ = fit_arm(tr, te, TINY, seed=0, frozen=True, epochs=3, lr=1e-3)
    assert np.isfinite(frozen)


def test_split_keys_by_session():
    """U11: trial numbers repeat across sessions, so a trial is (session_id, interval_id)."""
    ws = [{**w, "session_id": sess} for sess in ("a", "b") for w in synthetic_labeled(10, 5)]
    tr, te = split_trials(ws, 0.8, seed=0)
    keys = lambda part: {(w["session_id"], w["interval_id"]) for w in part}  # noqa: E731
    assert len(keys(tr)) == 16 and len(keys(te)) == 4  # 20 trials in total, not 10
    assert not keys(tr) & keys(te)


def test_label_scale_does_not_matter():
    """U27: labels are z-scored with train statistics, so raw velocity units (mm/s, cm/s) do not change R²."""
    ws = synthetic_labeled(200, 30, seed=0)
    tr, te = split_trials(ws, 0.8, seed=0)
    scale = lambda part, k: [{**w, "vel": w["vel"] * k} for w in part]  # noqa: E731
    r1, _ = fit_arm(tr, te, TINY, seed=0, epochs=15, lr=1e-3)
    r300, _ = fit_arm(scale(tr, 300), scale(te, 300), TINY, seed=0, epochs=15, lr=1e-3)
    print(f"\nscratch R2 x1 {r1:.3f}, x300 {r300:.3f}")
    assert r300 > 0.1 and abs(r300 - r1) < 0.02


def test_ridge_is_per_session():
    """Units are different neurons in every session. Session b has the same counts as session a but the opposite
    velocity: separate decoders fit both, a decoder shared across sessions cancels out (R² near 0). Different unit
    counts must not crash either (the Perich T-CO run has 6 sessions)."""
    a = [{**w, "session_id": "a"} for w in synthetic_labeled(150, 20, seed=1)]
    flipped = a + [{**w, "session_id": "b", "vel": -w["vel"]} for w in a]
    diff = [{**w, "session_id": "a"} for w in synthetic_labeled(150, 30, seed=1)] + \
           [{**w, "session_id": "b"} for w in synthetic_labeled(150, 20, seed=2)]
    for ws in (flipped, diff):
        tr, te = split_trials(ws, 0.8, seed=0)
        score, _ = ridge_r2(tr, te)
        assert score > 0.8, score


# ---------------- M5: Brain Treebank classification (U35) ----------------

def joint_checkpoint(path, sigs=(SPIKE, IEEG)):
    """A few pretraining steps on synthetic spikes and iEEG (or spikes only) -> path/final.pt."""
    write_synthetic(path / "data", (3, 8), n_windows=8)
    torch.manual_seed(0)
    loaders = {SPIKE: DataLoader(SpikeWindows(read_windows(path / "data", SYNTHETIC)), 4, shuffle=True,
                                 collate_fn=collate),
               IEEG: DataLoader(SyntheticIEEG((4, 5), n_windows=8), 4, shuffle=True, collate_fn=collate)}
    pretrain(IBrain(**TINY), [(s, loaders[s]) for s in sigs], steps=6, warmup=2, out_dir=path / "run")
    return path / "run" / "final.pt"


def test_auc_matches_pairs():
    rng = np.random.default_rng(0)
    s, y = rng.integers(0, 5, 60).astype(float), rng.integers(0, 2, 60)  # many ties
    pos, neg = s[y == 1], s[y == 0]
    pairs = (pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean()
    assert abs(auc(s, y) - pairs) < 1e-12 and auc([1, 2, 3], [0, 1, 1]) == 1.0
    with pytest.raises(ValueError):
        auc([1, 2], [1, 1])


def test_window_logit_shape_and_padding():
    """One logit per window for every U8 head, unaffected by padded channels; the iEEG input is (B, C, 10, 50)."""
    torch.manual_seed(0)
    m = IBrain(**TINY).eval()
    x = torch.randn(2, 6, 10, 50)
    valid = torch.ones(2, 6, dtype=torch.bool)
    valid[:, 4:] = False
    for name, H in HEADS.items():
        clf = Regressor(m, WindowLogit(H(TINY["d"], C=6, out=1)), sig=IEEG).eval()
        y = clf(x, valid)
        assert y.shape == (2,), name
        x2 = x.clone()
        x2[:, 4:] = 7.0
        assert torch.allclose(y, clf(x2, valid), atol=1e-5), name


def test_ieeg_finetune_continues_pretraining(tmp_path):
    """Fine-tuning a joint checkpoint on iEEG uses the iEEG encoder and iEEG type embedding of pretraining, with the
    shared time embedding and backbone. The spike encoder, spike type embedding, decoders and SimSiam head get no
    gradient and stay exactly as in the checkpoint (AdamW skips them, U24); frozen leaves the whole model unchanged."""
    ckpt = joint_checkpoint(tmp_path)
    assert ieeg_steps(ckpt) == 3  # 1:1 alternation over 6 steps
    state = load_checkpoint(ckpt)["model"]
    rng = np.random.default_rng(0)
    ws = [{"wave": rng.normal(0, 1, (5, 500)).astype(np.float32), "label": i % 2} for i in range(16)]
    dl = DataLoader(TreebankWindows(ws), 8, shuffle=True, collate_fn=collate_treebank)
    for frozen in (False, True):
        m = IBrain(**TINY)
        m.load_state_dict(state)
        clf = Regressor(m, WindowLogit(AttnPoolHead(TINY["d"], out=1)), frozen, sig=IEEG)
        train_regressor(clf, dl, 2, lr=1e-2, loss_fn=F.binary_cross_entropy_with_logits)
        same = {k for k, v in m.state_dict().items() if torch.equal(v, state[k])}
        moved = set(state) - same
        if frozen:
            assert not moved
            continue
        assert all(k.startswith(("enc.0.", "dec.", "proj.", "pred.", "type_emb.0")) for k in same), sorted(same)
        assert {"type_emb.1", "time_emb"} <= moved
        assert any(k.startswith("enc.1.") for k in moved) and any(k.startswith("backbone.") for k in moved)
        assert all(k in same for k in state if k.startswith(("enc.0.", "dec.", "proj.", "pred.", "type_emb.0")))


def test_ieeg_steps_of_spike_only_checkpoint(tmp_path):
    assert ieeg_steps(joint_checkpoint(tmp_path, sigs=(SPIKE,))) == 0


def test_classifier_learns_synthetic_treebank(tmp_path):
    write_treebank(tmp_path, trials=("trial000", "trial001"))
    train = read_trial(tmp_path, "sub_1", "trial000", "speech", electrodes=SEL, **POPT)
    test = read_trial(tmp_path, "sub_1", "trial001", "speech", electrodes=SEL, **POPT)
    score, hist = fit_classifier_arm(train, test, TINY, seed=0, epochs=15, lr=1e-3)
    print(f"\nscratch speech AUC {score:.3f} loss {hist[0]:.3f} -> {hist[-1]:.3f}")
    assert hist[-1] < hist[0] and score > 0.7
    assert fit_classifier_arm(train, test, TINY, seed=0, epochs=15, lr=1e-3)[0] == score  # U26


def test_finetune_script_on_treebank(tmp_path):
    """scripts/finetune.py --dataset on a synthetic Brain Treebank with a joint checkpoint: three neural arms, no
    ridge, per-subject AUC and the subject mean per seed."""
    ckpt = joint_checkpoint(tmp_path)
    write_treebank(tmp_path / "tb")
    cfg = yaml.safe_load((ROOT / "configs/tiny.yaml").read_text())
    cfg["model"] = {**cfg["model"], **TINY}
    cfg["finetune"].update(seeds=[0, 1], epochs=3)
    cfg["finetune"]["datasets"]["tb"] = {"format": "treebank", "root": str(tmp_path / "tb"), "split": "heldout",
                                         "tasks": ["speech"], "subjects": ["sub_1"], "stat_seconds": 30,
                                         "notch_hz": [60], "cache_dir": str(tmp_path / "cache")}
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg))
    out = tmp_path / "ft"
    r = subprocess.run([sys.executable, str(ROOT / "scripts/finetune.py"), "--config", str(tmp_path / "cfg.yaml"),
                        "--ckpt", str(ckpt), "--out", str(out), "--dataset", "tb", "--device", "cpu"],
                       capture_output=True, text=True, cwd=ROOT)
    print("\n" + r.stdout[-600:])
    assert r.returncode == 0, r.stderr[-2000:]
    assert "WARNING" not in r.stdout
    res = json.loads((out / "tb" / "results.json").read_text())
    meta = json.loads((out / "tb" / "meta.json").read_text())
    assert meta["ckpt_ieeg_steps"] == 3 and meta["split"] == "heldout"
    arms = res["tasks"]["speech"]["arms"]
    assert set(arms) == {"finetune", "finetune_frozen", "scratch"}
    sub = res["tasks"]["speech"]["subjects"]["sub_1"]
    for name, v in arms.items():
        assert v["auc_by_seed"] == sub["arms"][name]["auc"]  # one subject: the subject mean is that subject
        assert all(0.0 <= s <= 1.0 for s in v["auc_by_seed"]) and len(v["auc_by_seed"]) == 2
    assert len(list((tmp_path / "cache").glob("*.npy"))) == 2  # both trials cached once for the task
    assert res["complete"] and all(v["n_subjects"] == 1 for v in arms.values())
    for f in (tmp_path / "cache").glob("*.json"):  # the reader options reach the reader
        c = json.loads(f.read_text())
        assert c["stat_seconds"] == 30 and c["notch_hz"] == [60]


def test_finetune_script_rejects_unknown_option(tmp_path):
    cfg = yaml.safe_load((ROOT / "configs/tiny.yaml").read_text())
    cfg["finetune"]["datasets"]["tb"] = {"format": "treebank", "root": str(tmp_path), "stat_second": 675}
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg))
    r = subprocess.run([sys.executable, str(ROOT / "scripts/finetune.py"), "--config", str(tmp_path / "cfg.yaml"),
                        "--out", str(tmp_path / "ft"), "--dataset", "tb", "--device", "cpu"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode != 0 and "unknown option(s) ['stat_second']" in r.stderr
    assert not (tmp_path / "ft").exists()


def test_finetune_script_skips_variants_by_default(tmp_path):
    """An entry with variant: true (treebank_popt in paper.yaml) runs only when --dataset names it."""
    cfg = yaml.safe_load((ROOT / "configs/tiny.yaml").read_text())
    cfg["finetune"]["datasets"] = {"tb_popt": {"format": "treebank", "variant": True, "root": str(tmp_path),
                                               "reref": "laplacian"}}
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg))
    r = subprocess.run([sys.executable, str(ROOT / "scripts/finetune.py"), "--config", str(tmp_path / "cfg.yaml"),
                        "--out", str(tmp_path / "ft"), "--device", "cpu"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode != 0 and "unknown or no dataset" in r.stderr
