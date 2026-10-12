"""M4 checks on synthetic NWB files that copy the layouts of the real downstream data:
NLB (hand_vel in processing/behavior, starting_time + rate), Perich (cursor_vel in processing/behavior/Velocity,
timestamps linked from Position/cursor_pos), and NLB test files (no behavior)."""
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import yaml
from pynwb import NWBHDF5IO, NWBFile, TimeSeries
from pynwb.behavior import BehavioralTimeSeries, Position, SpatialSeries

from ibrain.data_nwb import MissingSeries, bin_counts, patch_labels, read_nwb, window_starts

ROOT = Path(__file__).resolve().parent.parent


def write_nwb(path, layout="nlb", n_trials=40, n_units=12, trial_len=2.3, gap=0.5, seed=0):
    """Synthetic session with velocity-tuned units. Velocity: 2-D Ornstein-Uhlenbeck at 1 kHz (time constant 1 s,
    std 1), stored ×100 as in mm/s. Rates 20 softplus(v W + b) Hz, spikes Bernoulli per 1 ms.
    Returns (ts, stored velocity, list of spike time arrays)."""
    rng = np.random.default_rng(seed)
    ts = np.arange(0, n_trials * (trial_len + gap), 0.001)
    noise = rng.normal(0, 0.045, (len(ts), 2))
    v = np.zeros((len(ts), 2))
    for t in range(1, len(ts)):
        v[t] = 0.999 * v[t - 1] + noise[t]
    W = rng.normal(0, 1, (2, n_units))
    rate = 20 * np.log1p(np.exp(v @ W + rng.normal(0, 0.5, n_units)))
    fired = rng.random(rate.shape) < rate * 0.001
    spikes = [ts[fired[:, c]] for c in range(n_units)]

    nwb = NWBFile(session_description="synthetic", identifier=path.stem,
                  session_start_time=datetime(2020, 1, 1, tzinfo=timezone.utc))
    for st in spikes:
        nwb.add_unit(spike_times=st)
    # NLB: every 5th trial has split "none"; Perich: every 5th trial failed ("F"). trial_mask must drop them.
    col = {"nlb": "split", "perich": "result"}.get(layout)
    if col:
        nwb.add_trial_column(name=col, description="synthetic")
    for i in range(n_trials):
        extra = {col: ("none" if col == "split" else "F") if i % 5 == 4 else ("train" if col == "split" else "R")} if col else {}
        nwb.add_trial(start_time=i * (trial_len + gap), stop_time=i * (trial_len + gap) + trial_len, **extra)
    if layout != "test":
        beh = nwb.create_processing_module("behavior", "behavior")
        if layout == "nlb":
            beh.add(TimeSeries(name="hand_vel", data=100 * v, starting_time=0.0, rate=1000.0, unit="mm/s"))
        else:
            pos = SpatialSeries(name="cursor_pos", data=np.cumsum(v, 0) * 1e-3, timestamps=ts, reference_frame="screen")
            beh.add(Position(spatial_series=pos))
            beh.add(BehavioralTimeSeries(name="Velocity",
                                         time_series=TimeSeries(name="cursor_vel", data=100 * v, timestamps=pos,
                                                                unit="cm/s")))
    with NWBHDF5IO(str(path), "w") as io:
        io.write(nwb)
    return ts, 100 * v, spikes


def test_windows_and_bins():
    """U21 windows and left-closed 20 ms bins, on hand-made numbers."""
    t0, k = window_starts([0.0, 3.0, 5.0], [2.5, 3.7, 6.0])
    assert t0.tolist() == [0.0, 1.0, 5.0] and k.tolist() == [0, 0, 2]  # 2.5 s -> 2 windows, 0.7 s -> none, 1.0 s -> 1
    c = bin_counts([np.array([0.0, 0.019, 0.02, 0.999, 1.0]), np.array([5.5])], t0)
    assert c.shape == (3, 50, 2) and c.dtype == np.uint32
    assert (c[0, 0, 0], c[0, 1, 0], c[0, 49, 0], c[1, 0, 0]) == (2, 1, 1, 1)  # [0, 0.02) holds 0.0 and 0.019
    assert c[:, :, 0].sum() == 5 and c[2, 25, 1] == 1 and c[:, :, 1].sum() == 1


@pytest.mark.parametrize("layout,name", [("nlb", "hand_vel"), ("perich", "cursor_vel")])
def test_read_nwb_layouts(tmp_path, layout, name):
    f = tmp_path / f"sub-X_ses-{layout}.nwb"
    ts, vel, spikes = write_nwb(f, layout, n_trials=6)
    ws = read_nwb(f, name)
    assert len(ws) == 10  # 6 trials of 2.3 s -> 2 windows each, trial 4 dropped by trial_mask (U29)
    assert "4" not in {w["interval_id"] for w in ws}
    assert {w["session_id"] for w in ws} == {f.stem} and ws[0]["counts"].shape == (50, 12)
    assert [w["interval_id"] for w in ws[:4]] == ["0", "0", "1", "1"]
    for w in ws[:3]:
        a = w["start_seconds"]
        expect = [((st >= a) & (st < a + 1.0)).sum() for st in spikes]
        assert w["counts"].sum(0).tolist() == expect  # every spike in the window is counted once
        assert np.allclose(w["vel"], patch_labels(a, ts, vel))  # labels from this file's own clock


def test_missing_behavior(tmp_path):
    f = tmp_path / "sub-X_ses-test.nwb"
    write_nwb(f, "test", n_trials=3)
    with pytest.raises(MissingSeries):
        read_nwb(f, "hand_vel")
    assert len(read_nwb(f)) == 6 and "vel" not in read_nwb(f)[0]  # spikes still readable without labels


def test_finetune_script_on_nwb_folder(tmp_path):
    """scripts/finetune.py on a folder with two labeled sessions and one test file: the test file is skipped,
    both sessions are used, and velocity is decodable from the windows."""
    d = tmp_path / "data"
    d.mkdir()
    for i in range(2):
        write_nwb(d / f"sub-X_ses-{i}_desc-train.nwb", "nlb", n_trials=40, seed=i)
    write_nwb(d / "sub-X_ses-0_desc-test.nwb", "test", n_trials=5, seed=9)
    cfg = yaml.safe_load((ROOT / "configs/tiny.yaml").read_text())
    cfg["finetune"].update(seeds=[0], epochs=10)
    cfg["finetune"]["datasets"]["nlb"] = {"format": "nwb", "root": str(d), "behavior": "hand_vel"}
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg))
    out = tmp_path / "ft"
    r = subprocess.run([sys.executable, str(ROOT / "scripts/finetune.py"), "--config", str(tmp_path / "cfg.yaml"),
                        "--out", str(out), "--dataset", "nlb", "--device", "cpu"],
                       capture_output=True, text=True, cwd=ROOT)
    print("\n" + r.stdout[-600:])
    assert r.returncode == 0, r.stderr[-2000:]
    assert "skip sub-X_ses-0_desc-test.nwb" in r.stdout and "sessions=2" in r.stdout
    assert not (out / "synthetic").exists()  # only the dataset that was asked for
    res = json.loads((out / "nlb" / "results.json").read_text())
    meta = json.loads((out / "nlb" / "meta.json").read_text())
    assert len(meta["nwb_files"]) == 2 and meta["behavior"] == "hand_vel"
    # ridge well above 0 means labels line up with spikes (misaligned labels give R² near 0); scratch just learns
    assert res["arms"]["ridge"]["mean"] > 0.3 and res["arms"]["scratch"]["mean"] > 0.0
