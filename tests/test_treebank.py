"""M5 downstream checks on a synthetic Brain Treebank with the real layout: per-electrode datasets data/electrode_i at
2048 Hz (plus an unlabeled one, as in sub_1), electrode_labels.json with '*' names, corrupted_elec.json,
subject_metadata, subject_timings with pause/unpause and end/beginning rows, and transcripts/<movie>/features.csv with
a NaN row. The PopT rules (nearest-row alignment, quartiles, non-word intervals, random.seed(42) balancing) are checked
against direct re-implementations of the PopT code."""
import json
import random

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
from scipy.signal import resample_poly

from ibrain.data_ieeg import FS, channel_normalize, channel_stats
from ibrain import data_treebank
from ibrain.data_treebank import (NOTCH_HZ, TASKS, TB_FS, TreebankWindows, collate_treebank, neighbours, nonword_centers,
                                  quartile_examples, read_trial, read_trials, read_words, sample_index,
                                  select_electrodes, split_subject, stem, task_examples)

ELECS = ["A1", "A2", "A3", "A4", "A5", "A6", "B1*", "B2", "B3"]
CORRUPTED = ["A4", "B1*"]  # "B1*" never matches the cleaned label "B1" under the BrainBERT rule
SEL = ["A2", "A3", "A4", "A5", "B2"]
DURATION, LAG = 150.0, 7.45  # s of recording; s from recording start to movie time 0 (word edges stay >= 0.05 s
# away from the 1 s grid, so the 2-sample trigger jitter cannot move a word across an interval boundary)
BREAKS = [(62.04, 4.0), (95.02, 3.0)]  # (movie time, s the recording runs on): a pause, then an end/beginning pair
N_BLOCKS, N_WORDS = 12, 10
POPT = dict(reref="laplacian", notch_hz=NOTCH_HZ)  # the PopT signal variant; the default is no notch, no re-reference


def rec_seconds(m):
    """Recording time (s) of movie time m, given the breaks."""
    m = np.asarray(m, dtype=np.float64)
    return LAG + m + sum(d * (m >= b) for b, d in BREAKS)


def write_treebank(root, trials=("trial000", "trial001"), seed=0):
    """sub_1 with one movie per trial. Speech comes in 6 s blocks (10 words of 0.3 s, every 0.6 s, the first a
    sentence onset) separated by 5 s of silence, and the breaks fall in silences. Electrode A3 gets a 0.25 s, 40 Hz
    burst at every word onset whose amplitude follows the word's rms; all channels share a slow component that the
    Laplacian removes. Returns {trial: (word table as written, raw signals {label: (T,)})}."""
    (root / "electrode_labels" / "sub_1").mkdir(parents=True)
    (root / "electrode_labels" / "sub_1" / "electrode_labels.json").write_text(json.dumps(ELECS))
    (root / "corrupted_elec.json").write_text(json.dumps({"sub_1": CORRUPTED}))
    for d in ("subject_metadata", "subject_timings"):
        (root / d).mkdir()
    out = {}
    for k, trial in enumerate(trials):
        rng = np.random.default_rng(seed + k)
        movie = f"movie-{k}"
        words = [{"text": "w", "start": 1.0 + 11 * b + 0.6 * i, "end": 1.0 + 11 * b + 0.6 * i + 0.3,
                  "is_onset": float(i == 0), "rms": rng.uniform(0, 1), "pitch": rng.uniform(80, 300), "extra": 1.0}
                 for b in range(N_BLOCKS) for i in range(N_WORDS)]
        words[3]["extra"] = np.nan  # dropped by the PopT dropna rule
        words = pd.DataFrame(words)
        n = int(DURATION * TB_FS)
        t = np.arange(n) / TB_FS
        x = {e: 50 * np.sin(2 * np.pi * 0.7 * t) + rng.normal(0, 1, n) for e in ELECS}
        burst = np.arange(int(0.25 * TB_FS))
        for s, r in zip(rec_seconds(words["start"]), words["rms"]):
            i = int(round(s * TB_FS))
            x["A3"][i:i + len(burst)] += 20 * (0.2 + r) * np.sin(2 * np.pi * 40 * burst / TB_FS)
        with h5py.File(root / f"sub_1_{trial}.h5", "w") as h:
            g = h.create_group("data")
            for i, e in enumerate(ELECS):
                g.create_dataset(f"electrode_{i}", data=x[e])
            g.create_dataset(f"electrode_{len(ELECS)}", data=np.zeros(n))  # unlabeled extra dataset, as in sub_1
        (root / "subject_metadata" / f"sub_1_{trial}_metadata.json").write_text(json.dumps({"filename": movie}))
        # Triggers every 1 s of movie time with up to 2 samples of jitter. Each break adds two rows that share a movie
        # time just below the last trigger (it steps back, as in the real files), the second one after the break.
        mt = np.arange(0.0, 134.0)
        rows = [{"type": "beginning", "movie_time": 0.0, "index": int(round(LAG * TB_FS))}]
        rows += [{"type": "trigger", "movie_time": m, "index": int(round(rec_seconds(m) * TB_FS)) + int(j)}
                 for m, j in zip(mt, rng.integers(-2, 3, len(mt)))]
        for (b, d), kind in zip(BREAKS, [("pause", "unpause"), ("end", "beginning")]):
            pos = next(i for i, r in enumerate(rows) if r["type"] == "trigger" and r["movie_time"] > b)
            at = int(round(rec_seconds(b - 1e-9) * TB_FS))
            rows[pos:pos] = [{"type": kind[0], "movie_time": np.floor(b) - 0.01, "index": at},
                             {"type": kind[1], "movie_time": np.floor(b) - 0.01, "index": at + int(d * TB_FS)}]
        rows.append({"type": "end", "movie_time": 133.5, "index": int(round(rec_seconds(133.5) * TB_FS))})
        pd.DataFrame(rows).to_csv(root / "subject_timings" / f"sub_1_{trial}_timings.csv", index=False)
        (root / "transcripts" / movie).mkdir(parents=True)
        words.to_csv(root / "transcripts" / movie / "features.csv")
        out[trial] = (words, x)
    return out


def expected_nonword_seconds():
    """Centers (s) of the 1 s recording intervals that overlap no word and lie more than 1 s from both ends, from the
    true recording times of the words (independent of the trigger alignment)."""
    s = rec_seconds(1.0 + 11 * np.arange(N_BLOCKS)[:, None] + 0.6 * np.arange(N_WORDS)).ravel()
    s = np.delete(s, 3)  # the NaN row
    hit = [((s < k + 1) & (s + 0.3 > k)).any() for k in range(int(DURATION))]
    return [k + 0.5 for k in range(int(DURATION)) if not hit[k] and k - 0.5 > 0 and k + 1.5 < DURATION]


def auc(score, y):
    pos, neg = score[y == 1], score[y == 0]
    return (pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean()


# ---------------- PopT rules against direct re-implementations ----------------

def test_sample_index_matches_popt():
    rng = np.random.default_rng(0)
    mt = np.sort(rng.uniform(0, 100, 300))
    ix = np.round(mt * TB_FS + 5000)
    # A pause as in the real files: a pause and an unpause row share a movie time just below the last trigger, and
    # the recording jumps ahead by 30 s at the unpause.
    k = 150
    mt = np.concatenate([mt[:k], [mt[k - 1] - 0.03] * 2, mt[k:]])
    ix = np.concatenate([ix[:k], [ix[k - 1] + 10, ix[k - 1] + 10 + 30 * TB_FS], ix[k:] + 30 * TB_FS])
    ts = np.concatenate([rng.uniform(-1, 101, 500), mt[k - 1:k + 3] + 0.001])
    trigs = pd.DataFrame({"movie_time": mt, "index": ix})
    for t, got in zip(ts, sample_index(ts, mt, ix)):
        j = (abs(trigs["movie_time"] - t)).idxmin()
        assert got == round(trigs.loc[j, "index"] + (t - trigs.loc[j, "movie_time"]) * TB_FS)


def test_quartiles():
    v = np.random.default_rng(0).normal(size=103)
    idx, y = quartile_examples(v)
    assert (y == 0).sum() == 25 and (y == 1).sum() == 103 - 77  # int(103 / 4) and 103 - int(3 * 103 / 4)
    assert v[idx[y == 0]].max() < v[idx[y == 1]].min()


def test_nonword_intervals_match_popt():
    rng = np.random.default_rng(0)
    start = np.sort(rng.integers(0, 200_000, 60))
    end = start + rng.integers(0, 3000, 60)
    L, n = 2048, 210_000
    words = list(zip(start, end))
    intersect = lambda a, b: a[0] < b[1] and a[1] > b[0]
    ref = []
    for i in range(n // L):
        a = (i * L, (i + 1) * L)
        c = int((a[0] + a[1]) / 2)
        if not any(intersect(a, w) for w in words) and c - L > 0 and c + L < n:
            ref.append(c)
    assert nonword_centers(start, end, n, L).tolist() == ref


def test_balancing_matches_popt():
    rng = np.random.default_rng(0)
    start = np.sort(rng.integers(0, 400_000, 80))
    words = {"start": start, "end": start + 500, "is_onset": rng.random(80) < 0.3}
    neg = nonword_centers(words["start"], words["end"], 410_000, 2048)
    for task, pos in [("onset", start[words["is_onset"]]), ("speech", start)]:
        c, y = task_examples(words, 410_000, task)
        n = min(len(pos), len(neg))
        random.seed(42)
        ni = random.sample(list(range(len(neg))), n)
        wi = random.sample(list(range(len(pos))), n)
        assert c.tolist() == pos[wi].tolist() + neg[ni].tolist()
        assert y.tolist() == [1] * n + [0] * n


def test_stem_and_neighbours():
    assert stem("T1cIf9") == ("T1cIf", 9) and stem("F10Fa13") == ("F10Fa", 13) and stem("DC") is None
    assert neighbours("A3", ["A2", "A3", "A4"]) == ["A2", "A4"]
    with pytest.raises(ValueError):
        neighbours("A1", ["A1", "A2"])


# ---------------- Files ----------------

def test_read_words(tmp_path):
    truth = write_treebank(tmp_path)
    words = truth["trial000"][0].drop(index=3)
    w = read_words(tmp_path, "sub_1", "trial000")
    assert len(w["start"]) == len(words) and w["is_onset"].sum() == N_BLOCKS
    # Alignment across the pause and the end/beginning pair: within the trigger jitter (2 samples) plus two roundings
    # (trigger index and word index).
    assert np.abs(w["start"] / TB_FS - rec_seconds(words["start"])).max() <= 3 / TB_FS
    assert np.allclose(w["rms"], words["rms"].to_numpy())  # CSV round trip


def test_select_electrodes(tmp_path):
    write_treebank(tmp_path, trials=("trial000",))
    # BrainBERT rule: only the raw name "A4" matches, so A3, A4 and A5 go; B2 keeps its neighbour B1.
    assert select_electrodes(tmp_path, "sub_1") == ["A2", "B2"]
    assert select_electrodes(tmp_path, "sub_1", drop_corrupted=True) == ["A2"]  # "B1*" cleaned: B2 goes too
    assert select_electrodes(tmp_path, "sub_1", SEL) == SEL
    assert select_electrodes(tmp_path, "sub_1", SEL, drop_corrupted=True) == ["A2"]
    with pytest.raises(ValueError):
        select_electrodes(tmp_path, "sub_1", ["Z9"])


def test_window_values(tmp_path):
    """Main condition (the defaults: no notch, no re-reference): a window is exactly the U4-normalized 500 Hz signal
    around the onset."""
    truth = write_treebank(tmp_path, trials=("trial000",))
    raw = truth["trial000"][1]["B2"]
    y = resample_poly(raw[None], 125, 512, axis=1)
    y = channel_normalize(y, *channel_stats(y))[0]
    ws = read_trial(tmp_path, "sub_1", "trial000", "speech", electrodes=["B2"])
    for w in ws[:5] + ws[-5:]:
        a = int(round(w["center_seconds"] * FS)) - FS // 2
        assert np.allclose(w["wave"][0], y[a:a + FS], atol=1e-4)


@pytest.mark.parametrize("task", TASKS)
def test_read_trial(tmp_path, task):
    write_treebank(tmp_path, trials=("trial000",))
    ws = read_trial(tmp_path, "sub_1", "trial000", task, electrodes=SEL, **POPT)
    y = np.array([w["label"] for w in ws])
    n_words, neg = N_BLOCKS * N_WORDS - 1, expected_nonword_seconds()
    expect = {"pitch": int(n_words / 4) + n_words - int(3 * n_words / 4), "volume": None,
              "onset": 2 * min(N_BLOCKS, len(neg)), "speech": 2 * min(n_words, len(neg))}
    expect["volume"] = expect["pitch"]
    assert len(ws) == expect[task]  # every example fits in the recording
    assert ws[0]["wave"].shape == (len(SEL), FS) and ws[0]["wave"].dtype == np.float32
    assert {w["session_id"] for w in ws} == {"sub_1_trial000"} and ws[0]["electrodes"] == tuple(SEL)
    if task in ("onset", "speech"):
        assert y.mean() == 0.5
        got = sorted(round(w["center_seconds"] - 0.5) + 0.5 for w in ws if w["label"] == 0)
        assert set(got) <= set(neg)  # negatives are word-free intervals of the true recording clock
    if task != "pitch":
        # A3 carries the burst, so the classes separate on it; windows are centred on the onset, so the burst fills
        # samples 250-375 of positives and the last 0.25 s is quiet.
        a3 = np.array([np.abs(w["wave"][SEL.index("A3")]).mean() for w in ws])
        assert auc(a3, y) > 0.9
        pos = np.stack([np.abs(w["wave"][SEL.index("A3")]) for w in ws if w["label"] == 1])
        assert pos[:, 250:375].mean() > 2 * pos[:, 375:].mean()


def test_laplacian_removes_common_signal(tmp_path):
    write_treebank(tmp_path, trials=("trial000",))
    ws = read_trial(tmp_path, "sub_1", "trial000", "speech", electrodes=["A2"], reref="laplacian")
    raw = read_trial(tmp_path, "sub_1", "trial000", "speech", electrodes=["A2"])
    # The shared 0.7 Hz component dominates the raw channel; after the Laplacian only white noise is left.
    lag1 = lambda x: np.mean([np.corrcoef(w["wave"][0, :-1], w["wave"][0, 1:])[0, 1] for w in x])
    assert lag1(raw) > 0.9 and lag1(ws) < 0.9


def test_one_pass_and_cache(tmp_path):
    write_treebank(tmp_path, trials=("trial000",))
    every = read_trials(tmp_path, "sub_1", "trial000", TASKS, electrodes=SEL, cache_dir=tmp_path / "cache")
    assert len(list((tmp_path / "cache").glob("*.npy"))) == len(TASKS)
    again = read_trials(tmp_path, "sub_1", "trial000", TASKS, electrodes=SEL, cache_dir=tmp_path / "cache")
    for task in TASKS:
        one = read_trial(tmp_path, "sub_1", "trial000", task, electrodes=SEL)
        for other in (every[task], again[task]):
            assert [w["label"] for w in one] == [w["label"] for w in other]
            assert all(np.array_equal(u["wave"], v["wave"]) for u, v in zip(one, other))


def test_stat_chunks_change_normalization(tmp_path):
    write_treebank(tmp_path, trials=("trial000",))
    whole = read_trial(tmp_path, "sub_1", "trial000", "speech", electrodes=["B2"])
    chunked = read_trial(tmp_path, "sub_1", "trial000", "speech", electrodes=["B2"], stat_seconds=30)
    assert len(whole) == len(chunked)
    assert not all(np.array_equal(u["wave"], v["wave"]) for u, v in zip(whole, chunked))


def overwrite(root, trial, edits):
    """Replace parts of written raw signals: edits {label: f(x) -> x}."""
    with h5py.File(root / f"sub_1_{trial}.h5", "r+") as h:
        for e, f in edits.items():
            d = h["data"][f"electrode_{ELECS.index(e)}"]
            d[:] = f(d[:])


def flat(a, b):
    def f(x):
        x[int(a * TB_FS):int(b * TB_FS)] = 0.0
        return x
    return f


def test_flat_laplacian_channel_is_left_out(tmp_path, capsys):
    """PopT variant: three neighbouring contacts constant for 60% of the trial make the middle one's Laplacian 0 there,
    so its scale is 0 (the review case: values up to ~2e6 after normalization). It becomes 0 and invalid in place, so
    the other channels keep their positions."""
    write_treebank(tmp_path, trials=("trial000",))
    overwrite(tmp_path, "trial000", {e: flat(0, 0.6 * DURATION) for e in ("A2", "A3", "A4")})
    ws = read_trial(tmp_path, "sub_1", "trial000", "speech", electrodes=SEL, **POPT)
    assert "A3 left out of 1 of 1 statistics chunk(s)" in capsys.readouterr().out
    ok = np.array([w["channel_ok"] for w in ws])
    assert (ok[:, SEL.index("A3")] == 0).all() and ok[:, [i for i, e in enumerate(SEL) if e != "A3"]].all()
    x, valid, _ = collate_treebank([TreebankWindows(ws)[i] for i in range(len(ws))])
    a3 = SEL.index("A3")
    assert x.shape[1] == len(SEL) and x.abs().max() < 1e3 and (x[:, a3] == 0).all()
    assert not valid[:, a3].any() and valid[:, [i for i in range(len(SEL)) if i != a3]].all()
    # With 30 s chunks only the chunks inside the flat stretch leave A3 out.
    ws = read_trial(tmp_path, "sub_1", "trial000", "speech", electrodes=SEL, stat_seconds=30, **POPT)
    late = np.array([w["center_seconds"] > 0.6 * DURATION for w in ws])
    a3 = np.array([w["channel_ok"][SEL.index("A3")] for w in ws])
    assert a3[late].all() and not a3[~late].any()


def test_nan_channel_is_left_out_and_drops_counted(tmp_path, capsys):
    """PopT variant: one NaN sample turns the rest of the notched trace into NaN, and the Laplacian spreads it to the
    two neighbours.
    Those channels are left out of the trial (under 30 s of finite samples) and the windows are kept; a NaN late in
    the trial keeps the channels and drops the windows after it, with the count printed."""
    write_treebank(tmp_path, trials=("trial000",))
    clean = read_trial(tmp_path, "sub_1", "trial000", "pitch", electrodes=SEL, **POPT)

    def nan_at(t):
        def f(x):
            x[int(t * TB_FS)] = np.nan
            return x
        return f
    overwrite(tmp_path, "trial000", {"A3": nan_at(10.0)})
    capsys.readouterr()
    ws = read_trial(tmp_path, "sub_1", "trial000", "pitch", electrodes=SEL, **POPT)
    assert len(ws) == len(clean)
    assert [e for e, k in zip(SEL, ws[0]["channel_ok"]) if not k] == ["A2", "A3", "A4"]
    assert "windows dropped" not in capsys.readouterr().out
    overwrite(tmp_path, "trial000", {"A3": lambda x: np.nan_to_num(x)})
    overwrite(tmp_path, "trial000", {"A3": nan_at(120.0)})
    ws = read_trial(tmp_path, "sub_1", "trial000", "pitch", electrodes=SEL, **POPT)
    late = sum(w["center_seconds"] > 119.5 for w in clean)
    assert late > 0 and len(ws) == len(clean) - late and all(w["channel_ok"].all() for w in ws)
    assert f"{late} of {len(clean)} windows dropped (0 outside the recording, {late} with a non-finite sample" \
        in capsys.readouterr().out


def test_cache_name_has_code_version(tmp_path, monkeypatch):
    write_treebank(tmp_path, trials=("trial000",))
    read_trials(tmp_path, "sub_1", "trial000", ("speech",), electrodes=SEL, cache_dir=tmp_path / "cache")
    meta = json.loads(next((tmp_path / "cache").glob("*.json")).read_text())
    assert meta["code"] == data_treebank.code_version() and "dropped" in meta
    monkeypatch.setattr(data_treebank, "_CODE", ["changed"])  # other code: the old cache is not loaded
    read_trials(tmp_path, "sub_1", "trial000", ("speech",), electrodes=SEL, cache_dir=tmp_path / "cache")
    assert len(list((tmp_path / "cache").glob("*.npy"))) == 2


# ---------------- Splits and Dataset ----------------

def test_split_heldout(tmp_path):
    write_treebank(tmp_path)  # trial001 is sub_1's PopT test trial
    train, val, test = split_subject(tmp_path, "sub_1", "speech", electrodes=SEL)
    assert {w["trial"] for w in train} == {"trial000"} and {w["trial"] for w in test} == {"trial001"} and val == []


def test_split_popt(tmp_path):
    write_treebank(tmp_path)
    train, val, test = split_subject(tmp_path, "sub_1", "speech", mode="popt", electrodes=SEL)
    n = len(train) + len(val) + len(test)
    assert {w["trial"] for w in train + val + test} == {"trial001"}
    assert abs(len(train) - 0.8 * n) <= 1 and abs(len(test) - 0.1 * n) <= 1
    assert len({w["center_seconds"] for w in train + val + test}) == n


def test_split_guards(tmp_path):
    write_treebank(tmp_path, trials=("trial001",))
    with pytest.raises(ValueError, match="no other trial"):
        split_subject(tmp_path, "sub_1", "speech", electrodes=SEL)
    with pytest.raises(ValueError, match="no PopT test trial"):
        split_subject(tmp_path, "sub_5", "speech")


def test_dataset_collate(tmp_path):
    write_treebank(tmp_path, trials=("trial000",))
    ws = read_trial(tmp_path, "sub_1", "trial000", "speech", electrodes=SEL)
    x, valid, y = collate_treebank([TreebankWindows(ws)[i] for i in range(4)])
    assert x.shape == (4, len(SEL), 10, 50) and valid.all() and y.dtype == torch.float32
    # Invalid channels (channel_ok False) and padding (a window with fewer channels) are both valid False.
    ws[1] = {**ws[1], "channel_ok": np.array([True, False, True, True, True])}
    ws[2] = {**ws[2], "wave": ws[2]["wave"][:3], "channel_ok": np.ones(3, bool)}
    x, valid, _ = collate_treebank([TreebankWindows(ws)[i] for i in range(3)])
    assert valid.tolist() == [[True] * 5, [True, False, True, True, True], [True] * 3 + [False] * 2]


def test_invalid_channel_equals_absent_channel():
    """A channel with valid False changes no other channel's output and no head output: the model gives the same window
    logit as with the channel removed (the pretraining streams' treatment), for any values in it."""
    from ibrain.finetune import HEADS, Regressor, WindowLogit
    from ibrain.model import IEEG, IBrain
    cfg = dict(d=32, H=4, ffn=64, L=2, d_proj=16)
    torch.manual_seed(0)
    model = IBrain(**cfg).eval()
    x = torch.randn(2, 5, 10, 50)
    for head in ("attn", "mean"):
        clf = Regressor(model, WindowLogit(HEADS[head](32, 5, out=1)), False, sig=IEEG).eval()
        valid = torch.ones(2, 5, dtype=torch.bool)
        valid[:, 2] = False
        keep = [0, 1, 3, 4]
        with torch.no_grad():
            a = clf(x, valid)
            b = clf(x[:, keep], torch.ones(2, 4, dtype=torch.bool))
            x2 = x.clone()
            x2[:, 2] = 1e6
            c = clf(x2, valid)
        assert torch.allclose(a, b, atol=1e-5) and torch.allclose(a, c, atol=1e-5)
