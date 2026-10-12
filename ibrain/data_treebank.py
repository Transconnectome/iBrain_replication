"""M5 downstream: Brain Treebank (Wang et al. 2024) -> labeled 1 s iEEG windows for the four iBrain tasks, Pitch,
Volume, Onset and Speech, scored by AUC (SPEC 1.7, 1.8, U34).
The paper says only that it follows the subject-specific protocol of BrainBERT and PopT. The label rules, electrodes
and filters below are those of the PopT code (github.com/czlwang/PopulationTransformer), cut to iBrain's 1 s window
at 500 Hz:
  pitch / volume: the words of one trial sorted by `pitch` / `rms`; the bottom quarter is 0, the top quarter is 1.
  onset: 1 = first word of a sentence (is_onset); 0 = a 1 s stretch of the movie grid that overlaps no word. Balanced.
  speech: 1 = any word; 0 = as for onset. Balanced.
A window is 1 s centered on the word onset (PopT centers its 5 s window) or on the non-word stretch. The non-word
grid and its margin from the recording ends are 1 s too (PopT: 5 s), so the negatives are not PopT's.
Signal (main condition, the same input as pretraining): the PopT electrodes (BrainBERT's clean Laplacian rule),
2048 -> 500 Hz, then U4 normalization. A channel that is flat or mostly NaN in a statistics chunk
(data_ieeg.usable_channels, the pretraining rule) is set to 0 there and marked channel_ok False, which collate turns
into valid False: the model then ignores it exactly as a padded channel, the same as the pretraining streams leaving
it out of the block, while every channel keeps its position (flatten head). No notch and no re-reference: the pretraining iEEG has neither
added, and a shaft Laplacian cannot be built for SWEC (no channel metadata) or the AJILE12 grids.
PopT variant (reref="laplacian", notch_hz=NOTCH_HZ): notch at 60 Hz and harmonics up to 360 Hz, then Laplacian
re-reference (minus the mean of the two neighbouring contacts on the same shaft), as the PopT code.
Each trial is one recording; window dicts have the same role as data_nwb's, with "wave" (C, 500), "channel_ok" (C,)
and "label" 0/1."""
import hashlib
import inspect
import json
import random
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ibrain import data_ieeg
from ibrain.data_ieeg import (FS, WINDOW_SAMPLES, channel_normalize, channel_stats, resample_to_fs, to_patches,
                              usable_channels)
from ibrain.data_spike import collate

TB_FS = 2048  # Hz, every Brain Treebank trial
TASKS = ("pitch", "volume", "onset", "speech")
FEATURE = {"pitch": "pitch", "volume": "rms"}  # features.csv column of the word-level tasks
NOTCH_HZ = (60, 120, 180, 240, 300, 360)  # PopT H5DataReader.freqs_to_filter
NOTCH_Q = 30
# One held-out trial per subject: PopT trial_selections/test_trials.json (github.com/czlwang/PopulationTransformer,
# commit d237755), the same trials as BrainBERT data/test_split_trials.json under the older subject ids.
TEST_TRIALS = {"sub_1": "trial001", "sub_2": "trial006", "sub_3": "trial000", "sub_4": "trial000",
               "sub_6": "trial004", "sub_7": "trial000", "sub_10": "trial000"}

# ---------------- Files and electrodes ----------------

def strip_label(name):
    """BrainBERT/PopT label cleaning: '*', '#' and '_' are removed ('RT3aHa1*' -> 'RT3aHa1')."""
    return name.replace("*", "").replace("#", "").replace("_", "")


def stem(name):
    """'T1cIf9' -> ('T1cIf', 9): the trailing digits are the contact number on the shaft (PopT stem_electrode_name).
    None for a label without trailing digits."""
    i = len(name)
    while i and name[i - 1].isdigit():
        i -= 1
    return (name[:i], int(name[i:])) if i < len(name) else None


def electrode_labels(root, subject):
    """Cleaned labels in file order: label i is dataset data/electrode_i. sub_1 has 156 datasets and 155 labels; the
    unlabeled last dataset is never read, as in PopT, which indexes datasets by label position."""
    path = Path(root) / "electrode_labels" / subject / "electrode_labels.json"
    return [strip_label(e) for e in json.loads(path.read_text())]


def neighbours(name, labels):
    """The two contacts next to `name` on its shaft (number - 1 and + 1). PopT requires both."""
    s, n = stem(name)
    out = [f"{s}{n + d}" for d in (-1, 1)]
    if any(o not in labels for o in out):
        raise ValueError(f"{name}: Laplacian needs both neighbours, {out} not all in the labels")
    return out


def laplacian_electrodes(labels):
    """Labels whose contacts number - 1 and + 1 on the same shaft are both in `labels` (BrainBERT
    get_all_laplacian_electrodes), in label order."""
    stems = {stem(e) for e in labels if stem(e)}
    return [e for e in labels if stem(e) and (stem(e)[0], stem(e)[1] - 1) in stems
            and (stem(e)[0], stem(e)[1] + 1) in stems]


def select_electrodes(root, subject, electrodes="popt", drop_corrupted=False):
    """Electrodes used for one subject, in label order (PopT get_ordered_electrodes).
    electrodes="popt": BrainBERT get_clean_laplacian_electrodes, the rule behind PopT's clean_laplacian.json (equal to
    that file for all 10 subjects, checked 2026-10-03): the corrupted_elec.json names are removed from the cleaned
    labels, then electrodes with both neighbours are kept. BrainBERT removes the raw names, so a corrupted name with
    '*' or '#' ('RT3aHa2*') never matches its cleaned label and 9 corrupted electrodes stay (sub_2, sub_7, sub_10).
    drop_corrupted=True removes the cleaned names instead, which drops them and the electrodes whose neighbours they
    are. A list of cleaned labels is used as given; with drop_corrupted, corrupted ones and their users are dropped."""
    labels = electrode_labels(root, subject)
    raw = json.loads((Path(root) / "corrupted_elec.json").read_text()).get(subject, [])
    bad = {strip_label(e) for e in raw} if drop_corrupted else set(raw)
    if electrodes == "popt":
        return laplacian_electrodes([e for e in labels if e not in bad])
    sel = set(electrodes)
    missing = sorted(sel - set(labels))
    if missing:
        raise ValueError(f"{subject}: electrodes not in the labels: {missing[:5]}")
    if drop_corrupted:
        sel = {e for e in sel if e not in bad and not bad.intersection(neighbours(e, labels))}
    return [e for e in labels if e in sel]


def subject_trials(root, subject):
    """Trials of a subject that have a recording, e.g. ['trial000', 'trial001']."""
    return sorted(p.stem.split("_")[-1] for p in Path(root).glob(f"{subject}_trial*.h5"))


# ---------------- Words and task examples ----------------

def sample_index(t, movie_time, index, fs=TB_FS, batch=256):
    """Movie time (s) -> native sample index: nearest row of the timings file, then linear in time from it (PopT
    estimate_sample_index). Every row counts (triggers, pause, unpause, beginning, end) and the first of equally near
    rows wins, as pandas idxmin. movie_time is not sorted: pause/unpause pairs share a movie time and step back by
    a few tens of ms, so this is a direct search, in batches to bound memory."""
    t = np.asarray(t, dtype=np.float64)
    j = np.concatenate([np.abs(movie_time[None] - t[i:i + batch, None]).argmin(1) for i in range(0, len(t), batch)]
                       or [np.zeros(0, dtype=np.int64)])
    return np.round(index[j] + (t - movie_time[j]) * fs).astype(np.int64)


def read_words(root, subject, trial):
    """Words of the movie shown in one trial, aligned to the recording. As in PopT: rows of features.csv with any NaN
    are dropped, then words from the first one that starts after the movie time of the last row onward. Returns
    arrays start, end (native sample index), is_onset (bool), rms, pitch."""
    import pandas as pd
    root = Path(root)
    movie = json.loads((root / "subject_metadata" / f"{subject}_{trial}_metadata.json").read_text())["filename"]
    df = pd.read_csv(root / "transcripts" / movie / "features.csv").set_index("Unnamed: 0").dropna()
    df = df.reset_index(drop=True)
    trig = pd.read_csv(root / "subject_timings" / f"{subject}_{trial}_timings.csv")
    mt, ix = trig["movie_time"].to_numpy(np.float64), trig["index"].to_numpy(np.float64)
    start, end = df["start"].to_numpy(np.float64), df["end"].to_numpy(np.float64)
    n = len(start) if (start <= mt[-1]).all() else int(np.argmax(start > mt[-1]))  # PopT breaks at the first one
    return {"start": sample_index(start[:n], mt, ix), "end": sample_index(end[:n], mt, ix),
            "is_onset": df["is_onset"].to_numpy()[:n].astype(bool),
            "rms": df["rms"].to_numpy(np.float64)[:n], "pitch": df["pitch"].to_numpy(np.float64)[:n]}


def quartile_examples(values):
    """PopT get_word_features_labels: bottom quarter -> 0, top quarter -> 1, words in between unused.
    Returns (word indices, labels) in that order. Stable sort; pandas' default quicksort may order ties differently."""
    order = np.argsort(values, kind="stable")
    lo, hi = order[:int(len(order) / 4)], order[int(3 * len(order) / 4):]
    return np.concatenate([lo, hi]), np.repeat([0, 1], [len(lo), len(hi)])


def nonword_centers(start, end, n_native, length):
    """PopT get_aligned_linguistic_control_matrix: the recording cut into consecutive `length`-sample intervals from
    sample 0; an interval counts when it overlaps no word (open overlap, s < b and e > a) and its center is more than
    `length` samples from both ends. Returns the centers."""
    k = np.arange(n_native // length)
    a, b = k * length, (k + 1) * length
    c = (a + b) // 2
    keep = (c - length > 0) & (c + length < n_native)
    if len(start) == 0:
        return c[keep]
    order = np.argsort(start, kind="stable")
    s, e = start[order], np.maximum.accumulate(end[order])  # e[i] = latest end among words starting at or before s[i]
    m = np.searchsorted(s, b, side="left")  # words with s < b
    hit = (m > 0) & (e[np.maximum(m - 1, 0)] > a)
    return c[~hit & keep]


def task_examples(words, n_native, task, fs=TB_FS, seconds=1.0):
    """One trial -> (centers (N,) native sample index, labels (N,) int) in PopT order. pitch / volume: word onsets of
    the two quartiles. onset / speech: positives (sentence onsets / all words) then non-word intervals, both subsampled
    to the smaller count with random.seed(42) and random.sample as PopT. The non-word intervals are `seconds` long and
    their centers more than `seconds` from the recording ends; PopT uses its 5 s window here, so with seconds = 1 the
    candidate negatives (and hence the sampled ones) differ from PopT's."""
    if task in FEATURE:
        idx, y = quartile_examples(words[FEATURE[task]])
        return words["start"][idx], y
    if task not in ("onset", "speech"):
        raise ValueError(f"unknown task {task!r}; expected one of {TASKS}")
    neg = nonword_centers(words["start"], words["end"], n_native, int(seconds * fs))
    pos = words["start"][words["is_onset"]] if task == "onset" else words["start"]
    n = min(len(pos), len(neg))
    random.seed(42)
    neg = neg[random.sample(range(len(neg)), n)]
    pos = pos[random.sample(range(len(pos)), n)]
    return np.concatenate([pos, neg]), np.repeat([1, 0], [n, n])


# ---------------- Signal ----------------

def notch(x, fs=TB_FS, freqs=NOTCH_HZ, q=NOTCH_Q):
    """PopT notch filter: scipy iirnotch at each frequency, applied causally with lfilter."""
    from scipy.signal import iirnotch, lfilter
    for f in freqs:
        b, a = iirnotch(f / (fs / 2), q)
        x = lfilter(b, a, x)
    return x


class _Traces:
    """Notched, 500 Hz traces of single contacts, read on demand with a small cache (Laplacian neighbours are adjacent
    contacts, so they are reused by the next electrodes). Notch and resampling are linear and time-invariant per
    channel, so applying them before the Laplacian gives the same signal as PopT's order (notch, re-reference)."""

    def __init__(self, path, labels, notch_hz=(), size=8):
        self.path, self.index, self.notch_hz, self.size = path, {e: i for i, e in enumerate(labels)}, notch_hz, size
        self.cache = OrderedDict()

    def __call__(self, name):
        if name in self.cache:
            self.cache.move_to_end(name)
            return self.cache[name]
        import h5py
        with h5py.File(self.path, "r") as h:
            x = np.asarray(h["data"][f"electrode_{self.index[name]}"][:], dtype=np.float64)
        if self.notch_hz:
            x = notch(x, freqs=self.notch_hz)
        y = resample_to_fs(x[None], TB_FS)[0].astype(np.float64)
        self.cache[name] = y
        if len(self.cache) > self.size:
            self.cache.popitem(last=False)
        return y


def stat_chunks(n, chunk_samples):
    """Chunk start indices for U4 statistics; None = the whole recording. A tail shorter than half a chunk joins the
    previous chunk, as in the AJILE12 stream (U31)."""
    if chunk_samples is None:
        return np.array([0])
    starts = list(range(0, n, chunk_samples))
    if len(starts) > 1 and n - starts[-1] < chunk_samples / 2:
        starts.pop()
    return np.asarray(starts)


def chunk_stats(trace, chunks):
    """U4 statistics of one channel in each chunk -> center, scale, number of finite samples, each (K,)."""
    bounds = list(chunks[1:]) + [len(trace)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # an all-NaN chunk gives NaN statistics; usable_channels drops it
        st = np.array([[v[0] for v in channel_stats(trace[None, a:z])] for a, z in zip(chunks, bounds)])
    return st[:, 0], st[:, 1], np.array([np.isfinite(trace[a:z]).sum() for a, z in zip(chunks, bounds)])


def extract(trace, starts, chunk_of, center, scale):
    """Windows of one channel, normalized with the statistics of the chunk each window's center falls in (U4).
    trace (T,), starts (N,) 500 Hz, center, scale (K,) from chunk_stats -> (N, 500) float32."""
    w = trace[starts[:, None] + np.arange(WINDOW_SAMPLES)]
    return channel_normalize(w, center[chunk_of], scale[chunk_of]).astype(np.float32)


_CODE = []


def code_version():
    """Hash of the code behind the cached arrays: this file and the data_ieeg functions and constants it uses. Part of
    every cache name, so a cache written by other code is never loaded (an edit to a comment here also renames it)."""
    if not _CODE:
        src = Path(__file__).read_text(encoding="utf-8") + "".join(
            inspect.getsource(f) for f in (channel_stats, channel_normalize, resample_to_fs, usable_channels))
        src += repr((FS, WINDOW_SAMPLES, data_ieeg.MIN_STAT_SECONDS, data_ieeg.MIN_SCALE_RATIO))
        _CODE.append(hashlib.md5(src.encode()).hexdigest()[:10])
    return _CODE[0]


# ---------------- One trial ----------------

def read_trial(root, subject, trial, task, **kw):
    """One trial, one task -> list of window dicts (see read_trials)."""
    return read_trials(root, subject, trial, (task,), **kw)[task]


def read_trials(root, subject, trial, tasks=TASKS, electrodes="popt", reref="none", notch_hz=(),
                stat_seconds=None, drop_corrupted=False, cache_dir=None):
    """One trial -> {task: list of window dicts {wave float32 (C, 500), channel_ok bool (C,), label 0/1,
    center_seconds, subject, trial, session_id, task, electrodes}}. The signal is read and filtered once for all tasks.
    channel_ok is False for the electrodes not usable in the statistics chunk of the window (usable_channels: flat, or
    under 30 s of finite samples); their samples are 0 and collate_treebank marks them invalid, so the model ignores
    them as the pretraining streams do (which leave them out of the block). Windows that do not fit
    in the recording, have a non-finite sample in a kept channel, or keep no channel are dropped; the counts are printed
    and kept in the cache metadata. reref: "none" (main) or "laplacian" (PopT variant); notch_hz: () (main) or
    NOTCH_HZ (PopT variant). stat_seconds: U4 statistics over chunks of
    this length (by window center) instead of the whole trial; it must match the pretraining unit (SPEC U4).
    cache_dir: keeps each task's arrays as .npy (memory-mapped on reuse) under a name that encodes every argument and
    the code version."""
    root = Path(root)
    elecs = select_electrodes(root, subject, electrodes, drop_corrupted)
    out, todo, names = {}, [], {}
    for task in tasks:
        cfg = {"subject": subject, "trial": trial, "task": task, "electrodes": elecs, "reref": reref,
               "notch_hz": list(notch_hz or ()), "stat_seconds": stat_seconds, "code": code_version()}
        names[task] = (cfg, f"{subject}_{trial}_{task}_{hashlib.md5(json.dumps(cfg).encode()).hexdigest()[:10]}")
        if cache_dir is not None and (Path(cache_dir) / f"{names[task][1]}.json").exists():
            meta = json.loads((Path(cache_dir) / f"{names[task][1]}.json").read_text())
            out[task] = (np.load(Path(cache_dir) / f"{names[task][1]}.npy", mmap_mode="r"), np.asarray(meta["labels"]),
                         np.asarray(meta["centers"]), np.asarray(meta["chunk_of"]), np.asarray(meta["channel_ok"]))
        else:
            todo.append(task)
    if todo:
        new = _compute_trial(root, subject, trial, todo, elecs, reref, notch_hz, stat_seconds)
        for task, (x, y, centers, chunk_of, ok, dropped) in new.items():
            if cache_dir is not None:
                cfg, name = names[task]
                Path(cache_dir).mkdir(parents=True, exist_ok=True)
                np.save(Path(cache_dir) / f"{name}.npy", x)
                meta = {**cfg, "labels": y.tolist(), "centers": centers.tolist(), "chunk_of": chunk_of.tolist(),
                        "channel_ok": ok.tolist(), "dropped": dropped}
                (Path(cache_dir) / f"{name}.json").write_text(json.dumps(meta))
            out[task] = (x, y, centers, chunk_of, ok)
    el = tuple(elecs)
    return {task: [{"wave": x[i], "channel_ok": ok[:, chunk_of[i]], "label": int(y[i]),
                    "center_seconds": float(centers[i]) / TB_FS, "subject": subject, "trial": trial,
                    "session_id": f"{subject}_{trial}", "task": task, "electrodes": el} for i in range(len(y))]
            for task, (x, y, centers, chunk_of, ok) in ((t, out[t]) for t in tasks)}


def _compute_trial(root, subject, trial, tasks, elecs, reref, notch_hz, stat_seconds):
    """-> {task: (x (N, C, 500) float32, labels (N,), centers (N,) native samples, chunk_of (N,), channel_ok (C, K),
    dropped window counts)}, one pass over the electrodes. Prints the electrodes left out of a chunk and the dropped
    windows."""
    import h5py
    if reref not in ("laplacian", "none"):
        raise ValueError(f"reref must be 'laplacian' or 'none', got {reref!r}")
    path = root / f"{subject}_{trial}.h5"
    labels = electrode_labels(root, subject)
    with h5py.File(path, "r") as h:
        n_native = h["data"]["electrode_0"].shape[0]
    words = read_words(root, subject, trial)
    n500 = int(round(n_native * FS / TB_FS))
    chunks = stat_chunks(n500, None if stat_seconds is None else int(round(stat_seconds * FS)))
    ex = {}
    for task in tasks:
        centers, y = task_examples(words, n_native, task)
        starts = np.round(centers * FS / TB_FS).astype(np.int64) - WINDOW_SAMPLES // 2
        fit = (starts >= 0) & (starts + WINDOW_SAMPLES <= n500)
        starts = starts[fit]
        chunk_of = np.searchsorted(chunks, starts + WINDOW_SAMPLES // 2, side="right") - 1
        ex[task] = (centers[fit], y[fit], starts, chunk_of,
                    np.empty((len(starts), len(elecs), WINDOW_SAMPLES), dtype=np.float32), int((~fit).sum()))
    scale, n_finite = np.empty((len(elecs), len(chunks))), np.empty((len(elecs), len(chunks)), dtype=np.int64)
    trace = _Traces(path, labels, notch_hz)
    for c, e in enumerate(elecs):
        t = trace(e)
        if reref == "laplacian":
            t = t - np.mean([trace(n) for n in neighbours(e, labels)], axis=0)
        center, scale[c], n_finite[c] = chunk_stats(t, chunks)
        for _, _, starts, chunk_of, x, _ in ex.values():
            x[:, c] = extract(t, starts, chunk_of, center, scale[c])
    ok = np.stack([usable_channels(n_finite[:, k], scale[:, k]) for k in range(len(chunks))], axis=1)  # (C, K)
    for c in np.flatnonzero(~ok.all(1)):
        print(f"{subject} {trial}: {elecs[c]} left out of {(~ok[c]).sum()} of {len(chunks)} statistics chunk(s) "
              f"(flat or under {data_ieeg.MIN_STAT_SECONDS} s of finite samples)")
    out = {}
    for task, (centers, y, _, chunk_of, x, outside) in ex.items():
        keep = ok[:, chunk_of].T  # (N, C)
        x[~keep] = 0.0  # unusable channel-chunks: 0, as a padded channel (a NaN or ~1e6 value would reach the model)
        bad = (keep & ~np.isfinite(x).all(axis=2)).any(axis=1)
        empty = ~keep.any(axis=1) & ~bad
        sel = ~bad & ~empty
        dropped = {"outside": outside, "nonfinite": int(bad.sum()), "no_channel": int(empty.sum())}
        if sum(dropped.values()):
            print(f"{subject} {trial} {task}: {sum(dropped.values())} of {len(y) + outside} windows dropped "
                  f"({dropped['outside']} outside the recording, {dropped['nonfinite']} with a non-finite sample, "
                  f"{dropped['no_channel']} with no usable channel)")
        out[task] = ((x, y, centers, chunk_of) if sel.all() else (x[sel], y[sel], centers[sel], chunk_of[sel])) + (
            ok, dropped)
    return out


# ---------------- Splits and Dataset ----------------

def split_subject(root, subject, task, mode="heldout", seed=42, **kw):
    """Train / val / test window lists for one subject (kw goes to read_trial).
    heldout (default, the iBrain text: "fine-tuned on a subset of recordings from each subject and evaluated on the
    remaining held-out recordings"): test = the subject's PopT test trial, train = its other trials, val empty.
    Subjects with a single trial (sub_5, sub_8, sub_9) cannot be split this way and raise.
    popt (the PopT code): only the test trial, its examples split 80/10/10 at random. Same proportions as PopT
    tasks/utils.py, not the same indices (PopT uses sklearn train_test_split with random_state 42)."""
    if subject not in TEST_TRIALS:
        raise ValueError(f"{subject} has no PopT test trial; subjects: {sorted(TEST_TRIALS)}")
    test_trial = TEST_TRIALS[subject]
    if mode == "heldout":
        others = [t for t in subject_trials(root, subject) if t != test_trial]
        if not others:
            raise ValueError(f"{subject}: only {test_trial}, no other trial to fine-tune on")
        train = [w for t in others for w in read_trial(root, subject, t, task, **kw)]
        return train, [], read_trial(root, subject, test_trial, task, **kw)
    if mode == "popt":
        ws = read_trial(root, subject, test_trial, task, **kw)
        idx = np.random.default_rng(seed).permutation(len(ws))
        a, b = int(round(0.8 * len(ws))), int(round(0.9 * len(ws)))
        return [ws[i] for i in idx[:a]], [ws[i] for i in idx[a:b]], [ws[i] for i in idx[b:]]
    raise ValueError(f"mode must be 'heldout' or 'popt', got {mode!r}")


class TreebankWindows(Dataset):
    """Window dicts -> ((C, S, P) float32, label float32 scalar, channel_ok (C,) bool). Waves are already normalized
    (U4), with 0 in the channels whose channel_ok is False (all True when the dict has no channel_ok)."""

    def __init__(self, windows):
        self.w = list(windows)

    def __len__(self):
        return len(self.w)

    def __getitem__(self, i):
        w = self.w[i]
        x = to_patches(w["wave"])
        ok = w.get("channel_ok")
        ok = torch.ones(len(x), dtype=torch.bool) if ok is None else torch.as_tensor(np.asarray(ok, dtype=bool))
        return x, torch.tensor(float(w["label"])), ok


def collate_treebank(batch):
    """-> x (B, C, S, P), valid (B, C), y (B,). Channels are padded when a batch mixes subjects; valid is False for
    padding and for channels with channel_ok False, so both are excluded the same way (V_c)."""
    x, valid = collate([b[0] for b in batch])
    ok = torch.nn.utils.rnn.pad_sequence([b[2] for b in batch], batch_first=True)
    return x, valid & ok, torch.stack([b[1] for b in batch])
