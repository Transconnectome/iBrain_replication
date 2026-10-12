"""M5: iEEG input (SPEC 1.1). 500 Hz, 1 s window = 500 samples, 10 patches of 100 ms x 50 samples.
One channel is one sequence. Channel padding reuses data_spike.collate.
Normalization is per channel with robust statistics of a whole recording or a long chunk of it, applied in the
Dataset, so input and Eq. (8) target are the same values and nothing depends on the mask (U4).
Real recordings are streamed one contiguous block at a time (IEEGStream), each block normalized with its own
statistics: AJILE12 NWB at 500 Hz (U31) and SWEC-ETHZ HDF5 at 1024 Hz with Blosc compression, resampled to 500 Hz
(U32). The offline preprocessing plan that may replace the stream is docs/260929_ieeg_offline_preprocessing.md.
SyntheticIEEG is for tests (U28)."""
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

FS, WINDOW_SAMPLES = 500, 500  # Hz, 1 s window
S, P = 10, 50  # 10 patches of 100 ms, 50 samples per patch
IQR_TO_STD = 1.349  # IQR of a standard normal


def to_patches(wave):
    """One window (C, 500) float -> (C, S, P) float32 (Eq. 1 input). x[c, s, p] = wave[c, 50s + p]."""
    x = torch.from_numpy(np.ascontiguousarray(wave, dtype=np.float32))
    return x.view(x.shape[0], S, P)


def channel_stats(wave):
    """Per-channel center and scale of one recording (U4). wave (C, T), T = the whole recording, not one window.
    center = median, scale = IQR / 1.349 (robust std; long recordings contain large artifacts)."""
    q25, q50, q75 = np.nanpercentile(wave, [25, 50, 75], axis=1)  # NaN samples ignored, not propagated
    return q50, (q75 - q25) / IQR_TO_STD


def channel_normalize(x, center, scale, eps=1e-6):
    """iEEG: normalize with the recording's per-channel statistics (U4). x has the channel axis first,
    e.g. (C, 500) or (C, S, P); center, scale (C,) from channel_stats. Applied in the Dataset, so the
    input and the Eq. (8) target are the same values and nothing depends on the mask."""
    shape = (-1,) + (1,) * (x.ndim - 1)
    return (x - center.reshape(shape)) / (scale.reshape(shape) + eps)


class SyntheticIEEG(Dataset):
    """Synthetic iEEG for tests. Each 'session' (= one recording) has a different number of channels;
    each channel is 3 sinusoids + noise + a DC offset. Per-channel scales differ, so each session is
    normalized with its own channel_stats over all of its windows (U4).
    The conv adapter (U1) is expensive per token, so CPU tests keep the channel count small."""

    def __init__(self, n_channels=(4, 8, 16), n_windows=16, seed=0):
        rng = np.random.default_rng(seed)
        t = np.arange(WINDOW_SAMPLES) / FS
        self.windows, self.session, self.stats = [], [], []
        for i, C in enumerate(n_channels):
            f = rng.uniform(1, 40, (C, 3, 1))
            a = rng.uniform(0.5, 2, (C, 3, 1))
            ph = rng.uniform(0, 2 * np.pi, (C, 3, 1))
            scale = rng.uniform(10, 500, (C, 1))  # uV-ish, per channel
            dc = rng.normal(0, 100, (C, 1))
            ws = []
            for _ in range(n_windows):
                off = rng.uniform(0, 1)
                w = (a * np.sin(2 * np.pi * f * (t + off) + ph)).sum(1)  # (C, 500)
                ws.append(scale * (w + rng.normal(0, 0.3, (C, WINDOW_SAMPLES))) + dc)
            self.stats.append(channel_stats(np.concatenate(ws, axis=1)))  # the whole recording (C, n_windows × 500)
            self.windows += ws
            self.session += [i] * n_windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        return to_patches(channel_normalize(self.windows[i], *self.stats[self.session[i]]))


# ---------------- Real recordings, streamed by block ----------------

@dataclass(frozen=True)
class Block:
    """A contiguous stretch of one recording. start/stop are native sample indices on the time axis.
    cols: channel columns to keep (None = all). bad: (start_s, stop_s) intervals, seconds from sample 0 of the
    recording, whose windows are dropped."""
    path: str
    key: str
    start: int
    stop: int
    fs: float
    time_axis: int
    cols: tuple = None
    bad: tuple = ()


def resample_to_fs(x, fs):
    """(C, L) at fs -> (C, L') at 500 Hz. Polyphase with its built-in anti-aliasing filter (1024 Hz: up 125, down 256)."""
    if fs == FS:
        return x
    from scipy.signal import resample_poly
    f = Fraction(FS / fs).limit_denominator(1000)
    return resample_poly(x, f.numerator, f.denominator, axis=1).astype(np.float32)


def read_block(b, pad_seconds=1.0):
    """Block -> (C, L) float32 at 500 Hz. When resampling, reads pad_seconds of context on both sides and trims it
    afterwards, so windows at block edges match resampling of the whole recording."""
    import h5py
    import hdf5plugin  # noqa: F401  registers the Blosc filter used by SWEC
    pad = int(round(pad_seconds * b.fs)) if b.fs != FS else 0
    with h5py.File(b.path, "r") as h:
        d = h[b.key]
        n = d.shape[b.time_axis]
        a, z = max(0, b.start - pad), min(n, b.stop + pad)
        x = np.asarray(d[a:z] if b.time_axis == 0 else d[:, a:z], dtype=np.float32)
    if b.time_axis == 0:
        x = x.T
    if b.cols is not None:
        x = x[list(b.cols)]
    y = resample_to_fs(x, b.fs)
    lo = int(round((b.start - a) * FS / b.fs))
    return y[:, lo:lo + int(round((b.stop - b.start) * FS / b.fs))]


MIN_STAT_SECONDS, MIN_SCALE_RATIO = 30, 1e-3


def usable_channels(n_finite, scale):
    """Channels whose statistics can be used (U4, U31): at least MIN_STAT_SECONDS of finite samples, and a finite
    scale of at least MIN_SCALE_RATIO times the median scale of the channels that pass those checks. n_finite, scale (C,) of one
    statistics unit (block or chunk) -> (C,) bool."""
    ok = (np.asarray(n_finite) >= MIN_STAT_SECONDS * FS) & np.isfinite(scale) & (scale > 0)
    if ok.any():
        ok &= scale >= MIN_SCALE_RATIO * np.median(scale[ok])
    return ok


def normalize_block(b, x):
    """U4 at chunk level for one streamed block, x (C, L) at 500 Hz -> (C', L) normalized, C' <= C.
    Statistics ignore NaN samples and samples inside bad intervals (their windows are dropped anyway, and non-NaN
    Blocklist spans or artifacts would otherwise set the scale). A channel is left out of this block when it has fewer
    than 30 s of usable samples (a short stretch lets a window's own patches set its statistics, which leaks masked
    content) or when its scale is below 1e-3 of the block's median channel scale (flat, zero-filled or heavily
    quantized; eps alone would blow such a channel up to ~1e8)."""
    import warnings
    u = x.astype(np.float64)
    t0 = b.start / b.fs
    for a, z in b.bad:
        i, j = max(0, int(np.floor((a - t0) * FS))), min(x.shape[1], int(np.ceil((z - t0) * FS)))
        u[:, i:j] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN channels give NaN statistics and are left out below
        c, sc = channel_stats(u)
    ok = usable_channels(np.isfinite(u).sum(1), sc)
    return channel_normalize(x[ok], c[ok], sc[ok]).astype(np.float32)


def block_windows(b, x):
    """1 s windows (C, S, P) of one block, from the block start. Windows that overlap a bad interval or contain a
    non-finite sample are dropped."""
    t0 = b.start / b.fs
    for k in range(x.shape[1] // WINDOW_SAMPLES):
        s = t0 + k * WINDOW_SAMPLES / FS
        if any(a < s + WINDOW_SAMPLES / FS and s < z for a, z in b.bad):
            continue
        w = x[:, k * WINDOW_SAMPLES:(k + 1) * WINDOW_SAMPLES]
        if np.isfinite(w).all():
            yield to_patches(w)


def _blocks(path, key, n, fs, time_axis, block_seconds, cols=None, bad=()):
    """Consecutive blocks of block_seconds. Windows are cut from each block start, so block_seconds must be a whole
    number of 1 s windows; otherwise every block would drop its tail and restart the window grid."""
    if abs(block_seconds - round(block_seconds)) > 1e-9:
        raise ValueError(f"block_seconds={block_seconds} is not a whole number of 1 s windows")
    L = int(round(block_seconds * fs))
    starts = list(range(0, n, L))
    if len(starts) > 1 and n - starts[-1] < L / 2:
        starts.pop()  # a short tail joins the previous block, so no block's statistics come from a few seconds
    out = []
    for i, a in enumerate(starts):
        z = starts[i + 1] if i + 1 < len(starts) else n
        hit = tuple((u, v) for u, v in bad if u < z / fs and a / fs < v)
        out.append(Block(str(path), key, a, z, float(fs), time_axis, cols, hit))
    return out


def ajile12_blocks(root, block_seconds=675):
    """AJILE12 NWB files (DANDI 000055) -> blocks (U31). The signal is acquisition/ElectricalSeries (T, C) at 500 Hz,
    stored in chunks of 84375 samples (168.75 s) per channel. 675 s = 4 chunks is the shortest block that reads whole
    chunks and is also a whole number of 1 s windows. Only channels with electrodes.good
    are kept. Epochs whose label contains "Blocklist" (e.g. "Blocklist (Data break)") are dropped."""
    from pynwb import NWBHDF5IO
    from pynwb.ecephys import ElectricalSeries
    files = sorted(Path(root).rglob("*.nwb"))
    if not files:
        raise FileNotFoundError(f"no .nwb files under {root}")
    out = []
    for f in files:
        with NWBHDF5IO(str(f), "r", load_namespaces=True) as io:
            nwb = io.read()
            es = [o for o in nwb.acquisition.values() if isinstance(o, ElectricalSeries)]
            if len(es) != 1:
                raise ValueError(f"{f.name}: expected one ElectricalSeries in acquisition, found {len(es)}")
            es = es[0]
            fs, t_start, n, key = float(es.rate), float(es.starting_time or 0.0), es.data.shape[0], es.data.name
            rows = np.asarray(es.electrodes.data[:])
            etab = nwb.electrodes
            good = np.asarray(etab["good"][:], dtype=bool)[rows] if "good" in etab.colnames else np.ones(len(rows), bool)
            bad = []
            if nwb.epochs is not None and len(nwb.epochs):
                df = nwb.epochs.to_dataframe()
                hit = df.apply(lambda r: any("Blocklist" in str(v) for v in r.values), axis=1).to_numpy()
                bad = [(u - t_start, v - t_start) for u, v in zip(df["start_time"][hit], df["stop_time"][hit])]
        out += _blocks(f, key, n, fs, 0, block_seconds, tuple(np.flatnonzero(good).tolist()), tuple(bad))
    return out


def swec_blocks(root, block_seconds=None, target_seconds=900):
    """SWEC-ETHZ part files -> blocks (U32). data/ieeg is (C, T) with sampling_rate in the root attributes.
    The signal is compressed in chunks (3 min in the dataset card), and reading any sample decompresses its whole chunk.
    By default a block is the whole number of chunks closest to target_seconds, read from each file's own chunk size,
    so blocks start on chunk boundaries; the 1 s resampling context then touches only the two neighbouring chunks
    (900 s = 5 chunks: 7 chunks decompressed per 5 used, instead of about 2.4 per 256 s block before). block_seconds
    overrides this (tests).
    IDxx_total.h5 files are virtual datasets over the parts and are skipped, because a virtual dataset over parts that
    are still downloading reads as zeros. There is no channel metadata, so every channel is kept; seizures are kept.
    Files that cannot be opened (incomplete download) are skipped with a message."""
    import h5py
    import hdf5plugin  # noqa: F401
    files = [f for f in sorted(Path(root).rglob("*.h5")) if not f.name.endswith("_total.h5")]
    if not files:
        raise FileNotFoundError(f"no SWEC part .h5 files under {root}")
    out, skipped = [], []
    for f in files:
        try:
            with h5py.File(f, "r") as h:
                d = h["data/ieeg"]
                n, fs, chunk = d.shape[1], float(h.attrs["sampling_rate"]), (d.chunks or (None, None))[1]
        except (OSError, KeyError) as e:
            skipped.append(f"{f.name} ({type(e).__name__})")
            continue
        seconds = block_seconds
        if seconds is None:
            chunk_s = chunk / fs if chunk else None
            if chunk_s and abs(chunk_s - round(chunk_s)) < 1e-9:  # whole seconds, so whole 1 s windows per block
                seconds = chunk_s * max(1, round(target_seconds / chunk_s))
            else:
                seconds = target_seconds
        out += _blocks(f, "data/ieeg", n, fs, 1, seconds)
    if skipped:
        print(f"SWEC: skipped {len(skipped)} unreadable files: {skipped[:5]}")
    return out


def select_hours(blocks, max_hours, seed=0):
    """A seeded random subset of blocks totalling at most max_hours (all blocks if max_hours is None)."""
    if max_hours is None:
        return list(blocks)
    out, total = [], 0.0
    for i in np.random.default_rng(seed).permutation(len(blocks)):
        h = (blocks[i].stop - blocks[i].start) / blocks[i].fs / 3600
        if total + h > max_hours:
            continue
        out.append(blocks[i])
        total += h
    return out


class IEEGStream(IterableDataset):
    """1 s windows (C, S, P) streamed from blocks, normalized per block (normalize=False gives raw values, for tests).
    Each pass visits the blocks in a new random order, and each
    DataLoader worker takes a disjoint share of that order. Windows go through a shuffle buffer so that a batch mixes
    several blocks. The order comes from the DataLoader's base seed, so seed_all makes a run reproducible."""

    def __init__(self, blocks, buffer=1024, normalize=True):
        self.blocks, self.buffer, self.normalize = list(blocks), buffer, normalize
        self.hours = sum((b.stop - b.start) / b.fs for b in self.blocks) / 3600

    def __iter__(self):
        info = get_worker_info()
        base = (info.seed - info.id) if info else int(torch.randint(0, 2 ** 62, (1,)))
        order = np.random.default_rng(base).permutation(len(self.blocks))
        if info:
            order = order[info.id::info.num_workers]
        rng = np.random.default_rng(base + (info.id if info else 0) + 1)
        buf = []
        for i in order:
            b = self.blocks[i]
            x = read_block(b)
            if self.normalize:  # U4 at chunk level: statistics of this block (hundreds of seconds), never of a window
                x = normalize_block(b, x)
            if not len(x):  # no usable channel in this block
                continue
            for w in block_windows(b, x):
                buf.append(w)
                if len(buf) >= self.buffer:
                    j = int(rng.integers(len(buf)))
                    buf[j], buf[-1] = buf[-1], buf[j]
                    yield buf.pop()
        for j in rng.permutation(len(buf)):
            yield buf[j]
