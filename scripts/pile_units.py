"""Units (channels) per row and per 1 s window of the Neural Pile train split, by source. CPU only.
Example: python scripts/pile_units.py --root <pile root> --out runs/pile_units.npz --workers 8"""
import argparse
import collections
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def file_rows(path):
    """(n_units, n_bins, source) per row of one parquet file."""
    t = pq.read_table(path, columns=["spike_counts", "source_dataset"])
    arr = t.column(0).combine_chunks()
    n = arr.value_lengths().to_numpy()
    t_unit = arr.flatten().value_lengths().to_numpy()
    first = np.cumsum(n) - n  # index of each row's first unit
    bins = np.where(n > 0, t_unit[np.minimum(first, len(t_unit) - 1)], 0)
    return list(zip(n.tolist(), bins.tolist(), t.column(1).to_pylist()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    files = sorted(Path(a.root).rglob(f"{a.split}-*.parquet"))
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(file_rows, files):
            rows += r
    n = np.array([r[0] for r in rows])
    bins = np.array([r[1] for r in rows])
    src = np.array([r[2] for r in rows])
    w = bins // 50
    per_window = np.repeat(n, w)
    print(f"files {len(files)} rows {len(n)} windows {w.sum()} uint8 GB {(n * bins).sum() / 1e9:.1f}")
    print("units per row     p50/p90/p99/max", np.percentile(n, [50, 90, 99]).round().tolist(), n.max())
    print("units per window  p50/p75/p90/p95/p99/max", np.percentile(per_window, [50, 75, 90, 95, 99]).round().tolist(),
          per_window.max())
    for c in (128, 256, 384, 512, 768, 1024, 2048):
        print(f"windows with units > {c:4d}: {(per_window > c).mean() * 100:5.1f}%")
    by = collections.defaultdict(lambda: [0, 0, 0])
    for k, b, s in rows:
        by[s][0] += b // 50
        by[s][1] = max(by[s][1], k)
        by[s][2] += 1
    for s, (wn, mx, r) in sorted(by.items(), key=lambda x: -x[1][0]):
        print(f"{s:28s} windows {wn:9d} max_units {mx:5d} rows {r}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, n_units=n, n_bins=bins, source=src)


if __name__ == "__main__":
    main()
