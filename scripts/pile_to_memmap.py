"""One-time conversion of the Neural Pile parquet files to the memory-mapped layout read by format pile_mmap (U30).
CPU only: run through scripts/slurm/pile_to_memmap.sbatch. Example:
python scripts/pile_to_memmap.py --root <pile root> --out /pscratch/sd/z/zres0710/iBrain/cache/neural-pile-primate-mmap"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, so `ibrain` imports without install

from ibrain.data_pile import convert_memmap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Neural Pile folder with the <split>-*.parquet files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    t0 = time.time()
    out = convert_memmap(a.root, a.out, a.split, a.workers)
    meta = json.loads((out / "meta.json").read_text())
    print(f"{out}: {meta['rows']} rows from {len(meta['files'])} files, {meta['bytes'] / 1e9:.1f} GB, "
          f"{(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
