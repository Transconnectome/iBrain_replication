"""Run pretraining. Example: python scripts/pretrain.py --config configs/tiny.yaml --seed 0 --out runs/tiny_s0
Spike data: Neural Pile (format pile), nhp-spike-corpus shards (corpus) or synthetic. iEEG data: a list of AJILE12 and
SWEC sources streamed by block, "synthetic", or null. --spike-only ignores iEEG (M3).
Outputs: out/meta.json, log.jsonl, ckpt.pt (latest), final.pt. Resume with --resume out/ckpt.pt."""
import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, so `ibrain` imports without install

import torch
import yaml
from torch.utils.data import DataLoader, IterableDataset

from ibrain.data_ieeg import IEEGStream, SyntheticIEEG, ajile12_blocks, select_hours, swec_blocks
from ibrain.data_pile import PileWindows
from ibrain.data_spike import SYNTHETIC, SpikeWindows, collate, read_windows, write_synthetic
from ibrain.model import IBrain, IEEG, SPIKE
from ibrain.pretrain import pretrain, total_steps
from ibrain.repro import run_meta, seed_all, write_json

IEEG_SOURCES = {"ajile12": ajile12_blocks, "swec": swec_blocks}


def spike_data(sc, out, seed):
    """data.spike -> (Dataset, hours)."""
    fmt = sc["format"]
    if fmt == "pile":
        ds = PileWindows(sc["root"], sc.get("split", "train"), sc.get("exclude_sources", ()), sc.get("max_rows"),
                         seed)
        return ds, ds.hours
    if fmt == "corpus":
        ds = SpikeWindows(itertools.chain.from_iterable(read_windows(sc["root"], s) for s in sc["sources"]))
    elif fmt == "synthetic":
        root = out / "data"
        if not root.exists():
            write_synthetic(root, (3, 8, 20), n_windows=16, seed=seed)
        ds = SpikeWindows(read_windows(root, SYNTHETIC))
    else:
        raise ValueError(f"unknown data.spike.format {fmt!r}")
    return ds, len(ds) / 3600


def ieeg_data(ic, seed, buffer):
    """data.ieeg -> (Dataset, hours)."""
    if ic == "synthetic":
        return SyntheticIEEG(n_windows=16, seed=seed), None
    blocks = []
    for e in ic:
        bl = select_hours(IEEG_SOURCES[e["format"]](e["root"]), e.get("max_hours"), seed)
        print(f"iEEG {e['format']}: {len(bl)} blocks, {sum((b.stop - b.start) / b.fs for b in bl) / 3600:.1f} h")
        blocks += bl
    ds = IEEGStream(blocks, buffer)
    return ds, ds.hours


def loader(ds, bs, g, workers):
    if isinstance(ds, IterableDataset):  # order comes from the stream itself (seeded by the DataLoader base seed)
        return DataLoader(ds, bs, collate_fn=collate, num_workers=workers)
    return DataLoader(ds, bs, shuffle=True, collate_fn=collate, generator=g, num_workers=workers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--spike-only", action="store_true", help="ignore data.ieeg (spike-only pretraining, M3)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    out = Path(a.out)
    seed_all(a.seed)
    meta = run_meta(cfg, a.seed)
    write_json(out / "meta.json", meta)

    pc, dc = cfg["pretrain"], cfg["data"]
    workers = pc.get("num_workers", 0)
    g = torch.Generator().manual_seed(a.seed)
    ds_spike, spike_h = spike_data(dc["spike"], out, a.seed)
    dl_spike = loader(ds_spike, pc["batch_size"], g, workers)
    loaders, ieeg_h = [(SPIKE, dl_spike)], None
    if dc.get("ieeg") and not a.spike_only:
        ds_ieeg, ieeg_h = ieeg_data(dc["ieeg"], a.seed, pc.get("shuffle_buffer", 1024))
        loaders.append((IEEG, loader(ds_ieeg, pc["batch_size"], g, workers)))
    accum = pc.get("grad_accum", 1)  # U33
    steps = pc["steps"] or total_steps(pc["epochs"], len(dl_spike), len(loaders), accum)  # U20: x2 under 1:1 alternation
    precision = pc.get("precision", "fp32")  # U36
    meta["data"] = {"spike_windows": len(ds_spike), "spike_hours": spike_h, "ieeg_hours": ieeg_h, "steps": steps,
                    "grad_accum": accum, "windows_per_step": pc["batch_size"] * accum, "precision": precision,
                    "token_budget": pc.get("token_budget"), "grad_checkpoint": pc.get("grad_checkpoint", False)}
    write_json(out / "meta.json", meta)
    print(f"steps={steps} spike batches/epoch={len(dl_spike)} grad_accum={accum} loaders={len(loaders)} device={a.device} "
          f"precision={precision}")
    if pc["warmup"] >= steps:
        print(f"WARNING: warmup {pc['warmup']} >= {steps} steps: lr never reaches its peak and the cosine decay never runs. "
              "Lower pretrain.warmup for small runs (U20, U33)")

    model = IBrain(**cfg["model"])
    pretrain(model, loaders, steps, pc["warmup"], a.device, out_dir=out, ckpt_every=pc["ckpt_every"],
             resume=a.resume, meta=meta, accum=accum, precision=precision,
             token_budget=pc.get("token_budget"), grad_checkpoint=pc.get("grad_checkpoint", False))


if __name__ == "__main__":
    main()
