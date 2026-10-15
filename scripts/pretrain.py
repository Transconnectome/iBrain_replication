"""Run pretraining. Example: python scripts/pretrain.py --config configs/tiny.yaml --seed 0 --out runs/tiny_s0
Spike data: Neural Pile (format pile, or pile_mmap for the converted copy of scripts/pile_to_memmap.py), nhp-spike-corpus
shards (corpus) or synthetic. iEEG data: a list of AJILE12 and SWEC sources streamed by block, "synthetic", or null.
--spike-only ignores iEEG (M3); --ieeg-only ignores spike data (iEEG pilots, U42).
Outputs: out/meta.json, log.jsonl, ckpt.pt (latest), final.pt. Resume with --resume out/ckpt.pt.
Several GPUs (U43): started by srun in scripts/slurm/pretrain_ddp.sbatch, each rank takes batch_size / world windows
of each step; only rank 0 prints and writes."""
import argparse
import itertools
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, so `ibrain` imports without install

import torch
import yaml
from torch.utils.data import DataLoader, DistributedSampler, IterableDataset

from ibrain import dist

from ibrain.data_ieeg import IEEGStream, SyntheticIEEG, ajile12_blocks, select_hours, swec_blocks
from ibrain.data_pile import PileMemmap, PileWindows
from ibrain.data_spike import SYNTHETIC, SpikeWindows, collate, read_windows, write_synthetic
from ibrain.model import IBrain, IEEG, SPIKE
from ibrain.pretrain import pretrain, total_steps
from ibrain.repro import run_meta, seed_all, write_json

IEEG_SOURCES = {"ajile12": ajile12_blocks, "swec": swec_blocks}


def spike_data(sc, out, seed):
    """data.spike -> (Dataset, hours)."""
    fmt = sc["format"]
    if fmt in ("pile", "pile_mmap"):  # pile_mmap: the converted copy of scripts/pile_to_memmap.py (U30)
        reader = PileWindows if fmt == "pile" else PileMemmap
        ds = reader(sc["root"], sc.get("split", "train"), sc.get("exclude_sources", ()), sc.get("max_rows"), seed)
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


def ieeg_data(ic, seed, buffer, open_blocks=8, shard=None):
    """data.ieeg -> (Dataset, hours). open_blocks: blocks each DataLoader worker mixes at once (U41).
    shard: (rank, world, seed) to split the blocks over several ranks (U43)."""
    if ic == "synthetic":
        return SyntheticIEEG(n_windows=16, seed=seed), None
    blocks = []
    for e in ic:
        bl = select_hours(IEEG_SOURCES[e["format"]](e["root"]), e.get("max_hours"), seed)
        print(f"iEEG {e['format']}: {len(bl)} blocks, {sum((b.stop - b.start) / b.fs for b in bl) / 3600:.1f} h")
        blocks += bl
    ds = IEEGStream(blocks, buffer, open_blocks=open_blocks, shard=shard)
    return ds, ds.hours


def loader(ds, bs, g, workers, seed=0):
    """bs windows per batch. With several ranks a map-style dataset goes through a DistributedSampler (every rank
    the same number of batches; pretrain() calls set_epoch at each pass). A stream drops each worker's last, partial
    batch, and so does a DistributedSampler loader: a batch of one window fails the head's BatchNorm (U7), and a rank
    with a short batch would leave the ranks' shares unequal (seen with 8 ranks, 2026-10-12)."""
    if isinstance(ds, IterableDataset):  # order comes from the stream itself (seeded by the DataLoader base seed)
        return DataLoader(ds, bs, collate_fn=collate, num_workers=workers, drop_last=True)
    if dist.world() > 1:
        sampler = DistributedSampler(ds, dist.world(), dist.rank(), shuffle=True, seed=seed)
        return DataLoader(ds, bs, sampler=sampler, collate_fn=collate, num_workers=workers, drop_last=True)
    return DataLoader(ds, bs, shuffle=True, collate_fn=collate, generator=g, num_workers=workers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", default=None)
    only = ap.add_mutually_exclusive_group()
    only.add_argument("--spike-only", action="store_true", help="ignore data.ieeg (spike-only pretraining, M3)")
    only.add_argument("--ieeg-only", "--iEEG-only", dest="ieeg_only", action="store_true",
                      help="ignore data.spike (iEEG-only pretraining, U42); a streamed iEEG source needs pretrain.steps")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    rank, world, local = dist.init()
    if world > 1:
        a.device = f"cuda:{local}" if torch.cuda.is_available() else "cpu"
        if rank:  # only rank 0 prints; errors still reach stderr
            sys.stdout = open(os.devnull, "w")
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    out = Path(a.out)
    seed_all(a.seed)
    meta = run_meta(cfg, a.seed)
    if dist.is_main():
        write_json(out / "meta.json", meta)

    pc, dc = cfg["pretrain"], cfg["data"]
    workers = pc.get("num_workers", 0)
    g = torch.Generator().manual_seed(a.seed)
    bs = pc["batch_size"]
    if world > 1:  # U43: a step's batch_size windows per type are shared out over the ranks
        group = cfg["model"].get("head_bn_group") if cfg["model"].get("head_bn") else None
        if bs % world or (group and (bs // world) % group):
            sys.exit(f"batch_size {bs} must split into {world} equal rank batches"
                     f"{f' that are multiples of head_bn_group {group}' if group else ''}")
        bs //= world
    if a.ieeg_only and not dc.get("ieeg"):
        sys.exit("--ieeg-only needs data.ieeg")
    loaders, ds_spike, spike_h, ieeg_h = [], None, None, None
    if not a.ieeg_only:
        ds_spike, spike_h = spike_data(dc["spike"], out, a.seed)
        loaders.append((SPIKE, loader(ds_spike, bs, g, workers, a.seed)))
    if dc.get("ieeg") and not a.spike_only:
        shard = (rank, world, a.seed) if world > 1 else None
        ds_ieeg, ieeg_h = ieeg_data(dc["ieeg"], a.seed, pc.get("shuffle_buffer", 1024), pc.get("open_blocks", 8), shard)
        if shard and isinstance(ds_ieeg, IEEGStream) and len(ds_ieeg.blocks) < world * max(workers, 1):
            sys.exit(f"{len(ds_ieeg.blocks)} iEEG blocks for {world} ranks x {max(workers, 1)} workers: every worker "
                     "needs at least one block")
        loaders.append((IEEG, loader(ds_ieeg, bs, g, workers, a.seed)))
    accum = pc.get("grad_accum", 1)  # U33
    steps = pc["steps"]
    if not steps:  # U20: epochs count passes over the spike loader, x2 under 1:1 alternation
        if not a.ieeg_only:
            steps = total_steps(pc["epochs"], len(loaders[0][1]), len(loaders), accum)
        elif isinstance(ds_ieeg, IterableDataset):
            sys.exit("--ieeg-only with a streamed iEEG source needs pretrain.steps: the stream has no length to count "
                     "epochs over")
        else:  # iEEG-only with a map-style source (synthetic): epochs count passes over the iEEG loader
            steps = total_steps(pc["epochs"], len(loaders[0][1]), 1, accum)
    precision = pc.get("precision", "fp32")  # U36
    mode = "spike_only" if a.spike_only or not dc.get("ieeg") else "ieeg_only" if a.ieeg_only else "joint"
    meta["data"] = {"mode": mode, "spike_windows": len(ds_spike) if ds_spike is not None else None,
                    "spike_hours": spike_h, "ieeg_hours": ieeg_h, "steps": steps,
                    "grad_accum": accum, "windows_per_step": pc["batch_size"] * accum, "precision": precision,
                    "token_budget": pc.get("token_budget"), "grad_checkpoint": pc.get("grad_checkpoint", False),
                    "compile": pc.get("compile", False), "world": world}
    if dist.is_main():
        write_json(out / "meta.json", meta)
    n_spike = len(loaders[0][1]) if ds_spike is not None else None
    print(f"mode={mode} steps={steps} spike batches/epoch={n_spike} grad_accum={accum} loaders={len(loaders)} "
          f"device={a.device} precision={precision} world={world} windows/rank={bs}")
    if pc["warmup"] >= steps:
        print(f"WARNING: warmup {pc['warmup']} >= {steps} steps: lr never reaches its peak and the cosine decay never runs. "
              "Lower pretrain.warmup for small runs (U20, U33)")

    model = IBrain(**cfg["model"])  # the same seed on every rank: the same initialization (pretrain broadcasts it)
    if world > 1:
        seed_all(a.seed + 1_000_003 * rank)  # masks, channel views and dropout differ across ranks
    pretrain(model, loaders, steps, pc["warmup"], a.device, out_dir=out, ckpt_every=pc["ckpt_every"],
             resume=a.resume, meta=meta, accum=accum, precision=precision,
             token_budget=pc.get("token_budget"), grad_checkpoint=pc.get("grad_checkpoint", False),
             keep_every=pc.get("keep_every"), use_compile=pc.get("compile", False))
    dist.finish()


if __name__ == "__main__":
    main()
