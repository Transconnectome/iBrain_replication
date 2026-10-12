"""Representation collapse check of a pretraining checkpoint against the same model at initialization.
GPU: run through scripts/slurm/check_collapse.sbatch. Real windows: spike from a few Neural Pile files, iEEG from a few
AJILE12/SWEC blocks. Per signal type and model, on n windows in eval mode (no dropout, no masking):
  q_std    per-dimension std of the l2-normalized projection q, averaged over dimensions (SimSiam's collapse metric:
           about 1/sqrt(d_proj) when spread out, 0 when every window gives the same q)
  q_cos    mean cosine between the q of two different windows (1 = one point)
  r_cos    the same for the pooled backbone representation r (before the SimSiam head)
  r_erank  effective rank of the centred r (exp of the entropy of the normalized singular values; 1 = one direction)
  D_same   SimSiam loss between two channel views of the same window (what pretraining minimizes)
  D_other  the same loss between views of two different windows; close to D_same means the head cannot tell windows apart
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
import yaml

from ibrain.data_ieeg import IEEGStream, ajile12_blocks, select_hours, swec_blocks
from ibrain.data_pile import pile_files, row_matrices
from ibrain.data_spike import WINDOW_BINS, collate, to_patches
from ibrain.model import IBrain, IEEG, SPIKE
from ibrain.pretrain import pack, sample_views
from ibrain.repro import load_checkpoint


def spike_windows(root, n, files, seed):
    """n random 1 s windows from `files` Neural Pile parquet files spread over the split (several sources)."""
    rng = np.random.default_rng(seed)
    paths = pile_files(root)
    paths = [paths[i] for i in np.linspace(0, len(paths) - 1, files).round().astype(int)]
    mats, srcs = [], []
    for p in paths:
        t = pq.read_table(p, columns=["spike_counts", "source_dataset"])
        mats += row_matrices(t.column(0))
        srcs += t.column(1).to_pylist()
    idx = [(m, w) for m, mat in enumerate(mats) for w in range(mat.shape[1] // WINDOW_BINS) if mat.shape[0] > 0]
    pick = rng.choice(len(idx), n, replace=False)
    out = [to_patches(mats[m][:, w * WINDOW_BINS:(w + 1) * WINDOW_BINS].T) for m, w in (idx[i] for i in pick)]
    print(f"spike: {n} windows from {len(paths)} files, sources {sorted({srcs[idx[i][0]] for i in pick})}")
    return out


def ieeg_windows(cfg, n, hours, seed):
    blocks = []
    for e in cfg["data"]["ieeg"]:
        reader = {"ajile12": ajile12_blocks, "swec": swec_blocks}[e["format"]]
        blocks += select_hours(reader(e["root"]), hours, seed)
    out = []
    torch.manual_seed(seed)
    for w in IEEGStream(blocks, buffer=4096):
        out.append(w)
        if len(out) >= 4 * n:
            break
    pick = np.random.default_rng(seed).choice(len(out), min(n, len(out)), replace=False)
    print(f"iEEG: {len(pick)} windows from {len(blocks)} blocks")
    return [out[i] for i in pick]


@torch.no_grad()
def embed(model, windows, sig, budget, device, views=None):
    """r (n, d) and q, p (n, d_proj) per window, computed in unit-count micro-batches. views: per-window channel mask."""
    x, valid = collate(windows)
    if views is not None:
        valid = views
    # width each window needs: its last valid channel + 1 (a view is not a prefix of the channels)
    width = (valid * torch.arange(1, valid.shape[1] + 1)).amax(1)
    r = torch.zeros(len(windows), model.backbone.norm.normalized_shape[0])
    for idx in pack(width, budget):
        c = int(width[idx].max())
        xc, vc = x[idx, :c].to(device), valid[idx, :c].to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            r[idx] = model.pool(model.encode(xc, vc, sig), vc).float().cpu()
    q, p = model.align(r.to(device))
    return r, q.float().cpu(), p.float().cpu()


def D(p, q):
    return -F.cosine_similarity(p, q, dim=-1).mean().item()


def metrics(model, windows, sig, budget, device, seed):
    r, q, _ = embed(model, windows, sig, budget, device)
    qn = F.normalize(q, dim=-1)
    n = len(q)
    off = ~torch.eye(n, dtype=torch.bool)
    rn = F.normalize(r, dim=-1)
    s = torch.linalg.svdvals(r - r.mean(0))
    pr = s / s.sum()
    torch.manual_seed(seed)
    _, valid = collate(windows)
    v1, v2 = sample_views(valid)
    _, q1, p1 = embed(model, windows, sig, budget, device, v1)
    _, q2, p2 = embed(model, windows, sig, budget, device, v2)
    perm = torch.roll(torch.arange(n), 1)  # pair each window with another one
    return {"q_std": qn.std(0).mean().item(), "q_cos": (qn @ qn.T)[off].mean().item(),
            "r_cos": (rn @ rn.T)[off].mean().item(), "r_erank": torch.exp(-(pr * pr.clamp_min(1e-12).log()).sum()).item(),
            "D_same": 0.5 * D(p1, q2) + 0.5 * D(p2, q1), "D_other": 0.5 * D(p1, q2[perm]) + 0.5 * D(p2, q1[perm])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default=None, help="default: the config stored in the checkpoint")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--pile-files", type=int, default=4)
    ap.add_argument("--ieeg-hours", type=float, default=0.3, help="per iEEG source")
    ap.add_argument("--budget", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = load_checkpoint(a.ckpt, "cpu")
    cfg = yaml.safe_load(Path(a.config).read_text()) if a.config else ck["meta"]["config"]
    trained = IBrain(**cfg["model"])
    trained.load_state_dict(ck["model"])
    torch.manual_seed(a.seed)
    init = IBrain(**cfg["model"])
    print(f"checkpoint {a.ckpt} step {ck['step']}; 1/sqrt(d_proj) = {cfg['model']['d_proj'] ** -0.5:.3f}")
    data = {SPIKE: spike_windows(cfg["data"]["spike"]["root"], a.n, a.pile_files, a.seed),
            IEEG: ieeg_windows(cfg, a.n, a.ieeg_hours, a.seed)}
    for sig, name in ((SPIKE, "spike"), (IEEG, "iEEG")):
        for label, m in (("init", init), ("trained", trained)):
            m.to(device).eval()
            res = metrics(m, data[sig], sig, a.budget, device, a.seed)
            print(f"{name:5s} {label:7s} " + " ".join(f"{k}={v:7.3f}" for k, v in res.items()))


if __name__ == "__main__":
    main()
