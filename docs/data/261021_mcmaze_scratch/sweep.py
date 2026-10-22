"""One-off (2026-10-20): why does the MC-Maze scratch arm reach R2 0.004 and ridge 0.338 (lambda at the grid edge)?
Scratch arm only (no pretraining checkpoint). Hyperparameters are chosen on an inner trial-level 80/20 split of the
training trials; the test split is scored once, for the chosen setting only.
  1. ridge: the U27 lambda grid, a wider grid, and +-1 / causal 2-patch context of binned counts (test R2; ridge picks
     its own lambda on an inner split)
  2. scratch sweep, 100 epochs, inner-validation R2 every 10 epochs: lr {1e-4, 3e-4, 1e-3} x {constant,
     5% warmup + cosine}; one fp32 run of the current setting (lr 1e-4 constant) to rule out bf16
  3. the best (setting, epoch) by validation R2, retrained on the whole training split, test R2 once
Results: results.json next to this file, rewritten after every run."""
import json
import math
import sys
import time
from pathlib import Path

REPO = Path("/pscratch/sd/z/zres0710/iBrain_replication")
sys.path.insert(0, str(REPO))

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from ibrain.data_nwb import MissingSeries, read_nwb
from ibrain.finetune import (PRECISION, AttnPoolHead, LabeledWindows, Regressor, _autocast, collate_labeled, predict,
                             r2, ridge_r2, split_trials)
from ibrain.model import IBrain
from ibrain.repro import seed_all

HERE = Path(__file__).resolve().parent
cfg = yaml.safe_load((REPO / "configs/paper.yaml").read_text())
fc, ds = cfg["finetune"], cfg["finetune"]["datasets"]["mc_maze"]
results = {"ridge": {}, "sweep": {}, "final": None}


def save():
    (HERE / "results.json").write_text(json.dumps(results, indent=1) + "\n")


def r2_of(reg, ws, mu, sd, amp):
    pred, y = predict(reg, DataLoader(LabeledWindows(ws, mu, sd), 64, collate_fn=collate_labeled), "cuda", amp)
    return r2(pred.reshape(-1, 2), y.reshape(-1, 2))


def fit(tr, ev, lr, sched, epochs, amp, eval_every=10):
    """Scratch arm on tr; R2 on ev every eval_every epochs. Same model, head, AdamW, wd, clip and batch as U27."""
    seed_all(0)
    C = max(w["counts"].shape[1] for w in tr + ev)
    reg = Regressor(IBrain(**cfg["model"]), AttnPoolHead(cfg["model"]["d"], C)).cuda()
    Y = np.concatenate([np.asarray(w["vel"], np.float64) for w in tr])
    mu, sd = Y.mean(0), Y.std(0) + 1e-8
    dl = DataLoader(LabeledWindows(tr, mu, sd), fc["batch_size"], shuffle=True, collate_fn=collate_labeled,
                    generator=torch.Generator().manual_seed(0), num_workers=4)
    opt = torch.optim.AdamW(reg.parameters(), lr=lr, weight_decay=fc["wd"])
    total, warm = epochs * len(dl), max(1, int(0.05 * epochs * len(dl)))
    rule = (lambda t: 1.0) if sched == "const" else \
        (lambda t: (t + 1) / warm if t < warm else 0.5 * (1 + math.cos(math.pi * (t - warm) / (total - warm))))
    sch = torch.optim.lr_scheduler.LambdaLR(opt, rule)
    curve, t0 = [], time.time()
    for ep in range(1, epochs + 1):
        reg.train()
        tot, n = 0.0, 0
        for x, valid, y in dl:
            x, valid, y = x.cuda(), valid.cuda(), y.cuda()
            with _autocast("cuda", amp):
                out = reg(x, valid)
            loss = F.mse_loss(out.float(), y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reg.parameters(), 1.0)
            opt.step()
            sch.step()
            tot, n = tot + loss.item() * len(x), n + len(x)
        if ep % eval_every == 0 or ep == epochs:
            curve.append({"epoch": ep, "train_loss": tot / n, "r2": r2_of(reg, ev, mu, sd, amp),
                          "train_r2": r2_of(reg, tr[:1000], mu, sd, amp), "sec": time.time() - t0})
            print(f"  epoch {ep}: train loss {tot / n:.4f}, train R2 {curve[-1]['train_r2']:.4f}, "
                  f"eval R2 {curve[-1]['r2']:.4f} ({curve[-1]['sec']:.0f} s)", flush=True)
    return curve


def context(ws, offsets):
    """Counts of the patches at the given offsets placed side by side as channels (zeros outside the window), so
    ridge_r2's patch features become several patches' bins."""
    P = 5
    out = []
    for w in ws:
        c = np.asarray(w["counts"], np.float64)
        parts = []
        for o in offsets:
            s = np.zeros_like(c)
            if o < 0:
                s[-o * P:] = c[:o * P]
            elif o > 0:
                s[:-o * P] = c[o * P:]
            else:
                s = c
            parts.append(s)
        out.append({**w, "counts": np.concatenate(parts, 1)})
    return out


def main():
    windows = []
    for f in sorted(Path(ds["root"]).glob(ds.get("glob", "**/*.nwb"))):
        try:
            windows += read_nwb(f, ds["behavior"])
        except MissingSeries:
            pass
    train, test = split_trials(windows, 0.8, fc["split_seed"])
    inner_tr, inner_va = split_trials(train, 0.8, fc["split_seed"])
    print(f"windows train={len(train)} test={len(test)}; inner train={len(inner_tr)} val={len(inner_va)}")
    v = np.concatenate([w["vel"] for w in train])
    print(f"velocity per dim: mean {v.mean(0)}, std {v.std(0)}; mean count per bin {np.mean([w['counts'].mean() for w in train]):.4f}")

    print("== ridge (test R2)")
    wide = tuple(10.0 ** k for k in range(-2, 7))
    for name, tr, te, lams in (("U27 grid", train, test, (1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)),
                               ("wide grid", train, test, wide),
                               ("context -1..+1", context(train, (-1, 0, 1)), context(test, (-1, 0, 1)), wide),
                               ("causal -2..0", context(train, (-2, -1, 0)), context(test, (-2, -1, 0)), wide)):
        score, lam = ridge_r2(tr, te, lams=lams, seed=fc["split_seed"])
        results["ridge"][name] = {"r2": score, "lambda": lam}
        print(f"  {name}: R2 {score:.4f} lambda {lam:g}", flush=True)
        save()

    print("== scratch sweep (inner validation R2)")
    runs = [(lr, s, 100, "bf16") for s in ("const", "cosine") for lr in (1e-4, 3e-4, 1e-3)] + [(1e-4, "const", 50, "fp32")]
    for lr, s, ep, prec in runs:
        key = f"lr={lr:g} {s} {prec} {ep}ep"
        print(f"-- {key}", flush=True)
        results["sweep"][key] = fit(inner_tr, inner_va, lr, s, ep, PRECISION[prec])
        save()

    best = max(((k, c) for k, cs in results["sweep"].items() for c in cs if "bf16" in k), key=lambda kc: kc[1]["r2"])
    lr, s = float(best[0].split()[0][3:]), best[0].split()[1]
    print(f"== final: {best[0]} at epoch {best[1]['epoch']} (val R2 {best[1]['r2']:.4f}), whole train -> test", flush=True)
    curve = fit(train, test, lr, s, best[1]["epoch"], PRECISION["bf16"], eval_every=best[1]["epoch"])
    results["final"] = {"setting": best[0], "epochs": best[1]["epoch"], "val_r2": best[1]["r2"],
                        "test_r2": curve[-1]["r2"], "train_r2": curve[-1]["train_r2"]}
    save()
    print(json.dumps(results["final"]))


if __name__ == "__main__":
    main()
