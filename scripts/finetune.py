"""Downstream evaluation (M4, M5). Downstream datasets are listed in the config under finetune.datasets, each with a
format and its own options; --dataset picks one or more of them (default: all). Results in out/<dataset>/.
    synthetic  M4 synthetic regression check, no files
    nwb        spike velocity regression from DANDI NWB files (U29): root, glob, behavior. Three arms (finetune /
               scratch / ridge) × seeds on one trial-level 80/20 split, R². Files without the behavior series (NLB
               test files) are skipped
    treebank   Brain Treebank Pitch / Volume / Onset / Speech per subject (U34, U35): root, split, cache_dir, and
               optionally tasks, subjects and the reader options of data_treebank.read_trials (electrodes, reref,
               notch_hz, stat_seconds, drop_corrupted). Neural arms × seeds, AUC averaged over subjects
An option a format does not know is an error, so a misspelled key cannot be ignored silently. An entry with
variant: true (e.g. treebank_popt, the PopT signal condition) runs only when --dataset names it.
Example: python scripts/finetune.py --config configs/paper.yaml --ckpt runs/joint_s0/final.pt --out runs/ft_s0 \\
             --dataset mc_maze treebank
Not yet run on real data (SPEC M4, M5)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, so `ibrain` imports without install

import numpy as np
import torch
import yaml

from ibrain.data_nwb import MissingSeries, read_nwb
from ibrain.data_treebank import TASKS, TEST_TRIALS, read_trials, split_subject, subject_trials
from ibrain.finetune import fit_arm, fit_classifier_arm, ieeg_steps, ridge_r2, split_trials, synthetic_labeled
from ibrain.repro import run_meta, write_json


def arms_of(ckpt):
    """(name, ckpt, frozen) of the neural arms: finetune and finetune_frozen need a checkpoint, scratch does not."""
    return ([("finetune", ckpt, False), ("finetune_frozen", ckpt, True)] if ckpt else []) + [("scratch", None, False)]


def run_regression(windows, cfg, a, out, meta):
    """M4: ridge and the neural arms × seeds on one trial-level 80/20 split, R² (U11, U12, U27)."""
    fc = cfg["finetune"]
    write_json(out / "meta.json", {**run_meta(cfg, None), "ckpt": a.ckpt, **meta})
    train, test = split_trials(windows, 0.8, fc["split_seed"])  # all arms use the same split (M4)
    print(f"sessions={len({w['session_id'] for w in windows})} windows train={len(train)} test={len(test)}")
    results = {"split_seed": fc["split_seed"], "n_train": len(train), "n_test": len(test), "arms": {}}
    score, lam = ridge_r2(train, test, seed=fc["split_seed"])
    results["arms"]["ridge"] = {"r2": [score], "lambda": lam}
    for name, ckpt, frozen in arms_of(a.ckpt):
        r2s = []
        for seed in fc["seeds"]:
            r2, hist = fit_arm(train, test, cfg["model"], seed, ckpt=ckpt, head=fc["head"], frozen=frozen,
                               epochs=fc["epochs"], lr=fc["lr"], wd=fc["wd"], batch_size=fc["batch_size"],
                               device=a.device)
            r2s.append(r2)
            print(f"{name} seed={seed} R2={r2:.4f} loss {hist[0]:.4f} -> {hist[-1]:.4f}")
        results["arms"][name] = {"r2": r2s, "seeds": fc["seeds"]}
    for k, v in results["arms"].items():
        v["mean"], v["std"] = float(np.mean(v["r2"])), float(np.std(v["r2"]))  # U12
        print(f"{k:16s} R2 = {v['mean']:.4f} ± {v['std']:.4f}")
    write_json(out / "results.json", results)


def run_synthetic(ds, cfg, a, out):
    run_regression(synthetic_labeled(200, 30, seed=cfg["finetune"]["split_seed"]), cfg, a, out, {"format": "synthetic"})


def run_nwb(ds, cfg, a, out):
    """ds: root, glob (default every .nwb below root), behavior."""
    files = sorted(Path(ds["root"]).glob(ds.get("glob", "**/*.nwb")))
    if not files:
        sys.exit(f"no NWB files under {ds['root']} matching {ds.get('glob', '**/*.nwb')!r}")
    windows, used = [], []
    for f in files:
        try:
            ws = read_nwb(f, ds["behavior"])
        except MissingSeries:
            print(f"skip {f.name}: no {ds['behavior']!r} series (no labels)")
            continue
        print(f"{f.name}: {len(ws)} windows, {ws[0]['counts'].shape[1] if ws else 0} units")
        windows += ws
        used.append(str(f))
    if not windows:
        sys.exit(f"no labeled windows found for behavior {ds['behavior']!r}")
    run_regression(windows, cfg, a, out, {"format": "nwb", "behavior": ds["behavior"], "nwb_files": used})


TREEBANK_READER = ("electrodes", "reref", "notch_hz", "stat_seconds", "drop_corrupted", "cache_dir")


def run_treebank(ds, cfg, a, out):
    """U35: per subject and task, the neural arms × seeds on the subject's split (U34). The task score of a seed is the
    mean AUC over subjects; the reported value is the mean ± std of that over seeds (U12). No linear baseline.
    A subject whose train or test windows lack a class is skipped with a warning and left out of the mean.
    results.json is rewritten after every subject ("complete": false until the end), so a crash keeps what is done.
    ds: root, split (heldout | popt), and optionally tasks, subjects and the read_trials options TREEBANK_READER."""
    fc = cfg["finetune"]
    root, split, tasks = ds["root"], ds.get("split", "heldout"), ds.get("tasks", list(TASKS))
    subjects = ds.get("subjects") or sorted(TEST_TRIALS, key=lambda s: int(s.split("_")[1]))
    n_ieeg = ieeg_steps(a.ckpt) if a.ckpt else None
    if n_ieeg == 0:
        print("WARNING: the checkpoint has no iEEG steps (spike-only pretraining). Its iEEG encoder and iEEG type "
              "embedding are untrained; the paper reports no Brain Treebank numbers for spike-only pretraining.")
    kw = {k: ds[k] for k in TREEBANK_READER if k in ds}
    write_json(out / "meta.json", {**run_meta(cfg, None), "ckpt": a.ckpt, "ckpt_ieeg_steps": n_ieeg, "format": "treebank",
                                   "root": root, "split": split, "tasks": tasks, "subjects": subjects, "reader": kw})
    if kw.get("cache_dir"):  # read and filter each trial once for all tasks; split_subject then loads the cached arrays
        for s in subjects:
            for t in subject_trials(root, s):
                read_trials(root, s, t, tasks, **kw)
    results = {"split": split, "seeds": fc["seeds"], "complete": False, "tasks": {}}
    for task in tasks:
        per_subject = {}
        results["tasks"][task] = {"subjects": per_subject}
        for s in subjects:
            train, _, test = split_subject(root, s, task, mode=split, **kw)  # val is not used (U27)
            per_subject[s] = {"n_train": len(train), "n_test": len(test), "arms": {}}
            classes = {part: sorted({w["label"] for w in ws}) for part, ws in (("train", train), ("test", test))}
            if any(c != [0, 1] for c in classes.values()):
                per_subject[s]["skipped"] = f"classes train={classes['train']} test={classes['test']}"
                print(f"WARNING: {task} {s} skipped, left out of the subject mean: {per_subject[s]['skipped']}")
                write_json(out / "results.json", results)
                continue
            print(f"{task} {s}: channels={test[0]['wave'].shape[0]} train={len(train)} test={len(test)}")
            for name, ckpt, frozen in arms_of(a.ckpt):
                aucs = []
                for seed in fc["seeds"]:
                    score, hist = fit_classifier_arm(train, test, cfg["model"], seed, ckpt=ckpt, head=fc["head"],
                                                     frozen=frozen, epochs=fc["epochs"], lr=fc["lr"], wd=fc["wd"],
                                                     batch_size=fc["batch_size"], device=a.device)
                    aucs.append(score)
                    print(f"{task} {s} {name} seed={seed} AUC={score:.4f} loss {hist[0]:.4f} -> {hist[-1]:.4f}")
                per_subject[s]["arms"][name] = {"auc": aucs}
            write_json(out / "results.json", results)
        used = [s for s in subjects if "skipped" not in per_subject[s]]
        summary = {}
        for name, _, _ in arms_of(a.ckpt) if used else ():
            by_seed = np.mean([per_subject[s]["arms"][name]["auc"] for s in used], axis=0)  # (seeds,)
            summary[name] = {"auc_by_seed": by_seed.tolist(), "mean": float(by_seed.mean()),
                             "std": float(by_seed.std()), "n_subjects": len(used)}
            print(f"{task:8s} {name:16s} AUC = {summary[name]['mean']:.4f} ± {summary[name]['std']:.4f} "
                  f"({len(used)} of {len(subjects)} subjects)")
        results["tasks"][task]["arms"] = summary
        write_json(out / "results.json", results)
    results["complete"] = True
    write_json(out / "results.json", results)


FORMATS = {"synthetic": run_synthetic, "nwb": run_nwb, "treebank": run_treebank}
OPTIONS = {"synthetic": (), "nwb": ("root", "glob", "behavior"),
           "treebank": ("root", "split", "tasks", "subjects") + TREEBANK_READER}  # besides "format" and "variant"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None, help="Pretraining checkpoint. If absent, only the scratch arm (and ridge)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", nargs="+", default=None,
                    help="names under finetune.datasets (default: all entries without variant: true)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    datasets = cfg["finetune"].get("datasets") or {}
    names = a.dataset or [n for n, ds in datasets.items() if not ds.get("variant")]
    unknown = [n for n in names if n not in datasets]
    if unknown or not names:
        ap.error(f"unknown or no dataset {unknown}; finetune.datasets in {a.config} has {list(datasets)}")
    for n in names:
        ds = datasets[n]
        if ds.get("format") not in FORMATS:
            ap.error(f"dataset {n!r}: format must be one of {list(FORMATS)}, got {ds.get('format')!r}")
        extra = sorted(set(ds) - {"format", "variant", *OPTIONS[ds["format"]]})
        if extra:
            ap.error(f"dataset {n!r}: unknown option(s) {extra} for format {ds['format']!r}; "
                     f"allowed {list(OPTIONS[ds['format']])}")
    for n in names:
        print(f"===== {n} ({datasets[n]['format']})")
        ds = {k: v for k, v in datasets[n].items() if k != "variant"}
        FORMATS[ds["format"]](ds, cfg, a, Path(a.out) / n)


if __name__ == "__main__":
    main()
