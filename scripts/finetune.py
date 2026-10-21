"""Downstream evaluation (M4, M5), one unit at a time or all of them (U44). Downstream datasets are listed in the
config under finetune.datasets, each with a format and its own options; --dataset picks one or more (default: all
entries without variant: true).
    synthetic  M4 synthetic regression check, no files
    nwb        spike velocity regression from DANDI NWB files (U29): root, glob, behavior. Arms finetune /
               finetune_frozen / scratch / ridge × seeds on one trial-level 80/20 split, R². Files without the
               behavior series (NLB test files) are skipped
    treebank   Brain Treebank Pitch / Volume / Onset / Speech per subject (U34, U35): root, split, cache_dir, and
               optionally tasks, subjects and the reader options of data_treebank.read_trials (electrodes, reref,
               notch_hz, stat_seconds, drop_corrupted, nonword_nan_rows, speech_negatives). Neural arms × seeds, AUC
A unit is one (dataset[, task, subject], arm, seed); --arm, --seed, --task, --subject restrict the run to some of
them, so a Slurm job array can run one unit per task (scripts/downstream_plan.py, scripts/slurm/finetune.sbatch).
Layout under --out (the summary over units is made by scripts/downstream_summary.py):
    <dataset>/meta.json                                       reading options, split sizes
    <dataset>/ridge/result.json                               M4
    <dataset>/<arm>/seed<k>/{result.json, params.pt, preds.npz}                    M4
    <dataset>/<task>/<subject>/<arm>/seed<k>/{result.json, params.pt, preds.npz}   M5
    <dataset>/<task>/<subject>/skipped.json                   a subject whose train or test lacks a class (U35 (e))
params.pt holds the parameters the fit trained, apart from the pretraining checkpoint (ibrain.finetune.save_fit).
A unit whose result.json exists is skipped unless --redo. --prepare reads the datasets (builds the Brain Treebank
cache) and trains nothing; run it once before a job array, which must not build the cache concurrently.
An option a format does not know is an error, so a misspelled key cannot be ignored silently. An entry with
variant: true (e.g. treebank_popt, the PopT signal condition) runs only when --dataset names it.
Example: python scripts/finetune.py --config configs/paper.yaml --ckpt runs/joint_s0/final.pt --out runs/ft_s0 \\
             --dataset mc_maze treebank"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, so `ibrain` imports without install

import torch
import yaml

from ibrain.data_nwb import MissingSeries, read_nwb
from ibrain.data_treebank import TASKS, TEST_TRIALS, read_trials, split_subject, subject_trials
from ibrain.finetune import PRECISION, fit_arm, fit_classifier_arm, ieeg_steps, ridge_r2, split_trials, synthetic_labeled
from ibrain.repro import git_hash, run_meta, write_json

ARMS = ("finetune", "finetune_frozen", "scratch", "ridge")


def arms_of(ckpt, wanted):
    """(name, ckpt, frozen) of the wanted neural arms: finetune and finetune_frozen need a checkpoint."""
    neural = ([("finetune", ckpt, False), ("finetune_frozen", ckpt, True)] if ckpt else []) + [("scratch", None, False)]
    return [t for t in neural if t[0] in wanted]


def fit_kwargs(cfg, a):
    fc = cfg["finetune"]
    return dict(head=fc["head"], epochs=fc["epochs"], lr=fc["lr"], wd=fc["wd"], batch_size=fc["batch_size"],
                device=a.device, amp=PRECISION[a.precision], num_workers=a.num_workers,
                cache_gib=fc.get("cache_gib", 12))


def done(unit, a):
    if (unit / "result.json").exists() and not a.redo:
        print(f"skip {unit}: result.json exists")
        return True
    return False


def write_meta_once(out, meta):
    if not (out / "meta.json").exists():
        write_json(out / "meta.json", meta)


def run_regression(windows, cfg, a, out, meta):
    """M4: ridge and the neural arms × seeds on one trial-level 80/20 split, R² (U11, U12, U27)."""
    fc = cfg["finetune"]
    train, test = split_trials(windows, 0.8, fc["split_seed"])  # all arms use the same split (M4)
    print(f"sessions={len({w['session_id'] for w in windows})} windows train={len(train)} test={len(test)}")
    if a.prepare:
        return
    write_meta_once(out, {**run_meta(cfg, None), **meta, "split_seed": fc["split_seed"], "n_train": len(train),
                          "n_test": len(test)})
    if "ridge" in a.arms and not done(out / "ridge", a):
        t0 = time.time()
        score, lam = ridge_r2(train, test, seed=fc["split_seed"])
        write_json(out / "ridge" / "result.json", {"metric": "r2", "score": score, "lambda": lam, "arm": "ridge",
                                                   "n_train": len(train), "n_test": len(test), "git": git_hash(),
                                                   "seconds": time.time() - t0})
        print(f"ridge R2={score:.4f} lambda={lam}")
    for name, ckpt, frozen in arms_of(a.ckpt, a.arms):
        for seed in a.seeds:
            unit = out / name / f"seed{seed}"
            if done(unit, a):
                continue
            r2, hist = fit_arm(train, test, cfg["model"], seed, ckpt=ckpt, frozen=frozen, out=unit, **fit_kwargs(cfg, a))
            print(f"{name} seed={seed} R2={r2:.4f} loss {hist[0]:.4f} -> {hist[-1]:.4f}")


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


TREEBANK_READER = ("electrodes", "reref", "notch_hz", "stat_seconds", "drop_corrupted", "cache_dir",
                   "nonword_nan_rows", "speech_negatives")


def run_treebank(ds, cfg, a, out):
    """U35: per subject and task, the neural arms × seeds on the subject's split (U34). The task score of a seed is the
    mean AUC over subjects (scripts/downstream_summary.py). No linear baseline. A subject whose train or test
    windows lack a class is recorded in skipped.json and left out of the mean.
    ds: root, split (heldout | popt), and optionally tasks, subjects and the read_trials options TREEBANK_READER;
    --task and --subject restrict them."""
    root, split = ds["root"], ds.get("split", "heldout")
    tasks = a.task or ds.get("tasks", list(TASKS))
    subjects = a.subject or ds.get("subjects") or sorted(TEST_TRIALS, key=lambda s: int(s.split("_")[1]))
    kw = {k: ds[k] for k in TREEBANK_READER if k in ds}
    if kw.get("cache_dir"):  # read and filter each trial once for all tasks; split_subject then loads the cached arrays
        for s in subjects:
            for t in subject_trials(root, s):
                read_trials(root, s, t, tasks, **kw)
    if a.prepare:
        return
    n_ieeg = ieeg_steps(a.ckpt) if a.ckpt else None
    if n_ieeg == 0:
        print("WARNING: the checkpoint has no iEEG steps (spike-only pretraining). Its iEEG encoder and iEEG type "
              "embedding are untrained; the paper reports no Brain Treebank numbers for spike-only pretraining.")
    write_meta_once(out, {**run_meta(cfg, None), "format": "treebank", "root": root, "split": split, "reader": kw})
    for task in tasks:
        for s in subjects:
            units = [(name, ckpt, frozen, seed) for name, ckpt, frozen in arms_of(a.ckpt, a.arms) for seed in a.seeds
                     if not done(out / task / s / name / f"seed{seed}", a)]
            if not units:
                continue
            train, _, test = split_subject(root, s, task, mode=split, **kw)  # val is not used (U27)
            classes = {part: sorted({w["label"] for w in ws}) for part, ws in (("train", train), ("test", test))}
            if any(c != [0, 1] for c in classes.values()):
                why = f"classes train={classes['train']} test={classes['test']}"
                print(f"WARNING: {task} {s} skipped, left out of the subject mean: {why}")
                write_json(out / task / s / "skipped.json", {"why": why, "n_train": len(train), "n_test": len(test)})
                continue
            print(f"{task} {s}: channels={test[0]['wave'].shape[0]} train={len(train)} test={len(test)}")
            for name, ckpt, frozen, seed in units:
                unit = out / task / s / name / f"seed{seed}"
                score, hist = fit_classifier_arm(train, test, cfg["model"], seed, ckpt=ckpt, frozen=frozen, out=unit,
                                                 **fit_kwargs(cfg, a))
                print(f"{task} {s} {name} seed={seed} AUC={score:.4f} loss {hist[0]:.4f} -> {hist[-1]:.4f}")


FORMATS = {"synthetic": run_synthetic, "nwb": run_nwb, "treebank": run_treebank}
OPTIONS = {"synthetic": (), "nwb": ("root", "glob", "behavior"),
           "treebank": ("root", "split", "tasks", "subjects") + TREEBANK_READER}  # besides "format" and "variant"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None, help="Pretraining checkpoint. If absent, only scratch (and ridge)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", nargs="+", default=None,
                    help="names under finetune.datasets (default: all entries without variant: true)")
    ap.add_argument("--arm", nargs="+", default=list(ARMS), choices=ARMS, help="arms to run (default: all)")
    ap.add_argument("--seed", nargs="+", type=int, default=None, help="run seeds (default: finetune.seeds)")
    ap.add_argument("--task", nargs="+", default=None, help="treebank: tasks (default: the entry's or all four)")
    ap.add_argument("--subject", nargs="+", default=None, help="treebank: subjects (default: the entry's or all)")
    ap.add_argument("--redo", action="store_true", help="rerun units whose result.json exists")
    ap.add_argument("--prepare", action="store_true", help="read the datasets (build the Brain Treebank cache), no fits")
    ap.add_argument("--precision", choices=list(PRECISION), default=None, help="default: finetune.precision or fp32")
    ap.add_argument("--num-workers", type=int, default=None, help="default: finetune.num_workers or 0")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    fc = cfg["finetune"]
    a.arms = set(a.arm)
    a.seeds = a.seed if a.seed is not None else fc["seeds"]
    a.precision = a.precision or fc.get("precision", "fp32")
    a.num_workers = a.num_workers if a.num_workers is not None else fc.get("num_workers", 0)
    datasets = fc.get("datasets") or {}
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
