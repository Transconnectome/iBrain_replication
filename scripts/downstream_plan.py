"""The units of one pretraining checkpoint's downstream evaluation (U44), one line of scripts/finetune.py arguments
each, for the job array of scripts/slurm/finetune.sbatch:
    python scripts/downstream_plan.py --config configs/paper.yaml --ckpt <ckpt.pt> --root <root> --protocol probe > plan.txt
    sbatch --array=1-$(wc -l < plan.txt) scripts/slurm/finetune.sbatch configs/paper.yaml plan.txt
--protocol probe: intermediate checkpoints, finetune.probe seeds and Brain Treebank subjects; full: finetune.seeds and
every subject (the final checkpoint). Checkpoint arms (finetune, finetune_frozen) go to <root>/step_<step>/; scratch
and ridge do not depend on the checkpoint and go to <root>/scratch/ and <root>/ridge/ once. Units whose result.json
exists are left out, so a plan after a failure holds only what remains; the count goes to stderr.
Without --ckpt only scratch and ridge are planned."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from ibrain.data_treebank import TASKS, TEST_TRIALS
from ibrain.repro import load_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--root", required=True, help="downstream root: <root>/step_<step>, <root>/scratch, <root>/ridge")
    ap.add_argument("--protocol", choices=("probe", "full"), default="full")
    ap.add_argument("--dataset", nargs="+", default=None, help="default: entries without variant: true")
    ap.add_argument("--arm", nargs="+", default=("finetune", "finetune_frozen", "scratch", "ridge"))
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    fc = cfg["finetune"]
    probe = fc.get("probe", {}) if a.protocol == "probe" else {}
    seeds = probe.get("seeds", fc["seeds"])
    root = Path(a.root)
    outs = {"scratch": root / "scratch", "ridge": root / "ridge"}
    if a.ckpt:
        step = load_checkpoint(a.ckpt)["step"]
        outs["finetune"] = outs["finetune_frozen"] = root / f"step_{step:07d}"
    datasets = fc["datasets"]
    names = a.dataset or [n for n, ds in datasets.items() if not ds.get("variant")]
    n_units, n_done = 0, 0

    def unit(out, dataset, arm, seed=None, task=None, subject=None):
        nonlocal n_units, n_done
        n_units += 1
        d = out / dataset
        if task:
            d = d / task / subject
        d = d / arm / (f"seed{seed}" if seed is not None else "")
        if (d / "result.json").exists():
            n_done += 1
            return
        line = (f"--ckpt {a.ckpt} " if arm.startswith("finetune") else "") + f"--out {out} --dataset {dataset} --arm {arm}"
        if seed is not None:
            line += f" --seed {seed}"
        if task:
            line += f" --task {task} --subject {subject}"
        print(line)

    for n in names:
        ds = datasets[n]
        arms = [m for m in a.arm if m in outs and (ds["format"] != "treebank" or m != "ridge")]
        if ds["format"] == "treebank":
            tasks = ds.get("tasks", list(TASKS))
            subjects = probe.get("subjects") or ds.get("subjects") or sorted(TEST_TRIALS, key=lambda s: int(s.split("_")[1]))
            for task in tasks:
                for s in subjects:
                    for arm in arms:
                        for seed in seeds:
                            unit(outs[arm], n, arm, seed, task, s)
        else:
            for arm in arms:
                for seed in (seeds if arm != "ridge" else [None]):
                    unit(outs[arm], n, arm, seed)
    print(f"{n_units - n_done} units to run, {n_done} done", file=sys.stderr)


if __name__ == "__main__":
    main()
