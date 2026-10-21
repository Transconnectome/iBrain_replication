"""Collects the result.json of every downstream unit under <root> (U44) into <root>/summary/results.csv (one row per
unit) and <root>/summary/table.md: per dataset (and Brain Treebank task), arms × pretraining checkpoint steps,
mean ± std over seeds. For Brain Treebank the value of a seed is its mean AUC over the subjects that have a result
for every arm and step shown (U35 (e)); skipped subjects (skipped.json) are listed.
    python scripts/downstream_summary.py --root /pscratch/sd/z/zres0710/iBrain_runs/downstream"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def rows_of(root):
    """One dict per result.json: group (step_<n> | scratch | ridge), dataset, task, subject, arm, seed, score."""
    out = []
    for p in sorted(root.rglob("result.json")):
        parts = p.relative_to(root).parts[:-1]  # group, dataset, [task, subject], arm, [seed<k>]
        if len(parts) < 3:
            continue
        r = json.loads(p.read_text())
        group, dataset, rest = parts[0], parts[1], parts[2:]
        task, subject = (rest[0], rest[1]) if len(rest) >= 4 else ("", "")
        arm = rest[-2] if rest[-1].startswith("seed") else rest[-1]
        seed = int(rest[-1][4:]) if rest[-1].startswith("seed") else -1
        step = int(group[5:]) if group.startswith("step_") else -1
        out.append({"group": group, "step": step, "dataset": dataset, "task": task, "subject": subject, "arm": arm,
                    "seed": seed, "metric": r["metric"], "score": r["score"], "seconds": r.get("seconds"),
                    "ckpt": (r.get("ckpt") or {}).get("path", ""), "git": r.get("git", "")})
    return out


def column(row):
    """Table column of a row: scratch and ridge first, then checkpoint steps ascending."""
    return (0, row["arm"]) if row["step"] < 0 else (1, row["step"])


def cell(scores):
    """scores: {seed: value}. mean ± std over seeds (n)."""
    v = np.array(list(scores.values()), dtype=float)
    return f"{v.mean():.3f} ± {v.std():.3f} (n={len(v)})" if len(v) > 1 else f"{v[0]:.3f}"


def table(rows, root):
    lines = []
    by_block = defaultdict(list)
    for r in rows:
        by_block[(r["dataset"], r["task"])].append(r)
    for (dataset, task), rs in sorted(by_block.items()):
        cols = sorted({column(r) for r in rs})
        names = [c[1] if c[0] == 0 else f"step {c[1]:,}" for c in cols]
        lines += [f"### {dataset}" + (f" / {task}" if task else "") + f" ({rs[0]['metric']})", ""]
        if task:  # Treebank: subject mean per seed, over the subjects present for every (arm, column)
            per = defaultdict(lambda: defaultdict(dict))  # (arm, col) -> seed -> subject -> score
            for r in rs:
                per[(r["arm"], column(r))][r["seed"]][r["subject"]] = r["score"]
            subjects = set.intersection(*[set(s) for d in per.values() for s in d.values()]) if per else set()
            skipped = sorted(p.parent.name for p in (root / "scratch" / dataset / task).rglob("skipped.json")) \
                if (root / "scratch" / dataset / task).exists() else []
            lines.append(f"subjects in the mean: {sorted(subjects, key=lambda s: int(s.split('_')[1]))}"
                         + (f"; skipped: {skipped}" if skipped else ""))
            lines.append("")
            get = lambda arm, c: {seed: float(np.mean([d[s] for s in subjects])) for seed, d in per[(arm, c)].items()
                                  if subjects <= set(d)}
        else:
            per = defaultdict(dict)  # (arm, col) -> seed -> score
            for r in rs:
                per[(r["arm"], column(r))][r["seed"]] = r["score"]
            get = lambda arm, c: per[(arm, c)]
        lines += ["| arm | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        for arm in ("ridge", "scratch", "finetune_frozen", "finetune"):
            cells = [cell(get(arm, c)) if get(arm, c) else "" for c in cols]
            if any(cells):
                lines.append(f"| {arm} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    root = Path(a.root)
    rows = rows_of(root)
    (root / "summary").mkdir(parents=True, exist_ok=True)
    with open(root / "summary" / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["group"])
        w.writeheader()
        w.writerows(rows)
    md = table(rows, root)
    (root / "summary" / "table.md").write_text(md)
    print(md)
    print(f"{len(rows)} units -> {root / 'summary'}")


if __name__ == "__main__":
    main()
