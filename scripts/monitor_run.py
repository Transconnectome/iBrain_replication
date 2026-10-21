"""Health check of a pretraining run: Slurm jobs, progress, losses and collapse metrics, checkpoints, backup.
Standard library only and reads only the end of log.jsonl, so it runs on a login node in about a second.
  .venv/bin/python scripts/monitor_run.py --out ../iBrain_runs/joint_s0 --job-name ibrain-joint-s0 \
      --report-dir ../iBrain_runs/monitor/joint_s0 --backup-log ~/iBrain_backup/backup.log
Per signal type (0 spike, 1 iEEG) the last --window logged steps ("cur") are compared with the --window before them
("prev"). Medians, not means: full-data iEEG has rare artifact batches with L_rec hundreds of times a normal one.
Each finding is OK / WARN / CRIT; the exit code is 0 / 1 / 2 for the worst one. With --report-dir, latest.md and
latest.json are rewritten and one line of metrics is appended to history.jsonl (long-term trends).
"""
import argparse
import json
import math
import re
import statistics as st
import subprocess
import time
from datetime import datetime
from pathlib import Path

LEVELS = {"OK": 0, "WARN": 1, "CRIT": 2}
TH = {
    "stall_min": 30,          # RUNNING but log.jsonl unchanged this long
    "ckpt_age_h": 2.0,        # RUNNING but ckpt.pt older than this (ckpt_every 1000 steps is about 12 min)
    "rec_rise": {0: 0.20, 1: 0.30},  # relative rise of the median L_rec, cur vs prev
    "outlier_x": 10.0,        # a batch with L_rec above this times the window median is an outlier
    "outlier_frac": {0: 0.005, 1: 0.05},  # seed 0's first segment: spike 0, iEEG about 2%
    "d_gap_warn": 0.3, "d_gap_crit": 0.1, "d_gap_drop": 0.2,  # U38: about 0 when windows cannot be told apart
    "r_std_drop": 0.5,        # cur median below this fraction of prev
    "sec_rise": 0.5,          # relative slowdown of the median step time
    "backup_age_h": 14.0,     # the backup cron runs every 12 h
}
FAILED = ("FAILED", "NODE_FAIL", "OUT_OF_MEMORY", "BOOT_FAIL", "DEADLINE", "CANCELLED")


def tail_lines(path, n, block=1 << 20):
    """The last n lines of a text file, read from the end."""
    with open(path, "rb") as f:
        f.seek(0, 2)
        end = pos = f.tell()
        data = b""
        while pos > 0 and data.count(b"\n") <= n:
            pos = max(0, pos - block)
            f.seek(pos)
            data = f.read(end - pos)
    return data.decode(errors="replace").splitlines()[-n:]


def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def slurm_jobs(name, since):
    """(queued [{id, state, start, reason, limit}], history [{id, state, start, end, elapsed}] oldest first)."""
    queued = []
    for line in sh(["squeue", "-h", "-n", name, "-o", "%i|%T|%S|%r|%l"]).splitlines():
        i, s, start, reason, limit = line.split("|")
        queued.append({"id": i, "state": s, "start": start, "reason": reason, "limit": limit})
    hist = []
    for line in sh(["sacct", "-n", "-X", "-P", "--name", name, "-S", since, "-D",
                    "-o", "JobID,State,Start,End,Elapsed"]).splitlines():
        i, s, start, end, el = line.split("|")
        hist.append({"id": i, "state": s.split()[0], "start": start, "end": end, "elapsed": el})
    return queued, hist


def window_stats(rows):
    rec = [r["rec"] for r in rows]
    med = st.median(rec)
    return {"n": len(rows), "first": rows[0]["step"], "last": rows[-1]["step"], "rec": med,
            "rec_max": max(rec), "outlier_frac": sum(x > TH["outlier_x"] * med for x in rec) / len(rec),
            **{k: st.median(r[k] for r in rows) for k in ("align", "d_gap", "r_std", "sec")}}


def check(a):
    out, now = Path(a.out), time.time()
    found = []

    def add(level, code, msg):
        found.append({"level": level, "code": code, "msg": msg})

    log = out / "log.jsonl"
    rows = [json.loads(x) for x in tail_lines(log, 4 * a.window + 10) if x.strip()]
    last = rows[-1]["step"]
    log_age = (now - log.stat().st_mtime) / 60
    meta = json.loads((out / "meta.json").read_text())
    pc = meta.get("config", {}).get("pretrain", {})

    # total steps from the latest Slurm stdout of this run ("mode=... steps=N ...")
    total = None
    outs = sorted(Path(a.slurm_logs).glob(f"{a.job_name}_*.out"), key=lambda p: p.stat().st_mtime)
    for p in reversed(outs):
        m = re.search(r"\bsteps=(\d+)", p.read_text(errors="replace"))
        if m:
            total = int(m.group(1))
            break

    # Slurm
    complete = (out / "final.pt").exists()
    queued, hist = slurm_jobs(a.job_name, a.since)
    running = [j for j in queued if j["state"] == "RUNNING"]
    if complete:
        add("OK", "complete", "final.pt exists")
    elif not queued:
        add("CRIT", "no_job", f"run not complete (step {last}) and no {a.job_name} job queued")
    for j in queued:
        add("OK", "queued", f"job {j['id']} {j['state']} limit {j['limit']} start {j['start']} ({j['reason']})")
    if hist and hist[-1]["state"].startswith(FAILED) and not running:
        lvl = "CRIT" if not queued else "WARN"
        add(lvl, "job_failed", f"last job {hist[-1]['id']} {hist[-1]['state']} at {hist[-1]['end']}")
    starts = sum(1 for j in hist if j["start"] not in ("Unknown", "None") and j["start"] >= a.since)
    if starts:
        add("OK", "segments", f"{starts} job record(s) since {a.since}")

    # errors in the stderr of a running job
    for j in running:
        err = Path(a.slurm_logs) / f"{a.job_name}_{j['id']}.err"
        if err.exists():
            text = "\n".join(tail_lines(err, 200))
            if re.search(r"Traceback|Error:", text):
                add("CRIT", "traceback", f"{err.name} contains a traceback")

    # progress
    if running and log_age > TH["stall_min"]:
        add("CRIT", "stall", f"job RUNNING but log.jsonl unchanged for {log_age:.0f} min (last step {last})")
    steps = [r["step"] for r in rows]
    if any(b != s + 1 for s, b in zip(steps, steps[1:])):
        add("WARN", "step_gap", "steps in the log tail are not consecutive")
    bad = [r["step"] for r in rows if any(isinstance(v, float) and not math.isfinite(v) for v in r.values())]
    if bad:
        add("CRIT", "nonfinite", f"NaN/Inf in {len(bad)} logged steps, first {bad[0]}")

    # losses and collapse metrics per signal type
    sig = {}
    for s in (0, 1):
        rs = [r for r in rows if r.get("sig") == s and all(math.isfinite(r[k]) for k in ("rec", "d_gap", "r_std"))]
        if len(rs) < 2 * a.window:
            continue
        cur, prev = window_stats(rs[-a.window:]), window_stats(rs[-2 * a.window:-a.window])
        sig[s] = {"cur": cur, "prev": prev}
        name = ("spike", "iEEG")[s]
        if cur["rec"] > prev["rec"] * (1 + TH["rec_rise"][s]):
            add("WARN", f"rec_rise_{s}", f"{name} median L_rec {prev['rec']:.3f} -> {cur['rec']:.3f}")
        if cur["outlier_frac"] > TH["outlier_frac"][s]:
            add("WARN", f"outliers_{s}", f"{name} {cur['outlier_frac']:.1%} of batches above "
                f"{TH['outlier_x']:.0f}x the median L_rec (max {cur['rec_max']:.1f})")
        if cur["d_gap"] < TH["d_gap_crit"]:
            add("CRIT", f"collapse_{s}", f"{name} d_gap {cur['d_gap']:.3f}: windows hardly told apart")
        elif cur["d_gap"] < TH["d_gap_warn"] or cur["d_gap"] < prev["d_gap"] - TH["d_gap_drop"]:
            add("WARN", f"d_gap_{s}", f"{name} d_gap {prev['d_gap']:.3f} -> {cur['d_gap']:.3f}")
        if cur["r_std"] < TH["r_std_drop"] * prev["r_std"]:
            add("WARN", f"r_std_{s}", f"{name} r_std {prev['r_std']:.4f} -> {cur['r_std']:.4f}")
        if cur["sec"] > prev["sec"] * (1 + TH["sec_rise"]):
            add("WARN", f"slow_{s}", f"{name} median s/step {prev['sec']:.2f} -> {cur['sec']:.2f}")

    # checkpoints
    ck = out / "ckpt.pt"
    ck_age = (now - ck.stat().st_mtime) / 3600 if ck.exists() else None
    if running and (ck_age is None or ck_age > TH["ckpt_age_h"]):
        add("WARN", "ckpt_old", f"ckpt.pt {'missing' if ck_age is None else f'{ck_age:.1f} h old'} while running")
    keep = pc.get("keep_every")
    snaps = sorted(int(p.stem.split("_")[1]) for p in out.glob("ckpt_[0-9]*.pt"))
    if keep:
        missing = [k for k in range(keep, last + 2, keep) if k not in snaps]  # ckpt_<steps done>.pt
        if missing:
            add("WARN", "snapshot_missing", f"epoch snapshots missing for steps {missing[:5]}")

    # backup
    if a.backup_log and Path(a.backup_log).expanduser().exists():
        line = tail_lines(Path(a.backup_log).expanduser(), 1)[0]
        m = re.match(r"(\S+)\s.*?(?:fail=(\d+))?\)?$", line)
        try:
            age = (now - datetime.fromisoformat(m.group(1)).timestamp()) / 3600
        except (AttributeError, ValueError):
            age = None
        if age is None or age > TH["backup_age_h"]:
            add("WARN", "backup_old", f"last backup line is {'unreadable' if age is None else f'{age:.1f} h old'}")
        if m and m.group(2) and int(m.group(2)) > 0:
            add("WARN", "backup_fail", f"backup reported fail={m.group(2)}")

    mean_sec = st.mean(r["sec"] for r in rows[-2 * a.window:] if math.isfinite(r["sec"]))
    eta_h = (total - last) * mean_sec / 3600 if total else None
    return {"time": datetime.now().astimezone().isoformat(timespec="seconds"), "out": str(out), "step": last,
            "total": total, "progress": last / total if total else None, "log_age_min": log_age,
            "ckpt_age_h": ck_age, "snapshots": snaps, "mean_sec": mean_sec, "eta_compute_h": eta_h,
            "queued": queued, "history": hist[-5:], "sig": sig,
            "level": max((f["level"] for f in found), key=LEVELS.get, default="OK"), "findings": found}


def markdown(r):
    p = f" / {r['total']:,} ({r['progress']:.2%})" if r["total"] else ""
    eta = f", about {r['eta_compute_h'] / 24:.1f} days of compute left" if r["eta_compute_h"] else ""
    lines = [f"# {r['level']}: {Path(r['out']).name} at {r['time']}", "",
             f"- step {r['step']:,}{p}, {r['mean_sec']:.2f} s/step{eta}",
             f"- log.jsonl updated {r['log_age_min']:.0f} min ago; ckpt.pt "
             + (f"{r['ckpt_age_h']:.1f} h old" if r["ckpt_age_h"] is not None else "missing")
             + f"; snapshots {r['snapshots'] or 'none'}", ""]
    if r["sig"]:
        lines += ["| signal | window steps | median L_rec | outliers | align | d_gap | r_std | s/step |",
                  "|---|---|---|---|---|---|---|---|"]
        for s, w in r["sig"].items():
            for k in ("prev", "cur"):
                x = w[k]
                lines.append(f"| {('spike', 'iEEG')[s]} {k} | {x['first']}-{x['last']} | {x['rec']:.3f} | "
                             f"{x['outlier_frac']:.1%} (max {x['rec_max']:.1f}) | {x['align']:.3f} | "
                             f"{x['d_gap']:.3f} | {x['r_std']:.4f} | {x['sec']:.2f} |")
        lines.append("")
    lines += [f"- **{f['level']}** `{f['code']}`: {f['msg']}" for f in
              sorted(r["findings"], key=lambda f: -LEVELS[f["level"]])]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="the run's output directory (log.jsonl, meta.json, ckpt.pt)")
    ap.add_argument("--job-name", required=True, help="Slurm job name of the run's segments")
    ap.add_argument("--slurm-logs", default=str(Path(__file__).resolve().parent.parent / "logs"),
                    help="directory of <job-name>_<id>.out/.err (the submit directory's logs/)")
    ap.add_argument("--since", default="2026-10-01", help="sacct start date for the job history")
    ap.add_argument("--window", type=int, default=2000, help="logged steps per signal type in each window")
    ap.add_argument("--backup-log", default=None)
    ap.add_argument("--report-dir", default=None)
    ap.add_argument("--json", action="store_true", help="print JSON instead of markdown")
    a = ap.parse_args()
    r = check(a)
    md = markdown(r)
    if a.report_dir:
        d = Path(a.report_dir)
        d.mkdir(parents=True, exist_ok=True)
        (d / "latest.md").write_text(md)
        (d / "latest.json").write_text(json.dumps(r, indent=1) + "\n")
        keep = {k: r[k] for k in ("time", "step", "level", "mean_sec", "log_age_min")}
        keep["sig"] = {s: w["cur"] for s, w in r["sig"].items()}
        keep["codes"] = [f["code"] for f in r["findings"] if f["level"] != "OK"]
        with open(d / "history.jsonl", "a") as f:
            f.write(json.dumps(keep) + "\n")
    print(json.dumps(r, indent=1) if a.json else md, end="")
    raise SystemExit(LEVELS[r["level"]])


if __name__ == "__main__":
    main()
