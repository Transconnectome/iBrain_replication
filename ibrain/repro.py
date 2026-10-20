"""Reproducibility helpers: seeds, run metadata, RNG state, checkpoints (SPEC U26)."""
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)  # numpy takes only 32 bits; a resume reseeds with 1_000_003 * step + rank
    torch.manual_seed(seed)  # seeds CUDA / MPS generators too


def git_hash():
    """HEAD hash, "-dirty" appended when the tree has uncommitted changes. None outside a git repo."""
    root = Path(__file__).resolve().parent.parent
    try:
        h = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, stderr=subprocess.DEVNULL, text=True)
        return h + ("-dirty" if dirty.strip() else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def run_meta(config, seed):
    """Everything needed to re-run: config, seed, code version, library versions, command line, time."""
    return {"config": config, "seed": seed, "git": git_hash(), "torch": torch.__version__,
            "python": sys.version.split()[0], "platform": platform.platform(), "argv": sys.argv,
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}


def rng_state():
    st = {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(st):
    # .cpu(): load_checkpoint(path, "cuda") maps these ByteTensors to the GPU, and torch accepts only CPU ones
    torch.set_rng_state(st["torch"].cpu())
    np.random.set_state(st["numpy"])
    random.setstate(st["python"])
    if "cuda" in st and torch.cuda.is_available():
        # a checkpoint saved where more GPUs were visible restores only the visible ones
        for i, s in enumerate(st["cuda"][:torch.cuda.device_count()]):
            torch.cuda.set_rng_state(s.cpu(), i)


def save_checkpoint(path, model, opt, step, hist, meta):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "hist": hist,
                "meta": meta, "rng": rng_state()}, path)


def load_checkpoint(path, device="cpu"):
    return torch.load(path, map_location=device, weights_only=False)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
