# ibrain-repro
Replication codebase for iBrain, written from the paper since no official code was released
[[arxiv]](https://arxiv.org/abs/2609.06960)
[[spec]](SPEC.md)

Everything described in the paper is implemented and tested on synthetic data. No real-data run has been done yet.
The choices the paper leaves open, and why we made them, are listed in `SPEC.md` section 2.

### Installation

First, install `uv` by following the steps [here](https://docs.astral.sh/uv/getting-started/installation/). Then, create your Python environment:

```bash
uv venv .venv -p 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
```

To check that everything works, run the test suite. It takes about two minutes on CPU.

```bash
pytest
```

### Pretraining on synthetic data
To see the whole pipeline run before you have any data, train on the built-in synthetic spikes and iEEG:

```bash
python scripts/pretrain.py --config configs/tiny.yaml --seed 0 --out runs/tiny_s0
```

This writes `meta.json`, `log.jsonl`, `ckpt.pt` and `final.pt` under `runs/tiny_s0`.
Checkout `configs/tiny.yaml` for all configurations available.

### Pretraining on spikes
To pretrain on real spikes you first need the [Neural Pile primate](https://huggingface.co/datasets/eminorhan/neural-pile-primate) parquet files, the corpus the paper used.
Set `data.spike.root` in `configs/paper.yaml` to the folder that holds them. As in the paper the whole pile is used, although it contains the Perich and Area2-Bump evaluation sessions; set `data.spike.exclude_sources: [perich, area2-bump]` for the variant without them. Each step uses 256 random windows per type, the paper's 8 GPUs x 32. Sessions differ in unit count (96 at the median, up to 1,734), so the 256 are computed in micro-batches of similar unit count with at most `token_budget` channel slots each; the gradient is the same as for the 256 at once (SPEC U33). Training runs in bf16 autocast (`precision`, U36), and `grad_checkpoint: true` trades about one extra forward for memory (U37).

Then you can pretrain on spikes only by running:

```bash
python scripts/pretrain.py --config configs/paper.yaml --seed 0 --out runs/spike_s0 --spike-only
```

On the lab server, submit the same run through SLURM instead. GPU jobs must not be launched from a login shell.

```bash
sbatch scripts/slurm/pretrain.sbatch configs/paper.yaml 0 runs/spike_s0 --spike-only
```

Before loading data, the SLURM script checks that the largest micro-batches of `token_budget` fit on the GPU and stops if they do not. For a first run on the lab server use `configs/smoke.yaml` (200 pile rows, 0.5 h of each iEEG set, 200 steps):

```bash
sbatch --time=02:00:00 --job-name=ibrain-smoke scripts/slurm/pretrain.sbatch configs/smoke.yaml 0 runs/smoke_s0
```

The pile is held in memory, so set `data.spike.max_rows` for a small first run. We run seeds 0, 1 and 2 for every configuration. To continue an interrupted run, add `--resume runs/spike_s0/ckpt.pt`. The SLURM script does this by itself when a checkpoint exists.
Checkout `configs/paper.yaml` for all configurations available.

### Downstream evaluation
Every downstream dataset is an entry under `finetune.datasets` in the config, with a `format` and the options of that format. `configs/paper.yaml` lists the eight benchmarks of the paper:

| Name | Format | Options |
|---|---|---|
| `mc_maze`, `area2_bump` | `nwb` | `root`, `behavior: hand_vel` |
| `perich_tco`, `perich_trt` | `nwb` | `root`, `glob: "sub-T/*ses-CO*.nwb"` (or `*ses-RT*`), `behavior: cursor_vel` |
| `treebank` | `treebank` | `root`, `split: heldout`, `cache_dir`, optional `tasks` and `subjects` |

`--dataset` picks one or more of them by name, and leaving it out runs all of them. Each one writes `out/<name>/meta.json` and `out/<name>/results.json`:

```bash
python scripts/finetune.py --config configs/paper.yaml --ckpt runs/joint_s0/final.pt --out runs/ft_s0 --dataset mc_maze treebank
```

For the spike benchmarks ([DANDI 000128](https://dandiarchive.org/dandiset/000128), [000127](https://dandiarchive.org/dandiset/000127), [000688](https://dandiarchive.org/dandiset/000688)), the spikes and the velocity labels are both read from the NWB files, and files without the behavior series, such as the NLB test file, are skipped. Three arms (fine-tuning from the checkpoint, the same model from scratch, and ridge regression on binned counts) share one trial-level 80/20 split and report R².
`configs/tiny.yaml` has a `synthetic` entry for checking the evaluation code without data.
The NWB reader has been tested on synthetic files in both layouts but not yet on the real files, so expect to touch it on the first try.

### Joint pretraining with iEEG
To add iEEG you first need AJILE12 ([DANDI 000055](https://dandiarchive.org/dandiset/000055)) and the [SWEC-ETHZ](https://huggingface.co/datasets/NeuroTec/SWEC_iEEG_Dataset) HDF5 files. Reading SWEC needs `hdf5plugin`, which is installed with this package.
Set the two `root` paths under `data.ieeg` in `configs/paper.yaml`. Recordings are streamed block by block, AJILE12 keeps only good channels and skips Blocklist epochs, and SWEC is resampled from 1024 to 500 Hz.

Then you can run the paper's joint pretraining, alternating spike and iEEG batches, by leaving out `--spike-only`:

```bash
sbatch scripts/slurm/pretrain.sbatch configs/paper.yaml 0 runs/joint_s0
```

### Brain Treebank
The `treebank` entry builds the Pitch, Volume, Onset and Speech examples of the [Brain Treebank](https://braintreebank.dev) with the PopT rules, cut to 1 s windows at 500 Hz (SPEC U34). Each of the 7 subjects with more than one trial is fine-tuned on its other trials and tested on its held-out PopT trial. The model encodes with the iEEG encoder and iEEG type embedding of pretraining, the head gives one logit per window, and the score is AUC averaged over subjects (SPEC U35). Only the three neural arms run; there is no linear baseline. With `cache_dir`, every trial is read and filtered once for all four tasks.

## Cite
This repository is not affiliated with the iBrain authors. Please cite [their paper](https://arxiv.org/abs/2609.06960) if you use this code in your own work:

```bibtex
@article{
    chen2026ibrain,
    title={iBrain: A Unified Foundation Model Reading the Brain from Surface to Spikes},
    author={Ying Chen and Tiou Wang and Zhifeng Yue},
    journal={arXiv preprint arXiv:2609.06960},
    year={2026},
}
```
