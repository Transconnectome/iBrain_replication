# MC-Maze scratch and ridge on real data: first results and a diagnostic sweep

Status: **done, 2026-10-21.** First downstream runs on real data (SPEC M4, U27, U44). No pretraining checkpoint is used: only the two arms that do not depend on one (scratch, ridge). Raw results, the sweep script and its log are in `docs/data/261021_mcmaze_scratch/`. The decision that follows is recorded in SPEC U27.

## 1. Setting

- Data: DANDI 000128, `sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb` (the NLB test file has no `hand_vel` and is skipped, U29). 5,689 windows of 1 s, 182 units, mean 0.059 spikes per 20 ms bin. Velocity std 140 / 126 per dimension.
- Split: trial-level 80/20 with `split_seed` 0 (U11): 4,563 train / 1,126 test windows. For the sweep the training trials are split again 80/20 by trial (3,642 inner train / 921 validation); hyperparameters are chosen on validation only and the test split is scored once.
- Model and head as in `configs/paper.yaml`: the paper model (6 blocks, d 256), U8 attention-pooling head, labels z-scored with train statistics (U27).
- Hardware: one A100 of a shared node, bf16 autocast unless stated.

## 2. Protocol runs (U27 settings)

| Arm | Test $R^2$ | Notes | Time |
|---|---|---|---|
| ridge | **0.338** | λ = 1000, the upper end of the U27 grid | 5 s |
| scratch, seed 0 | **0.004** | lr 1e-4 constant, 50 epochs; train loss 1.09 -> 0.89 | 11 min |

Job 60077144, commit 38194f5 (results under `iBrain_runs/downstream/{ridge,scratch}/mc_maze/`).

## 3. Ridge: λ grid and temporal context

ridge_r2 as in U27 (one decoder per session, λ chosen on an inner trial split, refit on all train), with the counts of neighbouring patches appended as extra channels. Test $R^2$:

| Features per 100 ms patch | λ grid | λ chosen | Test $R^2$ |
|---|---|---|---|
| this patch (U27) | 1e-2 to 1e3 | 1e3 | 0.338 |
| this patch | 1e-2 to 1e6 | 1e4 | 0.340 |
| patches t-1, t, t+1 | 1e-2 to 1e6 | 1e4 | 0.541 |
| patches t-2, t-1, t (causal) | 1e-2 to 1e6 | 1e4 | **0.565** |

The grid edge was not the cause (1e4 is inside the wide grid and changes $R^2$ by 0.002). Temporal context is: past activity adds most, as expected when neural activity leads movement. That ridge reaches 0.57 with context means labels and spikes are aligned and the reader is sound.

## 4. Scratch: hyperparameter sweep

Six settings, 100 epochs each, validation $R^2$ every 10 epochs, plus the U27 setting in fp32 for 50 epochs. Warmup + cosine: 5% linear warmup, cosine to 0. Everything else as U27 (AdamW, wd 5e-2, clip 1.0, batch 32).

Validation $R^2$ (train $R^2$ on 1,000 training windows in brackets):

| Setting | ep 10 | ep 20 | ep 50 | ep 100 | Best val |
|---|---|---|---|---|---|
| lr 1e-4 constant | 0.009 (0.03) | -0.021 (-0.01) | 0.027 (0.07) | -0.527 (0.51) | 0.027 |
| lr 3e-4 constant | 0.006 (0.01) | **0.032** (0.05) | -0.009 (0.08) | -0.569 (0.74) | **0.032** |
| lr 1e-3 constant | -0.003 (0.00) | 0.002 (0.00) | 0.013 (0.02) | 0.008 (0.01) | 0.013 |
| lr 1e-4 cosine | 0.021 (0.04) | 0.024 (0.08) | -0.461 (0.71) | -0.584 (0.90) | 0.024 |
| lr 3e-4 cosine | 0.022 (0.03) | 0.021 (0.04) | 0.025 (0.06) | -0.410 (0.35) | 0.025 |
| lr 1e-3 cosine | -0.054 (-0.05) | 0.013 (0.02) | -0.001 (0.01) | -0.003 (0.06) | 0.019 |
| lr 1e-4 constant, **fp32** | 0.019 (0.04) | 0.039 (0.08) | -0.315 (0.60) | – | 0.039 |

Best setting (lr 3e-4 constant, 20 epochs) retrained on all training windows: **test $R^2$ 0.026** (train 0.049).

Job 60078958, commit 38194f5, 2 h 22 min (0.59 node-hours).

## 5. Reading

1. **No setting learns anything that generalizes.** Validation $R^2$ never exceeds 0.04. For tens of epochs the loss stays near 1 (the z-scored labels' variance: predicting the mean). Then training $R^2$ rises to 0.5-0.9 while validation drops below 0: the model memorizes the training windows. lr 1e-3 does not train at all.
2. **Precision is not the cause**: fp32 follows the same course as bf16.
3. **It is not a lack of information in the input**: ridge on the same patch features reaches 0.34, and the model sees the whole 1 s window through its time attention.
4. **Likely cause: the model cannot tell units apart.** The spike encoder applies one MLP to every unit, and the only embeddings added are the signal type and the time patch (Eq. 2, U22); there is no unit or channel embedding. Channel attention has no positional information, so the backbone is equivariant to permuting the units, and the U8 attention-pooling head (like the mean head) is invariant to it. The prediction is therefore a function of the set of unit activity patterns, not of which unit fired. Decoding velocity direction needs the latter (each unit's preferred direction). Ridge has fixed feature positions per unit and does not have this limit. This is an inference from the architecture, not yet tested (a flatten head, U8 alternative 2, would test it).

## 6. Decision

Kyuhyung Lee, 2026-10-21: **replicate what the paper states; change neither the model nor the fine-tuning settings.** The paper names no unit or channel embedding, so none is added. The scratch and ridge numbers above are reported as they are (the paper has no scratch or ridge arm). Whether the finetune and frozen arms reach the paper's 0.914 / 0.904 with the same head shows whether pretraining overcomes the limit of point 4; if they do not, that points to a component the paper does not describe. Ridge stays on the U27 features (this patch only); the context variants are reference numbers only.
