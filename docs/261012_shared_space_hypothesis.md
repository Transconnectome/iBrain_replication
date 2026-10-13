# Shared space or concatenation? Experiments on what joint pretraining does to the backbone

Status: **proposal, not implemented.** Written 2026-10-12. Supersedes the counterfactual part (sections 5 to 7) of `docs/260929_modality_embedding_ablation.md`. Its diagnostics (4.1, 4.2) are reused here. Needs real-data checkpoints: joint (M6/M7), spike-only (M3) and iEEG-only (U42).

## 1. Claim and alternative

The paper's central observation is that pretraining one backbone on both signal types, alternating 1:1, improves downstream results on both. For MC-Maze, the paper reports $R^2$ 0.914 joint against 0.89 spike-only (SPEC 1.8, M4). The paper reads this as the two types being mapped into a shared representation, with transfer coming from common structure in neural activity.

The alternative we test:

- **H-concat (geometry).** The backbone does not put the two types in one space. It gives each type its own subspace and its own set of units. In effect it is two models stored side by side in one set of weights, a "concatenation" of a spike model and an iEEG model.
- **H-reg (source of gain).** The downstream gain does not come from the content of the other signal type. It comes from a generic regularization or optimization effect of training on a second, different stream: more optimizer steps, a longer lr schedule, gradient noise, implicit capacity limits.

The two hypotheses are logically separate, and both directions are possible:

| | Gain from content (not H-reg) | Gain from regularization (H-reg) |
|---|---|---|
| **Shared space** | The paper's claim | Shared geometry, but the content of the other type does not matter |
| **Concatenated (H-concat)** | Unlikely: little to transfer through if the paths are separate. Possible through shared attention patterns or shared low-level operations | Our working hypothesis |

So the proposal has two parts. Part A measures geometry (section 4), Part B measures the source of the gain (section 5). Neither part alone answers the question.

### 1.1 What "shared" can and cannot mean here

The strong sense of alignment, "the same latent content gives the same vector whatever the signal type", cannot be tested. There is no paired data: the spike corpus is macaque recordings, the iEEG corpora are human recordings, and no stimulus or behavior is common to both. Every test below uses the weak sense: **the two types use the same subspaces, the same units and the same computations of the backbone.** Results should be reported in these terms, and "aligned" should not be used without this qualifier.

Signal type is also completely confounded with species, dataset, recording hardware and subject. Any separation measured here is separation of "these two corpora", which is an upper bound on separation of the signal types.

## 2. Should the embedding swap be run at all?

The earlier proposal (`260929_modality_embedding_ablation.md` section 5) gives spike tokens the iEEG type embedding and the reverse, at inference, with all weights frozen. We reviewed whether to run it. **Recommendation: do not run Swap, Neutral, Zero or the random-direction control.**

### 2.1 Reasons

1. **It answers a different question.** Swap asks whether the model *uses the marker* $e_{\text{type}}$. H-concat is about whether the backbone *keeps the two types apart*. The two are independent. A backbone that keeps the types apart can ignore the marker entirely, because the per-type encoders already make the inputs separable (old doc 2.2). A backbone that shares one space can still use the marker as a per-type bias. No Swap result separates H-concat from a shared space.
2. **A null result is uninformative by construction.** Both encoders end in a `Linear` with a bias, so $W x + b + e_{\text{type}} = W x + (b + e_{\text{type}})$. The marker is redundant as a parameter. How the type offset splits between $b$ and $e_{\text{type}}$ depends on initialization and weight decay, not on what the model needs. "Swap changes nothing" then means "this vector does not carry the offset", not "the model needs no type information".
3. **A positive result is confounded with distribution shift.** Under 1:1 alternation the backbone never saw spike tokens with the iEEG marker. Any drop may be a reaction to out-of-distribution input of that size, not to the type direction. The random-direction control only partly fixes this: it matches the norm, not the specific off-manifold shift.
4. **The marker has little room to act.** Each window holds one type, so the marker is the same vector on every token. Added to every key, it shifts all attention logits of a query by the same amount, which softmax cancels. It acts only through LayerNorm, the value path and the FFN. A small effect is expected regardless of the hypothesis.
5. **A training-time variant does not help either.** Pretraining with no marker (old doc D4) is equivalent in capacity, because the encoder bias can absorb the offset. A matching result would only confirm reason 2, which follows from algebra.

### 2.2 What we give up, and what we keep

- **Given up:** an empirical statement that "the model does / does not use the type embedding". If a reviewer asks for one, answer with the algebraic argument in 2.1 (2) plus the norm numbers below, and run Swap then. It costs forward passes only, and the code plan in the old doc section 8 still applies.
- **Kept from the old doc:** the norm diagnostics of section 4.1 ($\lVert e_m \rVert$, $\lVert e_{\text{spike}} - e_{\text{iEEG}} \rVert$, $\lVert b_m \rVert$, $\cos(b_m, e_m)$, against mean $\lVert h \rVert$). They are descriptive, nearly free, and say how much of the type offset sits in the marker. The layer-wise probe of section 4.2 becomes test A1 below.

### 2.3 Arguments for keeping Swap, and why we still drop it

| Argument | Response |
|---|---|
| It is cheap (forward passes only) | Cost is not the issue. The problem is that no outcome changes our conclusion about H-concat or H-reg |
| It is the most intuitive test of "concatenation" | It is intuitive only if one assumes the marker is what keeps the types apart. In this architecture the encoders do that (2.1, 1) |
| Readers expect it | Better to explain why it is uninformative than to report a number that invites the wrong reading |

## 3. Prerequisites

| Item | Needed for | State |
|---|---|---|
| **Gate: does our replication reproduce the joint gain?** Joint vs spike-only on MC-Maze, joint vs iEEG-only on Brain Treebank, 3 seeds | Part B is moot without a gain. Part A still runs | Not run. Needs M6/M7 real-data runs |
| Joint checkpoint, plus **intermediate checkpoints** (`ckpt_every: 1000`) | A1 to A4 (A4 needs several training points) | Loop done (M6), real-data run pending |
| Initialization checkpoint of the same seed | Baseline for A1 to A3: separation that the encoders produce before any training | Free: save step 0 |
| Spike-only and iEEG-only checkpoints, same seeds | A1 to A3 references, B1 | `--spike-only` (M3), `--ieeg-only` (U42) |
| Held-out recordings and sessions of both types, never seen in pretraining | All measurements on pretraining-type data | Open decision D1 of the old doc. **Must be fixed before the pretraining runs** |
| Downstream splits | B1 to B3 | MC-Maze M4 split; Brain Treebank U34 split |

**Downstream leakage.** The Neural Pile contains the Perich and Area2-Bump evaluation sessions (U14); the pile survey does not list MC-Maze. Use **MC-Maze as the primary spike benchmark** for Part B, and report Perich and Area2-Bump only next to the `exclude_sources: [perich, area2-bump]` variant.

## 4. Part A: geometry (H-concat)

All measurements on held-out, unmasked tokens, with equal numbers per type. Every measurement is reported at each depth: encoder output $h$ (0a), $z_0 = h + e_{\text{type}} + e_{\text{time}}$ (0b), after each block (1 to 6), and after the final LayerNorm (7, U23). Every number is also computed on the **initialization checkpoint** and, where it applies, on the **single-type checkpoints**. The question is not whether the types are separated, which is nearly certain at 0a, but **whether joint training reduces the separation with depth, compared with these references.**

### A1. Subspace overlap

From the old doc 4.2, plus one measure that is easier to read:

| Measure | Definition | H-concat predicts | Shared space predicts |
|---|---|---|---|
| Linear modality probe, full and rank-1 (difference of means) | Old doc 4.2. Split by recording | Near 100% at every depth | Rank-1 accuracy falls with depth |
| Fisher ratio along the class-mean difference | Old doc 4.2 | Stays high or rises | Falls |
| Principal angles between top-$k$ PCA subspaces ($k$ at 90% variance) | Old doc 4.2 | Near 90° | Small |
| **Cross-explained variance** $\mathrm{EV}(A \mid B) = \mathrm{tr}(P_B \Sigma_A P_B) / \mathrm{tr}(\Sigma_A)$, $P_B$ the projector on the top-$k$ PCs of type $B$ | Fraction of type $A$'s variance inside type $B$'s subspace | Near the random-subspace level $k/d$ | Near the split-half level $\mathrm{EV}(A_1 \mid A_2)$ |

$\mathrm{EV}$ is bounded by two references computed on the same data: $k/d$ (unrelated subspaces) and the split-half value within one type (same subspace up to noise). Report it as a position between the two.

Pooled features are measured too (`IBrain.pool`), because the downstream heads read pooled features.

### A2. Unit usage

For each FFN hidden unit (6 blocks × 1024), the mean absolute post-GELU activation per type, $a_s$ and $a_i$, and the selectivity index $\mathrm{SI} = (a_s - a_i) / (a_s + a_i)$.

- H-concat predicts a bimodal SI distribution: most units near $\pm 1$, i.e. used by one type only.
- A shared space predicts most of the mass near 0.
- Report the histogram per block, and the fraction with $|\mathrm{SI}| > 0.9$, against the initialization checkpoint.

The same for attention heads, with the norm of each head's output per type in place of the activation.

### A3. Causal double dissociation

A1 and A2 are correlational. To test whether the separation matters for computation: zero the FFN units with $\mathrm{SI} > 0.9$ (spike-selective) and measure held-out masked reconstruction loss for **both** types, then the same for the iEEG-selective units.

| Result | Reading |
|---|---|
| Removing spike units hurts spike only, removing iEEG units hurts iEEG only | Double dissociation: separate paths (H-concat) |
| Removing units of either type hurts both | Shared computation |
| Either removal changes little | Selective units are not where the computation is. Repeat with the top PCA directions of each type (A1) in place of units |

Control: remove the same number of randomly chosen units, 10 draws. Losses are compared within a type only, as skill retained (old doc 6.1), never across types.

### A4. Gradient alignment during training

At each intermediate checkpoint, compute the gradient of the pretraining loss on the **shared** parameters (backbone, `time_emb`) for spike batches $g_s$ and iEEG batches $g_i$, averaged over $K$ batches each. Report $\cos(g_s, g_i)$ against the within-type reference $\cos(g_s^{(1)}, g_s^{(2)})$ from two disjoint sets of spike batches (and the same for iEEG).

| Result | Reading |
|---|---|
| $\cos(g_s, g_i) \approx 0$ throughout | The two types update orthogonal directions: consistent with H-concat |
| Clearly positive, a sizable fraction of the within-type value | An update for one type also helps the other: transfer |
| Negative | Interference. A gain then has to come from somewhere else (H-reg) |

Raw gradients are measured, not AdamW updates: the per-parameter scaling of Adam would mix the two types' statistics. This measurement is the only one in Part A that looks at training rather than the end point. It costs a few forward and backward passes per checkpoint.

## 5. Part B: source of the gain (H-reg)

All of Part B needs new pretraining runs, 3 seeds each, evaluated on MC-Maze ($R^2$) and Brain Treebank (AUC) with the existing `scripts/finetune.py`.

### B1. Step-matched single-type baselines

Under U20, a joint run gives spike exactly the spike-only run's 30 epochs, but it takes **twice as many optimizer steps**, and the warmup and cosine schedule are stretched over twice the length. Spike-only vs joint, as in the paper's comparison, therefore differs in more than the presence of iEEG.

| Run | Spike exposure | Optimizer steps | Schedule length |
|---|---|---|---|
| Joint (paper) | 30 epochs | $2N$ | $2N$ |
| Spike-only, paper | 30 epochs | $N$ | $N$ |
| **Spike-only, step matched** | 60 epochs | $2N$ | $2N$ |
| **Spike-only, schedule matched** | 30 epochs | $N$ steps of spike, each followed by one lr-schedule step with no update | $2N$ |

The last row gives spike the same lr at the same point in its exposure as in the joint run, with no second type. It needs a small change to the loop (advance the scheduler on skipped steps). If one of the two matched baselines reaches the joint result, the gain needs no iEEG content.

The same for iEEG-only vs joint on Brain Treebank.

### B2. Surrogate second type

Joint pretraining with the second type replaced by a surrogate that keeps its low-level statistics but not its neural structure. If the gain survives the surrogate, it does not come from the content of the other type.

| Surrogate for iEEG | Keeps | Removes |
|---|---|---|
| S1: per-channel phase randomization, independent phases per channel | Power spectrum of each channel, amplitude distribution approximately | Cross-channel structure, nonlinear temporal structure |
| S2: phase randomization with the same phases on all channels of a window | Spectra and linear cross-channel correlations | Nonlinear structure |
| S3: Gaussian noise with a $1/f$ spectrum | Approximate spectral shape | Everything recording-specific |

| Surrogate for spike | Keeps | Removes |
|---|---|---|
| S1: per-unit time shuffle of 20 ms bins within a window | Rates per unit | Temporal structure, correlations |
| S2: Poisson counts with each unit's mean rate | Rates | All structure |

Surrogates are generated on the fly in the Dataset, after channel normalization (U4), so that input and target are the same values. The surrogate runs evaluate only the **real** type's downstream task.

### B3. (Optional, expensive) Forced separation

Joint pretraining with a backbone that is separated by construction: shared attention, but per-type FFNs (hard routing by type), or two backbones of half width. The parameter count stays the same.

- If the forced separation reaches the joint result, sharing is not what produces the gain.
- If it falls to the single-type result, some sharing is needed.

Run only if Part A and B1/B2 leave the question open.

## 6. Reading the results together

| Part A | Part B | Conclusion |
|---|---|---|
| Separation stays at its initial level or rises with depth; SI bimodal; double dissociation; $\cos(g_s, g_i) \approx 0$ | Matched baselines or surrogates reach the joint result | H-concat and H-reg both supported: one set of weights holds two models, and the gain is regularization or optimization |
| Same as above | Gain remains only with the real second type | The types are kept apart, but something specific about the other type helps. Look at attention heads, the shared part under H-concat |
| Separation falls with depth; SI near 0; removals hurt both; $\cos > 0$ | Gain remains only with the real second type | Supports the paper |
| Separation falls with depth | Surrogates also give the gain | Shared geometry, but the content of the other type is not what helps. A shared space is not evidence of transfer |
| No joint gain in our replication (gate, section 3) | — | Report the failure to reproduce. Part A still describes the geometry, but there is no gain to explain |

## 7. Order of work and cost

1. **Free, as soon as the joint run exists:** norm diagnostics (2.2), A1, A2 and A4 on saved checkpoints. Forward passes, a few backward passes for A4, and sklearn on CPU.
2. **A3:** forward passes per removal condition.
3. **B1:** two extra spike-only runs and two extra iEEG-only runs per seed. B1 is the cheapest test of H-reg and needs no new data code.
4. **B2:** surrogate datasets (new code), then one joint run per surrogate and seed. Start with iEEG S1 and spike S2.
5. **B3:** only if needed.

GPU work goes through `sbatch` on NERSC, never from the login node. The probe fits and the SI and EV computations run in a CPU-node job.

## 8. Implementation sketch

No change to `model.py` for Part A: forward hooks on the blocks and on the FFN activation give all depths.

| File | Content |
|---|---|
| `ibrain/geometry.py` | Per-depth feature extraction with hooks; probes, Fisher ratio, principal angles, EV (A1); SI (A2); unit removal through a hook that zeros selected units (A3); per-type gradients on shared parameters (A4) |
| `scripts/shared_space.py` | CLI: checkpoint(s) + held-out data → `results.json` with `meta.json` (U26) |
| `ibrain/surrogate.py` | Surrogate wrappers for an iEEG or spike Dataset (B2), selected by config, e.g. `data.ieeg[*].surrogate: phase_independent` |
| `ibrain/pretrain.py` | Option for the schedule-matched baseline of B1 (advance the scheduler on skipped steps) |

Tests only for what can silently go wrong: the hooks reproduce `model.encode` exactly when no unit is removed; a surrogate keeps each channel's power spectrum within tolerance; the gradient of one type touches no parameter of the other type (as `test_type_isolation`).

## 9. Open decisions

| # | Question | Proposal |
|---|---|---|
| E1 | Held-out split (old doc D1) | Whole recordings and sessions, about 5% of each type, fixed before any of the runs above |
| E2 | Number of seeds | 3 for every run in Part B. Part A on all 3 joint seeds, since it costs little |
| E3 | Threshold for "selective" units (A2, A3) | $|\mathrm{SI}| > 0.9$, with 0.8 and 0.95 as a sensitivity check |
| E4 | Which B1 baseline is the main comparison | Step matched. Schedule matched as a second row |
| E5 | Whether to run Swap at all | No (section 2). Revisit only if a reviewer asks |
