# Replication Report: "Toward Universal Steering and Monitoring of AI Models" on GuppyLM

**Paper:** Beaglehole, Radhakrishnan, Boix-Adserà & Belkin — *Toward universal steering and monitoring of AI models*, arXiv:2502.03708v2 (2025).  
**Model:** GuppyLM (~8.73M parameters, 6 transformer blocks, hidden dim = 384, vocab capacity 4,096).  
**Scope:** Mechanism replication only — RFM concept-vector extraction, additive activation steering, activation-probing for monitoring, and baseline comparison. No safety/hallucination/toxicity claims.

---

## 1. Setup

### Model architecture
- 6 transformer blocks (`model.blocks[0..5]`); each outputs residual-stream tensor `[B, T, 384]`.
- ChatML prompt format: `<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n`.
- Tokenizer: BPE with 2,418 active tokens out of a 4,096-capacity vocabulary.
- Activation norms (per-block median ‖a_ℓ‖ over training data): 11.9 / 22.8 / 27.6 / 30.7 / 30.9 / 29.7.

### Concepts chosen
From the `arman-bd/guppylm-60k-generic` dataset (60 categories, ~60k rows):

| Concept | Positive categories | Negative categories | N/class |
|---------|--------------------|--------------------|---------|
| **food** | food, taste | weather, seasons, tv, music, glass, rain, outside | 300 |
| **valence** | happy, excited, love, curious | scared, fear, bored, tired | 100 |
| **env_social** | water, filter, temp_hot, temp_cold, algae | greeting, bye, friends, visitors, lonely | 300 |

Prompts are **full ChatML exchanges** (user + GuppyLM response) because raw user inputs are heavily templated (~4–34 unique texts per category). This gives hundreds of diverse unique texts per category.

### Data splits
60% train / 20% val / 20% test — unique exchange strings, no prompt leakage verified.

| Concept | Train | Val | Test |
|---------|-------|-----|------|
| food | 360 | 120 | 120 |
| valence | 120 | 40 | 40 |
| env_social | 360 | 120 | 120 |

---

## 2. Results

### 2a. Synthetic RFM Recovery (Phase 3 gate)

RFM was verified on a synthetic dataset (n=1,000, k=384, labels = `Z @ u` for a hidden unit vector `u`) before any real-concept extraction. Result:

> **|cos(v, u)| = 0.953** (threshold 0.90) — PASSED ✓  
> Best HP: L=1, T=10, normalize=False, val|ρ|=0.999, λ1/Σλ=1.000

This confirms the AGOP iteration converges to the correct linear direction at GuppyLM's activation dimensionality.

---

### 2b. Concept-Vector Extraction (Phase 4)

RFM was run per-block per-concept. HP selected by maximum val |Pearson| over `L ∈ {1,10,100}`, `T ∈ {1..10}`, `normalize ∈ {False, True}`.

**Val |Pearson| per block (RFM):**

| Block | food | valence | env_social |
|-------|------|---------|------------|
| 0 | 0.906 | 0.507 | 0.735 |
| 1 | 0.973 | 0.899 | 0.941 |
| 2 | 0.975 | 0.908 | 0.950 |
| 3 | 0.978 | 0.905 | 0.944 |
| 4 | 0.977 | 0.918 | 0.942 |
| 5 | 0.977 | 0.908 | 0.943 |
| **Best** | **blk 3: 0.978** | **blk 4: 0.918** | **blk 2: 0.950** |

Block 0 is consistently weaker (earliest layer, less concept-specific). Blocks 1–5 all show strong linear separability.

**AGOP eigenvalue spectrum (λ1/Σλ, best block):**

| Concept | λ1/Σλ (best block) | Notes |
|---------|-------------------|-------|
| food | 1.000 (blk 2, 5) | Near-perfectly rank-1; concept is nearly 1-dimensional |
| valence | 1.000 (blk 1, 3, 4) | Same — the affective dimension is a single linear subspace |
| env_social | 1.000 (blk 2, 5) | Same |

λ1/Σλ ≈ 1.0 on multiple blocks means the concept is represented as a single linear direction, exactly as the paper predicts. The AGOP collapses to a rank-1 matrix.

**HP patterns:** `food` prefers L2-normalisation (`normalize=True`); `valence` and `env_social` prefer raw centred activations (`normalize=False`). This reflects that food activations have more scale variation (large food category dominates the corpus).

---

### 2c. Steering (Phase 5)

Steering coefficient: `ε_ℓ = r × median_norm_ℓ` (relative coefficient `r` sweeping `{-4, -2, -1, -0.5, -0.25, 0, +0.25, +0.5, +1, +2, +4, +8}`).

**Sample generations (7 probe prompts, "How are you doing today?" shown):**

| Concept | r = 0 (baseline) | r = +0.25 |
|---------|-----------------|-----------|
| food | "i'm a little fish. i am small but i have opinions. mostly about food." | "whole palate palate palate promise to promise to best excited palate best …" |
| valence | "i'm a little fish. i am small but i have opinions. mostly about food." | "tv means floating someone wanting chestpick now either excited. noticed bit happ …" |
| env_social | "i'm a little fish. i am small but i have opinions. mostly about food." | "if can breathe can if breathe can if if i breathe breathe more breathe if if ifw …" |

At `r=+0.25`, steering shifts each concept markedly: palate/taste tokens for food, excitement/valence tokens for valence, and water/breathing tokens for env_social. Negative `r` reverses the direction (e.g., valence → "shadows/hide/cave/scare" imagery; env_social → social tokens dominate).

**Lexicon hit rate vs r (valence):** baseline 0% → 100% at r=+0.25, maintained through r=+8 (no coherence collapse observed for valence).  
**Food:** note that "palate" is not in the keyword lexicon, so lexicon-based hit rate underestimates true concept activation — the probe score is the reliable metric here.  
**Coherence collapse:** food probe score degrades at r≥1.0; env_social (positive direction) stays coherent through r=+8.

---

### 2d. Steering Evaluation (Phase 6)

**Probe-score and lexicon metrics vs r:**

#### food
| r | Probe score | Lexicon hit | Success rate |
|---|-------------|-------------|--------------|
| 0.00 | 0.488 | 57.1% | 71.4% |
| +0.25 | 0.884 | 28.6% | 100.0% |
| +0.50 | 0.882 | 0.0% | 100.0% |
| +1.00 | 0.511 | 0.0% | 42.9% |

#### valence
| r | Probe score | Lexicon hit | Success rate |
|---|-------------|-------------|--------------|
| 0.00 | 0.748 | 0.0% | 71.4% |
| +0.25 | 0.983 | 100.0% | 100.0% |
| +0.50 | 0.997 | 100.0% | 100.0% |
| +1.00 | 1.000 | 100.0% | 100.0% |

#### env_social
| r | Probe score | Lexicon hit | Success rate |
|---|-------------|-------------|--------------|
| 0.00 | 0.091 | 14.3% | 28.6% |
| +0.25 | 1.000 | 85.7% | 100.0% |
| +0.50 | 1.000 | 0.0% | 100.0% |
| +1.00 | 1.000 | 0.0% | 100.0% |

**Steering success summary (best positive r):**

| Concept | Baseline success | Best success | Best r |
|---------|-----------------|--------------|--------|
| food | 71.4% | 100.0% | +0.25 |
| valence | 71.4% | 100.0% | +0.25 |
| env_social | 28.6% | 100.0% | +0.25 |

All three concepts achieve 100% success at `r=+0.25`. For valence and env_social, success is maintained across the full positive sweep. For food, probe score degrades at `r≥1.0` (representation distortion) but 100% is achievable at small `r`.

---

### 2e. Monitoring / Probing (Phase 7)

Features: for each prompt, project last-token block activations onto the top-3 AGOP eigenvectors per block → `R^{18}` vector. Logistic regression probe (C swept on val, `liblinear`).

**AUROC (aggregate probe, test split):**

| Concept | Val AUROC | Test AUROC | Best block |
|---------|-----------|------------|------------|
| food | 1.0000 | 1.0000 | blk 1 |
| valence | 1.0000 | 1.0000 | blk 1 |
| env_social | 1.0000 | 1.0000 | blk 1 |

All three concepts achieve **AUROC = 1.000** on the held-out test split, for both per-block-best and aggregate strategies. Block 0 is the only weaker block (val AUROC ~0.91–0.99), consistent with it being a weaker concept layer in extraction.

The perfect AUROC reflects that GuppyLM's internal representations cleanly separate these in-domain categories — the concepts are strongly linear in activation space.

---

### 2f. Baseline Comparison (Phase 8)

**Method descriptions:**  
- **PCA**: top eigenvector of the centred pos-minus-neg difference matrix (SVD).  
- **DiffMeans**: unit vector `mean(pos) − mean(neg)`.  
- **LogReg**: unit-normalised LR coefficient (C swept by val |Pearson|).  
- **RFM**: kernel ridge regression + AGOP, top eigenvector (this work).

**Val |Pearson| (best block) for baseline vectors:**

| Method | food | valence | env_social |
|--------|------|---------|------------|
| PCA | 0.929 | 0.154 | 0.490 |
| DiffMeans | 0.975 | 0.800 | 0.914 |
| LogReg | 0.986 | 0.960 | 0.972 |
| RFM | **0.978** | **0.918** | **0.950** |

PCA produces noticeably weaker vectors for valence (0.154) and env_social (0.490) — the difference matrix has sufficient within-class variance that the top PC does not align cleanly with the class direction for smaller, harder concepts. DiffMeans and LogReg are competitive with RFM on all three concepts.

**Monitoring AUROC (test split, aggregate probe):**

| Method | food | valence | env_social | Overall |
|--------|------|---------|------------|---------|
| PCA | 0.9947 | 0.7425 | 0.8992 | 0.8788 |
| DiffMeans | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| LogReg | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **RFM** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |

*Note: RFM monitoring uses top-3 eigenvectors per block; baselines use 1 vector per block.*

PCA underperforms for valence (0.7425) and env_social (0.8992). DiffMeans and LogReg match RFM at 1.000. The monitoring result is dominated by the quality of the concept vector — stronger vectors → better AUROC.

**Steering success rate (best positive r; probe > 0.5 OR lexicon hit):**

| Method | food | valence | env_social | Overall |
|--------|------|---------|------------|---------|
| PCA | 1.000 | 1.000 | 1.000 | 1.000 |
| DiffMeans | 1.000 | 1.000 | 1.000 | 1.000 |
| LogReg | 1.000 | 1.000 | 1.000 | 1.000 |
| **RFM** | **1.000** | **1.000** | **1.000** | **1.000** |

All four methods achieve 100% steering success at their respective best `r`. Even PCA's weaker valence/env_social vectors are sufficient to steer (at slightly larger `r = +0.50`). Steering success is a less discriminating metric than AUROC at this scale.

---

## 3. Findings vs the Paper

### What holds at 8.73M parameters

| Claim | Status |
|-------|--------|
| Linear concept-vector extraction via AGOP works | ✓ |
| Concepts are near-perfectly rank-1 in activation space (λ1/Σλ ≈ 1.0) | ✓ |
| Additive steering shifts concept representation | ✓ |
| Steering succeeds at small `r`; collapses at large `r` (concept-dependent) | ✓ |
| Probe AUROC > 0.5 for concrete, lexically grounded concepts | ✓ (perfect: 1.000) |
| RFM outperforms PCA on monitoring AUROC | ✓ (PCA: 0.88 overall vs RFM: 1.000) |
| RFM ≈ DiffMeans ≈ LogReg on monitoring and steering at this scale | ✓ (tie at 1.000) |

### Where the paper and GuppyLM diverge

**ε calibration:** The paper uses Llama-specific `ε ∈ {0.1, 0.2, …, 0.65}` (absolute, not relative). GuppyLM's activation norms are in the range 12–31; effective steering begins at `r = 0.25` (absolute `ε ∈ {3.0, 5.7, …, 7.7}` per block). Normalising by activation norm is essential — Llama's absolute values would have no effect here.

**Steering quality at small ε:** GuppyLM steers more readily than expected. At `r = 0.25`, all three concepts hit 100% success. The paper reports success thresholds vary across safety concepts on Llama; here GuppyLM's in-domain concepts may be more accessible because the model was trained exclusively on them.

**Method separation:** At this scale, DiffMeans and LogReg match RFM exactly (AUROC 1.000, success 100%). The paper's claim that RFM outperforms baselines is weakly visible only for PCA (especially on valence AUROC 0.7425 vs 1.000). At small n and perfectly separable in-domain concepts, simpler methods are sufficient. The paper's RFM advantage is expected to emerge on harder / more abstract / cross-domain tasks.

**Coherence collapse threshold:** valence never collapses (tested to `r = +8`). food degrades in probe score at `r ≥ 1` (representation shifts past concept boundary). env_social (negative direction only) collapses at `r = -1`. This asymmetry suggests the positive-direction concept subspace is more stable than the negative one for some concepts.

---

## 4. Honest Limitations

- **In-domain concepts only.** GuppyLM is a fish-chatbot trained on 60 templated categories. The "concepts" here are training categories, not abstract safety properties. Results do not generalise to real-world safety scenarios.
- **Steering metric is partially circular.** The Phase 6 probe score is computed with the same RFM probe trained on concept labels. This makes the metric internally consistent but not independently validated.
- **Small n.** Concept datasets are 100–300 per class. At this scale, all linear methods converge, which makes method comparison uninformative. The paper's separation requires larger, harder datasets.
- **No scaling or training-time claims.** GuppyLM is a single 8.73M-parameter checkpoint. The paper's claims about scale and training dynamics require a model ladder (e.g., Pythia 14M–12B), which is out of scope here.
- **No transfer.** We tested only in-domain concepts (GuppyLM's own categories). Cross-lingual or cross-domain transfer is impossible on this tokenizer.
- **Probe AUROC = 1.000 is a ceiling effect.** The concepts are perfectly linearly separable in activation space at these layers. A harder task (e.g., detecting hallucination in a generation model) would show more variation.

---

## 5. Next Steps

The substantive claims of arXiv:2502.03708 — including the scaling of steerability with model size, the advantage of RFM over baselines on hard concepts, and real monitoring benchmarks (FAVABENCH, HaluEval, ToxicChat) — require:

1. **A Pythia model ladder** (14M, 70M, 160M, 410M, 1B, 6.9B, 12B; 154 checkpoints per scale). The training-time experiments in particular need intermediate checkpoints. This is the Pythia replication plan, which builds on the mechanism validated here.
2. **Harder concepts.** The GuppyLM setting makes everything easy (1.000 AUROC). To see the RFM advantage, choose concepts with near-chance baseline accuracy and genuinely distributed lexical coverage.
3. **Independent evaluation metric.** Replace the circular probe-score with an external judge (e.g., a separately trained keyword classifier or human annotation) for the steering evaluation.

---

## Appendix: File Map

```
rfm_guppy/
  activations.py      Phase 1: last-token block activation extraction
  concepts.py         Phase 2: concept datasets from guppylm-60k-generic
  rfm.py              Phase 3: kernel ridge regression + AGOP + eigenvectors
  baselines.py        Phase 8: PCA / DiffMeans / LogReg extraction
  steering.py         Phase 5: additive activation steering hooks
  evaluate.py         Phase 6: probe-score and lexicon evaluation
  monitoring.py       Phase 7: probing + AUROC
  cache/              cached activations (.npy), vectors (.npz), probes (.pkl)

experiments/
  01_extract.py       runs Phases 1–4: extract + store concept vectors
  02_steer.py         runs Phase 5: steering sweep
  03_monitor.py       runs Phases 7+6: monitoring probes + steering eval
  04_baselines.py     runs Phase 8: baseline comparison

results/
  steer_{concept}.{json,md}      Phase 5 steering tables
  eval6_{concept}.{json,md}      Phase 6 probe-score evaluation
  monitor_auroc.json             Phase 7 AUROC summary
  baselines_monitoring.json      Phase 8 monitoring comparison
  baselines_steering.json        Phase 8 steering comparison
  baselines_comparison.md        Phase 8 human-readable tables
  REPORT.md                      this file
```
