# GUPPY_PHASE8-9.md — Phase 8 (Baselines) and Phase 9 (Report)

This document describes what was implemented and what was found in the final two phases of the GuppyLM replication project. For background on the earlier phases (environment setup, activation extraction, concept datasets, RFM implementation, steering, and monitoring), see `GUPPY_PHASE0.md` through `GUPPY_PHASE6-7.md`.

---

## Recap: where we were

After Phase 7, we had:
- Three binary concept datasets built from GuppyLM's own training categories: **food**, **valence** (positive vs. negative affect), and **env_social** (tank/environment vs. social interaction).
- RFM concept vectors extracted per block (6 blocks × 384-dim) for each concept, with val |Pearson| ranging from 0.91–0.98.
- Activation-steering hooks verified to shift model output toward each concept at relative coefficient `r = +0.25` (steering success 100% for all three concepts).
- Monitoring probes achieving perfect AUROC = 1.000 on the held-out test split.

What remained: a **baseline comparison** (do simpler methods work as well as RFM?) and a **written report** synthesising all results.

---

## Phase 8: Baseline Comparison

### What the paper asks for

The paper compares RFM against three simpler concept-vector extraction methods:
1. **PCA** — top eigenvector of the pos-minus-neg difference matrix.
2. **DiffMeans** — normalised `mean(positive activations) − mean(negative activations)`.
3. **Logistic Regression** — the normalised weight vector of a trained LR classifier.

For each method, the paper reports two metrics:
- **Monitoring AUROC** — how well a probe using those vectors detects the concept in held-out prompts.
- **Steering success rate** — what fraction of steered generations are classified as on-concept (probe score > 0.5 or lexicon hit), at the best `ε` in the sweep.

### What we implemented

**`rfm_guppy/baselines.py`** — three extraction functions:

```
pca_vector(Z_train, y_train, Z_val, y_val)
diffmeans_vector(Z_train, y_train, Z_val, y_val)
logreg_vector(Z_train, y_train, Z_val, y_val, C_values)
```

Each function:
- Takes the `[n, 384]` activations for one block and binary labels.
- Mean-centres the activations (training mean stored for reuse on val/test).
- Returns a **unit vector** oriented so that positive projections correspond to label=1 (Pearson-sign flip, identical to RFM's `orient()` step).
- Also returns the training mean and the val |Pearson| for reporting.

A batch wrapper `extract_all_baseline_vectors(acts_train, y_train, acts_val, y_val)` loops over all 6 blocks and all 3 methods, returning a nested dict.

For monitoring, `baseline_monitoring_auroc(...)` projects each block's activations onto its baseline vector (one scalar per block), trains an LR probe on the resulting `[n, 6]` feature vectors (C swept on validation), and evaluates AUROC on the test split.

**`experiments/04_baselines.py`** — the experiment script:
1. **Part A**: extracts baseline vectors for all concepts/blocks, evaluates monitoring AUROC, appends RFM's Phase-7 AUROC for comparison.
2. **Part B**: loads the GuppyLM model; for each (concept, baseline method), runs a steering sweep over `r ∈ {-2, -1, -0.5, -0.25, 0, +0.25, +0.5, +1, +2}` × 7 neutral probe prompts; scores each generation using the same Phase-7 RFM probe and lexicon (identical to the Phase-6 scoring function); records best-positive-r success rate.
3. Prints formatted comparison tables and saves `results/baselines_monitoring.json`, `results/baselines_steering.json`, and `results/baselines_comparison.md`.

The steering in Part B uses **baseline vectors** to steer but the **RFM probe** to score. This is intentional: the probe measures concept presence in activation space and is the same scoring stick for all methods, so the comparison is fair.

### Results

**Vector quality (val |Pearson|, best block):**

| Method | food | valence | env_social |
|--------|------|---------|------------|
| PCA | 0.929 | 0.154 | 0.490 |
| DiffMeans | 0.975 | 0.800 | 0.914 |
| LogReg | 0.986 | 0.960 | 0.972 |
| RFM | 0.978 | 0.918 | 0.950 |

PCA struggles on **valence** (0.154) and **env_social** (0.490). The reason: PCA takes the top eigenvector of the difference matrix, but if the within-class activation variance is large relative to the between-class signal, the top PC reflects within-class structure rather than the class-separating direction. For small, emotionally-nuanced categories like valence (100 samples/class), this is pronounced. DiffMeans and LogReg, which directly target the class boundary rather than variance, perform much better.

**Monitoring AUROC (test split, aggregate probe):**

| Method | food | valence | env_social | Overall |
|--------|------|---------|------------|---------|
| PCA | 0.9947 | **0.7425** | **0.8992** | 0.8788 |
| DiffMeans | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| LogReg | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| RFM | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

PCA is the only method that falls short. DiffMeans and LogReg match RFM exactly. This is consistent with the paper's claim that RFM outperforms PCA; the DiffMeans/LogReg tie with RFM is expected at this small scale (see discussion below).

**Steering success rate (best positive r):**

| Method | food | valence | env_social | Overall |
|--------|------|---------|------------|---------|
| PCA | 1.000 | 1.000 | 1.000 | 1.000 |
| DiffMeans | 1.000 | 1.000 | 1.000 | 1.000 |
| LogReg | 1.000 | 1.000 | 1.000 | 1.000 |
| RFM | 1.000 | 1.000 | 1.000 | 1.000 |

All four methods achieve 100% steering success. Even PCA's weaker vectors (val|ρ| = 0.154 for valence) are sufficient to steer the model — they just need a slightly larger `r` (+0.50 instead of +0.25). Steering success is a less discriminating metric than AUROC at this scale.

### Key interpretation

At GuppyLM's scale and with these in-domain concepts, the concepts are so strongly linear in activation space that multiple methods converge. The paper's RFM advantage over baselines is most visible (a) on harder/more abstract concepts, (b) at larger scale, and (c) on monitoring (AUROC) rather than steering (which is a coarser binary metric). Our data shows the monitoring advantage (PCA 0.88 vs others 1.00) but not the DiffMeans/LogReg advantage, because GuppyLM's representations are unusually clean.

---

## Phase 9: Report

**`results/REPORT.md`** is a self-contained document covering all nine phases.

### Structure

1. **Setup** — GuppyLM architecture, hook target, activation norms, concept definitions with actual category names and sample counts, train/val/test split sizes.

2. **Synthetic RFM recovery** — confirms the AGOP implementation is correct before any real data is used (|cos(v,u)| = 0.953, threshold 0.90).

3. **Concept-vector extraction** — per-block val |Pearson| table for all three concepts; AGOP eigenvalue spectrum (λ1/Σλ ≈ 1.000 on multiple blocks, confirming the concepts are nearly rank-1 linear subspaces).

4. **Steering** — sample generations at r=0 and r=+0.25 showing the concept shift; lexicon and probe metrics vs r; coherence-collapse threshold per concept.

5. **Monitoring** — AUROC table (1.000 for all three concepts, both per-block and aggregate probes).

6. **Baseline comparison** — the tables from Phase 8; discussion of why PCA underperforms on valence/env_social.

7. **Findings vs the paper** — a table of which claims hold, plus the key calibration difference: the paper uses Llama-specific absolute `ε` values (0.1–0.65), which would have no effect on GuppyLM's activations (median norm 12–31); we normalise by activation norm and use a relative coefficient `r`.

8. **Honest limitations** — in-domain only, small n, probe circularity, ceiling effects, no scaling claims.

9. **Next steps** — points to the Pythia replication plan as the path to the paper's scaling, training-time, and real benchmark claims.

### Notable findings stated plainly

- The mechanism works at 8.73M parameters. AGOP converges, concepts are rank-1, steering succeeds, probes achieve perfect AUROC.
- The results are at ceiling: AUROC = 1.000 and 100% steering success are ceiling effects from using perfectly in-domain, lexically grounded categories.
- The paper's RFM-vs-baselines advantage is partially visible (vs PCA on monitoring) but DiffMeans and LogReg are just as good at this scale. This is not a failure of the replication — it is the expected result when concepts are perfectly linearly separable.
- `ε` calibration is critical. The paper's Llama-scale values are meaningless on GuppyLM; normalising by activation norm is required.

---

## Files created in Phases 8–9

```
rfm_guppy/
  baselines.py                    Phase 8 extraction methods (PCA, DiffMeans, LogReg)

experiments/
  04_baselines.py                 Phase 8 experiment script

results/
  baselines_monitoring.json       AUROC per method × concept (Phase 8)
  baselines_steering.json         Steering success per method × concept (Phase 8)
  baselines_comparison.md         Human-readable comparison tables (Phase 8)
  REPORT.md                       Full replication report (Phase 9)
```

---

## Project complete

All nine phases from the guide are done. The project root now contains:

```
Remake-Paper-Towards/
├── GUPPYLM_GUIDE.md       original plan
├── GUPPY_PHASE0.md        phase 0 notes (environment setup)
├── GUPPY_PHASE1.md        phase 1 notes (activations)
├── GUPPY_PHASE2.md        phase 2 notes (concepts)
├── GUPPY_PHASE3.md        phase 3 notes (RFM)
├── GUPPY_PHASE4.md        phase 4 notes (vector extraction)
├── GUPPY_PHASE5.md        phase 5 notes (steering)
├── GUPPY_PHASE6-7.md      phases 6–7 notes (eval + monitoring)
├── GUPPY_PHASE8-9.md      this file
├── guppylm/               cloned GuppyLM repo
├── rfm_guppy/             our replication package
│   ├── activations.py
│   ├── baselines.py
│   ├── concepts.py
│   ├── evaluate.py
│   ├── monitoring.py
│   ├── rfm.py
│   ├── steering.py
│   └── cache/             activations, vectors, probes
├── experiments/
│   ├── 01_extract.py
│   ├── 02_steer.py
│   ├── 03_monitor.py
│   └── 04_baselines.py
└── results/
    ├── steer_*.{json,md}
    ├── eval6_*.{json,md}
    ├── monitor_auroc.json
    ├── baselines_*.{json,md}
    └── REPORT.md
```
