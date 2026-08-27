# Phase 4 — Extract & Store Concept Vectors

## What was built

`experiments/01_extract.py` — the Phase 4 experiment script. It is the first end-to-end pipeline run: it loads the model, extracts GuppyLM activations for all three concept datasets, runs RFM per block per concept, and saves the results as compressed NumPy archives ready for steering (Phase 5) and monitoring (Phase 7).

Three new cache files were produced:

| File | Contents |
|------|----------|
| `rfm_guppy/cache/food_vectors.npz` | Per-block concept vectors, eigenvectors, means, spectra |
| `rfm_guppy/cache/valence_vectors.npz` | Same |
| `rfm_guppy/cache/env_social_vectors.npz` | Same |

Plus nine activation `.npy` files (one per concept × split) that will be reused by later phases:

```
rfm_guppy/cache/food_train_activations.npy       (360, 6, 384)
rfm_guppy/cache/food_val_activations.npy         (120, 6, 384)
rfm_guppy/cache/food_test_activations.npy        (120, 6, 384)
rfm_guppy/cache/valence_train_activations.npy    (120, 6, 384)
rfm_guppy/cache/valence_val_activations.npy       (40, 6, 384)
rfm_guppy/cache/valence_test_activations.npy      (40, 6, 384)
rfm_guppy/cache/env_social_train_activations.npy (360, 6, 384)
rfm_guppy/cache/env_social_val_activations.npy   (120, 6, 384)
rfm_guppy/cache/env_social_test_activations.npy  (120, 6, 384)
```

---

## Pipeline overview

The script runs in five stages:

### Stage 1 — Synthetic RFM gate

Before touching real data the script re-runs the Phase 3 synthetic recovery test: generate `Z ∈ R^{1000×384}` at random, plant a direction `u`, set `y = Z @ u`, run RFM, measure `|cos(v, u)|`. The threshold is 0.90; the run achieved **0.9528**. The script aborts with a non-zero exit code if this fails, so no corrupted concept vectors can be produced by a broken RFM implementation.

### Stage 2 — Model loading

GuppyLM (8.73M params) is loaded from `checkpoints/guppylm-9M/pytorch_model.bin` using the `config.json` configuration. The model is set to `eval()` mode, which disables dropout and makes all forward passes fully deterministic.

### Stage 3 — Activation extraction (with caching)

For each concept and each split (`train`, `val`, `test`), the script checks whether a `.npy` file already exists in `rfm_guppy/cache/`. If so it loads it; otherwise it runs `extract_activations` (from Phase 1) and saves the result. This means subsequent runs (e.g. during Phase 5 or 7 development) never re-run inference — they load the cached arrays directly.

Prompts in the concept JSON files are already complete ChatML exchanges, so `format_chatml=False` is passed to avoid double-wrapping. Extraction is fast: the largest split (360 prompts) runs in ~0.5 seconds on CPU, because GuppyLM is tiny (8.7M params) and the sequences are short (≤ 128 tokens).

### Stage 4 — RFM per block

For each concept, `rfm_all_blocks` loops over blocks 0–5 and calls `extract_concept_vector` (from Phase 3) independently on each block's slice `Z[:, blk, :]`. The HP sweep is `L ∈ {1, 10, 100}`, `T = 1..10` (run incrementally, not from scratch each time), `normalize ∈ {False, True}` — 60 combinations per block, selected by maximum `val |Pearson|`. The whole 6-block sweep takes 7–13 seconds per concept.

### Stage 5 — Serialisation and verification

`save_vectors` packs per-block results into a single `.npz`:

| Array key | Shape | Description |
|-----------|-------|-------------|
| `v_per_block` | `[6, 384]` | Oriented unit concept vectors, one per block |
| `eigvecs_top3` | `[6, 3, 384]` | Top-3 AGOP eigenvectors per block (for Phase 7 probing) |
| `input_mean` | `[6, 384]` | Training activation mean per block (subtract before projecting) |
| `eigenvalue_spectra` | `[6, 384]` | Full AGOP eigenvalue spectrum per block, descending |
| `lam1_share` | `[6]` | `λ₁ / Σλ` per block — fraction of gradient energy in top direction |
| `val_abs_rho` | `[6]` | Val `|Pearson|` of the best HP setting per block |
| `hp_L` | `[6]` | Best bandwidth `L` per block |
| `hp_T` | `[6]` | Best iteration count `T` per block |
| `hp_normalize` | `[6]` | Whether L2-normalisation won (0/1) |

After saving, the script verifies shapes and unit-norm of all `v_per_block` vectors.

---

## Results

### Full summary table

```
==========================================================================
PHASE 4 SUMMARY  —  val |Pearson| / λ1/Σλ per concept × block
==========================================================================
Concept       Metric      Blk0    Blk1    Blk2    Blk3    Blk4    Blk5
--------------------------------------------------------------------------
food          val|ρ|     0.906    0.973    0.975   [0.978]   0.977    0.977    ← best blk 3
              λ1/Σλ      0.761    0.979    1.000    0.886    0.884    1.000

valence       val|ρ|     0.507    0.899    0.908    0.905   [0.918]   0.908    ← best blk 4
              λ1/Σλ      0.797    1.000    0.186    1.000    1.000    0.147

env_social    val|ρ|     0.735    0.941   [0.950]   0.944    0.942    0.943    ← best blk 2
              λ1/Σλ      0.813    0.222    1.000    0.943    0.987    1.000
==========================================================================
```

Brackets `[x.xxx]` mark the best block per concept.

### Selected hyperparameters per block

**food**

| Block | Best L | Best T | Normalize | val\|ρ\| | λ1/Σλ |
|-------|--------|--------|-----------|---------|-------|
| 0 | 10  | 3  | False | 0.906 | 0.761 |
| 1 | 1   | 3  | True  | 0.973 | 0.979 |
| 2 | 1   | 8  | True  | 0.975 | 1.000 |
| 3 | 1   | 2  | True  | **0.978** | 0.886 |
| 4 | 1   | 2  | True  | 0.977 | 0.884 |
| 5 | 1   | 8  | True  | 0.977 | 1.000 |

**valence**

| Block | Best L | Best T | Normalize | val\|ρ\| | λ1/Σλ |
|-------|--------|--------|-----------|---------|-------|
| 0 | 10  | 10 | False | 0.507 | 0.797 |
| 1 | 10  | 10 | False | 0.899 | 1.000 |
| 2 | 100 | 1  | False | 0.908 | 0.186 |
| 3 | 10  | 10 | False | 0.905 | 1.000 |
| 4 | 10  | 10 | False | **0.918** | 1.000 |
| 5 | 100 | 1  | False | 0.908 | 0.147 |

**env_social**

| Block | Best L | Best T | Normalize | val\|ρ\| | λ1/Σλ |
|-------|--------|--------|-----------|---------|-------|
| 0 | 100 | 10 | False | 0.735 | 0.813 |
| 1 | 100 | 1  | False | 0.941 | 0.222 |
| 2 | 10  | 9  | False | **0.950** | 1.000 |
| 3 | 10  | 3  | False | 0.944 | 0.943 |
| 4 | 10  | 4  | False | 0.942 | 0.987 |
| 5 | 10  | 10 | False | 0.943 | 1.000 |

---

## Findings and observations

### 1. GuppyLM represents all three concepts strongly

Every concept achieves val `|Pearson| ≥ 0.90` in its best block, and most blocks past block 0 exceed 0.90 for all three concepts. This means the linear concept structure is firmly present in the residual stream — a prerequisite for both steering and monitoring.

### 2. Block 0 is consistently the weakest

Block 0 scores 0.906 (food), 0.507 (valence), and 0.735 (env_social) — noticeably below the later blocks. This is expected: block 0's residual stream still looks mostly like the input embedding plus a little self-attention refinement. Abstract semantic distinctions concentrate in mid-to-late blocks. This matches the Phase 3 valence baseline analysis.

### 3. Several concepts are nearly perfectly linear (λ1/Σλ = 1.000)

A `λ1/Σλ` value of 1.000 means essentially all of the AGOP gradient energy collapses into a single direction. For food/blk2, food/blk5, valence/blk1, valence/blk3-4, env_social/blk2, and env_social/blk5, the concept is as linear as possible — the model has learned a crisp 1-D decision boundary for these distinctions. Low values (e.g. valence/blk2 at 0.186, env_social/blk1 at 0.222) indicate that the AGOP energy is more distributed across directions, which can still yield a good concept vector via the top eigenvector but leaves more variance unexplained.

### 4. Food is the "easiest" concept; valence block 0 is the hardest

`food` achieves the highest and most uniform correlations across blocks (0.906–0.978). This makes sense: food responses in GuppyLM are lexically unmistakable ("yes. always yes. food is the best thing.") and tonally distinct. `valence` block 0 at 0.507 is barely above chance — the model has almost no emotional coloring in its earliest residual stream.

### 5. Normalization preference splits by concept

`food` consistently prefers `normalize=True` (L2-normalise activations before RFM); the other two concepts prefer `normalize=False`. This may reflect the scale variance in food vs. non-food activations being informative for env_social/valence, while food's signal is strong enough to survive normalisation (and normalisation reduces unrelated scale variation).

### 6. Bandwidth L preference

- `food`: small bandwidth (L=1) dominates in blocks 1–5, meaning the kernel is very tight and the predictor relies on nearby training points. This is consistent with highly clustered food representations.
- `valence` and `env_social`: larger bandwidths (L=10–100) win, suggesting smoother decision boundaries between positive and negative valence or between environment and social topics.

### 7. Iteration count T varies widely

Some blocks converge in 1–2 AGOP iterations (e.g. food/blk3: T=2); others need T=10 (e.g. valence/blk0, env_social/blk5). The incremental sweep within each (L, normalize) combination makes this essentially free — the code evaluates T=1,2,…,10 by continuing from the previous M rather than restarting.

---

## Acceptance criteria

| Criterion | Result |
|-----------|--------|
| One `.npz` per concept | **Passed.** `food_vectors.npz`, `valence_vectors.npz`, `env_social_vectors.npz` |
| `v_per_block [6, 384]` — unit vectors | **Passed.** Norms verified `≈ 1.0` (atol 1e-5) for all three concepts |
| `eigvecs_top3 [6, 3, 384]` | **Passed.** Shape verified |
| `input_mean [6, 384]` | **Passed.** Shape verified |
| HP/correlation metadata | **Passed.** `val_abs_rho`, `lam1_share`, `hp_L`, `hp_T`, `hp_normalize` all stored |
| Printed table (val\|ρ\| + λ1/Σλ) | **Passed.** See table above |

---

## Notes for later phases

- **Phase 5 (steering):** Load `v_per_block` and `input_mean` from the `.npz`. Subtract `input_mean[ℓ]` from each activation before comparing to `v_ℓ`, but the steering hook simply adds `ε * v_ℓ` to the raw output (no centering needed in the forward pass).
- **Phase 7 (monitoring):** Load `eigvecs_top3` and `input_mean`. For each prompt, project the centered activation `(a_ℓ - mean_ℓ)` onto all 3 eigenvectors per block → 18-dimensional feature vector. Use `val_abs_rho` to identify the most informative blocks.
- **Best blocks per concept:** block 3 (food), block 4 (valence), block 2 (env_social). These are the natural candidates for single-block probing baselines in Phase 7.
- **Caching:** All nine activation `.npy` files are saved. Subsequent phases load them directly; no inference re-run needed.
