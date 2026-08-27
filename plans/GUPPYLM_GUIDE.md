# GUPPYLM_GUIDE.md — Replicating the Core Mechanisms of *"Toward Universal Steering and Monitoring of AI Models"* on GuppyLM

**Audience:** an AI coding agent executing this plan end-to-end.
**Goal:** reproduce the *mechanism* of the paper (arXiv:2502.03708) — RFM concept-vector extraction, additive activation steering, and activation probing for monitoring — on the tiny, fully-controllable **GuppyLM** (~8.7M params, 6 blocks, hidden dim 384, vocab 4,096).

Read this whole file before writing any code. Work phase by phase. Each phase ends with **Acceptance criteria** — do not advance until they pass. If a criterion fails, stop and report what you found rather than forcing a result.

---

## 0. Scope: what we replicate and what we deliberately do NOT

The paper has two separable parts. We replicate **the mechanism**, not the headline safety findings.

### IN scope (replicable on GuppyLM)
1. **RFM concept-vector extraction** — kernel ridge regression + Average Gradient Outer Product (AGOP), concept vector = top eigenvector of the AGOP matrix `M_T`.
2. **Additive steering** — add `ε·v_ℓ` to each block's output during the forward pass and observe generation shift.
3. **Monitoring / probing** — project activations onto top-`p` AGOP eigenvectors, train a classifier, evaluate by AUROC.
4. **Baseline comparison** — RFM vs PCA, difference-in-means, logistic regression (the paper's four extraction methods).

### OUT of scope (impossible on GuppyLM — do not attempt)
- Anti-refusal / jailbreaks, deception, political stances — GuppyLM has no such representations.
- Cross-language transfer, coding, reasoning — capabilities absent from a 4,096-token fish-domain model.
- Hallucination/toxicity benchmarks (FAVABENCH, HaluEval, ToxicChat) — GuppyLM's tokenizer cannot even encode them.
- The scaling claim ("larger = more steerable") — needs a model ladder; that is a separate Pythia plan, not this one.

### Reframed objective
> *Does the linear-concept extraction + additive-steering + probing mechanism hold at <10M parameters, using only concepts the model actually represents (its own training categories)?*

A **negative or weak** result for abstract concepts is a legitimate finding (the paper itself shows steerability collapses at small scale). Report it honestly; do not fabricate success.

---

## 1. Background: the four mechanisms (precise definitions)

Per-block activations: GuppyLM maps a token sequence through `L = 6` transformer blocks. For prompt `X`, let `A_ℓ(X) ∈ R^{t×k}` be block `ℓ`'s output (`k = 384`). Following the paper, the per-prompt feature is the **last real (non-pad) token row** `a_ℓ ∈ R^k`.

### 1a. RFM (concept-vector extraction)
Given block data `Z = [a_ℓ^(1), …, a_ℓ^(n)] ∈ R^{n×k}` and labels `y ∈ {0,1}^n` (or real-valued):

Mahalanobis–Laplace kernel, for `M ⪰ 0`, bandwidth `L > 0`:
```
K_M(x, z) = exp( -(1/L) * sqrt( (x - z)^T M (x - z) ) )
```
Initialize `M_0 = I`. For `t = 0 … T-1`:
- **Step 1 (kernel ridge regression):** `α_t = y · [K_{M_t}(Z, Z) + λI]^{-1}`, predictor `f_t(z) = α_t · K_{M_t}(Z, z)`.
- **Step 2 (AGOP):** `M_{t+1} = (1/n) Σ_i ∇_z f_t(a^(i)) ∇_z f_t(a^(i))^T`.

Concept vector `v` = top eigenvector of `M_T`. **Orient** it: compute Pearson `ρ` between `{⟨a^(i), v⟩}` and `{y^(i)}`; final vector = `sign(ρ) · v`.

Gradient needed for AGOP (derive once, implement carefully). With `d_i(z) = sqrt((z - a^(i))^T M (z - a^(i)))`:
```
∇_z f_t(z) = Σ_i α_i · K_M(a^(i), z) · (-1/L) · (1/d_i(z)) · M (z - a^(i))
```
Handle `d_i = 0` (set that term's contribution to 0).

Hyperparameters (paper): `λ = 1e-3`, `L ∈ {1, 10, 100}`, `T ≤ 10`. Mean-center inputs `a^(i)` and mean-center gradients before forming the AGOP. Optionally normalize activations to the unit sphere. **Selection:** hold out 20% as validation; pick `(L, T, normalize)` maximizing `|Pearson(⟨a, v⟩, y)|` on validation.

### 1b. Steering
With per-block vectors `{v_ℓ}`, during the forward pass replace `A_ℓ` by `A_ℓ,i + ε·v_ℓ` for **every** token row `i`. `ε` is the control coefficient (sweep it). Multi-concept = steer with a linear combination of vectors.

### 1c. Monitoring (probing)
Extract the **top-`p` eigenvectors** `{v_{ℓ,j}}` per block (not just the top one). Feature for prompt `i` = projections `⟨a_ℓ^(i), v_{ℓ,j}⟩` for all `ℓ, j`. Two classifier strategies: (1) per-block, keep the best block; (2) aggregate all blocks into one vector `h ∈ R^{(L)·p}` and train one classifier. Evaluate by **AUROC**.

### 1d. Baselines (extraction methods to compare against RFM)
- **PCA:** pair label-1 with label-0 samples, top eigenvector of the differences.
- **Difference in means:** `v = mean(S_1) − mean(S_0)`.
- **Logistic regression:** normalized coefficient vector (sklearn, sweep `C ∈ {1000,100,1,0.1}`, `liblinear`).
All oriented via Pearson sign like RFM.

---

## 2. Environment & repository layout

Work inside the project root: `/Users/msc/Documents/Remake-Paper-Towards/`.

**Create this structure** (clone GuppyLM as a subdir; put new code in `rfm_guppy/`):
```
Remake-Paper-Towards/
├── GUPPYLM_GUIDE.md                # this file
├── guppylm/                        # cloned: github.com/arman-bd/guppylm
├── rfm_guppy/                      # NEW — our replication package
│   ├── __init__.py
│   ├── activations.py              # Phase 1: hooks → per-block last-token activations
│   ├── concepts.py                 # Phase 2: build labeled concept datasets
│   ├── rfm.py                      # Phase 3: RFM (kernel ridge + AGOP + eig)
│   ├── baselines.py                # Phase 8: PCA / diff-means / logistic
│   ├── steering.py                 # Phase 5: steering hooks + generation
│   ├── monitoring.py               # Phase 7: probing + AUROC
│   ├── evaluate.py                 # Phase 6: steering-effect metric
│   └── cache/                      # saved activations, vectors, results (.npy/.pt/.json)
├── experiments/
│   ├── 01_extract.py
│   ├── 02_steer.py
│   ├── 03_monitor.py
│   └── 04_baselines.py
└── results/                        # figures, tables, REPORT.md
```

**Setup steps:**
1. `git clone https://github.com/arman-bd/guppylm` into the root (giving `./guppylm`).
2. Create a venv. Install GuppyLM's deps (`pip install torch tokenizers` plus whatever its `requirements`/`pyproject` lists). Add: `numpy scipy scikit-learn matplotlib pandas`.
3. Obtain a trained model. Prefer the **pretrained checkpoint from HuggingFace** referenced in GuppyLM's README/notebook (fast). Fallback: run GuppyLM's own training pipeline (`generate_data.py` → `prepare_data.py` → `train.py`); it trains in ~5 min on one GPU and is fine on CPU for our small inference needs.
4. **Smoke test:** run `python -m guppylm chat` (or its inference entrypoint) and confirm it produces fish-themed text.

**Acceptance criteria (Phase 0):**
- [ ] `./guppylm` imports; model + tokenizer load in a Python REPL.
- [ ] A forward pass on a sample prompt returns logits of shape `[B, T, 4096]`.
- [ ] Inference produces coherent in-character output.

---

## 3. Phase 1 — Activation extraction harness (`rfm_guppy/activations.py`)

**Task:** capture per-block, last-real-token activations for a batch of prompts.

1. **Inspect `guppylm/model.py`.** Identify the `nn.Module` list of the 6 transformer blocks (e.g., `model.blocks`, `model.layers`, or similar). Record the exact attribute path — every later phase depends on it.
2. Register `forward_hook`s on each block. Each hook stores the block **output** tensor `[B, T, 384]`.
3. For each prompt: tokenize, build an attention mask / record true length, run forward, and from each block's captured output take the row at the **last non-pad position** → `a_ℓ ∈ R^{384}`.
4. Return an array `A ∈ R^{n × L × 384}` (n prompts, L=6 blocks). Save to `rfm_guppy/cache/` as `.npy` keyed by dataset name.

Provide a function:
```python
def extract_activations(prompts: list[str], model, tokenizer, batch_size=64) -> np.ndarray:
    """Returns array [n, L, k] of last-token block activations."""
```

**Acceptance criteria:**
- [ ] Output shape `[n, 6, 384]`; no NaNs.
- [ ] Two distinct prompts yield distinct activations; the same prompt twice yields identical activations (determinism — `model.eval()`, no dropout).
- [ ] Padding is handled (a prompt's activations are independent of right-padding length — verify by padding to two different lengths and comparing).

---

## 4. Phase 2 — Concept datasets (`rfm_guppy/concepts.py`)

GuppyLM can only be steered toward concepts it represents. Use its **own training categories** as concepts.

1. Load `arman-bd/guppylm-60k-generic` from HuggingFace (fields: `input`, `output`, `category`; 60 categories). Fall back to the repo's local generated data if HF is unavailable.
2. **Enumerate the actual 60 categories** and print counts. Do not assume names — read them from the data.
3. Define **at least 3 binary concepts** as category contrasts (pick real categories from step 2; examples assume typical Guppy categories — adjust to what exists):
   - **C1 — Food/hunger:** label 1 = food/hunger categories; label 0 = a neutral mix of other categories.
   - **C2 — Mood (positive vs negative):** label 1 = happy/excited; label 0 = scared/anxious. (Maps to the paper's "moods" class.)
   - **C3 — Topic (tank/water-conditions vs social/greetings):** label 1 = environment topics; label 0 = social topics.
4. For each concept, build a balanced set (e.g. 200–400 per class if available; minimum ~100/class). Use the `input` text as the prompt. Hold out 20% per concept for validation and a separate 20% test split for monitoring.
5. Save split manifests (prompt + label + concept) as JSON in `cache/`.

```python
def build_concept_dataset(name: str, pos_categories: list[str], neg_categories: list[str],
                          n_per_class: int) -> dict:  # {train, val, test} of (prompt, label)
```

**Acceptance criteria:**
- [ ] ≥3 concepts built, each class-balanced, with train/val/test splits and no prompt leakage across splits.
- [ ] Printed category inventory matches the dataset (counts add up).

---

## 5. Phase 3 — RFM implementation (`rfm_guppy/rfm.py`)

Implement RFM exactly as in §1a. Operate **per block** (loop over the 6 blocks independently).

Required functions:
```python
def laplace_kernel(Z1, Z2, M, L):           # K_M matrix
def rfm_fit(Z, y, L, lam=1e-3, T=5, center=True, normalize=False):
    """Returns M_T (k×k AGOP matrix) after T iterations."""
def agop_grad(alpha, Z_train, z_eval, M, L):  # ∇_z f_t at eval points (vectorized)
def top_eigvectors(M, p=1):                   # top-p eigenvectors, descending eigenvalue
def orient(v, Z, y):                          # sign(Pearson(<a,v>, y)) * v
def extract_concept_vector(Z, y, sweep) -> (v, info):  # full pipeline + HP selection
```

Implementation notes:
- Mean-center `Z` (store the mean; reuse it later for projections and steering consistency).
- AGOP gradients: vectorize over eval points; guard `d_i = 0`.
- Eigendecomposition via `numpy.linalg.eigh` (M is symmetric PSD, 384×384 — trivial).
- HP selection per §1a: sweep `L ∈ {1,10,100}`, `T ∈ {1..10}`, `normalize ∈ {False, True}`; pick max `|Pearson|` on the concept's **val** split. Log the winner.

**Sanity check (do this before trusting anything):** build a synthetic dataset where the label depends on a single known direction `u` in `R^{384}` plus noise. RFM's top eigenvector should align with `u` (`|cos(v, u)| > 0.9`). Gate the rest of the project on this passing.

**Acceptance criteria:**
- [ ] Synthetic recovery test passes (`|cos| > 0.9`).
- [ ] On each real concept, RFM produces a unit vector per block with val `|Pearson|` reported; at least the best block clearly exceeds the diff-means baseline's correlation.
- [ ] Runtime is seconds per concept (n is small).

---

## 6. Phase 4 — Extract & store concept vectors (`experiments/01_extract.py`)

For each of the ≥3 concepts: extract activations (Phase 1) → run RFM (Phase 3) per block → save `{v_ℓ}` (6 oriented unit vectors), the input mean, and the top-`p` eigenvectors (`p=3`, for monitoring) to `cache/<concept>_vectors.npz`. Also record AGOP eigenvalue spectra (to see how low-rank/linear the concept is).

**Acceptance criteria:**
- [ ] One `.npz` per concept containing `v_per_block [6,384]`, `eigvecs_top3 [6,3,384]`, `input_mean [6,384]`, and HP/correlation metadata.
- [ ] A printed table: per concept × block, val `|Pearson|` and top eigenvalue share (`λ1 / Σλ`).

---

## 7. Phase 5 — Steering (`rfm_guppy/steering.py`, `experiments/02_steer.py`)

**Task:** add `ε·v_ℓ` to every token row of block `ℓ`'s output during generation.

1. Register forward hooks that **modify** each block's output: `output = output + ε * v_ℓ` (broadcast over the token axis). Make ε and the active blocks configurable; support steering a subset of blocks if all-blocks is unstable.
2. **Calibrate ε to GuppyLM's activation scale** (do NOT reuse the paper's 0.1–0.65 — those are Llama-specific). Measure the mean per-block activation norm `‖a_ℓ‖` on a sample. Sweep ε across a relative range, e.g. `ε ∈ {0.25, 0.5, 1, 2, 4, 8} × (median ‖a_ℓ‖ / ‖v_ℓ‖)` — but since `v_ℓ` is unit norm, simply sweep ε spanning a fraction of typical activation norm up to a few × it. Stop increasing ε once outputs become repetitive/nonsensical (the paper observed this past a threshold).
3. For each concept, generate responses to a small fixed probe set (5–10 generic prompts) at ε=0 (baseline) and across the ε sweep, both `+ε` and `−ε`.
4. Save a side-by-side table (prompt × ε × output) to `results/`.

```python
def generate_steered(prompt, model, tokenizer, vectors_per_block, eps,
                     blocks="all", max_new_tokens=60) -> str
```

**Acceptance criteria:**
- [ ] At some ε, steered output visibly shifts toward the concept (e.g. food concept → outputs mention eating/food more) vs the ε=0 baseline, while remaining mostly coherent.
- [ ] `+ε` and `−ε` push in opposite directions for at least one concept.
- [ ] The coherence-collapse threshold in ε is identified and recorded.
- [ ] If no concept steers at any ε: report this as the finding (with the ε sweep and norms), and check (a) hook is actually mutating output, (b) correct block attribute, (c) sign/orientation.

---

## 8. Phase 6 — Steering evaluation (`rfm_guppy/evaluate.py`)

We have no GPT-4o judge and don't need one at this scale. Use a **lightweight automatic metric**, plus optional manual spot-check.

1. **Primary metric — probe score:** train the monitoring probe (Phase 7) for the concept, then score steered generations by feeding *the generated text* back through the model and measuring the probe's predicted concept probability. Report mean probe-score vs ε (expect monotone-ish rise with `+ε`). This is self-consistent and fully automatic.
2. **Secondary metric — lexicon hit-rate:** define a small keyword list per concept (e.g. food → {eat, food, hungry, flakes, bite, nibble}); report fraction of generations containing ≥1 keyword vs ε.
3. **Steering-success rate:** a generation counts as steered if probe-score crosses a threshold (e.g. >0.5) OR lexicon hit; report % across the probe set per ε, and "best over ε sweep" (mirrors the paper's "any coefficient worked" scoring).

**Acceptance criteria:**
- [ ] A plot/table of probe-score and lexicon-hit vs ε per concept.
- [ ] At least one concept exceeds baseline steering-success by a clear margin at some ε.

---

## 9. Phase 7 — Monitoring / probing (`rfm_guppy/monitoring.py`, `experiments/03_monitor.py`)

Reframe monitoring as: *from activations, predict the concept label of a held-out prompt.* (Hallucination/toxicity benchmarks are out of scope.)

1. Using top-`p=3` eigenvectors per block (from Phase 4), build features: for each prompt, `⟨a_ℓ, v_{ℓ,j}⟩` for all `ℓ∈{1..6}, j∈{1..3}` → vector in `R^{18}`.
2. Two probes (paper's two strategies):
   - **Per-block:** train a classifier per block (on its 3 projections), keep the best block by val AUROC.
   - **Aggregate:** one classifier on the full `R^{18}` vector.
   Use logistic regression or a small RFM-classifier; tune on val.
3. Evaluate on the held-out **test** split by **AUROC**. Report both strategies.

**Acceptance criteria:**
- [ ] AUROC reported per concept for per-block-best and aggregate probes.
- [ ] AUROC clearly > 0.5 (chance) for concrete concepts; abstract concepts may be weaker — report as-is.

---

## 10. Phase 8 — Baselines & comparison (`rfm_guppy/baselines.py`, `experiments/04_baselines.py`)

Implement PCA, difference-in-means, logistic regression extraction (§1d). For each concept and method:
- **Steering:** repeat Phase 5/6 success metric → success-rate table (method × concept).
- **Monitoring:** use each method's top vector(s) as the probe directions → AUROC table (method × concept).

Produce the paper's headline comparison shape: a table of **% successfully steered** and **AUROC** with rows = {PCA, Diff-Means, Logistic, RFM} and columns = concepts (and an overall column).

**Acceptance criteria:**
- [ ] One steering-success table and one AUROC table, all four methods, all concepts, overall column.
- [ ] RFM is compared fairly (identical data, splits, ε sweep, probe type). State whether RFM wins, ties, or loses — either is a valid result at this scale.

---

## 11. Phase 9 — Report (`results/REPORT.md`)

Write a concise report:
1. **Setup:** GuppyLM specs, concepts chosen (with the real category mapping), data sizes, splits.
2. **Mechanism results:** synthetic RFM recovery; per-concept extraction correlations + AGOP spectra; steering examples (side-by-side ε table); steering-success and monitoring-AUROC tables; baseline comparison.
3. **Findings vs the paper:** which mechanisms held at 8.7M, which were weak/absent, and the ε calibration vs the paper's Llama coefficients.
4. **Honest limitations:** in-domain concepts only; no scaling/transfer/coding/safety claims; small n.
5. **Next step pointer:** the substantive claims (scale, training-time, real monitoring benchmarks) require **Pythia** (14M–12B, 154 checkpoints) — out of scope here.

**Acceptance criteria:**
- [ ] `results/REPORT.md` exists, references actual generated tables/figures in `results/`, and states results plainly (including negatives).

---

## 12. Global guardrails for the executing agent

- **Determinism:** `model.eval()`, fixed seeds, no dropout during extraction.
- **Don't reuse Llama hyperparameters blindly** — recalibrate ε and bandwidth `L` to GuppyLM's activation scale.
- **Verify the hook target** against `guppylm/model.py` before trusting any activation/steering result; a wrong attribute path silently produces garbage.
- **Gate on the synthetic RFM test** (Phase 3) before extracting real concepts.
- **Report negatives.** Weak steering at 8.7M is expected and is itself a result; do not tune until a success appears and then present it as typical.
- **Cache everything** (`rfm_guppy/cache/`) so phases are re-runnable without recomputation.
- **Keep concepts in-domain.** If a chosen concept has near-chance probe AUROC, the model likely doesn't represent it — pick a more lexical/concrete category instead.

## 13. References
- Paper: Beaglehole, Radhakrishnan, Boix-Adserà, Belkin — *Toward universal steering and monitoring of AI models*, arXiv:2502.03708v2 (2025).
- Original method code: https://github.com/dmbeaglehole/neural_controllers (reference for RFM/AGOP + steering hooks).
- RFM origin: Radhakrishnan et al., *Mechanism for feature learning…*, Science 383(6690), 2024.
- GuppyLM: https://github.com/arman-bd/guppylm — model, dataset (`arman-bd/guppylm-60k-generic`), training/inference pipeline.

---

### Suggested execution order (TL;DR)
`Phase 0 (setup)` → `1 (activations)` → `2 (concepts)` → `3 (RFM + synthetic gate)` → `4 (extract vectors)` → `5 (steer)` → `6 (steer eval)` → `7 (monitor)` → `8 (baselines)` → `9 (report)`. Do not skip acceptance checks.
