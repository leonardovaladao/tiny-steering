# WORK_LOG.md — Step-by-Step Record of Work Done

**Project:** Replicating the core mechanisms of *"Toward Universal Steering and Monitoring of AI Models"* (Beaglehole, Radhakrishnan, Boix-Adserà & Belkin, arXiv:2502.03708) on **GuppyLM**, an 8.73M-parameter fish-chatbot LM.

**Status:** Complete. Phases 0–9 all executed and documented; final report + a Medium-style write-up (EN + PT-BR) produced.

---

## Overall Goal

The paper claims that (i) concepts live as **linear directions** in a transformer's residual stream, (ii) those directions can be extracted with the **Recursive Feature Machine (RFM)** — kernel ridge regression + Average Gradient Outer Product (AGOP), taking the top eigenvector of `M_T` — and (iii) once extracted they support both **additive steering** (`a_ℓ → a_ℓ + ε·v_ℓ`) and **monitoring** (probing activations projected onto AGOP eigenvectors). The paper demonstrates this at Llama 8B–70B scale on safety concepts.

This project asks the **isolated mechanism question**:

> Does linear concept extraction + additive steering + activation probing still hold at **<10M parameters**, restricted to concepts the model actually represents (its own training categories)?

Explicitly **out of scope** (and never attempted): jailbreak/deception/political-stance concepts, cross-language transfer, hallucination/toxicity benchmarks, and the scaling claim — none are representable by a 4,096-token fish-domain model. The plan for all of this is in `plans/GUPPYLM_GUIDE.md`; each phase had pre-declared acceptance criteria that had to pass before advancing.

---

## Steps

### Phase 0 — Environment & smoke test  → `plans/phase_plans/GUPPY_PHASE0.md`
- Cloned GuppyLM (`arman-bd/guppylm`) into `guppylm/`; created `.venv` with torch, tokenizers, numpy, datasets, scipy, scikit-learn, matplotlib, pandas, huggingface_hub.
- Downloaded pretrained checkpoint `arman-bd/guppylm-9M` → `checkpoints/guppylm-9M/`.
- Confirmed architecture: **8,726,016 params**, 6 pre-norm blocks, hidden dim 384, 6 heads, vocab capacity 4,096 (2,418 active BPE tokens), max seq len 128, ChatML prompt format.
- **Checks passed:** model + tokenizer load; forward pass gives `[1,10,4096]` logits; generation is coherent and in-character.
- Established the hook target for everything downstream: `model.blocks[0..5]`, whose output is the residual stream `[B, T, 384]`.

### Phase 1 — Activation extraction harness  → `plans/phase_plans/GUPPY_PHASE1.md`, `rfm_guppy/activations.py`
- Built `extract_activations(prompts, model, tokenizer, ...) -> [n, 6, 384]`: forward hooks on all 6 blocks, batched tokenization with right-padding, extraction at the **last non-pad token** (the paper's per-prompt feature), hooks always removed in `finally`.
- **Checks passed:** correct shape, no NaNs; distinct prompts → distinct activations; same prompt twice → bit-identical (`model.eval()`); padding-invariant to within 7×10⁻⁷ (float32 noise only).
- Key observation recorded for later: activation norms **grow across blocks** (≈3 → ≈23), so the paper's absolute Llama ε values cannot be reused.

### Phase 2 — Concept datasets  → `plans/phase_plans/GUPPY_PHASE2.md`, `rfm_guppy/concepts.py`
- Inventoried `arman-bd/guppylm-60k-generic`: 57,000 rows, 60 categories, ~950 rows each.
- **Problem found:** the `input` (user message) field is heavily templated — only 4–34 unique texts per category — so leak-free splits of ≥100/class were impossible on inputs alone.
- **Fix:** use the **full ChatML exchange** (user turn + Guppy's response) as the prompt, since responses are template-composed and yield 40–940+ unique strings per category. Downstream code passes `format_chatml=False`.
- Defined three binary concepts and built 60/20/20 splits:

| Concept | Positive cats | Negative cats | n/class | train/val/test |
|---|---|---|---|---|
| food | food, taste | weather, seasons, tv, music, glass, rain, outside | 300 | 360/120/120 |
| valence | happy, excited, love, curious | scared, fear, bored, tired | 100 | 120/40/40 |
| env_social | water, filter, temp_hot, temp_cold, algae | greeting, bye, friends, visitors, lonely | 300 | 360/120/120 |

- **Checks passed:** ≥3 concepts, every split exactly class-balanced, zero prompt overlap across splits (asserted programmatically). Also defined `CONCEPT_LEXICONS` for later keyword-based evaluation.

### Phase 3 — RFM implementation  → `plans/phase_plans/GUPPY_PHASE3.md`, `rfm_guppy/rfm.py`
- Implemented the Mahalanobis–Laplace kernel `K_M(x,z)=exp(−√((x−z)ᵀM(x−z))/L)`, kernel ridge regression, the analytic gradient, mean-centred AGOP update `M ← (1/n)G_cᵀG_c`, top-eigenvector extraction, and Pearson-sign orientation.
- **Memory-efficient gradient:** rewrote the naive `[n,n,k]` formulation (≈400 MB) as `G = −(W_sum⊙Z_c − W·Z_c)·M`, needing only `W [n,n]`, `M [k,k]`, `Z_c [n,k]`.
- Mahalanobis distances computed via eigendecomposition (`M_half = diag(√D₊)Vᵀ`) + `cdist`, with negative eigenvalues clamped.
- HP sweep `L∈{1,10,100} × T=1..10 × normalize∈{F,T}` run **incrementally in T** (continue from previous `M` instead of restarting) — 60 settings per block in ~1.1 s.
- **Gate test passed:** synthetic recovery (n=1000, k=384, `y = Z·u`) → `|cos(v,u)| = 0.953` vs 0.90 threshold. Documented why regression labels rather than sign labels are the right gate at k=384.
- **Sanity check on real data:** on `valence`, RFM beat difference-in-means at every block (best block 0.918 vs 0.793).

### Phase 4 — Extract & store concept vectors  → `plans/phase_plans/GUPPY_PHASE4.md`, `experiments/01_extract.py`
- Pipeline: re-run the synthetic gate (aborts on failure) → load model → extract + cache activations for 3 concepts × 3 splits (9 `.npy` files) → run RFM on all 6 blocks per concept → serialise.
- Saved `cache/{concept}_vectors.npz` with `v_per_block [6,384]`, `eigvecs_top3 [6,3,384]`, `input_mean [6,384]`, `eigenvalue_spectra`, `lam1_share`, `val_abs_rho`, and the selected HPs.
- **Results — val |Pearson| per block:**

| Block | food | valence | env_social |
|---|---|---|---|
| 0 | 0.906 | 0.507 | 0.735 |
| 1 | 0.973 | 0.899 | 0.941 |
| 2 | 0.975 | 0.908 | **0.950** |
| 3 | **0.978** | 0.905 | 0.944 |
| 4 | 0.977 | **0.918** | 0.942 |
| 5 | 0.977 | 0.908 | 0.943 |

- **Findings:** block 0 is consistently weakest (early residual stream ≈ embeddings); several blocks hit `λ₁/Σλ = 1.000`, i.e. the concept collapses to a **single linear direction**, exactly as the paper predicts. `food` prefers `normalize=True`, the other two `normalize=False`.
- **Checks passed:** one `.npz` per concept, all `v_per_block` unit-norm, all metadata stored.

### Phase 5 — Additive steering  → `plans/phase_plans/GUPPY_PHASE5.md`, `rfm_guppy/steering.py`, `experiments/02_steer.py`
- Forward hooks return `output + ε_ℓ·v_ℓ` on every block, firing at every autoregressive step (GuppyLM has no KV cache). Context manager guarantees hook removal.
- **ε calibration (a necessary deviation from the paper):** measured per-block median activation norms — 11.85 / 22.77 / 27.55 / 30.72 / 30.86 / 29.71 — and swept a **relative** coefficient `ε_ℓ = r × median_norm_ℓ`, `r ∈ {−4,−2,−1,−0.5,−0.25,0,+0.25,+0.5,+1,+2,+4,+8}`. The paper's absolute Llama values (0.1–0.65) would have essentially no effect here.
- Fixed `torch.manual_seed(42)` per generation so runs differ only by the steering perturbation.
- **Bug found and fixed:** lexicon matching used substrings (`"hi" in "anything"` → false positive). Switched to word-boundary regex and **recomputed all results**.
- 7 neutral probe prompts × 12 r values × 3 concepts = 252 generations, saved to `results/steer_{concept}.{json,md}`.
- **Results:** valence 0% → **100%** lexicon hit at r=+0.25, no collapse even at r=+8, and clean bidirectionality ("excited/happiness" vs "shadows/hide/cave/scare"). env_social 14% → **86%**, collapsing into repetition at r=+1.0. food is already saturated at baseline (57%) and positive steering pushes it toward "palate"/"mouth" — a genuine taste register that the keyword lexicon simply doesn't cover.
- **Checks passed:** visible concept shift, opposite-direction behaviour demonstrated, collapse threshold identified.

### Phases 6 & 7 — Steering evaluation + monitoring probes  → `plans/phase_plans/GUPPY_PHASE6-7.md`, `rfm_guppy/monitoring.py`, `rfm_guppy/evaluate.py`, `experiments/03_monitor.py`
Implemented together because Phase 6's main metric depends on Phase 7's probes.

- **Phase 7 (monitoring):** per prompt, centre each block activation by `input_mean`, optionally L2-normalise per the Phase-4 flag, project onto the **top-3 AGOP eigenvectors** per block → an 18-dim feature. Trained `StandardScaler → LogisticRegression(liblinear)` probes, C swept on validation AUROC, in two variants (per-block-best and aggregate).
  - **Result: test AUROC = 1.0000 for all three concepts**, both variants; best block = 1 in each case. Only block 0 is weaker (val ≈0.91–0.99). Probes cached to `cache/{concept}_probe.pkl`, summary in `results/monitor_auroc.json`.
- **Phase 6 (steering evaluation):** re-fed every steered generation to the model as a complete ChatML exchange (matching Phase-2 formatting) and scored it three ways — probe score, word-boundary lexicon hit, and success = `probe > 0.5 OR lexicon hit`.
  - **Result: all three concepts reach 100% success at r = +0.25**, from baselines of 71.4% / 71.4% / 28.6%. Probe score proved more sensitive than the lexicon at large ε; negative r drives the probe to exactly 0 for valence and env_social (true bidirectional control); food's probe degrades at r ≥ 1.0 (representation pushed off-manifold).
- **Checks passed:** AUROC ≫ chance for all concepts; every concept beats its baseline success by ≥28.6 pp (criterion was >20 pp for at least one).

### Phase 8 — Baseline comparison  → `plans/phase_plans/GUPPY_PHASE8-9.md`, `rfm_guppy/baselines.py`, `experiments/04_baselines.py`
- Implemented the paper's three alternatives — **PCA** (top eigenvector of the centred difference matrix), **DiffMeans** (`mean(pos) − mean(neg)`), **LogReg** (unit-normalised LR coefficient, C swept on val) — each returning a unit vector oriented identically to RFM's.
- Ran both downstream evaluations for every method: monitoring AUROC, and a steering sweep (`r ∈ {−2 … +2}` × 7 prompts). Steering used the **baseline vectors** but the **RFM probe** as the scorer, so all methods are measured with the same stick.
- **Results:**

| Method | Monitoring AUROC (overall) | Steering success (overall) | Weakest val \|ρ\| |
|---|---|---|---|
| PCA | 0.8788 | 1.000 | 0.154 (valence) |
| DiffMeans | 1.0000 | 1.000 | 0.800 |
| LogReg | 1.0000 | 1.000 | 0.960 |
| **RFM** | **1.0000** | **1.000** | 0.918 |

- **Interpretation:** RFM's advantage over **PCA** reproduces clearly (0.88 vs 1.00 AUROC; PCA's top PC tracks within-class variance rather than the class boundary on small/nuanced concepts). RFM's advantage over DiffMeans/LogReg does **not** appear — at this scale, with perfectly in-domain concepts, everything ties at ceiling. This matches the paper's own framing that the separation requires harder concepts and larger models.

### Phase 9 — Final report  → `results/REPORT.md`
- Consolidated all phases into a self-contained report: setup, synthetic gate, extraction tables, steering samples + sweeps, monitoring AUROC, baseline comparison, a claim-by-claim "what holds / where we diverge" section, limitations, and next steps.
- **Honest limitations stated:** concepts are in-domain training categories, not safety properties; the Phase-6 probe metric is partially circular (same probe family used to train and score); n is small (100–300/class); AUROC = 1.000 is a **ceiling effect**; no scaling, training-time, or transfer claims are supported by this work.
- **Next step identified:** a Pythia model ladder (14M → 12B with intermediate checkpoints) plus genuinely hard concepts and an external (non-circular) judge, which is where the paper's scaling and RFM-vs-baseline claims can actually be tested.

### Write-up
- `MEDIUM_POST.md` — a public-facing narrative version of the whole replication (English).
- `MEDIUM_POST_PTBR.md` — Portuguese translation of the same.
- (Two rendered PDFs of an earlier LaTeX version sit in `.trash/`, superseded.)

---

## Repository Map

```
Remake-Paper-Towards/
├── README.md                     project overview: origin of the idea and the goal
├── requirements.txt              Python dependencies (CPU-only)
├── plans/
│   ├── GUPPYLM_GUIDE.md          the pre-registered 9-phase plan with acceptance criteria
│   └── phase_plans/              per-phase execution notes (what was built, what was found)
│       └── GUPPY_PHASE0.md … GUPPY_PHASE8-9.md
├── WORK_LOG.md                   this file
├── MEDIUM_POST{,_PTBR}.md        public write-up (EN / PT-BR)
├── Toward universal ... .pdf     the source paper
├── guppylm/                      upstream GuppyLM repo as a submodule, pinned at a30df30
├── checkpoints/guppylm-9M/       pretrained weights + tokenizer
├── rfm_guppy/                    the replication package
│   ├── activations.py            Ph1 — last-token per-block activation extraction
│   ├── concepts.py               Ph2 — concept datasets + lexicons
│   ├── rfm.py                    Ph3 — kernel ridge + AGOP + eigenvectors
│   ├── steering.py               Ph5 — additive steering hooks
│   ├── evaluate.py               Ph6 — probe-score / lexicon steering evaluation
│   ├── monitoring.py             Ph7 — probe features, training, AUROC
│   ├── baselines.py              Ph8 — PCA / DiffMeans / LogReg
│   └── cache/                    activations (.npy), vectors (.npz), probes (.pkl)
├── notebooks/                    executed notebook view of the same four experiments
│   ├── nbcommon.py               shared boilerplate (paths, checkpoint loading, plot style)
│   └── 01_extract … 04_baselines.ipynb
├── experiments/
│   ├── 01_extract.py             Ph1–4  extract + store concept vectors
│   ├── 02_steer.py               Ph5    steering sweep
│   ├── 03_monitor.py             Ph7+6  probes, then steering evaluation
│   └── 04_baselines.py           Ph8    four-way method comparison
└── results/
    ├── steer_{concept}.{json,md}      Ph5 generations per r
    ├── eval6_{concept}.{json,md}      Ph6 probe/lexicon/success vs r
    ├── monitor_auroc.json             Ph7 AUROC summary
    ├── baselines_{monitoring,steering}.json, baselines_comparison.md   Ph8
    └── REPORT.md                      Ph9 final report
```

**Setup:** `pip install -r requirements.txt`, then `hf download arman-bd/guppylm-9M --local-dir checkpoints/guppylm-9M`. Everything runs on CPU.

**Reproduction order:** `01_extract.py` → `02_steer.py` → `03_monitor.py` → `04_baselines.py`. Every stage caches, so re-runs never repeat inference.

---

## Bottom Line

The mechanism replicates at 8.73M parameters. AGOP converges (synthetic `|cos| = 0.953`), the three concepts are near-rank-1 linear directions (`λ₁/Σλ ≈ 1.000`), additive steering at `r = +0.25` gives 100% success on all three, and AGOP-eigenvector probes hit AUROC = 1.000. Two honest caveats stand: the ε coefficient **must** be calibrated to the model's own activation norms rather than borrowed from the paper, and at this scale the results sit at ceiling — which reproduces RFM > PCA but leaves RFM ≈ DiffMeans ≈ LogReg, exactly as the paper's own reasoning would predict for easy, in-domain concepts.
