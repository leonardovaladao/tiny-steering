# Revision 2026-10 — what was asked and what the tests answered

Source: the executed notebooks and data files in `run_20261009/` (the `.md` reports were not read).
Setting for every check: GuppyLM (8.73M), cached last-token residual activations $a_\ell\in\mathbb R^{384}$,
blocks $\ell=0..5$, concepts `food`, `valence`, `env_social` (train 360/120/360, balanced).
The 2026-10-09 rerun reproduces the 2026-10-02 CSVs of Checks I, Ia, II, III **byte for byte**.
Check IV exists only in the 10-09 run.

---

## Check I — Is $\lambda_1/\sum_j\lambda_j\approx1$ evidence of a concept, or does RFM produce it on any labels?

**How it was done.**
- Null model: replace $y_{\text{train}}$ by a random permutation. Class counts stay the same, but the
  link between label and activation is destroyed. Then run the *same* RFM code
  (`_agop_step`, `extract_concept_vector`) and record the share $s=\lambda_1(M_T)/\operatorname{tr}M_T$.
- Gate: recompute the 18 stored shares (3 concepts × 6 blocks) at the stored HPs; they must match
  the cache to $10^{-4}$.
- Regime A: HPs frozen at the values chosen with real labels, $B=100$ permutations per cell. Also
  recorded: $|\cos(u_{\text{shuf}},v_{\text{real}})|$ and the val $|\rho|$ of the shuffled direction.
- Regime B: the full HP sweep ($\sigma\in\{1,10,100\}$, $T=1..10$, normalize ∈ {F,T}, selection by
  val $|\rho|$) is rerun on shuffled train labels, $B=20$. This is the null for the number the paper
  reports.
- Regime C: $s(T)$ curves, real labels vs 10 permutations, at each concept's best block.
- Side check E: val $|\rho|$ of 2000 random unit directions, of PC1 and of DiffMeans, to calibrate
  $|\rho|$.
- Criterion (fixed before running): "concept-informative" is refuted at a cell if
  $p=\frac{1+\#\{s_{\text{shuf}}\ge s_{\text{real}}\}}{1+B}\ge0.05$, or if $\ge5\%$ of shuffled runs
  reach $s\ge0.95$.

**Answer.**
- Shuffled labels already give large shares: median 0.43–0.76 at the stored HPs, max up to 0.99.
  So a high share by itself is **not** evidence of a single concept direction.
- The real share still beats the null in most cells: refuted in **5/18** cells (A) and **4/18** (B),
  all in `valence` (blocks 0, 2, 5) and `env_social` (blocks 0–1). In 3 of those, the real share is
  itself small (0.15–0.22, $T=1$, $\sigma=100$).
- In (B) the smallest possible $p$ is $1/21=0.048$, so "holds" there sits on the resolution floor.
- Side result: val $|\rho|$ is a weak criterion. RFM on **shuffled** labels still reaches
  $|\rho|\approx0.86$ on `food`; plain PC1 reaches 0.94–0.97 and DiffMeans 0.95–0.98 on `food`.
  On `valence`/`env_social`, shuffled RFM falls to random-direction level (0.1–0.5).

## Check Ia — Does the RFM iteration itself concentrate the spectrum? (all blocks)

**How it was done.**
- Grid: every cell $(c,\ell,\text{normalize},\sigma)$, i.e. $3\times6\times2\times3=108$ cells, not just
  the best block as in Check I.
- In each cell: start from $M_0=I_{384}$, iterate $M_{t+1}=\mathrm{AGOP}(Z_\ell,y;M_t,\sigma,\lambda=10^{-3})$,
  and record $s(T)$ for $T=1..10$. This is done for the real labels and for $B=20$ permutations
  (seed 20261002).
- Gate: the real-label $s(T^\star)$ at the stored HPs must match the cache to $10^{-4}$ (18 pairs).
- Test of $H_0$ ($s_{\text{shuf}}(T)=s_{\text{shuf}}(1)$): take the paired differences
  $\Delta_b(T)=s_b(T)-s_b(1)$. $H_0$ is refuted at (cell, $T$) if $|\text{median}_b\,\Delta_b|>0.05$
  **and** a two-sided sign test gives $p<0.05$. That makes $108\times9=972$ tests.
- At the stored HPs: compare real vs shuffled at $T^\star$, and also at $T=1$.
- Cross-check: compare against Check I, Regime C, which used other seeds.

**Answer.**
- $H_0$ refuted in **972/972** (cell, $T$) pairs, already at $T=2$ in every cell.
  Under shuffled labels the share grows by +0.15 to +0.59 between $T=1$ and $T$.
- So **the iteration produces concentration on its own**. Part of every reported $s\approx1$ is an
  artifact of $T$, not of the concept.
- At $T=1$ the real share is often **not** above the shuffled one ($p_{\text{perm}}(1)\ge0.05$ in
  14/18 stored cells). The real–shuffled gap appears only after iterating.
- Real labels reach $s\ge0.999$ in 71/108 cells; shuffled never pass 0.989.
- Cross-check with Check I, Regime C: real curves identical; shuffled medians agree within MC noise.

## Check II — Is the PCA baseline wrong by construction?

**Question.** The code centres the paired-difference matrix $D$ before the SVD; Beaglehole et al.
(App. H) do not. Since $\tilde D^\top\tilde D=D^\top D-m\,\bar d\bar d^\top$ and $\bar d=\hat\mu_+-\hat\mu_-$,
centring removes exactly the DiffMeans direction. Does the reported "RFM ≫ PCA" survive the fix?

**How it was done.**
- Gates: the current `pca_vector` must reproduce the published Phase-8 numbers. The identity above
  must hold numerically, and $\bar d=\hat\mu_+-\hat\mu_-$ must hold, in all 18 (concept, block) cells.
- Variants compared, all with the same preprocessing (centre with the train mean, orient by Pearson
  sign, unit norm):
  - `PCA_c`: the current code (sequential pairing, $D$ centred).
  - `PCA_u`: $D$ not centred, with sequential pairing and with 20 random pairings.
  - `DM`.
  - RFM, using the stored values.
- Per block: val $|\rho|$, $|\cos|$ with DM, and the share of $\|D\|_F^2$ carried by the rank-one term
  $m\|\bar d\|^2$ (the part that centring discards).
- Headline: best-block val $|\rho|$ and test AUROC of the 6-feature aggregate probe.
- Criterion (fixed before running): uncentred PCA "breaks even" with DiffMeans if, for all 3 concepts,
  AUROC $\ge$ DM $-0.005$ and best-block $|\rho|\ge$ DM $-0.02$, worst case over the pairings.
- Supplementary: a paired bootstrap over the val samples, with the directions held fixed, for
  $|\rho|_{\text{RFM}}-|\rho|_{\text{PCA}_u}$ and $|\rho|_{\text{DM}}-|\rho|_{\text{PCA}_u}$.

**Answer.**
- Gates passed: old numbers reproduced; identity verified numerically in all 18 cells.
- **The old baseline was mis-specified.** Uncentred PCA is almost parallel to DiffMeans
  ($|\cos|\approx0.95$–$0.99$ from block 1 on) and jumps up:

  | concept | $|\rho|$ PCA centred → uncentred | DM | RFM | test AUROC PCA centred → uncentred |
  |---|---|---|---|---|
  | food | 0.929 → 0.972 | 0.975 | 0.978 | 0.995 → 1.000 |
  | valence | 0.154 → 0.694 | 0.800 | 0.918 | 0.743 → 0.995 |
  | env_social | 0.490 → 0.886 | 0.914 | 0.950 | 0.899 → 1.000 |

  Mean test AUROC: 0.879 (old) → 0.998 (fixed).
- Formal break-even: **not reached** (fails on `valence` and on `env_social` $|\rho|$), so uncentred
  PCA stays slightly below DM.
- Bootstrap over val samples: RFM − PCA$_u$ gap is positive for `valence` (+0.22, CI [0.10, 0.40])
  and `env_social` (+0.06, [0.04, 0.09]); not significant for `food`.
- Net: "RFM beats PCA by a wide margin" does not hold; the remaining gap is small and only in $|\rho|$.

## Check III — Does the 100 % steering success survive an activation-blind judge?

**Question.** The old success metric (probe $>0.5$ or lexicon hit) reads the same AGOP basis that the
steering vector comes from, so the metric is partly circular.

**How it was done.**
- No new steered generation: the stored 7 prompts × 12 values of $r$ per concept
  (`steer_<concept>.json`) are re-scored.
- Blind judge $g_c$:
  - TF-IDF on word 1–2-grams + logistic regression with balanced class weights.
  - Trained on the deduplicated assistant outputs of the GuppyLM dataset, labelled by the concept's
    pos/neg categories.
  - Score: $g_c=2P(\text{pos})-1\in[-1,1]$. Text with no known n-grams scores ≈ 0.
- Coherence:
  - $\mathrm{PPL}(y\mid x)$ under the unsteered model, and distinct-2 (unique bigrams / bigrams).
  - "Coherent" means PPL $\le q_{95}$ and distinct-2 $\ge q_{05}$. The quantiles come from 100
    unsteered generations on prompts from categories no concept uses.
- Gates:
  - G1: 5-fold CV AUROC of $g_c\ge0.95$. A concept that fails it is not interpreted.
  - G2: the lexicon hit rates recompute exactly.
  - G3: the 7 regenerated $r=0$ texts match the stored ones character for character.
- H1: exact one-sided sign tests against $r=0$ at $r^\star\in\{0.25,0.5\}$. Supported only if 7/7
  prompts go up at $+r^\star$ **and** 7/7 go down at $-r^\star$ (Bonferroni across the two $r^\star$).
- H2: "no coherent steering" is refuted if, at some $r>0$, $\ge4/7$ generations are both
  $g_c>0$ and coherent.
- H3 (descriptive): Spearman correlation and binary agreement between the probe and $g_c$ over the
  84 generations per concept.
- Post-hoc (not pre-registered): number of distinct texts and bigram Jaccard similarity across
  prompts.

**Answer.**
- Gates: G2 (lexicon rates, 36/36) and G3 (regenerated $r=0$ texts, 7/7) passed.
  **G1 failed for `valence`** (judge CV AUROC 0.78 < 0.95; only 40+40 unique outputs), so `valence` is
  not interpretable. `food` (0.998) and `env_social` (≈1.000) pass.
- **H1 (direction moves the judge both ways at $r^\star=\pm0.25,\pm0.5$, 7/7 each):** not supported
  for `food` (5/7 up) or `env_social` (7/7 up, but 6/7 down). The single "supported" cell is
  `valence` at $r=0.5$, which G1 says not to interpret.
- **H2 (coherent steering):** **0 coherent generations at any $r\neq0$**, for every concept
  (lowest steered PPL ≈ 2 559 vs threshold 2.25). Joint "on-concept ∧ coherent" = 0/7 everywhere.
  So the claim "no coherent steering" is not refuted, and the claim "fluency degrades at every
  $r\neq0$" is confirmed.
- **H3 (probe vs blind judge):** they agree on `valence`/`env_social` (Spearman ≈0.81, binary
  agreement 99 %), but **not on `food`** (Spearman 0.41, agreement 57 %; 35 cases where the probe
  says "yes" and the judge says "no").
- Old vs new table, best positive $r$: old metric 7/7 everywhere; blind judge 3/7 (`food`), 7/7
  (`env_social`, `valence`); blind judge ∧ coherent **0/7** for all.
- Post-hoc caveat: steered texts are near-copies across prompts (bigram Jaccard 0.85–1.00 at
  $|r|\ge0.5$). The 7 prompts are not 7 independent observations.

## Check IV — With few examples per class, does RFM beat DiffMeans? (pre-registered)

**Question.** $\Delta_c(n)=\mathbb E_S[\text{AUROC}(v_{\text{RFM}})-\text{AUROC}(v_{\text{DM}})]$ for balanced
subsamples with $n$ examples per class.

**How it was done.**
- Step 0: freeze the HPs using train + val only, before the test split is opened.
  - RFM: $L\in\{1,10,100\}$, $T=1..10$, normalize = False, selected by val $|\rho|$.
  - LogReg: $C$ is re-selected the same way.
- Subsamples: for each concept and $n\in\{1,2,4,8,16,32\}$, draw $R=200$ balanced subsamples from the
  train pool, plus the full pool $n_{\max}$ (one draw).
  - The same subsamples are used for all four methods and all blocks (paired design).
- Methods on each subsample (centred on the subsample mean, oriented by train Pearson sign, unit norm):
  - RFM with frozen $L,T$. The top eigenvector of every $M_t$, $t=1..10$, is also kept.
  - DM.
  - Uncentred PCA on randomly paired differences.
  - LogReg ($\ell_2$, liblinear).
- Metric: score $\langle a_\ell(x),v\rangle$ on the full test split; test AUROC (primary) and
  $|\rho|$ (secondary). Primary block 3; all 6 blocks reported, without selection.
- C0 sanity check, enforced inside the loop (a violation stops the run):
  - At $n=1$, RFM = DM: $1-|\cos|<10^{-8}$ and $|\Delta|<10^{-8}$.
  - Every direction lies in the row space of the data: residual $<10^{-6}$.
- Statistics:
  - $B=10\,000$ bootstrap resamples over the $R$ draws, percentile 95% CIs.
  - One-sided $p=(1+\#\{\hat\Delta^*\le0\})/(1+B)$.
  - Holm within each block over $k=10$ cells (2 main concepts × $n\in\{2,..,32\}$), $\alpha=0.025$.
- Hypotheses (fixed before running):
  - H3a is refuted if some main concept at $n\in\{2,4\}$ has a lower CI bound $>0$.
  - H-artigo is supported if Holm declares some cell with $\Delta>0$.
  - H3b is supported if, in the same concept, one cell has an upper CI bound $<0$ and a larger $n$ has
    a lower CI bound $>0$.
  - Also reported, outside the hypotheses: $\Delta(n,T)$ for every $T$, including the oracle $T^\star$.

**Answer.**
- **C0 sanity passed** in the final run: at $n=1$ RFM ≡ DM ≡ PCA ≡ LogReg (gap $\le 10^{-15}$),
  and every direction lies in the span of the data (residual $\le 3\times10^{-12}$).
  An earlier run had stopped on a C0 failure (`IV-small-n_run_failedC0.log`, RFM residual = 1 at
  $n=1$); the final notebook passes it.
- **H3a (RFM is not better at $n\in\{2,4\}$): not refuted, in all 6 blocks.** RFM is clearly *worse*:
  $\Delta=-0.05$ to $-0.21$ (CIs entirely below 0). Even the best $T$ chosen with oracle knowledge
  stays below 0.
- **H-artigo (RFM wins somewhere): refuted at the primary block 3** (and block 5). It is "supported"
  in blocks 0, 1, 2, 4, but only at $n=32$. The only practically relevant win is `valence`
  block 0 ($\Delta=+0.077$); the others are $\le+0.016$ (`valence` b1) or $\approx+10^{-4}$ (`env_social`).
- **H3b (RFM loses at small $n$, wins at larger $n$): supported in every block.** The crossover is at
  $n\approx32$.
- RFM also loses to LogReg at every $n\le32$ and to uncentred PCA at $n\le4$.
- `food` (ceiling control): RFM worse up to $n=16$, tied at $n=32$.

---

## One-line answers

| Check | Question | Answer |
|---|---|---|
| I | Is a high top-eigenvalue share concept-specific? | Partly. Shuffled labels already give 0.4–0.8; the real share beats the null in 13–14/18 cells. |
| Ia | Does iterating RFM inflate the share? | Yes, in every cell, from $T=2$ on. |
| II | Was the PCA baseline fair? | No. Centring removed the DiffMeans direction; fixed PCA ≈ DM, and the RFM gap nearly closes. |
| III | Is steering really 100 % successful? | No. Blind judge: partial at best; coherent ∧ on-concept = 0/7 at every $r\neq0$. |
| IV | Does RFM help when data is scarce? | No: worse than DM at $n\le16$; a small advantage appears only at $n\approx32$. |
