# Check III — Does the steering result survive a judge that never sees activations?

Revision of 2026-10-02. Notebook: `III-steering-metric.ipynb`.
Tables: `III-steering-metric_*.csv` (`per_r`, `H1`, `H2`, `H3`, `old_vs_new`, `judge_validity`,
`duplication`, `generations`, `calibration`).

This is the **minimal version**: the $7 \times 12$ RFM generations per concept already stored in
`results/steer_<concept>.json` are re-scored. No new steered text was generated. The only new
inference is a reproduction gate, 100 unsteered generations used to calibrate coherence, and
perplexities. The model is loaded from `.trash/guppylm-9M/`, and nothing outside this folder
was touched.

## Short answer

1. **The "100% steering success" does not survive.** If success means "the blind judge says
   on-concept **and** the text is coherent", every concept scores **0/7 at every $r \neq 0$**.
2. **Coherence is now measured, and it fails everywhere.** All 231 steered generations
   ($r \neq 0$) are incoherent. The lowest steered perplexity is **2,559**; the highest
   unsteered perplexity is **2.8**. No threshold between those two values changes the result.
   The write-up's "fluency degrades at every $r \neq 0$" is confirmed.
3. **The direction claim survives on one concept only, and on none under the pre-fixed rule.**
   - **`env_social`:** the blind judge moves from $-0.66$ to $+0.76$ at $r = +0.25$ on 7/7
     prompts, and it agrees with the probe on 83 of 84 generations. So the probe was not
     inflating this concept. The mirror test at $-r$ gives 6/7, so the pre-fixed criterion
     (7/7 both ways) is not met.
   - **`food`:** the blind judge sees the score move from $-0.34$ to about $0$, which is
     neutral, not food. The probe says "food" on 35 generations where the judge says "not
     food". Here the circular metric *was* manufacturing the result.
   - **`valence`:** this judge cannot be built from the dataset (CV AUROC 0.78; gate G1 fails).
     The only remaining evidence is the lexicon (*hide* vs *excited*).
4. **This was not planned: the 7 prompts are not 7 observations.** At $|r| \ge 0.5$ the steered
   texts are near-copies across prompts (mean pairwise bigram Jaccard 0.71–1.00). The steering
   overrides the prompt, and the seed is fixed. Every "7/7" in the write-up, and every sign-test
   $p$ below, therefore overstates the evidence. The effective sample size is close to 1.

## 1. What was measured

For a generation $y$ to prompt $x$:

- **Blind judge.** $g_c(y) = 2\,P(\text{pos} \mid y) - 1 \in [-1, 1]$. This is a TF-IDF
  (word 1–2 grams) + logistic regression classifier, trained only on the **assistant outputs**
  of the GuppyLM dataset, deduplicated and labelled by the concept's pos/neg categories (the
  same categories as `rfm_guppy/concepts.py`). Class weights are balanced. The judge never
  sees an activation, the AGOP, or the probe. A second variant drops the concept's lexicon
  words from the vocabulary, to test whether the judge only re-learns the lexicon.
- **Perplexity.** $\mathrm{PPL}(y \mid x) = \exp\big(-\tfrac{1}{|y|}\sum_t \log p_\theta(y_t \mid x, y_{<t})\big)$
  under the **unsteered** model, with $x$ the ChatML prefix.
- **Distinct-2.** Unique word bigrams divided by total bigrams.
- **Coherent** $:\Leftrightarrow \mathrm{PPL} \le 2.25$ and distinct-2 $\ge 1.0$. These are
  the $q_{95}$ and $q_{05}$ of 100 unsteered generations on prompts from the 33 categories
  that no concept uses.
- **On-concept** $:\Leftrightarrow g_c > 0$. **Joint** $:=$ on-concept $\wedge$ coherent.

## 2. Gates

| Gate | Result |
|---|---|
| G2: lexicon hit rates recomputed from `steer_*.json` match `eval6_*.json` | pass (36/36 cells) |
| G3: the 7 regenerated $r=0$ texts match the stored ones exactly | pass (7/7) |
| G1: judge 5-fold CV AUROC $\ge 0.95$ | food 0.998 pass · **valence 0.783 FAIL** · env_social 1.000 pass |

**Why valence fails.** The dataset has **10 unique assistant outputs per category**. The
valence classes therefore have 40 vs 40 texts, which is too few to learn a valence judge. The
same is true for food's negative class (70 texts against 647). The large categories (`food`,
`water`, `temp_*`, `greeting`, `bye`, `lonely`) have hundreds. This matters beyond this check:
the "unique exchanges" counts in `concepts.py` count (input, output) pairs, and they hide how
few distinct answers there really are.

Following the rule fixed before running, **valence results from the blind judge are reported
but not interpreted.**

## 3. Limits of the judge

- **Its zero point is not neutral for food.** On unsteered text from unrelated categories,
  mean $g_{\text{food}} = -0.51$, while valence gives $-0.01$ and env_social $-0.07$. The food
  judge contrasts "food" with "weather/tv/music/…", and the fish's ordinary self-talk lands on
  the negative side. For example, *"i'm a little fish … mostly about food."* scores $-0.54$. So
  the displacement $\Delta g$ (H1) is the reliable read for food. The absolute "on-concept"
  count is biased against food.
- **"palate" carries almost no weight.** It occurs in the food training outputs, but its weight
  is $+0.09$, against $+0.53$ for *mouth*. The `palate palate palate promise to…` generations
  score between $-0.25$ and $+0.15$. The write-up calls these "squarely on-concept"
  (`03_replication.tex:259`). That is our reading of the text, not something a judge trained
  on GuppyLM's own food answers confirms.
- **Coverage is full.** Every generation contains words the judge knows, so the judge never
  abstains.
- **Removing the lexicon words changes little.** CV AUROC is 0.998 → 0.998 for food and
  1.000 → 1.000 for env_social, and the H1 counts move by at most one prompt. The judge is not
  just the lexicon in disguise.

## 4. Results per $r$ (counts out of 7)

Values of $r$ near zero are shown here. The full sweep is in `III-steering-metric_per_r.csv`.
`old` is the published "probe > 0.5 OR lexicon" success.

| concept | $r$ | mean $g$ | on | coherent | joint | median PPL | probe mean | lexicon | old |
|---|---|---|---|---|---|---|---|---|---|
| food | −0.50 | −0.81 | 0 | 0 | 0 | 557,695 | 0.28 | 0 | 2 |
| food | −0.25 | −0.81 | 0 | 0 | 0 | 114,187 | 0.18 | 0 | 1 |
| food | 0 | −0.34 | 2 | 7 | 2 | 1.6 | 0.49 | 4 | 5 |
| food | +0.25 | −0.06 | 3 | 0 | 0 | 216,419 | 0.88 | 2 | **7** |
| food | +0.50 | −0.04 | 0 | 0 | 0 | 580,401 | 0.88 | 0 | **7** |
| valence* | −0.50 | −0.29 | 0 | 0 | 0 | 207,778 | 0.00 | 7 | 7 |
| valence* | −0.25 | −0.30 | 0 | 0 | 0 | 15,385 | 0.00 | 2 | 2 |
| valence* | 0 | −0.00 | 4 | 7 | 4 | 1.6 | 0.75 | 0 | 5 |
| valence* | +0.25 | +0.13 | 7 | 0 | 0 | 28,424 | 0.98 | 7 | **7** |
| valence* | +0.50 | +0.19 | 7 | 0 | 0 | 3,148,854 | 1.00 | 7 | **7** |
| env_social | −0.50 | −0.92 | 0 | 0 | 0 | 132,325 | 0.00 | 0 | 0 |
| env_social | −0.25 | −0.90 | 0 | 0 | 0 | 39,966 | 0.00 | 0 | 0 |
| env_social | 0 | −0.66 | 0 | 7 | 0 | 1.6 | 0.09 | 1 | 2 |
| env_social | +0.25 | **+0.76** | **7** | 0 | 0 | 4,003 | 1.00 | 6 | **7** |
| env_social | +0.50 | +0.75 | 7 | 0 | 0 | 16,391 | 1.00 | 0 | **7** |

\* The valence judge failed G1.

Perplexities in the thousands to millions are above the vocabulary size (4,096). The unsteered
model is very peaked (its median PPL on its own text is 1.4), so it gives the steered tokens
less probability than a uniform guess would. Valence at $r > 0$ shows why perplexity was
needed alongside distinct-2: those texts have distinct-2 ≈ 1.0 (nothing repeats), yet they
read as word salad (`"tv means floating someone wanting chestpick…"`).

## 5. H1 — displacement, paired sign test against $r = 0$

Rule fixed before running: *supported* if $g(y_{+r^\star}) > g(y_0)$ on 7/7 prompts **and**
$g(y_{-r^\star}) < g(y_0)$ on 7/7 prompts, for $r^\star = 0.25$ or $0.5$.

| concept | $r^\star$ | mean $\Delta g$ at $+r^\star$ | up at $+r^\star$ | mean $\Delta g$ at $-r^\star$ | down at $-r^\star$ | supported |
|---|---|---|---|---|---|---|
| food | 0.25 | +0.28 | 5/7 | −0.47 | 6/7 | no |
| food | 0.50 | +0.30 | 5/7 | −0.47 | 6/7 | no |
| valence* | 0.25 | +0.13 | 6/7 | −0.29 | 7/7 | no |
| valence* | 0.50 | +0.19 | 7/7 | −0.29 | 7/7 | (yes) |
| env_social | 0.25 | +1.42 | 7/7 | −0.24 | 6/7 | no |
| env_social | 0.50 | +1.41 | 7/7 | −0.26 | 6/7 | no |

How to read this:

- **env_social.** The positive push is the strongest effect in the project, with
  $\Delta g = +1.42$ on a scale of width 2. The negative push misses 7/7 because of a floor:
  the baseline already sits at $-0.66$, and one prompt's baseline (*"oh hello…"*, $g = -0.96$)
  cannot go lower.
- **food.** The sign does follow $r$ on average. But $+r$ moves the text to neutral, not to
  food, and two of the seven baselines are already on-concept.
- **valence.** The single "supported" row is formally met, but it comes from an invalid judge.
- In every row, the sign test assumes 7 independent observations. §8 shows that assumption is
  false for the steered texts.

## 6. H2 — coherence and capability

- **"Fluency degrades at every $r \neq 0$":** not refuted anywhere. The coherent count is 0/7
  at all 33 (concept, $r \neq 0$) cells, against 7/7 at $r = 0$.
- **"No coherent steering was demonstrated":** not refuted. The joint count is 0/7 at every
  $r > 0$ for every concept.

This verdict does not depend on the thresholds. Note that the distinct-2 threshold came out as
1.0 because 99 of the 100 calibration texts have no repeated bigram, so it is very strict.
Perplexity alone would give the same verdict: the gap between the worst unsteered text (2.8)
and the best steered text (2,559) is three orders of magnitude.

## 7. H3 — was the probe circular in practice?

| concept | Spearman(probe, $g$) | binary agreement | probe yes / judge no | probe no / judge yes |
|---|---|---|---|---|
| food | 0.41 | 57% | **35** | 1 |
| valence* | 0.81 | 99% | 1 | 0 |
| env_social | 0.82 | 99% | 1 | 0 |

- **On env_social, the probe and the blind judge agree on the direction.** The probe's
  weakness there was never the direction. It was its blindness to fluency.
- **On food, the probe disagrees with the judge on 35 of 84 generations, nearly always by
  saying "food" when the judge does not.** The `palate` text gets probe scores of 0.66–1.00
  but $g \approx 0$. This is the concrete case of the circularity the write-up describes in
  the abstract: a model pushed along $v$ produces tokens that, read back, project onto $v$,
  and that is not the same as producing text about food.

## 8. Effective sample size (not pre-declared)

This was added after reading the generations. Mean pairwise Jaccard similarity of word-bigram
sets across the 7 prompts (1 = identical texts):

| $r$ | food | valence | env_social |
|---|---|---|---|
| −0.50 | 0.77 | 0.88 | 0.91 |
| −0.25 | 0.26 | 0.64 | 0.47 |
| 0 | 0.05 | 0.05 | 0.05 |
| +0.25 | 0.46 | 0.55 | 0.89 |
| +0.50 | 0.85 | 0.71 | 0.98 |
| ≥ +1 | 0.94–1.00 | 0.84–1.00 | 1.00 |

At $r = 0$ the 7 answers are different texts. Once steering is on, the prompt barely matters,
and with `seed=42` fixed for every generation the sampling path is the same too. So the
7 steered texts at a given $r$ are close to **one** sample, repeated. This affects:

- every "7 of 7" in the write-up's steering section (e.g. *hide* 7/7, *excited* 7/7, *cold* 6/7);
- the sign-test $p$-values in §5, which treat the 7 texts as independent.

**Consequence for the full version:** 100 prompts alone will not fix this. The design needs
**several seeds per prompt**, and the unit of analysis should be the (prompt, seed) pair.

## 9. The published table, re-scored

Counts out of 7, baseline versus the best positive $r$:

| concept | old metric $r=0$ → best | blind judge on-concept $r=0$ → best | on-concept ∧ coherent $r=0$ → best |
|---|---|---|---|
| food | 5 → 7 ($+0.25$) | 2 → 3 ($+0.25$) | 2 → **0** |
| valence* | 5 → 7 ($+0.25$) | 4 → 7 ($+0.25$) | 4 → **0** |
| env_social | 2 → 7 ($+0.25$) | 0 → 7 ($+0.25$) | 0 → **0** |

## 10. What this changes in the write-up

Your proposed phrasing was "the steering intervention displaces the lexical distribution in
the direction of the concept". After this check, it holds **for env_social only** (blind judge,
positive direction), for valence **only through the lexicon**, and **not for food**. In every
case it rests on an effective sample of about one.

Below are drop-in replacements at the places that carry the claim. The surrounding prose is
yours to rewrite.

**`03_replication.tex:296–310`, Table `tab:steering_success`.** Replace the table and its
caption with:

```latex
\begin{table}[htbp]
\centering
\caption{Steering re-scored by an activation-blind judge: a text classifier trained on
GuppyLM's own assistant outputs. Counts out of 7 prompts, at $r=0$ and at the best positive
$r$ (always $+0.25$). Coherent means perplexity under the unsteered model and distinct-2
within the range of unsteered text. The \texttt{valence} judge failed its validity gate
(CV AUROC 0.78) and is shown for completeness only.}
\label{tab:steering_success}
\begin{tabular}{@{}lccc@{}}
\toprule
\textbf{Concept} & \textbf{probe or lexicon} & \textbf{blind judge} & \textbf{blind judge and coherent} \\
\midrule
food          & 5 $\to$ 7 & 2 $\to$ 3 & 2 $\to$ 0 \\
valence$^{*}$ & 5 $\to$ 7 & 4 $\to$ 7 & 4 $\to$ 0 \\
env\_social   & 2 $\to$ 7 & 0 $\to$ 7 & 0 $\to$ 0 \\
\bottomrule
\end{tabular}
\end{table}
```

**`03_replication.tex:327–328`.**
- *Original:* "The 100\% figure is therefore not independent evidence that steering works, and it should not be read as such."
- *Replacement:* "Re-scored by a classifier that never sees activations (Table~\ref{tab:steering_success}), the shift survives on \verb|env_social|, where the judge agrees with the probe on 83 of 84 generations, and disappears on \verb|food|, where the probe calls 35 generations on-concept that the judge does not. No steered generation, on any concept, is coherent."
- *Why:* this replaces a caveat about a number with the measurement itself (§7, §9).

**`03_replication.tex:259`.**
- *Original:* "squarely on-concept and simply absent from the thirteen-word lexicon."
- *Replacement:* "absent from the thirteen-word lexicon, and nearly absent from the model's own food answers as well, since a classifier trained on them gives \emph{palate} a weight of $+0.09$ and scores these generations at $g \approx 0$."
- *Why:* "on-concept" was our reading of the text, not a measurement (§3).

**`03_replication.tex:536`, limitation bullet.** Keep it, and add after it:

```latex
\item \textbf{The steered samples are not independent.} At $|r| \geq 0.5$ the seven
generations are near-copies of one another (mean pairwise bigram Jaccard 0.71--1.00), since
the steering overrides the prompt and the sampling seed is fixed, so every ``7 of 7'' in
this section is closer to a single observation.
```

**`04_conclusion.tex:3–4` and `0_summary.tex:5`** ("directions are causal when written back",
"transfers as a mechanism"). These still stand, but the support is narrower than the text
implies. If you want the sentence to carry its own evidence:
- *Replacement for the conclusion:* "…and writing them back into the residual stream shifts the output toward the concept's training categories on \verb|env_social|, as judged by a classifier that never sees activations, while on no concept and at no coefficient is the steered text coherent."

## 11. Not done, and what the full version needs

- **Phase-8 baseline methods.** Their generations were never saved (`baselines_steering.json`
  stores only aggregates), so they could not be re-scored. The "broken measurement" remark in
  the write-up stands unchanged.
- **The full version (100 prompts, dev/test split) needs two changes the minimal run exposed:**
  1. **Several seeds per prompt**, so that the samples are not copies (§8).
  2. **A different valence judge.** The dataset cannot supply one, with 40 unique answers per
     class. The options are a hand-labelled set of generations or an external LLM judge.
     Without one of them, valence steering can only be evaluated through the lexicon.
- **Coherence calibration.** It used 100 *new* unsteered generations, which is a small departure
  from "no new generation". It affects only the thresholds, and per §6 the verdict does not
  depend on them.
- **Code.** `rfm_guppy/evaluate.py` and `results/eval6_*` are unchanged.
