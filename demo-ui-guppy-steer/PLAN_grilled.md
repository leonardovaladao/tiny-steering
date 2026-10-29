# Demo UI — GuppyLM steering, side by side (grilled version)

*Written 2026-10-20, after a grilling session on `PLAN.md`. Planning document only: nothing here
has been built or run. Where this file and `PLAN.md` disagree, this file wins.*

**Status (2026-10-27):** step 0 done (`test_step0.py`, 17/17). Step 1 done (`calibrate.py` →
`CALIBRATION.md`): H-demo refuted for all 6 (c, m). Step 2 done (`export.py`, 11/11). Steps 3–4
done (`web/engine.js`, `test_engine.mjs` 14/14). Step 5: `web/index.html`, published as a private
Artifact; not yet checked by hand in the browser.

---

## 0. In one sentence

> A browser page where a user types a prompt and sees two GuppyLM answers next to each other:
> the normal one ($r = 0$) and one steered along a direction, with a slider for the steering
> strength $r$, a block selector, a position selector and an RFM / DiffMeans / random toggle.

---

## 1. Decisions taken

| Topic | Decision |
|---|---|
| Purpose | Teaching piece for students and the ACS group. Not a thesis item |
| Audience | Graduate-level students who study steering |
| Message | (i) a concept is a direction, and pushing along it changes behavior; (ii) steering trades concept strength against fluency |
| Scope | The demo only. No ordering against Check IV, no hour cap |
| Calibration (step 1) | Internal to the demo, written in `CALIBRATION.md`. Not added to `SUMMARY_OF_TESTS.md` |
| If step 1 confirms nothing | The demo is still built, with the "how steering breaks" wording (§7) |
| Main control | Slider over $r$, left column fixed at $r = 0$ |
| Other controls | Concept $c$, source $m \in \{\text{RFM}, \text{DM}, \text{random}\}$, blocks $S$, positions $P$ |
| Concepts | The existing three (`food`, `valence`, `env_social`); the page is data-driven |
| Hosting | Artifact (static page, no server); private, shared by link |
| Decoding | Sampled as in Check III (T = 0.7, top-$k$ = 50, visible seed, default 42); greedy as an option |
| Weights | fp32, split into 3 files |
| Language | English |

---

## 2. Why the demo needs a calibration first

Check III (`revision_20261002/SUMMARY_OF_TESTS.md`, lines 108–153) found **0 coherent steered
generations at any $r \neq 0$**, for every concept. The setup was all 6 blocks, all positions,
$|r| \ge 0.25$; the lowest steered PPL was ≈ 2559 against a threshold of 2.25. Steered outputs are
also near-copies across prompts (bigram Jaccard 0.85–1.00 at $|r| \ge 0.5$).

Two facts about that setup matter here.

1. **The intervention acts on the residual stream.** `Block.forward` returns
   `x + attn(norm1(x)) + ffn(norm2(x))` (pre-LN), so the hook's $\varepsilon_\ell v_\ell$ stays
   in the unnormalised residual and is carried to every later block. Steering all 6 blocks at $r$
   leaves roughly $\sum_\ell r\,\mathrm{med}_\ell v_\ell$ in the final residual, up to ≈ 6× the
   dose of one block at the same $r$. The smallest Check III intervention was therefore not small.
2. **The coherence metric penalises topic change.** $\mathrm{PPL}_\theta(y \mid x)$ under the
   unsteered model is high for any $y$ that leaves the prompt's topic, including fluent on-concept
   text. Check III cannot separate "word salad" from "fluent but about food".

Message (i) needs a range $0 < |r| < r_{\text{coh}}$ where the text stays Guppy-like but tilts
toward the concept:

```
r:  -r_max ······ -r_coh ····· 0 ····· +r_coh ······ +r_max
    word salad    tilted (−)  normal  tilted (+)    word salad
                 └── message (i) ──┘              └ message (ii) ┘
```

Step 1 looks for that range over small $|r|$, single blocks and token positions, with the two
problems above fixed.

---

## 3. The steering operation

Fix a concept $c$, a source $m$, a block set $S \subseteq \{0, \dots, 5\}$, a position set $P$
and a coefficient $r \in \mathbb{R}$. Let $T_0$ be the length of the ChatML-wrapped prompt. For
each $\ell \in S$ and each token position $t$, the output of block $\ell$ is replaced by

$$
h_{\ell,t} \;\mapsto\; h_{\ell,t} + \mathbb 1[t \in P]\; r \cdot \operatorname{med}_\ell \cdot v^{(c,m)}_\ell,
\qquad \lVert v^{(c,m)}_\ell \rVert_2 = 1,
$$

where

- $\operatorname{med}_\ell$ is the median of $\lVert a_\ell \rVert_2$ over the train, val and test
  activations of all three concepts (every `*_activations.npy` read by
  `steering.measure_activation_norms`); it does not depend on $c$;
- $v^{(c,\text{RFM})}_\ell$ is the leading eigenvector of the AGOP $M_T$ at block $\ell$;
- $v^{(c,\text{DM})}_\ell \propto \hat\mu_+ - \hat\mu_-$ at block $\ell$;
- RFM and DM vectors are oriented so that the positive direction points to the label-1 class;
- $v^{(\text{random})}_\ell = u_\ell$, drawn once from the uniform distribution on $S^{383}$
  (fixed seed), independent of $c$;
- the position sets partition $\{0, \dots, T_0 + 63\}$:

$$
P_{\text{prompt}} = \{\, t : t < T_0 - 1 \,\}, \qquad
P_{\text{gen}} = \{\, t : t \ge T_0 - 1 \,\}, \qquad
P_{\text{all}} = P_{\text{prompt}} \sqcup P_{\text{gen}}.
$$

$P_{\text{gen}}$ starts at the last prompt token (`assistant\n`), so every one of the generated
tokens is predicted from a steered state.

With $P = P_{\text{all}}$ and $m \in \{\text{RFM}, \text{DM}\}$ this is the operation of
`rfm_guppy/steering.py`. The left column is the same model with $r = 0$, which is bitwise equal to
the unhooked model ($x + 0 = x$ in IEEE arithmetic; Check III's G3 confirmed it on 7/7 texts).

**Effective dose.** Every cell reports

$$
\rho \;=\; \frac{\bigl\lVert \sum_{\ell \in S} r\,\operatorname{med}_\ell\, v^{(c,m)}_\ell \bigr\rVert_2}{\operatorname{med}_5},
$$

so that single-block and all-block settings can be compared on one scale.

**Block control:** a selector over $\{0\}, \dots, \{5\}$ and "all". "All" with $P = P_{\text{all}}$
reproduces the Check III setup.

---

## 4. Architecture: model in the browser

Model (`guppylm/guppylm/config.py`): 6 blocks, $d = 384$, vocabulary 4096, context 128,
≈ 8.7M parameters, ≈ 17 MFLOP per token.

Components:

1. **Weights:** fp32, ≈ 35 MB, split into 3 files of ≤ 15 MB.
2. **Tokenizer** in JS, ported from `guppylm/docs/tokenizer.json` (byte-level BPE, no normalizer).
3. **Forward pass** in JS, hand-written, mirroring `guppylm/guppylm/model.py`, **with a KV cache**.
   Without a cache, 64 new tokens after a 20-token prompt cost
   $\sum_{t=20}^{84} t \cdot 17\,\text{MFLOP} \approx 58$ GFLOP (tens of seconds in JS); with it,
   ≈ 1.4 GFLOP (≈ 1 s). The steering term is one line in the block loop, gated by the mask
   $\mathbb 1[t \in P]$. Under $P_{\text{prompt}}$ the prompt's K/V are computed steered and the
   rest clean; under $P_{\text{gen}}$ the prompt's K/V are clean and steering starts at
   $t = T_0 - 1$.
4. **Concept bundles** loaded as data (§5).
5. **Judge** $g_c$ evaluated in JS (§6). The TF-IDF tokenisation must copy scikit-learn's
   (lowercase, `token_pattern = (?u)\b\w+\b` so single-character tokens are kept, 1–2-grams,
   `sublinear_tf=True` i.e. term weight $1 + \log \mathrm{tf}$, L2-normalised rows), as in the
   Check III judge.

Generation runs **when the slider is released**, not while it is dragged. Tokens are streamed, and
the two columns are generated in parallel.

Fallback: **A. Precomputed static page** (fixed prompts, outputs on the step 1 grid), used if the
step 3 gate fails.

---

## 5. Data files (data-driven, extensible)

```
shared.json : {
  med:   [6],                      // per-block median activation norm
  u:     [6 × 384],                // random unit directions, fixed seed
}

concept_<name>.json : {
  name, pos_label, neg_label,
  validated: bool,                          // G1 of Check III
  blocks: [ { v_RFM, v_DM, mu } × 6 ],
  bands:  { "<m>|<S>|<P>": [r_coh_minus, r_coh_plus] },        // from step 1, every cell
  ticks:  { "<m>|<S>|<P>": { r_star, k_of_20, p_mcnemar } },   // confirmed cells only
  presets: [ ... ],                                            // §7
  judge:  { vocab, idf, weights, bias }     // TF-IDF + LogReg from Check III
}
```

Bands and ticks are read by the export script from the step 1 JSON, so the page and
`CALIBRATION.md` cannot disagree. Adding a concept means running the existing pipeline, step 1 and
the export script, then dropping the file in. The UI code does not change.

---

## 6. Monitoring panel: avoid circularity

Reading the projection at a steered block is circular. With $\lVert v \rVert = 1$,

$$
\langle a + \varepsilon v - \mu,\, v \rangle = \langle a - \mu,\, v \rangle + \varepsilon .
$$

Because the intervention stays in the residual stream (§2), this is also true at any block after
a steered one: steering at block $\ell < 5$ adds $\varepsilon_\ell \langle v_\ell, v_5 \rangle$ to
the projection at block 5, for every $S$. The page therefore shows two scores that do not read the
steered activations:

1. **Clean re-read.** Prompt plus generated text $y$ are fed through the *unsteered* model, and
   $p_t = \langle a_{5,t} - \mu_5,\, v_5 \rangle$ is plotted per generated token as a strip under
   the text. It measures where the text sits in concept space, not where the injected vector went.
   It uses $v_5^{(c,m)}$ for $m \in \{\text{RFM}, \text{DM}\}$, and $v_5^{(c,\text{RFM})}$ when
   $m$ = random, so the strip always measures the concept. *(The last choice was made while
   writing this file; it was not discussed in the grilling.)*
2. **Activation-blind judge** $g_c = 2P(\text{pos}) - 1 \in [-1, 1]$ on the generated text
   (TF-IDF over 1–2-grams + logistic regression, as in Check III).

The gap between the two ("the text reads as on-concept to the model, but the judge disagrees",
or the reverse) and the coherence band carry message (ii).

---

## 7. UI

- **Slider scale:** $r = \operatorname{sign}(u)\, r_{\max}\, |u|^3$, $u \in [-1, 1]$,
  $r_{\max} = 4$.
- **Bands:** for the current $(c, m, S, P)$ the slider track shades
  $[r^-_{\text{coh}}, r^+_{\text{coh}}]$, labelled "exploratory (7 prompts)".
- **Ticks:** if $(c, m, S, P)$ is a confirmed cell, a tick marks $r^\star$, with $k/20$ and the
  McNemar $p$ in a tooltip.
- **Presets:**
  1. *Check III setup*: `food`, RFM, all blocks, $P_{\text{all}}$, $r = +0.25$.
  2. *Best regime*: the confirmed cell, if one exists; otherwise the cell and $r$ with the widest
     coherence band.
  3. *Random control*: the same $(S, P, r)$ as preset 2 with $m$ = random.

  Each preset sets the controls and shows a one-line caption.
- **`valence`** carries a badge, "judge not validated (G1 failed: CV AUROC 0.78)", and its $g_c$ is
  greyed out.
- **Decoding:** sampled by default (T = 0.7, top-$k$ = 50) with a visible seed field (default 42);
  a "greedy" checkbox.
- **Max new tokens:** 64.
- **Prompt input:** at most 56 tokens of user text (context 128 − ChatML ≈ 8 − 64 new), with a live
  counter; Generate is disabled above the cap.
- **Formula panel:** the update rule of §3, with the current numerical $\varepsilon_\ell$,
  $\operatorname{med}_\ell$ and $\rho$.
- **Message wording if step 1 confirms no cell:** *"The representation and the vocabulary drift
  toward the concept, but no tested regime does so fluently and better than a random direction."*

---

## 8. Steps and gates

Each step stops the chain if its gate fails.

| Step | What | Gate | Est. |
|---|---|---|---|
| 0 | **Position-mask wrapper** in `demo-ui-guppy-steer/` (does not touch `rfm_guppy/steering.py`) | With $P = P_{\text{all}}$, text identical character for character to `steering_hooks` on the 7 Check III prompts | 1 h |
| 1 | **Calibration** (§8.1) | Pre-registered criterion; the result is written up either way | 6–8 h |
| 2 | **Export**: fp32 weights (3 files), tokenizer, `shared.json`, concept bundles | Files round-trip in Python (reload → identical tensors) | 2 h |
| 3 | **JS forward pass + tokenizer + KV cache** | On 20 prompts: $\max_i \lvert z^{\text{JS}}_i - z^{\text{PT}}_i \rvert < 10^{-3}$; identical greedy generations; logits with cache equal logits without cache | 6–8 h |
| 4 | **JS steering + mask + judge + clean re-read** | Steered JS greedy outputs equal Python on 5 cells covering all three $P$; JS $g_c$ equals Python $g_c$ to $10^{-6}$ | 3 h |
| 5 | **UI** (§7) | — | 4–5 h |

Total ≈ 22–27 h. If the step 3 gate fails: fallback A, built from the step 1 outputs.

### 8.1 Step 1 in detail — calibration

**Question.** Does some intervention (single block or all, any position set, small $|r|$) produce
generations that are coherent, on-concept, prompt-dependent, and better than a random direction?

**Model.** GuppyLM, unchanged checkpoint. Vectors as in §3.

**Decoding.** Exactly Check III's `GEN_KW`: 60 new tokens, T = 0.7, top-$k$ = 50, seed 42.
`generate_steered` sets the seed inside every call, so for a given prompt all $r$, all $m$ and
$r = 0$ share the same random stream (paired design).

**Prompt sets (frozen in a file before anything runs).**
- Exploration: the 7 Check III prompts.
- Confirmation: 20 prompts drawn with a fixed seed from the GuppyLM dataset categories that no
  concept uses, disjoint from the 7 exploration prompts, the 100 calibration prompts of Check III,
  and the data used to fit the vectors and the judge.

**Metrics.**
- *Coherent:* $\mathrm{PPL}_\theta(y \mid \varnothing) \le q_{95}$ (unconditional: $y$ scored
  after the ChatML template with an empty user turn) **and** distinct-2 $\ge q_{05}$. The
  quantiles are recomputed the same way on the 100 Check III calibration generations. Check III's
  conditional $\mathrm{PPL}_\theta(y \mid x)$ is reported as a secondary metric.
- *On-concept:* $\operatorname{sign}(r)\,[\,g_c(y_r) - g_c(y_0)\,] > 0.2$, where $y_0$ is the
  $r = 0$ generation on the same prompt and seed.
- *Prompt-dependent:* mean pairwise bigram Jaccard across prompts $< 0.5$.

**Exploration grid.**
- $c \in \{\text{food}, \text{valence}, \text{env\_social}\}$
- $m \in \{\text{RFM}, \text{DM}, \text{random}\}$ (random is in the grid only to give it bands;
  it is never selected)
- $S \in \{\{0\}, \dots, \{5\}, \text{all}\}$
- $P \in \{P_{\text{all}}, P_{\text{prompt}}, P_{\text{gen}}\}$
- $r \in \{\pm 0.005, \pm 0.01, \pm 0.02, \pm 0.05, \pm 0.1, \pm 0.15, \pm 0.25\}$

That is $3 \times 3 \times 7 \times 3 \times 14 = 2646$ cells, i.e. 18 522 generations on the 7
prompts. Each cell also records $\rho$. *(The figure of 1 512 cells given during the grilling was
an arithmetic slip; the grid above is the agreed one.)*

**Bands.** For each $(c, m, S, P)$:
$r^+_{\text{coh}} = \max\{\, r > 0 \text{ in the grid} : \text{every grid } r' \in (0, r]
\text{ is coherent on} \ge 4/7 \text{ prompts} \,\}$, and symmetrically for $r^-_{\text{coh}}$.
If no positive $r$ qualifies, $r^+_{\text{coh}} = 0$.

**Selection.** For each $(c, m)$ with $m \in \{\text{RFM}, \text{DM}\}$, pick the cell
$(S^\star, P^\star, r^\star)$ with the most generations that are coherent ∧ on-concept, breaking
ties by smallest $\rho$. That gives 6 selected cells.

**Confirmation (pre-registered).** Re-run the 6 selected cells on the 20 confirmation prompts,
together with the random control: the same $(S^\star, P^\star, r^\star)$ with 5 random directions
(fixed seeds). For prompt $i$, let $X_i = 1$ if the concept-direction generation is coherent ∧
on-concept, and $Y_i = 1$ if at least 3 of the 5 random-direction generations are.

H-demo is **supported for $(c, m)$** if all three hold:
1. $\sum_i X_i \ge 10$;
2. mean bigram Jaccard across the 20 concept-direction generations $< 0.5$;
3. the exact one-sided McNemar test rejects at $\alpha = 0.05/6$ (Bonferroni over the 6 cells):
   with $b = \#\{X_i = 1, Y_i = 0\}$ and $c' = \#\{X_i = 0, Y_i = 1\}$,
   $p = \Pr[\mathrm{Bin}(b + c', 1/2) \ge b]$.

Otherwise H-demo is **refuted for $(c, m)$**. Only supported cells get a tick on the page. For
`valence` the result is reported but not interpreted (G1 failed).

**Output.** `CALIBRATION.md` (frozen prompt lists, metric definitions, grid summary, selected
cells, confirmation statistics) and `calibration.json` (the raw grid, bands and ticks) in
`demo-ui-guppy-steer/`.

---

## 9. Open items

None at the design level. These are minor and can be decided during implementation:

- the exact caption text of the three presets;
- the colour scale of the clean re-read strip.
