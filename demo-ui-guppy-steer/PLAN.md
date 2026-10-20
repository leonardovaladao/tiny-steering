# Demo UI — GuppyLM steering, side by side

*Written 2026-10-20. Planning document only: nothing here has been built or run.*

**Status:** design agreed, no code. Steps 1–5 below are not started. Open questions are in §9.

---

## 0. In one sentence

> A browser page where a user types a prompt and sees two GuppyLM answers next to each other:
> the normal one ($r = 0$) and one steered along a concept direction, with a live slider for the
> steering strength $r$, a block selector and an RFM / DiffMeans toggle.

---

## 1. Decisions taken

| Topic | Decision |
|---|---|
| Audience | Graduate-level students who study steering |
| Message | (i) a concept is a direction, and pushing along it changes behavior; (ii) steering trades concept strength against fluency |
| Main control | Slider over $r$, left column fixed at $r = 0$ |
| Other controls | Block $S$, direction source $m \in \{\text{RFM}, \text{DM}\}$, concept $c$ |
| Concepts | The existing three (`food`, `valence`, `env_social`); more may be added later, so the page must be data-driven |
| Hosting | Artifact (static page, no server, free); private, shared by link |
| Decoding | Greedy by default; "sample" checkbox |
| Time budget | Not a constraint |

---

## 2. Why the demo needs an experiment first

Check III (`revision_20261002/SUMMARY_OF_TESTS.md`, lines 108–153) found **0 coherent steered
generations at any $r \neq 0$**, for every concept. The setup was all 6 blocks, all positions,
$|r| \ge 0.25$, and the lowest steered PPL was ≈ 2559 against a threshold of 2.25. Steered outputs
are also near-copies across prompts (bigram Jaccard 0.85–1.00 at $|r| \ge 0.5$).

Example at $r = +0.25$ (`results/steer_food.md`):
*"whole palate palate palate promise to promise to best excited palate…"*

Message (i) needs a range $0 < |r| < r_{\text{break}}$ where the text stays Guppy-like but tilts
toward the concept:

```
r:  -r_max ······ -r_break ····· 0 ····· +r_break ······ +r_max
    word salad    tilted (−)   normal   tilted (+)    word salad
                 └── message (i) ──┘                └ message (ii) ┘
```

Three choices in the original setup were never varied: $|r| < 0.25$, a single block instead of
all six, and the token positions steered. Step 1 tests the first two.

---

## 3. The steering operation

Fix a concept $c$, a source $m \in \{\text{RFM}, \text{DM}\}$, a block set
$S \subseteq \{0, \dots, 5\}$ and a coefficient $r \in \mathbb{R}$. For each $\ell \in S$ and each
token position $t$, the output of block $\ell$ is replaced by

$$
h_{\ell,t} \;\mapsto\; h_{\ell,t} + r \cdot \operatorname{med}_\ell \cdot v^{(c,m)}_\ell,
\qquad \lVert v^{(c,m)}_\ell \rVert_2 = 1,
$$

where

- $\operatorname{med}_\ell$ is the median of $\lVert a_\ell \rVert_2$ over the cached training
  activations (`steering.measure_activation_norms`);
- $v^{(c,\text{RFM})}_\ell$ is the leading eigenvector of the AGOP $M_T$ at block $\ell$;
- $v^{(c,\text{DM})}_\ell \propto \hat\mu_+ - \hat\mu_-$ at block $\ell$;
- both are oriented so that the positive direction points to the label-1 class.

This is the same operation as `rfm_guppy/steering.py`. The left column is the same model with
$r = 0$.

**Block control:** a selector over $\{0\}, \dots, \{5\}$ and "all". "All" reproduces the Check III
setup exactly.

---

## 4. Architecture: model in the browser

Model (`guppylm/guppylm/config.py`): 6 blocks, $d = 384$, vocabulary 4096, context 128,
≈ 8.7M parameters. A forward pass costs ≈ 17 MFLOP per token, which plain JavaScript handles.

| Weights | Size | Fits the artifact (≤ 15 MB per binary file)? |
|---|---|---|
| fp32 | ≈ 35 MB | no |
| fp16 | ≈ 17 MB | yes, split into 2 files (**default**) |
| int8 | ≈ 9 MB | yes, 1 file (only if it passes the equivalence check) |

Components:

1. **Tokenizer** in JS, ported from `guppylm/docs/tokenizer.json`.
2. **Forward pass** in JS, hand-written: embedding → 6 × (attention + MLP) → LM head, mirroring
   `guppylm/guppylm/model.py`. The steering term is one line inside the block loop.
3. **Concept bundles** loaded as data (§5).
4. **Judge** $g_c$ evaluated in JS (§6).

Alternatives considered:

- **A. Precomputed static page.** Fixed prompts, outputs on a grid of $r$. Kept as the fallback
  if the step 3 equivalence check fails.
- **C. Hugging Face Space (Gradio, free CPU).** Rejected: cold starts after inactivity, and more
  moving parts.

---

## 5. Concept bundles (data-driven, extensible)

One file per concept:

```
concept_<name>.json : {
  name, pos_label, neg_label,
  validated: bool,                          // G1 of Check III
  blocks: [ { v_RFM, v_DM, mu, med } × 6 ],
  zones:  { "<m>|<S>": [r_break_minus, r_break_plus] },   // from step 1
  judge:  { vocab, idf, weights, bias }     // TF-IDF + LogReg from Check III
}
```

Adding a concept means running the existing pipeline plus the export script and dropping the
file in. The UI code does not change.

---

## 6. Monitoring panel: avoid circularity

Reading the projection at the steered block is circular. With $\lVert v \rVert = 1$,

$$
\langle a + \varepsilon v - \mu,\, v \rangle = \langle a - \mu,\, v \rangle + \varepsilon ,
$$

so the score goes up by construction. Check III found the same problem in the old success
metric. The page shows two scores instead:

1. **Downstream projection** $p_t = \langle a_{\ell',t} - \mu_{\ell'},\, v_{\ell'} \rangle$ at
   $\ell' = 5$ (last block), per generated token, drawn as a strip under the text. If
   $5 \in S$, it is labelled "circular".
2. **Activation-blind judge** $g_c = 2P(\text{pos}) - 1 \in [-1, 1]$ on the generated text
   (TF-IDF over 1–2-grams + logistic regression, as in Check III). It never sees activations.

The gap between the two ("the representation moved, but the text is not on-concept / not
coherent") carries message (ii).

---

## 7. UI defaults

- **Slider scale:** $r = \operatorname{sign}(u)\, r_{\max}\, |u|^3$, $u \in [-1, 1]$,
  $r_{\max} = 4$. Most of the slider's travel then covers small $|r|$.
- **Zones** from step 1 are shaded on the slider track for the current $(c, m, S)$.
- **`valence`** carries a badge, "judge not validated (G1 failed: CV AUROC 0.78)", and its
  $g_c$ is greyed out.
- **Decoding:** greedy by default. "Sample" uses $T = 0.8$, top-$k = 40$, and a visible seed
  field.
- **Max new tokens:** 64.
- **Formula panel:** the update rule of §3, with the current numerical $\varepsilon_\ell$ and
  $\operatorname{med}_\ell$.
- **Language:** English.

---

## 8. Steps and gates

Each step stops the chain if its gate fails.

| Step | What | Gate | Est. |
|---|---|---|---|
| 1 | **H-demo / Check V**: exploration grid, then confirmation on fresh prompts (§8.1) | Pre-registered criterion; the result is reported either way | 4–6 h |
| 2 | **Export**: weights (fp16), tokenizer, concept bundles | Files round-trip in Python (reload → identical tensors) | 2 h |
| 3 | **JS forward pass + tokenizer** | On 20 prompts: $\max_i \lvert z^{\text{JS}}_i - z^{\text{PT}}_i \rvert < 10^{-3}$ and identical greedy generations | 5–7 h |
| 4 | **JS steering + judge** | Steered JS outputs equal `steering.py` outputs on 5 sampled $(c, m, S, r)$ cells; JS $g_c$ equals Python $g_c$ to $10^{-6}$ | 2 h |
| 5 | **UI** (§7) | — | 3–4 h |

Total ≈ 16–21 h. Steps 1–2 are useful even if the UI is never built.
**Minimal version:** step 1 plus fallback A (static page with the best regime). ≈ 6–8 h.

### 8.1 Step 1 in detail — H-demo (Check V)

**Question.** Does some single-block or small-$|r|$ intervention produce generations that are
coherent, on-concept and prompt-dependent?

**Model.** GuppyLM, unchanged checkpoint. Vectors as in §3.

**Exploration grid.**
- $c \in \{\text{food}, \text{valence}, \text{env\_social}\}$
- $m \in \{\text{RFM}, \text{DM}\}$
- $S \in \{\{0\}, \dots, \{5\}, \text{all}\}$
- $r \in \{\pm 0.02, \pm 0.05, \pm 0.1, \pm 0.15, \pm 0.25\}$

This gives $3 \times 2 \times 7 \times 10 = 420$ cells on the 7 Check III prompts, with greedy
decoding.

**Metrics (from Check III):**
- coherent: $\mathrm{PPL}(y \mid x) \le q_{95}$ and distinct-2 $\ge q_{05}$ (same unsteered
  reference set);
- on-concept: $g_c > 0$ for $r > 0$ ($g_c < 0$ for $r < 0$);
- prompt-dependent: mean pairwise bigram Jaccard across prompts $< 0.5$.

**Selection.** For each $(c, m)$, pick the cell $(S^\star, r^\star)$ with the most generations
that are coherent ∧ on-concept, breaking ties by smallest $|r|$. That gives 6 selected cells.

**Confirmation (pre-registered).** Re-run only the 6 selected cells on a **new, disjoint** set
of 20 prompts.

- **H-demo is supported for $(c, m)$** if ≥ 10/20 generations are coherent ∧ on-concept **and**
  the mean bigram Jaccard is $< 0.5$.
- **H-demo is refuted for $(c, m)$** otherwise.

The confirmation stage is needed because, among 420 exploration cells, some will pass by chance.
Only confirmed cells support the claim "a coherent regime exists". For `valence`, the result is
reported but not interpreted (G1 failed).

**Output.**
- the zones $[r^-_{\text{break}}, r^+_{\text{break}}]$ for each $(c, m, S)$ (largest $|r|$ that is
  still coherent on ≥ 4/7 prompts);
- a Check V entry (see Q11).

---

## 9. Open questions

- **Q11.** Should Check V become part of the revision, next to Checks I–IV in
  `SUMMARY_OF_TESTS.md`, or stay an internal calibration for the demo?
  *Recommendation: the revision. It extends Check III's negative result and answers the obvious
  committee question, "did you try smaller $r$ or a single layer?".*
- **Q12.** If no $(c, m)$ is confirmed, does the demo still go ahead as a "how steering breaks"
  exhibit (message (ii) only, with message (i) weakened to "the vocabulary drifts toward the
  concept, never fluently")? Or stop and revisit with another intervention (projection /
  ablation instead of addition)?
- **Q13.** Order relative to Check IV (`FOCO_ATUAL.md`, designed but not run). Suggestion: run
  step 1 together with Check IV, since both use the same cached activations and vectors, and do
  steps 2–5 afterwards.
- **Q14 (future).** Should the token positions steered (prompt only / generated only / all) be
  an extra control? Not tested anywhere yet.
