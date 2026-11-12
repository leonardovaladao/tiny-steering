# Edit Pass 3 — Conceptual review for `docs/latex/`

**For the applying agent.** This is pass 3 of 3. Pass 1 (`EDIT_PASS1_EQUATIONS.md`,
equations) and pass 2 (`EDIT_PASS2_LATEX.md`, compilation) are already applied; every
line number below refers to the files **as they stand now**.

## Scope

Pass 3 fixes **claims**: statements that are false about this project's own results,
statements that are stronger than the evidence, explanations the data do not support, and
caveats whose absence makes a correct number read as a stronger finding than it is.

It does **not** change mathematics (pass 1) and does **not** change structure, packages,
floats or characters (pass 2). Where a replacement below introduces new math or a new
`\verb|...|`, it is written in the style pass 2 settled on; nothing here should alter the
page count by more than a page or reopen a compilation issue.

## How every item was verified

Nothing here is a stylistic opinion. Each item was checked against the artifacts that
produced the numbers — `results/*.json`, the executed notebooks, and the code that wrote
them. The anchor is named in each item as **Evidence**. Several items include numbers
that are *not yet anywhere in the document*; each of those was recomputed directly from
`results/` and the exact command is given so it can be re-checked before it is written
into the paper.

Two facts established up front, because five separate items depend on them:

1. **Steering perturbs all six blocks at once, not the best block.**
   `rfm_guppy/steering.py:steering_hooks` registers a hook on *every* entry of
   `model.blocks`, and `experiments/02_steer.py:198` sets
   `eps_per_block = r * block_norms` for all `L`. Monitoring likewise uses all six blocks
   (`R^{18} = 6 × 3`). "Best block" appears in the artifacts only as the argmax of
   validation `|ρ|` in *extraction*, and is reported for description, never used as a
   selection.

2. **The success criterion is a disjunction.** `rfm_guppy/evaluate.py` scores a
   generation as steered when `probe_score > 0.5` **OR** the text hits the concept
   lexicon. Two independent failure modes therefore inflate it, and the document
   currently attributes everything to one of them.

## Priority tiers

- **Tier A (K1–K11)** — the document states something that is **false** about its own
  results, or inverts its own meaning. These must all be applied.
- **Tier B (K12–K26)** — overclaims, unsupported explanations, and imprecise terminology.
  Apply all; each is a small, local rewrite.
- **Tier C (K27–K31)** — missing caveats. These *add* text. Apply K27–K30; K31 is marked
  optional.
- **Tier D** — decisions for Maxwell. **Do not apply. Do not decide.**

---

# TIER A — factual errors about this project's own results

## K1 — "the best block … the one that steering and monitoring actually use" is false

**File:** `tex_files/03_replication.tex`, lines 104–107 (inside the rank-1 paragraph).

Steering adds `ε_ℓ v_ℓ` at **all six blocks simultaneously**, each with that block's own
vector; monitoring's aggregate probe reads **all six blocks × 3 eigenvectors**. Nothing in
the pipeline consumes only the best block. As written, a reader concludes the whole result
rests on one layer, which would also make the six-row Table `tab:rfm-val-pearson`
pointless.

**Evidence:** `rfm_guppy/steering.py:steering_hooks` (`for blk_idx, block in
enumerate(model.blocks)`); `experiments/02_steer.py:198`; `experiments/03_monitor.py` /
`rfm_guppy/monitoring.py` aggregate strategy; notebook `03_monitor.ipynb` prints
`aggregate (18 features)` per concept.

**FIND**
```latex
broad --- \verb|valence| block 5 sits at $\lambda_1 / \sum_j \lambda_j = 0.147$. At the
best block of each concept, the one that steering and monitoring actually use,
$\lambda_1 / \sum_j \lambda_j$ is $0.886$ (\verb|food|), $1.000$ (\verb|valence|) and
$1.000$ (\verb|env_social|): there the concept really is a single direction.
```

**REPLACE**
```latex
broad --- \verb|valence| block 5 sits at $\lambda_1 / \sum_j \lambda_j = 0.147$. At the
block where validation $|\rho|$ is highest for each concept,
$\lambda_1 / \sum_j \lambda_j$ is $0.886$ (\verb|food|, block 3), $1.000$
(\verb|valence|, block 4) and $1.000$ (\verb|env_social|, block 2): there the concept
really is a single direction. That block is reported for description only --- steering
perturbs all six blocks at once, each with its own $v_\ell$, and the monitoring probe
reads all six, so nothing downstream selects on it.
```

## K2 — `env_social` does **not** collapse at every $r \geq 1$

**File:** `tex_files/03_replication.tex`, line 253 (Main Findings). Handled together with
K19, which is in the same sentence — apply the single combined block below.

The collapse detector flags 7/7 at $r = +1, +2, +4$ and **0/7 at $r = +8$**. The claim
"$r \geq 1$" is contradicted by the largest coefficient in the sweep. Worse, the $r = +8$
text is the *same* degenerate string as $r = +1$ ("if can breathe can if …"); it escapes
the flag only because its most frequent word takes 0.59 of the output and the rule fires
above 0.60. Quoting "$r \geq 1$" is therefore not a harmless rounding — it hides the fact
that the detector, not the model, changed behaviour.

**Evidence:**
```bash
python3 -c "
import json; d=json.load(open('results/steer_env_social.json'))
[print(v['r'], v['n_collapsed'], '/', v['n_total']) for v in d['results_per_r'].values()]"
# +1.0 7/7   +2.0 7/7   +4.0 7/7   +8.0 0/7
```
Max repeated-word share per generation: 0.61 at $r = +4$, 0.59 at $r = +8$; the rule is
`experiments/02_steer.py:99` (`> 0.60`).

## K3 — "the best block are all the first one" is wrong twice over

**File:** `tex_files/03_replication.tex`, line 193.

`blk 1` is the **second** of six blocks (they are 0-indexed, and the document's own
Table `tab:rfm-val-pearson` starts at block 0). And it is not a winner: blocks 1–5 all
score validation AUROC exactly 1.0000, so `blk 1` is simply the first index of a five-way
tie that `np.argmax` returned. Reporting a tie-break as a finding invites the reader to
look for meaning in "the earliest block suffices" that the data do not carry.

**Evidence:** notebook `03_monitor.ipynb` per-block output — `food` blk 0 = 0.9983 and
blks 1–5 = 1.0000; `valence` blk 0 = 0.9125, blks 1–5 = 1.0000; `env_social` blk 0 =
0.9394, blks 1–5 = 1.0000.

**FIND**
```latex
For the three concepts food, valence and env\_social, the Test AUROC returned 1.0, and the best block are all the first one.
```

**REPLACE**
```latex
For all three concepts the test AUROC is $1.000$. The per-block strategy reports block 1
as best for each of them, but that is a tie-break and not a result: on validation, blocks
1--5 all score exactly $1.000$ and only block 0 is weaker ($0.998$ for \verb|food|,
$0.913$ for \verb|valence|, $0.939$ for \verb|env_social|), so block 1 is merely the first
index of a five-way tie.
```

## K4 — the second/third eigenvector AUROC range is misquoted

**File:** `tex_files/03_replication.tex`, line 202.

The stated range `0.60–0.71` silently drops the lowest value. The measured numbers are
$u_2 = 0.679 / 0.617 / 0.599$ and $u_3 = 0.519 / 0.645 / 0.705$ for
food / valence / env\_social. `food`'s third eigenvector at **0.519** on a 120-example
test split is not distinguishable from chance, so "above chance" cannot be asserted of the
whole set.

**Evidence:** notebook `03_monitor.ipynb`, "single-eigenvector test AUROC (best RFM
block)": `food blk3 u1=1.000 u2=0.679 u3=0.519` · `valence blk4 u1=1.000 u2=0.617
u3=0.645` · `env_social blk2 u1=1.000 u2=0.599 u3=0.705`.

**FIND**
```latex
The second and third are much weaker but not empty (0.60–0.71, above chance), which is a small qualification of the strict rank-1 picture: the trailing eigenvectors carry some residual class signal.
```

**REPLACE**
```latex
The second and third eigenvectors are much weaker: $u_2$ scores $0.679$, $0.617$ and
$0.599$ and $u_3$ scores $0.519$, $0.645$ and $0.705$ on \verb|food|, \verb|valence| and
\verb|env_social| respectively. Four of those six are far enough above $0.5$ to say the
trailing directions carry some residual class signal --- a small qualification of the
strict rank-1 picture --- while \verb|food|'s $u_3$ at $0.519$, on a 120-example test
split, is indistinguishable from chance.
```

## K5 — the baselines finding is stated backwards

**File:** `tex_files/03_replication.tex`, line 224. (Already flagged in both hand-offs.)

"except for the PCA" says the separation appears against every method *but* PCA. The
finding is the exact opposite, and the rest of the same paragraph — including "logistic
regression actually edges RFM out on all three concepts" — contradicts the opening
sentence.

**FIND**
```latex
The separation we expected does in fact appear, except for the PCA, and only on hard concepts.
```

**REPLACE**
```latex
The separation the paper predicts does appear --- but only against PCA, and only on the
two harder concepts.
```

## K6 — "does not report uniform success on any concept" inverts the intended meaning

**File:** `tex_files/03_replication.tex`, line 86. Combined with K16 and K18, which are in
the same sentence and the same line — apply the single block below.

As written it says the mechanism fails everywhere, which would contradict the table
directly above it. The intended claim is that it does not succeed *uniformly across*
concepts, which is what makes `valence` informative.

**Combined FIND (K6 + K16 + K18)**
```latex
Block 0 is consistently weaker for every concept, so one block of attention and Multi-Layer Perception is not enough to build the feature, and the stream there still looks largely like token embeddings. Blocks 1-5 are all strongly linear. \verb|food| and \verb|env_social| are topic distinctions, exactly the kind of thing a small model encodes strongly; \verb|valence| is the weak one, which is itself a finding, since the mechanism does not report uniform success on any concept, which is expected from such a measurement.
```

**Combined REPLACE**
```latex
Block 0 is consistently weaker for every concept: one block of attention and MLP is
apparently not enough to build the feature. Blocks 1--5 all give a projection strongly
correlated with the label. \verb|food| and \verb|env_social| are topic distinctions,
exactly the kind of thing a small model encodes bluntly; \verb|valence| is the weak one
throughout, and that is itself a finding --- the mechanism does not report the same
success on every concept it is handed, which is what one wants from a measurement rather
than from a demonstration.
```

*(K16: `|ρ|` is a correlation between the projection and the label, not separability;
separability is established later, by the AUROC of §Monitoring, and should not be claimed
here. K18: "the stream there still looks largely like token embeddings" is an
interpretation nothing in this work measured — no comparison against the embedding matrix
was run — so it is dropped rather than hedged.)*

## K7 — the explanation of the broken steering comparison is wrong as generalised

**File:** `tex_files/03_replication.tex`, line 243.

The document says the baselines are credited with success "only because the probe is built
from the RFM eigenbasis, so the sign convention lines up for RFM alone." That is not what
the data show. On `valence` and `env_social`, PCA's probe score under negative steering is
$0.10$–$0.13$ and $0.000$ — the probe reads the sign correctly for PCA there. The
inflation has **two** distinct causes, and which one fires depends on the concept and the
method:

1. **Probe sign-blindness, concept-specific.** On `food`, PCA scores $1.000$ at
   $r = -2$, DiffMeans $0.841$, LogReg $0.889$; on `valence`, DiffMeans scores $1.000$ at
   every negative $r$.
2. **The `OR lexicon` half of the criterion.** LogReg on `valence` scores probe $0.000$ at
   $r = -2, -1, -0.5$ and still records 100% success, purely on lexicon hits; the same
   happens for LogReg and DiffMeans on `env_social` at $r = -1$ and $-0.5$.

**Evidence:**
```bash
python3 -c "
import json; d=json.load(open('results/baselines_steering.json'))
for c in d:
  for m in d[c]:
    print(c,m,[(k,round(v['mean_probe_score'],3),round(v['lexicon_hit_rate'],2),
      v['steering_success_rate']) for k,v in sorted(d[c][m]['results_per_r'].items(),
      key=lambda x: float(x[0])) if float(k)<0])"
```

**FIND**
```latex
and those methods are credited with ``success'' for steering in the wrong direction only because the probe is built from the RFM eigenbasis, so the sign convention lines up for RFM alone.
```

**REPLACE**
```latex
and those methods are credited with ``success'' for steering in the wrong direction. Two
separate defects produce that, and both belong to the metric rather than to the methods.
The first is that the probe is built from the RFM eigenbasis and is scored on the
\emph{regenerated} text, so it need not respond antisymmetrically to a perturbation along
someone else's direction: on \verb|food| it saturates at both signs for PCA and
DiffMeans, while on \verb|env_social| it reads every baseline's negative sweep correctly
at $0.000$. The second is the disjunction in the criterion itself --- success is
``probe $>0.5$ \emph{or} lexicon hit'' --- so LogReg on \verb|valence| records 100\%
success at $r = -2, -1, -0.5$ on lexicon hits alone, with a probe score of exactly
$0.000$. Only the RFM row is measured by a stick that is aligned with it, at $0.454$ for
$r = -2$ and $0.436$ for $r = -1$ on \verb|food|.
```

## K8 — the two-poled-lexicon claim is both too narrow and too strong

**File:** `tex_files/03_replication.tex`, line 160 (the `valence` bullet).

Two corrections, and the second improves the paper's argument rather than weakening it.

*Too narrow:* `env_social`'s lexicon is **also** two-poled. It is
`water, filter, temperature, warm, cold, oxygen, clean, cloudy, algae, hello, hi, friend,
bye` — the last four are the *negative* categories (greeting, bye, friends). So the
property attributed to `valence` alone holds for two of the three concepts, and it is what
lets the baselines score `env_social` lexicon hits under negative steering (K7).

*Too strong:* the valence lexicon is **not** sign-blind at the word level. Negative $r$
fires `hide` in 7/7 generations; positive $r$ fires `excited` in 7/7. The pole information
is intact; it is the scalar hit-rate that throws it away by counting both as a hit. This
matters for the document's argument: it is a **non-probe, non-circular** confirmation that
the steering is bidirectional, which is exactly what §Steering evaluation says it lacks.

**Evidence:**
```bash
python3 -c "
import json,re
lex=['excited','happy','joy','wiggle','fast','scared','afraid','fear','nervous','tense','stiff','hide']
d=json.load(open('results/steer_valence.json'))
for k,v in sorted(d['results_per_r'].items(), key=lambda x: float(x[0])):
  h={w:sum(bool(re.search(r'\b'+w+r'\b',o['text'].lower())) for o in v['outputs']) for w in lex}
  print(v['r'], {w:c for w,c in h.items() if c})"
# r<=-0.5 -> {'hide': 7} ; r>=+0.25 -> {'excited': 7}
```
Lexicon definitions: `rfm_guppy/concepts.py:CONCEPT_LEXICONS`.

**FIND**
```latex
  \item valence reports 100\% at $r = \pm 0.5$ and beyond — in both directions. That is not symmetric steering; it is the metric failing. The valence lexicon deliberately contains both poles (happy, excited, joy alongside scared, fear, tense, stiff), so any push along the valence axis, either sign, scores a hit.
```

**REPLACE**
```latex
  \item \verb|valence| reports 100\% at $r = \pm 0.5$ and beyond --- in both directions.
  That is not symmetric steering; it is the scalar metric discarding a sign it in fact
  recorded. The \verb|valence| lexicon spans both poles (\emph{happy, excited, joy}
  alongside \emph{scared, fear, tense, stiff, hide}), so a push either way scores a hit
  --- but the words themselves separate cleanly: every negative $r \leq -0.5$ fires
  \emph{hide} in 7 generations out of 7, and every positive $r \geq +0.25$ fires
  \emph{excited} in 7 out of 7, with no crossover. The hit \emph{rate} is sign-blind; the
  hits are not. That is worth stating plainly, because it is bidirectional evidence that
  does not come from the probe, and so is not subject to the circularity discussed below.
  The same two-poled construction applies to \verb|env_social|, whose lexicon also
  contains its negative categories (\emph{hello, hi, friend, bye}).
```

## K9 — the `env_social` bullet rests on one keyword, and its baseline is a negative-pole word

**File:** `tex_files/03_replication.tex`, line 158. Apply together with K2's Main-Findings
fix; they are the same underlying measurement.

Both endpoints of "14\% $\to$ 86\%" need a qualifier the document does not give:

- the 14\% baseline is one generation containing **`hello`** — a word from the concept's
  *negative* categories, not evidence of the positive concept at all;
- the 86\% is a single keyword, **`cold`**, appearing in 6 of 7 generations. No other
  positive-pole word fires anywhere in the sweep.

The bullet's conclusion ("exactly the described shape") survives, but it is one token wide
and the paper should say so.

**Evidence:**
```bash
python3 -c "
import json,re
lex=['water','filter','temperature','warm','cold','oxygen','clean','cloudy','algae','hello','hi','friend','bye']
d=json.load(open('results/steer_env_social.json'))
for k,v in sorted(d['results_per_r'].items(), key=lambda x: float(x[0])):
  h={w:sum(bool(re.search(r'\b'+w+r'\b',o['text'].lower())) for o in v['outputs']) for w in lex}
  print(v['r'], {w:c for w,c in h.items() if c})"
# r=0 -> {'hello': 1} ; r=+0.25 -> {'cold': 6} ; everything else empty
```

**FIND**
```latex
  \item env\_social behaves as the paper would predict. Baseline 14\% $\to$ 86\% at r = +0.25, then outright collapse (7/7 degenerate at r = +1, +2, +4). A narrow usable band with a hard ceiling — exactly the described shape.
```

**REPLACE**
```latex
  \item \verb|env_social| behaves as the paper would predict: baseline 14\% $\to$ 86\% at
  $r = +0.25$, then collapse (7 of 7 degenerate at $r = +1, +2, +4$). A narrow usable band
  with a hard ceiling, exactly the described shape --- but a thin one. The 14\% baseline
  is a single generation containing \emph{hello}, a word from the concept's \emph{negative}
  categories, and the 86\% is a single keyword, \emph{cold}, in 6 generations out of 7. No
  other positive-pole word fires anywhere in the sweep.
```

## K10 — "the lexicon is measuring fluency loss, not concept absence" is not what happened on `food`

**File:** `tex_files/03_replication.tex`, line 159.

The dominant word in the $r = +0.25$ `food` generations is **`palate`** (56 occurrences
across 7 outputs; 119 at $r = +0.5$) — a food word that is simply not in the 13-word
lexicon (`eat, food, hungry, flakes, bite, nibble, pellet, feed, feeding, meal, snack,
bloodworm, brine`). So the 57\% $\to$ 29\% drop is chiefly the lexicon failing to contain
the vocabulary the steering actually produced, and only secondarily fluency loss. This
also makes the `food` case the cleanest demonstration in the paper that a fixed keyword
list cannot judge steering.

**Evidence:**
```bash
python3 -c "
import json,collections; d=json.load(open('results/steer_food.json'))
v=d['results_per_r']['0.25']
print(collections.Counter(' '.join(o['text'] for o in v['outputs']).lower().split()).most_common(5))"
# [('palate', 56), ('promise', 34), ('to', 17), ('best', 9), ('thank', 8)]
```
Also corroborated by `results/REPORT.md` §2c: "`palate` is not in the keyword lexicon, so
lexicon-based hit rate underestimates true concept activation."

**FIND**
```latex
Steering cannot add a concept that is already saturated; it only breaks the sentences that carried it, so the hit rate goes down: 57\% $\to$ 29\% at r = +0.25, then 0\% everywhere else. The lexicon is measuring fluency loss, not concept absence.
```

**REPLACE**
```latex
Steering cannot add a concept that is already saturated; it can only break the sentences
that carried it, so the hit rate goes \emph{down}: 57\% $\to$ 29\% at $r = +0.25$, then
0\% everywhere else. And the drop is mostly not even fluency: the word that dominates the
$r = +0.25$ generations is \emph{palate}, 56 occurrences across the seven outputs, which
is squarely on-concept and simply absent from the thirteen-word lexicon. What this curve
measures is the keyword list missing the vocabulary the steering produced.
```

## K11 — "the AGOP re-weighting is not what finds the direction" is not what the gate shows

**File:** `tex_files/03_replication.tex`, line 62.

In the implementation, `T` counts AGOP steps taken **from** $M_0 = I$: `T = 1` means the
predictor was fitted with the plain Laplace kernel and then **one full AGOP** was formed
from its gradients (`rfm.py:extract_concept_vector`, `for T in range(1, T_max+1): M, _, _
= _agop_step(...)`). So the direction at `T = 1` *is* an AGOP — what the flat curve shows
is that the **recursion** ($T \geq 2$) adds nothing here, not that the AGOP is idle. As
written, the sentence says the paper's central object did no work, which is stronger than
the experiment and, read literally, false.

The paragraph also drops the numbers that make the point, and the conclusion the sweep
actually licenses (that what it buys is the *bandwidth*).

**Evidence:** notebook `01_extract.ipynb` — `L=1: |cos| after T=1 → 0.809, after T=10 →
0.809` · `L=10: 0.860 → 0.856` · `L=100: 0.945 → …`, best HP `L=100, T=2, normalize=False`
at $|\cos| = 0.9528$.

**FIND**
```latex
We find then that the AGOP re-weighting is not what finds the direction. The first iterations starts from $M=I$ and since $y$ is exactly linear in $(z, u)$ a plain kernel at the right bandwidth already solves it.
```

**REPLACE**
```latex
What is flat here is the \emph{recursion}, not the AGOP. Since $T$ counts steps taken from
$M_0 = I$, the vector at $T = 1$ is already a full AGOP --- one formed from the gradients
of a predictor fitted with the ordinary Laplace kernel --- and because $y$ is exactly
linear in $\langle z, u \rangle$, that single step is enough: $\sigma = 100$ reaches
$|\cos(v,u)| = 0.945$ at $T = 1$ and the following nine iterations change nothing, while
$\sigma = 1$ sits at $0.809$ and $\sigma = 10$ at $0.860$ and neither climbs out. What the
sweep buys on this gate is therefore the \emph{bandwidth}, not the iteration count, which
is exactly why $\sigma$ is swept here rather than copied from the original paper.
```

---

# TIER B — overclaims, unsupported explanations, imprecise terminology

## K12 — RFM's advantage over a linear separator is asserted, then refuted by our own table

**File:** `tex_files/1_fundamentals.tex`, line 59.

The document's Table `tab:baselines_val_rho` has LogReg above RFM on all three concepts.
The paper's claim is conditional (hard, abstract concepts; large models) and must be stated
as one here, with the forward reference, or the reader meets a flat contradiction 150 lines
later. Also "in opposition to" $\to$ "in preference to", and "test it's" $\to$ "test its".

**FIND**
```latex
The RFM is chosen in opposition to a simple logistic regression because the AGOP can capture the directions that a nonlinear predictor is sensitive to, so the RFM can surface concept structure that a linear separator cannot. The original paper claims that this is the case for abstract concepts and large language models, so we bring it here to test it's applicability to a tiny model.
```

**REPLACE**
```latex
RFM is preferred to a plain logistic regression because the AGOP reports the directions a
\emph{nonlinear} predictor is sensitive to, so it can in principle surface concept
structure that a single linear separator misses. The original paper advances that claim
under two conditions --- abstract concepts, and large models --- and it is worth keeping
the conditions attached, because they decide the outcome of §\ref{sec:baselines-anchor}
below: on the three easy, in-domain concepts available here, logistic regression matches
and slightly exceeds RFM.
```

> The `\ref` needs a target. Add `\label{sec:baselines-anchor}` on the line immediately
> after `\subsection{Baselines}` in `tex_files/03_replication.tex` (line 204). If a label
> there collides with anything, drop the `\ref{...}` and write "the baseline comparison
> below" instead — do **not** leave an undefined reference and do **not** hard-code a
> section number; pass 2's end state requires zero undefined references.

## K13 — "the only concepts GuppyLM can represent" is stronger than anything tested

**File:** `tex_files/2_guppy.tex`, line 20.

Nothing here tested whether the model represents anything outside its 60 categories; the
README says "plausibly represent" for that reason. As stated it is an unfalsifiable claim
about a model's whole representational content, drawn from its training data alone.

**FIND**
```latex
Those categories are the only concepts GuppyLM can represent due to its training. We restrain to these inherent concepts on this work. 
```

**REPLACE**
```latex
Those categories are the only concepts GuppyLM can plausibly be expected to represent,
given that they are the whole of its training distribution; we restrict this work to them.
Whether the model also carries structure outside them is not something we tested, and not
something a probe restricted to these labels could see.
```

## K14 — the Summary claims a scaling conclusion a single checkpoint cannot support

**File:** `tex_files/0_summary.tex`, line 1 (final sentence).

One model at one size cannot establish how anything scales; the report's own limitations
say so, and the ladder is a *prerequisite* for asking the scaling question, not a
conclusion this work can reach. Fix `Largue` $\to$ `Large` and `Does these` $\to$ `Do
these` in the same block.

**FIND**
```latex
the concepts "known" by Largue Language Models
```
**REPLACE**
```latex
the concepts ``known'' by Large Language Models
```

**FIND**
```latex
One may question if these results are replicable at much smaller scales. Does these findings endure at a model as small as $\approx 9$ million parameters? The findings on this question could answer how this machinery scales and whether scaling it would require a ladder of model sizes.
```

**REPLACE**
```latex
One may question whether these results are replicable at much smaller scales. Do these
findings endure on a model as small as $\approx 9$ million parameters? A single
checkpoint cannot say how the machinery scales --- that needs a ladder of model sizes,
which is a separate experiment --- but it can say whether the mechanism survives at the
bottom of the range on its own merits, and a negative answer there would bound the whole
claim.
```

## K15 — "the small context the model have" is an explanation the evidence does not support

**File:** `tex_files/03_replication.tex`, line 146.

Context length was never varied, so it cannot be the attributed cause. Two supported
explanations are available, and the second is specific to this setup and worth stating:
the concept vectors are fitted on the last-token activation of a **complete** ChatML
exchange (`rfm_guppy/concepts.py:_CHATML_TEMPLATE`, user turn + assistant turn), whereas
steering applies $\varepsilon v_\ell$ at every position of a **partial** assistant turn
being generated. The direction is being written into a state distribution it was never
fitted on. The paragraph also garbles the Medium source's "tank-and-water vocabulary" into
"the simple vocabulary".

**FIND**
```latex
Even though the intervention produces results, they are blunt. A direction fitted to pairs of (prompt, label) that never saw a single generation drags the output onto the simple vocabulary from a prompt that mentions no idea of it. Even though they are blunt results, this is a repetitive and barely grammatical vocabulary drift, which makes sense for the small context the model have.
```

**REPLACE**
```latex
The intervention unmistakably works, and it is unmistakably blunt. A direction fitted to
$(\text{prompt}, \text{label})$ pairs, which never saw a single generation, drags the
output onto tank-and-water vocabulary from a probe prompt that mentions no such thing:
the causal claim replicates. But read the text rather than the keywords and it is
repetitive, barely grammatical vocabulary drift, not a fluent fish talking about its tank.
Two things plausibly contribute, and we can separate neither with one model. The
perturbation may simply be larger than an 8.7M-parameter residual stream has slack to
absorb; and the vectors are fitted on the last-token activation of a \emph{complete}
ChatML exchange while steering writes them into every position of a \emph{partial} one, so
the direction is being applied to a state distribution it was never estimated on.
```

## K16, K18 — see K6 (same line, combined there).

## K17 — rank-1-ness is attributed to $T$, but $T$ and $\sigma$ are confounded in the selected settings

**File:** `tex_files/03_replication.tex`, lines 251 (Main Findings) and, more weakly, 100.

Every block whose spectrum stayed broad was selected at $(\sigma = 100, T = 1)$ —
`valence` blocks 2 and 5, `env_social` block 1 — and every collapsed one at
$\sigma \in \{1, 10\}$ with larger $T$. Bandwidth and iteration count therefore move
together across the selected settings, and the sweep as run cannot attribute the collapse
to $T$ alone. The *observation* (broad spectrum, good direction) is untouched; only the
causal sentence needs to go.

**Evidence:** notebook `01_extract.ipynb` per-block HP dump —
`valence blk2 λ1/Σλ=0.186 best=(L=100, T=1)`, `valence blk5 0.147 (L=100, T=1)`,
`env_social blk1 0.222 (L=100, T=1)`, against `valence blk4 1.000 (L=10, T=10)`,
`env_social blk2 1.000 (L=10, T=9)`, `food blk2 1.000 (L=1, T=8)`.

**FIND**
```latex
But that number depends on which T the validation sweep happened to select; blocks stopped at T = 1 still yield a good direction from a spectrum that is nowhere near collapsed. Rank-1-ness turns out to be a consequence of enough AGOP iterations, not a precondition for the top eigenvector being useful.
```

**REPLACE**
```latex
But that number depends on the hyperparameters the validation sweep happened to select:
every block whose spectrum stayed broad was selected at $(\sigma = 100, T = 1)$ and every
collapsed one at $\sigma \in \{1, 10\}$ with a larger $T$, so bandwidth and iteration
count move together and this sweep cannot attribute the collapse to either alone. What the
data do show is the direction that matters: a block stopped at $T = 1$, with a spectrum
nowhere near collapsed, still yields a good concept vector. Rank-1-ness is not a
precondition for the top eigenvector being useful. Isolating what causes it would need a
sweep over $T$ at fixed $\sigma$, which we did not run.
```

## K19 — "antisymmetric in the activations" states a construction, not a finding

**File:** `tex_files/03_replication.tex`, line 253. **Apply this combined block, which also
carries K2.**

$+\varepsilon v$ and $-\varepsilon v$ are antisymmetric by definition; nothing was
measured. The finding is that the *downstream effect* — the probe score on the regenerated
text, and now (K8) the lexicon words themselves — tracks the sign.

**Combined FIND (K19 + K2)**
```latex
Yes, unambiguously — output vocabulary drags hard toward the concept, $+\varepsilon$/$-\varepsilon$ are antisymmetric in the activations (the probe score tracks the sign cleanly, even where the lexicon can't see it because its word list spans both poles), and the paper's small-coefficient-works, large-coefficient-collapses pattern shows up exactly as described (\verb|env_social| collapses 7 for 7 at $r \geq 1$).
```

**Combined REPLACE**
```latex
Yes, unambiguously --- output vocabulary drags hard toward the concept, and the effect
carries the sign of the perturbation rather than merely its size: the probe score tracks
$\operatorname{sign}(r)$ on all three concepts, and on \verb|valence| the generated words
do too (\emph{hide} under every negative $r$, \emph{excited} under every positive one),
which the aggregate hit rate hides because the word list spans both poles. The paper's
small-coefficient-works, large-coefficient-collapses pattern also shows up
(\verb|env_social| is flagged degenerate 7 for 7 at $r = +1, +2, +4$), with the caveat
recorded above that the flag itself stops firing at $r = +8$ on text that has not
recovered.
```

## K20 — the baseline comparison is not run through an *identical* pipeline

**File:** `tex_files/03_replication.tex`, line 206.

Splits, probe and threshold are identical; the steering sweep is not. The baselines are
swept over $r \in \{0, \pm 0.25, \pm 0.5, \pm 1, \pm 2\}$ (9 values,
`experiments/04_baselines.py:68`) while §Steering swept 12 values out to $-4$ and $+8$
(`experiments/02_steer.py:59`). It changes nothing in the conclusions — both best positive
coefficients land at $r = +0.25$ — but "identical" is checkable and currently false.

**FIND**
```latex
We ran PCA, difference-in-means, logistic regression and RFM through an identical pipeline.
```

**REPLACE**
```latex
We ran PCA, difference-in-means, logistic regression and RFM through the same pipeline:
the same splits, the same probe, the same success threshold, and a steering sweep over
$r \in \{0, \pm 0.25, \pm 0.5, \pm 1, \pm 2\}$ --- narrower at the extremes than the
twelve-point sweep used above, though every method's best positive coefficient falls at
$r = +0.25$, well inside it.
```

## K21 — extraction, steering and monitoring are conflated in one sentence

**File:** `tex_files/1_fundamentals.tex`, line 1 (final sentence). The property belongs to
**extraction**: fitting the concept vector needs no generated text. Steering is what you do
afterwards, and "to detect the concepts" describes monitoring. Also fixes `run run`.

**FIND**
```latex
then run run each prompt through the model
```
**REPLACE**
```latex
then run each prompt through the model
```

**FIND**
```latex
Note that steering process does not need text outputs, only pairs (prompt, label) to detect the concepts, meaning the data collection is cheap (the original paper steers a concept in under a minute with fewer than $500$ samples).
```

**REPLACE**
```latex
Note that extracting a concept vector requires no generated text at all --- only
$(\text{prompt}, \text{label})$ pairs and one forward pass each --- which is what makes
the data collection cheap, and what the steering and monitoring stages downstream then
build on (the original paper extracts and steers a concept in under a minute with fewer
than $500$ samples).
```

## K22 — "RFM is a supervised nonlinear generalization of PCA" needs its hedge back

**File:** `tex_files/1_fundamentals.tex`, line 5.

PCA is not a special case of the RFM iteration, so this is an analogy, not a
generalisation. The source text (`docs/MEDIUM_POST.md` §1) carries "informally"; the
LaTeX dropped it, which turns a useful intuition into a false mathematical statement.

**FIND**
```latex
As of \cite{radhakrishnan2024mechanism}, the Recursive Feature Machine (RFM) is a supervised nonlinear generalization of Principal Component Analysis (PCA). It learns which directions in input space the predictive function depends on using Average Gradient Outer Product (AGOP).
```

**REPLACE**
```latex
The Recursive Feature Machine (RFM) of \cite{radhakrishnan2024mechanism} can be read,
informally, as a supervised and nonlinear counterpart to Principal Component Analysis
(PCA): where PCA returns the directions of greatest \emph{variance} in the data, RFM
returns the directions to which a fitted \emph{predictor} is most sensitive, via the
Average Gradient Outer Product (AGOP). It is not a generalisation of PCA in the strict
sense --- PCA is not a special case of the iteration below --- but the analogy is the
right one for what the matrix $M$ is doing.
```

## K23 — GuppyLM is not "a building block for language models"

**File:** `tex_files/2_guppy.tex`, line 1. It is a teaching implementation of a complete
small model, not a component other models are built from.

**FIND**
```latex
It is created with educational purposes as a building block for language models.
```
**REPLACE**
```latex
It was built for teaching --- a complete, readable reference implementation of a small
instruction-tuned transformer.
```

## K24 — the activation-norm sentence has the wrong connective and no magnitude

**File:** `tex_files/2_guppy.tex`, line 22. The claim's *direction* is correct and was
checked (block 0 median 11.9, block 5 median 29.7, so a fixed absolute $\varepsilon$ is a
far larger relative perturbation at block 0). But it is a consequence, not an example, and
the growth should be quantified since the whole $\varepsilon$-calibration argument rests on
it. Growth is also not monotone — it peaks at block 4 and dips at block 5.

**Evidence:** `results/steer_env_social.json:block_norms` = 11.85, 22.77, 27.55, 30.72,
30.86, 29.71.

**FIND**
```latex
One convenient consequence of GuppyLM's small scale is that its per-block activation norms are tiny, can be known and grow across depth. For example, a perturbation that is a nudge at block 5 is a wrecking ball at block 0, which is why the original paper's absolute steering coefficients cannot be copied over.
```

**REPLACE**
```latex
One convenient consequence of GuppyLM's small scale is that its per-block activation norms
are small and can simply be measured: the median $\|a_\ell\|_2$ over the cached training
activations runs $11.9,\ 22.8,\ 27.6,\ 30.7,\ 30.9,\ 29.7$ from block 0 to block 5, a
roughly $2.6$-fold rise that peaks at block 4. Consequently a fixed absolute perturbation
is a nudge at block 5 and a wrecking ball at block 0, which is why the original paper's
absolute steering coefficients cannot be copied over.
```

## K25 — terminology: "databases" and "constraints"

**File:** `tex_files/03_replication.tex`, lines 3 and 5. The objects are datasets, and the
concepts are built as category *contrasts* — which is what the caption of
Table `tab:concept_categories` already says. The same line also drops the number that
justifies using full exchanges.

**FIND**
```latex
\subsection{Concept databases}
```
**REPLACE**
```latex
\subsection{Concept datasets}
```

**FIND**
```latex
Because raw GuppyLM inputs are templated, we use as prompts user + model responses, which yields hundreds of diverse texts per category. We then build three binary concepts as category constraints:
```
**REPLACE**
```latex
Because the raw user inputs are heavily templated --- only about 4 to 34 unique strings
per category --- we use the full ChatML exchange (user turn plus GuppyLM's response) as
the prompt, which yields hundreds of distinct texts per category. On those we build three
binary concepts as contrasts between groups of categories:
```

## K26 — "prompt-leakage across splits verified" reads as if leakage was found

**File:** `tex_files/03_replication.tex`, line 23. What was verified is its *absence*, and
it is asserted at build time rather than checked after the fact. Adding the split sizes
here also pays off in K28.

**FIND**
```latex
Each of these concepts are class-balanced and split 60/20/20 between train, validation, and test sets on unique exchange strings, with prompt-leakage across splits verified.
```

**REPLACE**
```latex
Each concept is class-balanced and split 60/20/20 into train, validation and test sets
over \emph{unique} exchange strings, with the absence of prompt leakage across splits
asserted at build time. That gives 360/120/120 examples for \verb|food| and
\verb|env_social| and 120/40/40 for \verb|valence|.
```

---

# TIER C — missing caveats (these add text)

## K27 — the synthetic gate uses real-valued labels, contradicting $y \in \{0,1\}$

**File:** `tex_files/03_replication.tex`, after line 53 (end of the synthetic-gate
paragraph).

§1 defines labels as $y \in \{0,1\}$; the gate uses $y_i = \langle z_i, u \rangle \in
\mathbb{R}$. A careful reader hits that as an inconsistency. It is deliberate, and the
reason is worth one sentence — it is also the honest admission that the gate is easier than
the real task.

**Evidence:** `rfm_guppy/rfm.py:312-327` — `y = Z @ u  # regression: label = projection
onto u`, docstring "Binary labels require n >> k to converge past 0.9; regression labels
converge at n=1000 with k=384"; split `n_tr = int(n*0.8)`, so 800 train / 200 validation.

**Insert after** `criterion is $|\cos(v, u)| > 0.90$; the run achieves $|\cos(v, u)| = 0.9528$.`

```latex
Two details of the gate are deliberate. The 1{,}000 points are split 800/200 into train
and validation, so the reported $|\cos(v,u)|$ is obtained from the hyperparameters that
maximise validation $|\rho|$, exactly as on the real concepts. And the labels are
real-valued here rather than the $y \in \{0,1\}$ of §2: recovering a direction in
$\mathbb{R}^{384}$ from 800 points is a question about the AGOP machinery, and thresholding
$\langle z, u \rangle$ to a sign would discard most of the information about $u$ unless
$n \gg k$. The gate is therefore an easier problem than the concepts that follow, by
construction --- it is a correctness check on the implementation, not a difficulty
benchmark.
```

## K28 — the AUROC ceiling is reported without the reasons it is easy

**File:** `tex_files/03_replication.tex`, after line 202 (end of the Monitoring
subsection).

The document reports 1.000 six times and only concedes "this is a ceiling" much later, in
Main Findings. The three reasons it is a ceiling are all facts already established in the
document, and one of them — the prompts being full exchanges — is not stated anywhere and
genuinely bounds the interpretation: the probe reads the last-token activation of a text
that already contains GuppyLM's own topical answer, so part of what AUROC 1.000 certifies
is topic classification of the text, not the recovery of a latent concept.

**Insert after** the paragraph K4 rewrote — i.e. as a new paragraph following
`\ldots is indistinguishable from chance.`, immediately before `\subsection{Baselines}`.
**Apply K4 first**, or the anchor will not exist.

```latex
A ceiling is not the same as a strong result, and three properties of this setup make it
an easy one. The concepts are the model's \emph{own} training categories rather than
emergent properties; the test splits are small (120, 40 and 120 examples); and the
positive and negative classes are drawn from disjoint, plainly distinct groups of
categories. A fourth is specific to the way the data were built: because a prompt is a
complete exchange containing GuppyLM's own answer, the probe is reading the last-token
state of a text that already states its topic lexically, so some of what it certifies is
topic classification rather than the recovery of a latent concept. A perfect score says
that the linear read-out mechanism is intact at 8.7M parameters. It says nothing about how
the method behaves on a genuinely hard concept, because none was presented.
```

## K29 — "collapsed = 0" is used as evidence of coherence, and it is not

**File:** `tex_files/03_replication.tex`, after the bullet list (insert before line 163,
"The honest result for this steering phase …").

The collapse column appears in the argument three times, and the detector is never
defined. It flags a generation when more than 60\% of its words are the same token
(`experiments/02_steer.py:99`), which is a very low bar. The strongest single illustration
available is the one K2 uncovers.

```latex
One caveat is easy to miss, and it matters for reading every ``collapsed'' figure above.
The detector flags a generation only when a single word takes more than 60\% of the
output. The \verb|food| generations at $r = +0.25$ sit between 0.28 and 0.41 by that
measure and are flagged clean, while reading as
\emph{``whole palate palate palate promise to promise to best\ldots''}; and
\verb|env_social| at $r = +8$ scores 0.59 and is flagged clean while producing the same
degenerate \emph{``if can breathe can if\ldots''} string that scores 0.61 and is flagged
collapsed at $r = +4$. A zero in the collapsed column means the text is not maximally
repetitive. It does not mean the text is coherent.
```

## K30 — the document has no Limitations subsection

**File:** `tex_files/03_replication.tex`, at the very end (after line 255).

Every limitation in `results/REPORT.md` §4 is currently scattered through the body or
absent, and the document ends on the comparative claim. For a research-group audience a
short closing list is the standard place to put what the work cannot support. Also fixes
the missing full stop at the end of line 255, which pass 2 deferred here.

**FIND** (the final characters of the document)
```latex
it was never a fair test to begin with, not a failed one
```
**REPLACE**
```latex
it was never a fair test to begin with, not a failed one.

\subsection{Limitations}

\begin{itemize}
  \item \textbf{In-domain concepts only.} These are GuppyLM's training categories, not
  abstract or safety-relevant properties, and nothing here transfers to that setting.
  \item \textbf{Both automatic judges are unreliable, in known and opposite ways.} The
  probe reads the same AGOP eigenbasis the steering vector comes from, so it is internally
  consistent rather than independently validated, and it cannot see fluency; the lexicon
  is saturated on \verb|food|, spans both poles on \verb|valence| and \verb|env_social|,
  and misses on-concept vocabulary such as \emph{palate}. Measuring steering, not
  performing it, was the hard part at this scale.
  \item \textbf{Small $n$ washes out method differences.} With 100--300 samples per class
  and 40--120 test examples, all supervised linear extractors converge, which is why the
  RFM-versus-baselines question could not be answered here.
  \item \textbf{$\mathrm{AUROC} = 1.000$ is a ceiling.} The concepts are perfectly
  linearly separable at these blocks; a harder task would show variance.
  \item \textbf{No coherent steering was demonstrated}, only causal vocabulary drift.
  \item \textbf{No scaling, training-time or transfer claims.} One 8.73M-parameter
  checkpoint cannot speak to any of them.
\end{itemize}
```

## K31 (OPTIONAL) — extraction space versus steering space

**File:** `tex_files/03_replication.tex`, as a footnote on the $\varepsilon_\ell$ formula
(line 115).

The concept vector is estimated on **mean-centred** activations, and for `food` on
**L2-normalised** ones as well (`rfm.py:_preprocess`; the sweep selected
`normalize=True` for every `food` block), but steering adds it to the **raw** residual
stream. The direction is a unit vector in $\mathbb{R}^{384}$ either way, so the operation
is well defined, and the calibration by median norm is what keeps the magnitude sensible —
but a reader who has followed the preprocessing will want the sentence. Apply only if
Maxwell wants this level of detail; it is a gap in exposition, not an error.

```latex
\footnote{The vectors are estimated on mean-centred activations, and for \texttt{food}
on L2-normalised ones as well, whereas the perturbation is added to the raw residual
stream. Since $v_\ell$ is a unit vector in $\mathbb{R}^{384}$ either way this is well
defined, and calibrating $\varepsilon_\ell$ against the median raw norm is what fixes the
scale.}
```

---

# TIER D — for Maxwell, not for the applying agent

**Do not act on these. They are recorded so they are not lost.**

1. **`results/REPORT.md` disagrees with the notebooks**, and the LaTeX (correctly) follows
   the notebooks. §2a of the report gives the synthetic gate's best HP as `L=1, T=10,
   val|ρ|=0.999`, while `01_extract.ipynb` prints `L=100, T=2, val|ρ|=0.954`; §2b gives
   $\lambda_1/\Sigma\lambda = 1.000$ for `food`'s best block, while the notebook gives
   $0.886$ at block 3. Both look like the report predating a re-run. The paper is right;
   the report should be reconciled separately.
2. **`imgs/07_baselines_val_rho.png`** is still unused — pass 2 raised this and it is still
   open.
3. **Whether to state the paper's monitoring-versus-GPT-4o result** in §2. It is the
   headline of the paper's monitoring section and the document currently omits it, which
   makes the monitoring motivation thinner than the steering one.

---

# Verification protocol

Run from `docs/latex/`.

```bash
# 1. No Tier-A error text survives
grep -n 'steering and monitoring actually use' tex_files/*.tex        # empty
grep -n 'collapses 7 for 7 at \$r \\geq 1\$' tex_files/*.tex          # empty
grep -n 'best block are all the first one' tex_files/*.tex            # empty
grep -n '0.60–0.71' tex_files/*.tex                                   # empty
grep -n 'except for the PCA' tex_files/*.tex                          # empty
grep -n 'uniform success on any concept' tex_files/*.tex              # empty
grep -n 'lines up for RFM alone' tex_files/*.tex                      # empty
grep -n 'measuring fluency loss' tex_files/*.tex                      # empty
grep -n 'AGOP re-weighting is not what finds' tex_files/*.tex         # empty
grep -n 'small context the model have' tex_files/*.tex                # empty
grep -n 'only concepts GuppyLM can represent' tex_files/*.tex         # empty
grep -n 'building block for language models' tex_files/*.tex          # empty
grep -n 'an identical pipeline' tex_files/*.tex                       # empty
grep -n 'Largue\|Does these\|run run\|Concept databases' tex_files/*.tex  # empty

# 2. Tier-C additions landed
grep -c 'Limitations' tex_files/03_replication.tex                    # >= 1
grep -c '60\\%' tex_files/03_replication.tex                          # >= 1

# 3. No new undefined reference (K12 adds a \ref)
grep -n 'sec:baselines-anchor' tex_files/*.tex                        # must appear twice

# 4. Rebuild, four passes, with the hang guard from pass 2
rm -f main.aux main.bbl main.blg main.log main.out main.pdf
for i in 1 2; do
  ( pdflatex -interaction=nonstopmode -file-line-error main.tex >/dev/null 2>&1 & \
    P=$!; for s in $(seq 1 90); do kill -0 $P 2>/dev/null || break; sleep 1; done; \
    kill -9 $P 2>/dev/null && echo "HANG" )
  [ $i = 1 ] && bibtex main
done
pdflatex -interaction=nonstopmode -file-line-error main.tex > final.log 2>&1

grep -E '^(\./|!)' final.log             # empty
grep -c 'undefined' final.log            # 0
grep -o 'Output written.*' final.log     # ~15-16 pages after the Tier-C additions
```

**Expected end state:** 0 errors, 0 undefined citations or references, a PDF one to two
pages longer than pass 2's fourteen (Tier C adds roughly a page of text).

## What this pass must NOT change

- Any equation, symbol convention, or the $\sigma$/$L$ split — pass 1 settled those.
- Any table's numbers. Every figure in every table was checked against `results/` and the
  notebooks during this review and all of them are correct.
- Any float placement, package, or table structure — pass 2 settled those.
- The document's overall thesis. Nothing in Tier A or B reverses a conclusion: the
  corrections make the steering section's argument *stronger* (K8 supplies non-circular
  bidirectional evidence the document currently says it lacks) and leave extraction,
  monitoring and the baseline comparison landing exactly where they already do.
