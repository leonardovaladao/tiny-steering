# Pass 1 — Equation corrections for `docs/latex/`

**For the agent applying these edits.** This is pass 1 of 3 (1 = equations, 2 = LaTeX
compilation, 3 = concepts). Apply **only** what is listed here. Two sections at the end
(“Hand-off to pass 2” / “Hand-off to pass 3”) exist so those issues are on record — **do
not fix them in this pass.**

Every correction below was checked against the implementation that produced the results
(`rfm_guppy/rfm.py`, `rfm_guppy/baselines.py`, `rfm_guppy/steering.py`) and against the
pre-registered method derivation (`plans/GUPPYLM_GUIDE.md` §1a–1d). The code anchor is
named in each item — if a `FIND` block does not match the file, re-read the file and
re-derive from the anchor rather than guessing.

**Two document-wide decisions already made by the author:**

1. **Bandwidth symbol.** `$L$` currently means both *number of transformer blocks* and
   *Laplace kernel bandwidth*. The bandwidth is renamed to `$\sigma$` throughout; `$L$`
   keeps the block count.
2. **AGOP form.** The equation is changed to the **mean-centred** version, because that
   is what the code computes.

Both are folded into the individual edits below.

---

## A. `tex_files/1_fundamentals.tex`

### E1 — Kernel definition is wrong in two ways (line 9)

As written, the `\exp` closing parenthesis lands before the square root, so the formula
reads $e^{-1/L}\cdot\sqrt{\cdot}$ rather than a kernel; and the quadratic form contains
`(x-x)`, which is identically zero — so the whole expression evaluates to $0$.

**FIND**
```latex
$$K_M(x,z) = \exp(-\frac{1}{L}) \sqrt{ (x-z)^\top M (x-x)}$$
```

**REPLACE**
```latex
\[
  K_M(x,z) \;=\; \exp\!\left(-\frac{1}{\sigma}\sqrt{(x-z)^{\top} M (x-z)}\right)
\]

Here $\sigma>0$ is the kernel bandwidth. We write $\sigma$ where the original paper
writes $L$, to avoid a clash with the number of transformer blocks $L$.
```

*Anchor:* `rfm_guppy/rfm.py:laplace_kernel` — `D = cdist(Z1 @ M_half.T, Z2 @ M_half.T)`
then `np.exp(-D / L)`, with `M_half` from `_mahal_half` so that
`D[i,j] = sqrt((z_i - z_j)^T M (z_i - z_j))`. Also `plans/GUPPYLM_GUIDE.md:41`.

### E2 — Rename the bandwidth to `$\sigma$` (lines 7, 18, 19)

Line 9 is already handled by E1. The three remaining occurrences:

**FIND** (line 7)
```latex
parametrized by a Positive Semidefinite (PSD) matrix $M$ with bandwidth $L$:
```
**REPLACE**
```latex
parametrized by a Positive Semidefinite (PSD) matrix $M$ with bandwidth $\sigma$:
```

Lines 18–19 are rewritten wholesale in **E4** below; make sure the `$\sigma$` rename
lands there too. One further occurrence lives in `03_replication.tex` — see **E11**.

Do **not** touch `$L$` on line 1 (`composed by $L$ blocks`) or on line 31 (the
monitoring feature dimension) — those are the block count and are now unambiguous.

### E3 — KRR step mixes row- and column-vector conventions (lines 11–15)

`$\alpha_t = y(\cdot)^{-1}$` treats $\alpha_t$ and $y$ as row vectors, while every other
vector in the document is a column — and the gradient on line 18 then indexes
$\alpha_i$ as if it were a column. `rfm.py:_agop_step` solves
$(K+\lambda I)\alpha = y$ for a column $\alpha$.

### E4 — AGOP gradient is missing its minus sign, and E5 — AGOP is uncentred

E3, E4 and E5 all live in the same contiguous block, so replace it in one go.

**FIND** (lines 11–19, the whole block from `Initialize` down to the hyperparameter
sentence)
```latex
Initialize $M_0=I$ and for $t=0,...,T-1$:
1. KRR: Fit $\alpha_t=y (K_{M_t} (Z,Z) + \lambda I)^{-1}$, giving the predictor $f_t(z)=\alpha_t K_{M_t}(Z,z)$.
2. AGOP: Recompute $M$ as average outer product of the predictor's input-gradients over the training points with
$$M_{t+1} = \frac{1}{n} \sum_i \nabla_z f_t(a^{(i)}) \nabla_z f_t(a^{(i)})^{\top}$$
after $T$ iterations, the concept vector $v$ is the top eigenvector of $M_T$.  It is oriented by computing the Pearson correlation $\rho$ between projections $\{\langle a^{(i)},v \rangle \}$ and labels $\{y^{(i)}\}$ and taking $sign(\rho) \cdot v$ so that positive projection means "more concept".

The gradient that drives AGOP for $d_i(z) = \sqrt{(z-a^{(i)})^{\top} M (z-a^{(i)})}$ is 
$$\nabla_z f_t (z) = \sum_i \alpha_i \cdot K_M (a^{(i)},z) \cdot (\frac{1}{L}) \cdot (\frac{1}{d_i (z)}) \cdot M(z-a^{(i)})$$
where typical hyperparameters are $\lambda = 10^{-3}, L \in \{1,10,100\}, T \leq 10$.
```

**REPLACE**
```latex
Initialize $M_0 = I$ and, for $t = 0, \dots, T-1$:

\begin{enumerate}
  \item \textbf{Kernel ridge regression.} Fit
  \[
    \alpha_t = \left(K_{M_t}(Z,Z) + \lambda I\right)^{-1} y \;\in\; \mathbb{R}^{n},
    \qquad
    f_t(z) = \alpha_t^{\top} K_{M_t}(Z,z) = \sum_{i=1}^{n} \alpha_{t,i}\, K_{M_t}\!\left(a^{(i)}, z\right).
  \]

  \item \textbf{AGOP.} Recompute $M$ as the average outer product of the predictor's
  input-gradients over the training points:
  \[
    M_{t+1} = \frac{1}{n}\sum_{i=1}^{n}
      \left(g^{(i)} - \bar{g}\right)\left(g^{(i)} - \bar{g}\right)^{\top},
    \qquad
    g^{(i)} = \nabla_z f_t\!\left(a^{(i)}\right),
    \quad \bar{g} = \frac{1}{n}\sum_{j=1}^{n} g^{(j)} .
  \]
\end{enumerate}

After $T$ iterations, the concept vector $v$ is the top eigenvector of $M_T$, taken with
$\|v\|_2 = 1$. It is oriented by computing the Pearson correlation $\rho$ between the
projections $\{\langle a^{(i)}, v\rangle\}$ and the labels $\{y^{(i)}\}$ and replacing
$v$ by $\operatorname{sign}(\rho)\, v$, so that a positive projection means ``more
concept''.

The gradient that drives the AGOP, with
$d_i(z) = \sqrt{(z - a^{(i)})^{\top} M (z - a^{(i)})}$, is
\[
  \nabla_z f_t(z) \;=\; -\frac{1}{\sigma}\sum_{i=1}^{n}
    \alpha_{t,i}\, K_{M_t}\!\left(a^{(i)}, z\right)\,
    \frac{1}{d_i(z)}\; M\left(z - a^{(i)}\right),
\]
with the convention that terms for which $d_i(z) = 0$ are dropped. Typical
hyperparameters are $\lambda = 10^{-3}$, $\sigma \in \{1, 10, 100\}$, $T \leq 10$.
```

Then add this footnote, attached to the word ``AGOP'' in step 2 (i.e. write
`\textbf{AGOP.}\footnote{...}`):

```latex
\footnote{We mean-centre the gradients before forming the outer product, as
pre-registered in \texttt{plans/GUPPYLM\_GUIDE.md} §1a and as implemented in
\texttt{rfm\_guppy/rfm.py}. Without centring, the top eigenvector of $M_{t+1}$ is pulled
toward the mean gradient direction rather than the direction of greatest gradient
variation.}
```

*Anchors:*
- Sign: `rfm.py:_agop_step` builds `W = alpha[None,:] * K / (L * safe_D)` (zeroed where
  `D < 1e-12`) and returns `G = -SZ @ M` — the overall sign is **negative**. The current
  LaTeX has `$(\frac{1}{L})$` with no minus.
- Centring: `rfm.py:_agop_step`, `G_c = G - G.mean(axis=0)` immediately before
  `M_new = (1.0/n) * (G_c.T @ G_c)`.
- KRR direction: `rfm.py:_agop_step`, `alpha = np.linalg.solve(K + lam*np.eye(n), y)`.

> The two KRR forms agree numerically because $K$ is symmetric; the reason to switch is
> that the gradient formula's $\alpha_{t,i}$ only type-checks under the column
> convention.

### E6 — Stacking notation gives the transposed shape (line 1)

A horizontal concatenation of $n$ vectors in $\mathbb{R}^k$ is $k \times n$, not
$n \times k$.

**FIND**
```latex
stack the per-block last-token activations into $Z_l = [a_l^{(1)}, ..., a_l^{(n)}] \in \mathbb{R}^{n \times k}$
```
**REPLACE**
```latex
stack the per-block last-token activations into the matrix $Z_\ell \in \mathbb{R}^{n \times k}$ whose $i$-th \emph{row} is $\left(a_\ell^{(i)}\right)^{\top}$
```

### E7 — Monitoring feature dimension contradicts the document's own numbers (line 31)

The text says $\mathbb{R}^{(L-1)p}$, which for $L=6$, $p=3$ predicts $\mathbb{R}^{15}$.
But `03_replication.tex:142` reports $\mathbb{R}^{18} = 6\times 3$, and
`plans/GUPPYLM_GUIDE.md:62` specifies `h ∈ R^{(L)·p}`. All six blocks are used.

**FIND**
```latex
a vector in $\mathbb{R}^{(L-1)p}$
```
**REPLACE**
```latex
a vector in $\mathbb{R}^{Lp}$
```

> If the original paper genuinely drops one block, that must become an explicit remark
> (``the paper uses $L-1$ blocks; we use all $L=6$''), never a silent mismatch. Leave
> that call to the author — for now the document must at least agree with itself.

### E8 — Steering equation uses literal Unicode inside math (line 26)

`Ã` and `ε` are raw Unicode characters; `·` likewise.

**FIND**
```latex
$$Ã_{l,i}(X) = A_{l,i}(X) + ε·v_l$$
for all token positions $i$, where $\varepsilon$ is the control coefficient.
```
**REPLACE**
```latex
\[
  \tilde{A}_{\ell,i}(X) = A_{\ell,i}(X) + \varepsilon\, v_\ell
  \qquad \text{for every token position } i,
\]
where $\varepsilon$ is the control coefficient and the concept vector is taken with
$\|v_\ell\|_2 = 1$.
```

The unit-norm statement is not cosmetic: without it, the relative coefficient
$\varepsilon_\ell = r \cdot (\text{median activation norm})$ in `03_replication.tex:71`
has no meaning. *Anchors:* `rfm.py:top_eigvectors` returns unit eigenvectors
(`np.linalg.eigh`); `baselines.py:_unit` normalises all three baseline vectors.

### E9 — The baselines are described in words but never defined (lines 35–38)

**FIND**
```latex
RFM is compared against three cheaper extration methods, all oriented by the same Pearson-sign rule:
- PCA: top eigenvector of the (paired) pos-minus-neg difference matrix.
- Difference-in-means: $v = \text{mean}(S_1)-\text{mean}(S_0)$.
- Logistic regression: The normalized LR coefficient vector.
```
**REPLACE**
```latex
RFM is compared against three cheaper extraction methods. All three operate on the
mean-centred training activations, return a unit vector, and are oriented by the same
Pearson-sign rule as RFM.

\begin{itemize}
  \item \textbf{PCA.} Pair the $j$-th positive sample with the $j$-th negative sample,
  $j = 1, \dots, m$ with $m = \min(n_+, n_-)$, and form the difference matrix $D$ with
  rows $D_j = a^{(\pi^+(j))} - a^{(\pi^-(j))}$. Mean-centre it, $\tilde{D} = D - \bar{D}$;
  then $v$ is the top right singular vector of $\tilde{D}$, equivalently the top
  eigenvector of $\tilde{D}^{\top}\tilde{D}$.

  \item \textbf{Difference-in-means.}
  $v = \dfrac{\mu_1 - \mu_0}{\left\|\mu_1 - \mu_0\right\|_2}$, where
  $\mu_c = \operatorname{mean}(S_c)$ is the mean activation of class $c$.

  \item \textbf{Logistic regression.} $v = w / \|w\|_2$, for the fitted coefficient
  vector $w$, with the inverse-regularisation strength $C$ swept on the validation split.
\end{itemize}
```

*Anchors:* `baselines.py:pca_vector` (`D = Z_c[pos] - Z_c[neg]`, `D -= D.mean(axis=0)`,
`np.linalg.svd(D)`, `Vt[0]`), `baselines.py:diffmeans_vector`,
`baselines.py:logreg_vector`, and `baselines.py:_orient` / `_unit` / `_center` for the
three shared properties.

---

## B. `tex_files/03_replication.tex`

### E10 — The vectorised gradient uses the wrong product and undefined symbols (line 23)

Three problems in `$G= -(W_\sum \otimes Z_C - W \cdot Z_C) \cdot M$`:
`\otimes` is the Kronecker/tensor product, but the operation is a **row-scaling** of
$Z_c$ by a vector; `W_\sum` is not a defined symbol; and $W$ itself is never introduced.
(`Z_C` should also be `Z_c`, the mean-centred activations.)

**FIND**
```latex
The naive gradient $[n, n, k]$ is rewritten as $G= -(W_\sum \otimes Z_C - W \cdot Z_C) \cdot M$, and the $T$ sweep runs incrementally, for iteration $T+1$ continues from iteration's $T$ matrix M.
```
**REPLACE**
```latex
Writing the gradient of every training point at once as an $[n, n, k]$ tensor would cost
roughly 400\,MB, so we rewrite it as a pair of matrix products. Define
\[
  W_{ij} = \frac{\alpha_j \, K_M\!\left(a^{(i)}, a^{(j)}\right)}
                {\sigma \, d_j\!\left(a^{(i)}\right)},
  \qquad
  W_{ij} = 0 \ \text{ when } d_j\!\left(a^{(i)}\right) = 0,
\]
so that the stacked gradient matrix $G \in \mathbb{R}^{n \times k}$, whose $i$-th row is
$\nabla_z f_t(a^{(i)})^{\top}$, is
\[
  G = -\left(\operatorname{diag}\!\left(W \mathbf{1}\right) Z_c \;-\; W Z_c\right) M ,
\]
with $\mathbf{1} \in \mathbb{R}^{n}$ the all-ones vector and $Z_c$ the mean-centred
activations. Peak memory is then $O(nk + n^2)$ rather than $O(n^2 k)$. The $T$ sweep also
runs incrementally: iteration $T+1$ continues from iteration $T$'s matrix $M$.
```

*Anchor:* `rfm.py:_agop_step` — `W_sum = W.sum(axis=1)`,
`SZ = W_sum[:,None] * Z_c - (W @ Z_c)`, `G = -SZ @ M`. Note that
`W_sum[:,None] * Z_c` is exactly $\operatorname{diag}(W\mathbf{1})Z_c$. The memory
figures are from the module docstring at `rfm_guppy/rfm.py:7-9`.

### E11 — Malformed set braces in the hyperparameter sweep (line 23)

`${\{1, 10, 100}\}$` has mismatched brace nesting (as does the `$T$` set), and this is
the last occurrence of the bandwidth symbol needing the `$\sigma$` rename.

**FIND**
```latex
and hyperparameter selection by maximum validation using Pearson over $L \in {\{1, 10, 100}\}$, $T \in {\{1, ..., 10}\}$, $\text{normalize} \in \{ \text{False}, \text{True} \}$.
```
**REPLACE**
```latex
and hyperparameter selection by maximising the validation $|\rho|$ --- the absolute
Pearson correlation between $\langle a, v \rangle$ and $y$ --- over
$\sigma \in \{1, 10, 100\}$, $T \in \{1, \dots, 10\}$, and
$\mathrm{normalize} \in \{\mathrm{False}, \mathrm{True}\}$.
```

*Anchor:* `rfm.py:extract_concept_vector` — `L_values = [1,10,100]`, `T_max = 10`,
`normalize = [False, True]`, selection on `best_abs_rho`.

### E12 — The synthetic-gate label is not written as an equation (line 25)

`$y=(z_i, u)$` is a pair, not an inner product; the index is also missing from $y$.

**FIND**
```latex
Before running on real data, we gate on the synthetic recovery test: We generate $n=1000$, $k=384$ standard-normal data with labels $y=(z_i, u)$ for a hidden unit direction $u$ and we confirm whether RFM's top eigenvector aligns with $u$.
```
**REPLACE**
```latex
Before running on real data, we gate on a synthetic recovery test: we generate
$n = 1000$ standard-normal points $z_i \in \mathbb{R}^{k}$ with $k = 384$, and label them
by $y_i = \langle z_i, u \rangle$ for a hidden direction $u$ with $\|u\|_2 = 1$. We then
confirm that RFM's top eigenvector $v$ aligns with $u$. The pre-declared acceptance
criterion is $|\cos(v, u)| > 0.90$; the run achieves $|\cos(v, u)| = 0.9528$.
```

The threshold and the achieved value currently appear only ~170 lines later, in the
Main Findings section; stating them at the gate is what makes it a gate.

### E13 — The steering-coefficient formula does not compile and is undefined (line 71)

`\text{median_norm_l}` puts bare underscores inside `\text{}` (a hard error), and the
quantity is never defined anywhere in the document.

**FIND**
```latex
Instead we choose $\varepsilon_l = r \times \text{median_norm_l}$, and sweep
```
**REPLACE**
```latex
Instead we choose
\[
  \varepsilon_\ell = r \cdot \operatorname*{median}_{i} \left\| a_\ell^{(i)} \right\|_2 ,
\]
the median being taken over all cached training activations, pooled across the three
concepts. Since $\|v_\ell\|_2 = 1$, the relative coefficient $r$ is exactly the size of
the perturbation as a fraction of the typical residual-stream norm at that block --- and
it is $r$, not $\varepsilon$, that ports across models. We sweep
```

*Anchor:* `rfm_guppy/steering.py:measure_activation_norms` — per-sample
`np.linalg.norm(acts, axis=2)` over every cached activation file, concatenated, then
`np.median(combined, axis=0)` per block. Docstring at `steering.py:9-11` confirms
`ε_ℓ = r · median_norm_ℓ`.

### E14 — The stated spectrum ratio is not the ratio actually reported (line 65)

The text says the eigenvalues are divided **by the largest eigenvalue**, but every number
quoted in this section and in Main Findings is $\lambda_1 / \sum_j \lambda_j$. These are
different quantities and the reader cannot reconcile them.

**FIND**
```latex
We also turn to study how rank-1 are the concepts. We divide each eigenvalue by the largest eigenvalue of the matrix $M$, and trace the curve under eigenvalue index. Each block is plotted at whatever iteration $T$ the validation  sweep selected, and the matrix spectrum depends strongly on that $T$. Where it chose many iterations the colapse is virtually total, numerically rank 1. Where it chose $T=1$ the matrix has barely been re-weighted and the spectrum is still broad. At the best block of each concept the concept is virtually a single direction.
```
**REPLACE**
```latex
We also turn to study how rank-1 the concepts are, using two distinct quantities. The
curve in the figure is the normalised spectrum $\lambda_j / \lambda_1$ against the
eigenvalue index $j$; the summary statistic quoted below is
$\lambda_1 / \sum_j \lambda_j$, the fraction of the spectral mass carried by the top
direction. Each block is plotted at whatever iteration $T$ the validation sweep selected
for it, and the spectrum depends strongly on that $T$. Where the sweep chose many
iterations the collapse is virtually total: \verb|env_social| block 2, selected at
$T = 9$, has $\lambda_2 / \lambda_1 \approx 3 \times 10^{-15}$, i.e. numerically rank 1.
Where it chose $T = 1$ the matrix has barely been re-weighted and the spectrum is still
broad --- \verb|valence| block 5 sits at $\lambda_1 / \sum_j \lambda_j = 0.147$. At the
best block of each concept, the one that steering and monitoring actually use,
$\lambda_1 / \sum_j \lambda_j$ is $0.886$ (\verb|food|), $1.000$ (\verb|valence|) and
$1.000$ (\verb|env_social|): there the concept really is a single direction.
```

*Anchor:* all five numbers are verified present in `notebooks/01_extract.ipynb`
(0.147, 0.186, 0.886) and in `docs/MEDIUM_POST.md` §Concept-vector extraction.

> Note: the `\verb|env_social|` form is used here deliberately — it is what the rest of
> this file already does and it sidesteps the bare-underscore problem. Pass 2 will
> normalise this across the document.

### E15 — Rank-1 footnote, minor (line 33)

$M = \lambda_1 v_1 v_1^{\top}$ is correct, but the scalar $\lambda_1$ is only
unambiguous once $v_1$ is declared to be a unit vector.

**FIND**
```latex
which is equivalent to saying M factors as a single outer product $M=\lambda_1 \cdot v_1 v_1^{\top}$
```
**REPLACE**
```latex
which is equivalent to saying $M$ factors as a single outer product $M = \lambda_1 v_1 v_1^{\top}$ with $\|v_1\|_2 = 1$
```

---

## C. `tex_files/2_guppy.tex`, `tex_files/0_summary.tex`, `tex_files/1_fundamentals.tex`

### E16 — `~` is a non-breaking space, not “approximately”

`~8.73M` typesets as “8.73M”: the approximation is silently dropped, so a stated
quantity changes meaning. Fix each occurrence.

| File | Line | FIND | REPLACE |
|---|---|---|---|
| `2_guppy.tex` | 7 | `Parameters &  ~8.73M \\` | `Parameters & $\approx 8.73$M \\` |
| `2_guppy.tex` | 15 | `{~60k rows over 60 categories}` | `{$\approx 60$k rows over 60 categories}` |
| `0_summary.tex` | 1 | `as small as ~9 million parameters` | `as small as $\approx 9$ million parameters` |
| `0_summary.tex` | 3 | `GuppyLM, a ~8.7M parameter language model` | `GuppyLM, an $\approx 8.7$M-parameter language model` |
| `1_fundamentals.tex` | 1 | `in under a minute with ~500 samples` | `in under a minute with fewer than $500$ samples` |

The last one also corrects the sense: `docs/MEDIUM_POST.md` §Setup and the source paper
both say *fewer than* 500 samples, not *approximately* 500.

---

## Verification before handing to pass 2

1. This must return **nothing**:
   ```bash
   grep -n 'x-x\|\\otimes\|W_\\sum\|(\\frac{1}{L})\|Ã\|median_norm\|(L-1)p\|{\\{' docs/latex/tex_files/*.tex
   ```
2. This must return **nothing** (no bandwidth `$L$` survives):
   ```bash
   grep -n 'bandwidth \$L\$\|L \\in \\{1, *10, *100\\}' docs/latex/tex_files/*.tex
   ```
3. Re-read each replaced block against the code anchor named in its item.
4. **Do not expect a clean `pdflatex` run yet.** Several structural table errors remain
   by design; they are pass 2's job. If a compile is attempted, ignore failures
   originating from the items listed immediately below.

---

## Hand-off to pass 2 — LaTeX compilation (DO NOT FIX HERE)

Recorded so the pass-2 agent does not have to rediscover them:

- `1_fundamentals.tex:36–38` — markdown `-` bullets that LaTeX runs together into one
  paragraph. (E9 above happens to fix this one as a side effect of rewriting the block;
  E4 likewise fixes the `1.`/`2.` list on lines 12–13. No others were touched.)
- `2_guppy.tex:3` — `\begin{tabular}{c c c}` declares three columns for two-column rows.
- `03_replication.tex:8` — `\begin{tabular}` with **no** column specification at all, and
  unescaped `env_social`, `temp_hot`, `temp_cold` inside the cells.
- `03_replication.tex:175–186` — `\begin{table}` containing `\toprule` but **no**
  `\begin{tabular}`, closed by a stray `\end{tabular}`. Also uses `\makecell`
  (package `makecell` not loaded) and `booktabs` rules (package not loaded, although
  `\toprule`/`\midrule`/`\bottomrule` are used in three other tables too).
- `03_replication.tex:1` — unbalanced parenthesis at the end of the first sentence.
- Unicode math sitting in prose and needing `$…$`: lines 108–110 (`→`, `±`), 138, 142
  (`R^{18}`, `6 × 384`), 150, 188 (`|ρ|`, bare `env_social`), 194–200 (`λ₁/Σλ`,
  `+ε/−ε`, `r ≥ 1`, `≫`, bare `env_social`).
- `main.tex` — loads `abntex2cite` and calls `\bibliography{references}` with no visible
  `\bibliographystyle`; confirm the keys `beaglehole2026toward` and
  `radhakrishnan2024mechanism` resolve in `references.bib`.

## Hand-off to pass 3 — Conceptual review (DO NOT FIX HERE)

Noticed while checking the mathematics; these are claims, not equations:

- `1_fundamentals.tex:21` — “the RFM can surface concept structure that a linear
  separator cannot” is asserted flatly, but this document's own baseline table
  (`03_replication.tex:164–167`) shows LogReg beating RFM on all three concepts. The
  paper's conditional (hard concepts, large models) needs to be attached.
- `03_replication.tex:57` — “does not report uniform success on **any** concept” inverts
  the intended meaning; should be “on **every** concept”.
- `03_replication.tex:97` — “which makes sense for the small context the model have”
  offers an explanation the evidence does not support (the Medium source attributes the
  degradation to lack of representational slack, not to context length).
- `03_replication.tex:172` — “The separation we expected does in fact appear, except for
  the PCA” inverts the finding: the separation appears **only** against PCA.
- `2_guppy.tex:18` — “the only concepts GuppyLM can represent” is stronger than the
  README's “can *plausibly* represent”.
