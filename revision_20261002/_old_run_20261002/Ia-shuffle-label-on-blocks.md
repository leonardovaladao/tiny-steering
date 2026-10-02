# Check Ia — Under shuffled labels, is $\lambda_1/\sum\lambda$ at $T=1$ the same as at $T>1$?

Revision of 2026-10-02. Notebook: `Ia-shuffle-label-on-blocks.ipynb` (executed, outputs saved).
Tables: `Ia-shuffle-label-on-blocks_{H0_tests,per_cell,stored_hp}.csv`.
Raw curves: `Ia-shuffle-label-on-blocks_sT.npz`. Figure: `Ia-shuffle-label-on-blocks_share_vs_T.png`.
Everything uses the cached activations in `rfm_guppy/cache/`. No model inference was run.

This check extends Check I. Check I's Regime C traces $s(T)$ only at each concept's best block
with 10 permutations. Here the trace covers **all 6 blocks** and every $(\sigma,\text{normalize})$,
with $B=20$ permutations and a formal test of $T$-invariance.

## Short answer

**No.** With shuffled labels, the share at $T=1$ is **not** the same as at later $T$. It rises
with every RFM iteration, in every one of the 108 cells. The pre-registered criterion rejects
"same result" in **972 of 972** $(\text{cell},T)$ pairs. The rejection already happens at $T=2$
in all 108 cells.

Typical numbers: the shuffled-label median moves from about 0.07–0.40 at $T=1$ to about
0.46–0.92 at $T=10$. The median increase $s(10)-s(1)$ lies between +0.28 and +0.59.

Consequences for the paper:

1. **Part of every reported share comes from the iteration, not the concept.** A share of
   0.76–0.90 with no reference distribution says nothing.
2. **Real labels still separate from the null, but only when $T$ is large enough.** The real
   curve reaches $\ge 0.999$ in 71/108 cells. Across all 108 cells, all 20 permutations and all
   $T$, the largest shuffled value is **0.989** (valence, block 0). So "shuffled never gets to 1"
   is true, but only just, and only for valence.
3. **At $T=1$ the share is not label-informative.** In most cells the real $s(1)$ sits inside the
   shuffled range, and for valence it is often *below* the shuffled median.
4. **Four of the 18 stored (concept, block) shares cannot be told apart from shuffled labels at
   their own hyperparameters:** valence b0, valence b2, valence b5, env_social b1 (Section 4).

## 1. Object

Fix a concept $c\in\{\texttt{food},\texttt{valence},\texttt{env\_social}\}$ and a block
$\ell\in\{0,\dots,5\}$. Let $Z_\ell\in\mathbb{R}^{n\times 384}$ be the training-split last-token
residual activations, centred by the training mean and, when normalize is on, row-normalised.
Here $n=360$ for food and env_social and $n=120$ for valence, all balanced. Let $y\in\{0,1\}^n$
be the labels.

The RFM iteration with Laplace bandwidth $\sigma$ and ridge $\lambda=10^{-3}$ is
$$
M_0 = I_{384},\qquad M_{t+1} = \frac1n\sum_{i=1}^n \big(g_i^{(t)}-\bar g^{(t)}\big)\big(g_i^{(t)}-\bar g^{(t)}\big)^\top,
\qquad g_i^{(t)}=\nabla f_t\big(z^{(i)}\big),
$$
where $f_t$ is the kernel ridge predictor with kernel $K_{M_t}(a,b)=\exp(-\|a-b\|_{M_t}/\sigma)$
fit to $(Z_\ell,y)$. This is exactly `rfm_guppy.rfm._agop_step`. The statistic is
$$
s(T)=\frac{\lambda_1(M_T)}{\operatorname{tr}(M_T)}\in\Big[\tfrac1{384},\,1\Big].
$$
A **cell** is $(c,\ell,\text{normalize},\sigma)$, with $\text{normalize}\in\{F,T\}$ and
$\sigma\in\{1,10,100\}$. That gives 108 cells, and in each $s(T)$ is computed for $T=1..10$.

## 2. Design and falsification criterion

In each cell, $s(T)$ is computed for the real labels and for $B=20$ permuted label vectors
$y^{\pi_b}$. A permutation keeps the class counts but destroys the link between labels and
activations. The permutations are drawn once per concept (seed `20261002`) and shared across
blocks and hyperparameters, so the 20 shuffled runs are paired across $T$.

**$H_0$ ("same result").** Under shuffled labels, $s$ does not depend on $T$. Define the paired
differences $\Delta_b(T)=s_b(T)-s_b(1)$ for $T=2..10$.

**Criterion (fixed before running).** $H_0$ is refuted at $(\text{cell},T)$ if both of the
following hold:

- $|\operatorname{median}_b \Delta_b(T)|>0.05$
- a two-sided sign test on $\Delta_1(T),\dots,\Delta_{20}(T)$ gives $p<0.05$

The 0.05 margin is the smallest change in the share treated as practically relevant.

**Reproduction gate.** At the stored hyperparameters $(\sigma^\star,T^\star,\text{normalize}^\star)$,
the real-label value must match the cached `lam1_share` to within $10^{-4}$ for all 18
(concept, block) pairs. **Passed, 18/18.**

## 3. Result: $s_{\text{shuf}}(T)$ is increasing in $T$

| Quantity | Value |
|---|---|
| $(\text{cell},T)$ pairs where $H_0$ is refuted | **972 / 972** |
| First $T$ at which $H_0$ is refuted | $T=2$ in all 108 cells |
| Range of $\operatorname{median}_b\Delta_b(T)$ over all pairs | $+0.149$ to $+0.587$ |
| Pairs where fewer than 20/20 permutations increased | 5 of 972 |

Every median difference is positive, so the effect always points the same way. In all but 5
pairs, every one of the 20 shuffled runs has a higher share at $T$ than at $T=1$. For the sign
test that gives $p=2\cdot2^{-20}\approx 2\times10^{-6}$, far below 0.05. The decision is driven by
effect size, not by a borderline p-value.

**Range of the shuffled-label share (medians over $b$, across the 108 cells).** At $T=1$ it is
0.07–0.40. At $T=10$ it is 0.46–0.92. For comparison, an isotropic $M$ gives $1/384\approx0.003$.

## 4. The 18 stored shares, against the shuffled null at the same hyperparameters

Here $p_{\text{perm}}=(1+\#\{b: s_b \ge s_{\text{real}}\})/(1+B)$. With $B=20$ its smallest
possible value is $1/21\approx0.048$. Hyperparameters are frozen; re-selecting them on shuffled
labels is Check I, Regime B.

| concept | block | $\sigma^\star$ | norm | $T^\star$ | real $s(T^\star)$ | shuffled median [min, max] at $T^\star$ | $p_{\text{perm}}$ |
|---|---|---|---|---|---|---|---|
| food | 0 | 10 | F | 3 | 0.761 | 0.480 [0.371, 0.544] | 0.048 |
| food | 1 | 1 | T | 3 | 0.979 | 0.627 [0.469, 0.699] | 0.048 |
| food | 2 | 1 | T | 8 | 1.000 | 0.606 [0.470, 0.867] | 0.048 |
| food | 3 | 1 | T | 2 | 0.886 | 0.526 [0.303, 0.651] | 0.048 |
| food | 4 | 1 | T | 2 | 0.884 | 0.472 [0.371, 0.600] | 0.048 |
| food | 5 | 1 | T | 8 | 1.000 | 0.557 [0.386, 0.701] | 0.048 |
| valence | 0 | 10 | F | 10 | **0.797** | **0.783 [0.533, 0.952]** | **0.524** |
| valence | 1 | 10 | F | 10 | 1.000 | 0.672 [0.483, 0.867] | 0.048 |
| valence | 2 | 100 | F | 1 | **0.186** | **0.210 [0.153, 0.301]** | **0.762** |
| valence | 3 | 10 | F | 10 | 1.000 | 0.723 [0.493, 0.932] | 0.048 |
| valence | 4 | 10 | F | 10 | 1.000 | 0.617 [0.476, 0.789] | 0.048 |
| valence | 5 | 100 | F | 1 | **0.147** | **0.196 [0.161, 0.285]** | **1.000** |
| env_social | 0 | 100 | F | 10 | 0.813 | 0.592 [0.397, 0.767] | 0.048 |
| env_social | 1 | 100 | F | 1 | **0.222** | **0.202 [0.162, 0.236]** | **0.095** |
| env_social | 2 | 10 | F | 9 | 1.000 | 0.497 [0.380, 0.670] | 0.048 |
| env_social | 3 | 10 | F | 3 | 0.943 | 0.393 [0.292, 0.592] | 0.048 |
| env_social | 4 | 10 | F | 4 | 0.987 | 0.469 [0.321, 0.634] | 0.048 |
| env_social | 5 | 10 | F | 10 | 1.000 | 0.535 [0.418, 0.755] | 0.048 |

How to read the four highlighted rows:

- **valence b0.** The real curve (figure, row 2, panel 1) tracks the shuffled median at every
  $T$. At this block, RFM finds nothing about valence that the iteration would not have produced
  from noise.
- **valence b2, valence b5, env_social b1.** The sweep chose $T^\star=1$ because val $|\rho|$ was
  highest there, and at $T=1$ the share is uninformative (point 3 above). At their own
  $(\sigma^\star,\text{normalize}^\star)$, the real curves still climb to $\approx1$ by $T\approx4$
  (figure). So the low reported share in these blocks comes from the $T$ selection. It is not a
  property of the block.

**At $T=1$**, the real share beats every shuffled run ($p=0.048$) in only 4 of the 18 stored cells:
food b0, food b1, valence b0 and env_social b0. For valence b3, b4 and b5, the real $s(1)$ is *below
all 20* shuffled values ($p_{\text{perm}}=1.0$).

## 5. Figure

`Ia-shuffle-label-on-blocks_share_vs_T.png` is a $3\times6$ grid with concepts as rows and blocks
as columns, each panel at the stored $(\sigma^\star,\text{normalize}^\star)$. It shows:

- red: the real labels
- grey band: min–max over the 20 permutations
- grey line: shuffled median
- dashed vertical line: $T^\star$
- dotted horizontal line: 0.95

Typical pattern: at $T=1$ the red and grey curves start together. The grey band then rises to
about 0.5–0.7 and flattens, while the red curve saturates at 1. The exceptions are valence b0 and
env_social b0, where red stays below about 0.82.

## 6. Cross-check with Check I, Regime C

The two checks share 18 configurations: each concept's best block × 6 $(\sigma,\text{normalize})$.

- **Real-label curves agree exactly**, with max difference 0, as expected for a deterministic
  computation.
- **Shuffled medians at $T=10$ differ by up to 0.14**, for example food b3, norm=T, $\sigma=100$:
  0.683 here vs 0.544 in Check I. Different permutations were drawn (different seeds, $B=20$ vs
  $B=10$), and the shuffled distribution is wide, with min–max spans of about 0.3. A median from
  10–20 draws therefore moves by this much. Both checks reach the same qualitative conclusion.

## 7. Interpretation

**Conjecture (not proved; supported by this check).** The RFM map $M_t\mapsto M_{t+1}$
concentrates the spectrum regardless of the labels, in a way similar to power iteration. With
kernel $K_{M_t}$, distances are measured mostly along the top eigenvectors of $M_t$. The predictor
$f_t$, and hence its gradients $g_i^{(t)}$, then vary mostly within that subspace. So the AGOP
$M_{t+1}$ inherits and sharpens whichever directions dominated $M_t$, even when those directions
came from fitting noise. A precise statement would need conditions on $\sigma$, $\lambda$, $n$
and the spectrum of $Z_\ell^\top Z_\ell$. This check does not provide them.

**What the paper can safely say.** The defensible version of the claim is relative, not absolute:

> For cell $(c,\ell,\sigma^\star,T^\star,\text{normalize}^\star)$, the observed share
> $s_{\text{real}}(T^\star)$ exceeds the shuffled-label null at the same hyperparameters, with
> $p_{\text{perm}}\le 1/(B+1)$.

That holds for 14 of the 18 stored cells. It does **not** hold for valence b0, b2, b5 or
env_social b1. The reading "$s\approx1 \Rightarrow$ rank-1 concept" needs both the null and the
hyperparameter selection taken into account.

## 8. Limits

- $B=20$, so $p_{\text{perm}}\ge0.048$. The permutation p-values in Section 4 are therefore at
  their floor, not "highly significant". More permutations would sharpen them. Each extra
  permutation over all 108 cells costs about 35–65 s of CPU (the slower end when the machine is
  busy with other runs).
- Hyperparameters are frozen. Re-selecting $(\sigma,T,\text{normalize})$ on shuffled labels
  (Check I, Regime B) is the right null for the full pipeline and is not repeated here.
- One random seed. The Section 6 comparison gives a rough idea of the Monte-Carlo variability of
  the shuffled medians (up to 0.14 at $B=10$–$20$).
- This check says nothing about whether the top *direction* is label-driven. It is only about the
  *share*. Check I records $|\cos(u_{\text{shuf}},v_{\text{real}})|$ for that question.
