# Check II — Is the PCA baseline wrong by definition?

Revision of 2026-10-02. Notebook: `II-PCA-baseline.ipynb`. Tables: `II-PCA-baseline_*.csv`.

## Short answer

**Yes, the PCA baseline is wrong by definition.** Our code mean-centres the difference
matrix before the SVD. Beaglehole et al. (SI Appendix H) do not. The centring step removes
exactly the component of the data that carries the class signal.

**But uncentred PCA does not fully break even with difference-in-means**, so under the
criterion fixed before running, the claim "RFM is better than PCA" is **not refuted**.
What changes is its size and its support:

| Claim in the write-up | Status after the fix |
|---|---|
| RFM beats PCA on monitoring (AUROC 1.000 vs 0.879) | **Drop it.** Correct PCA scores 0.990–1.000. |
| RFM beats PCA on val \|ρ\| (0.92 vs 0.15 on valence; 0.95 vs 0.49 on env_social) | **Keep it, with new numbers.** The gap is 0.92 vs 0.70 and 0.95 vs 0.89. No gap on food. |
| PCA fails because "variance is not the class boundary" | **Wrong explanation.** PCA failed because the code subtracted the class boundary. |

## What the code does vs. what the paper does

Both versions pair positive and negative samples and stack the differences into a matrix
$D \in \mathbb{R}^{m\times 384}$, with one row $D_j = a_{+,j} - a_{-,j}$ per pair.

- **Paper (Appendix H):** take the top eigenvector of $D^\top D$ (equivalently, the top
  right singular vector of $D$).
- **Our code** (`rfm_guppy/baselines.py:86`, `D -= D.mean(axis=0)`): first subtract the
  mean row $\bar d$, then take the top eigenvector of $\tilde D^\top \tilde D$.

These differ by exactly one rank-one term:

$$\tilde D^\top \tilde D \;=\; D^\top D \;-\; m\,\bar d\,\bar d^\top .$$

All training sets are balanced (180/180 for `food` and `env_social`, 60/60 for `valence`),
so every training point is used in exactly one pair, and

$$\bar d \;=\; \hat\mu_+ - \hat\mu_- ,$$

the unnormalised difference-in-means vector. **Centring therefore deletes the
difference-in-means direction from the matrix before PCA looks at it.** What remains,
$\tilde D^\top\tilde D = m\,\widehat{\mathrm{Cov}}(D)$, is roughly the sum of the two
within-class covariances. Its top eigenvector is a direction of within-class spread, and
the label enters only through the final Pearson sign flip.

Both facts (the identity and $\bar d = \hat\mu_+ - \hat\mu_-$) are checked numerically in
the notebook for all 18 (concept, block) cells.

### Evidence that this is what happened

The alignment with difference-in-means at blocks 2–5 shows it directly:

| | food | valence | env_social |
|---|---|---|---|
| \|cos(PCA centred, DiffMeans)\| | 0.50 – 0.58 | 0.08 – 0.19 | 0.24 – 0.32 |
| \|cos(PCA uncentred, DiffMeans)\| | 0.99 | 0.95 – 0.97 | 0.98 – 0.99 |

The rank-one term holds only 14–34% of $\|D\|_F^2$, yet it decides the top eigenvector:
it is concentrated in one direction, while the remaining variance is spread over 384.

## Results

All numbers come from cached activations. The current code reproduces the published
Phase-8 numbers exactly (notebook gate (a)), so "centred" below is the PCA in the paper.

**Best-block validation |ρ|**

| Method | food | valence | env_social |
|---|---|---|---|
| PCA, centred (published) | 0.929 | 0.154 | 0.490 |
| PCA, uncentred, sequential pairing | 0.972 | 0.694 | 0.886 |
| PCA, uncentred, random pairing (median / worst of 20) | 0.972 / 0.971 | 0.698 / 0.667 | 0.888 / 0.882 |
| DiffMeans | 0.975 | 0.800 | 0.914 |
| RFM | 0.978 | 0.918 | 0.950 |

**Test AUROC (6-feature aggregate probe)**

| Method | food | valence | env_social | mean |
|---|---|---|---|---|
| PCA, centred (published) | 0.995 | 0.743 | 0.899 | 0.879 |
| PCA, uncentred, sequential | 1.000 | 0.995 | 1.000 | 0.998 |
| PCA, uncentred, random (worst of 20) | 1.000 | 0.990 | 0.999 | 0.997 |
| DiffMeans / RFM | 1.000 | 1.000 | 1.000 | 1.000 |

Random and sequential pairing agree to within 0.03, so the pairing scheme does not matter.
Only the centring does.

## Verdict under the pre-fixed criterion

The criterion: uncentred PCA breaks even with DiffMeans if, for **all three** concepts,
AUROC ≥ DiffMeans − 0.005 **and** best-block |ρ| ≥ DiffMeans − 0.02, in the worst case
over the pairings.

| Concept | AUROC worst / DM | pass | \|ρ\| worst / DM (gap) | pass | Break-even |
|---|---|---|---|---|---|
| food | 1.000 / 1.000 | yes | 0.971 / 0.975 (0.004) | yes | **yes** |
| valence | 0.990 / 1.000 | no | 0.667 / 0.800 (0.132) | no | **no** |
| env_social | 0.999 / 1.000 | yes | 0.882 / 0.914 (0.032) | no | **no** |

**Not break-even.** By the rule we set, "RFM is better than PCA" is not refuted.

Caveat on the valence AUROC: the valence test set has only 40 samples (20 + 20), so there
are 400 positive–negative pairs. An AUROC of 0.990 means 4 of those pairs are misordered.
The fail sits at the resolution limit of that test set.

## Is the remaining |ρ| gap real?

This is a supplementary check, not part of the criterion. It is a paired bootstrap over the
validation samples (2000 resamples, directions held fixed, 95% intervals):

| Concept | RFM − PCA unc. | 95% CI | DM − PCA unc. | 95% CI |
|---|---|---|---|---|
| food | 0.006 | [−0.005, 0.025] | 0.003 | [−0.003, 0.013] |
| valence | 0.224 | [0.099, 0.395] | 0.105 | [0.055, 0.170] |
| env_social | 0.065 | [0.036, 0.095] | 0.029 | [0.017, 0.041] |

- **food:** no detectable difference between RFM, DiffMeans and correct PCA.
- **valence** and **env_social:** RFM beats correct PCA by more than validation-sample noise.
  DiffMeans also beats correct PCA. The valence interval is wide because the validation set
  has only 40 samples.
- This interval covers only the noise from the validation sample. It does not cover
  training-set variability or the fact that RFM's hyperparameters were selected on this
  same validation set, which favours RFM. A cleaner number would be |ρ| on the test split.

## Corrections to what I said earlier in this session

1. I said all three concepts have 180/180 training points. **Wrong:** `valence` has 60/60
   (val and test: 20/20). The algebra is unaffected because the classes are still balanced.
2. I said break-even failed only on |ρ|. **Wrong:** in the worst case over random pairings,
   the valence AUROC (0.990) also misses the 0.005 tolerance.
3. I said the env_social |ρ| gaps were "roughly at noise level". **Wrong:** the bootstrap
   intervals exclude zero for both RFM − PCA and DM − PCA.

## What this changes in the write-up

Places that use the centred-PCA numbers or its explanation:

- `docs/latex_en/tex_files/1_fundamentals.tex:122–126`: the PCA definition says
  "Mean-centre it". Remove the centring and cite Appendix H.
- `docs/latex_en/tex_files/03_replication.tex`
  - Table `tab:baselines_val_rho`: PCA row.
  - Lines 402–408: the "variance is not the class boundary" explanation.
  - Table `tab:baselines_auroc`: PCA column.
  - Lines 430–433: "only PCA separating from the pack".
  - Lines 455–458 and 479–480: "RFM beats PCA … 1.000 against an overall 0.879".
- `docs/latex_en/tex_files/04_conclusion.tex:9–11`: the same claim.
- The Portuguese version (`docs/latex_pt/`) and the Medium post (`medium/tiny-steering.md`)
  probably repeat these numbers. Not checked.

A remark worth keeping: the centring footnote added to the AGOP step in
`1_fundamentals.tex` is correct for RFM (it centres gradients). Applied to the PCA
difference matrix, the same step removes the signal. The two are not analogous.

## Not done

- `rfm_guppy/baselines.py` is **unchanged**, and `results/baselines_*` still hold the
  centred-PCA numbers. Fixing the code means deleting line 86 and, to match the paper,
  shuffling `pos_idx` / `neg_idx` before pairing.
- Phase-8 steering with PCA vectors was not rerun, because it needs the model checkpoint
  (now in `.trash/guppylm-9M/`). That comparison is already flagged in the write-up as a
  broken measurement, so its conclusion does not change.
