# rfm_checks: limitation checks on RFM/AGOP

## Purpose

The replication in [`../replication/`](../replication/) made four claims about RFM on GuppyLM: the AGOP matrix concentrates on one eigenvalue (top-eigenvalue share near 1), RFM beats PCA by a wide margin, steering succeeds 100% of the time, and RFM is useful with few labelled examples. Each check here tests one of those claims against a control, with a refutation criterion fixed before the run. Setting for every check: GuppyLM (8.73M), cached last-token residual activations in R^384, blocks 0 to 5, concepts `food`, `valence`, `env_social` (the default; a new run sets its concepts in its `run.json`, see [Recipe](#recipe)).

The full results are in [`runs/2026-10-09/SUMMARY_OF_TESTS.md`](runs/2026-10-09/SUMMARY_OF_TESTS.md). Where that summary disagrees with the CSVs, this README uses the corrected numbers from the report page (see [Caveats](#caveats), item b).

## Layout

```
rfm_checks/
├── I-shuffled-label-control.ipynb    Check I notebook (with stored outputs)
├── Ia-shuffle-label-on-blocks.ipynb  Check Ia notebook (with stored outputs)
├── II-PCA-baseline.ipynb             Check II notebook (with stored outputs)
├── III-steering-metric.ipynb         Check III notebook (with stored outputs)
├── IV-small-n.ipynb                  Check IV notebook (no stored outputs)
├── IV-small-n_design.md              Check IV pre-registered design, in Portuguese, with Amendment 1
├── checkrun.py                       creates a check run (run.json with its config); loaded by every notebook
├── kappa_side_run.py                 side run on the AGOP eigenvalue trajectory (caveat c)
├── runs/
│   ├── 2026-10-02/                   first run of I, Ia, II, III, with one .md report per check
│   └── 2026-10-09/                   rerun of I-III, Ia rerun, IV; executed notebooks; SUMMARY_OF_TESTS.md
└── reports/
    ├── checks_I-IV/                  private web page on Checks I-IV (English)
    └── checks_I-Ia_ptbr/             private web page on Checks I and Ia (pt-BR)
```

The top-level notebooks are templates. `checkrun.py new` copies them, with outputs cleared, into a new run folder, and the copies are executed there; started in `rfm_checks/` itself they refuse to run. Their code differs from the executed copies in `runs/2026-10-09/` in the setup cell (paths from `rfm_guppy/paths.py` instead of hand-built paths; run folder, concepts and replication run from `run.json`), in the lines that read the replication run (II, III) or write an output (every write goes into the run folder through `out_file`), and in the figure grids of I, Ia and IV, whose size follows the concept list. Their markdown and one code comment (III) also name the current paths. Older documents call this folder `revision_20261002/` and the second run `run_20261009/` (renamed in commit `b1ac054`).

## The checks

`s = λ1(M_T) / tr(M_T)` is the top-eigenvalue share of the AGOP matrix after T RFM iterations. "Stored HPs" are the hyperparameters saved in `data/cache/<concept>_vectors.npz`. All evidence files are in `runs/2026-10-09/`.

| Check | Question | Notebook | Metric | Refuted if (pre-registered) | Latest verdict | Evidence |
|---|---|---|---|---|---|---|
| I, Regime A | Is a high share s evidence of a concept, or does RFM produce it on any labels? | [I](runs/2026-10-09/I-shuffled-label-control.ipynb) | s with real labels vs B=100 label permutations, HPs frozen at stored values. Gate: the 18 stored shares recompute to 1e-4. | At a cell: `p = (1 + #{s_shuf >= s_real}) / (1 + B) >= 0.05`, or >= 5% of shuffled runs reach s >= 0.95 | Refuted in 5/18 cells, holds in 13/18. Shuffled median 0.43–0.76 in the 15 cells with T*>1, 0.20–0.21 in the 3 cells with T*=1; max 0.99. | [regimeA.csv](runs/2026-10-09/I-shuffled-label-control_regimeA.csv), [results.npz](runs/2026-10-09/I-shuffled-label-control_results.npz) |
| I, Regime B | Same, against the number the paper reports | [I](runs/2026-10-09/I-shuffled-label-control.ipynb) | Full HP sweep rerun on shuffled labels, B=20 | Same criterion as Regime A | Refuted in 4/18 cells, holds in 14/18. The smallest possible p is 1/21 = 0.048, so "holds" sits on the resolution floor. | [regimeB.csv](runs/2026-10-09/I-shuffled-label-control_regimeB.csv), [resultsB.npz](runs/2026-10-09/I-shuffled-label-control_resultsB.npz) |
| I, side check E | How strong a criterion is val \|ρ\|? | [I](runs/2026-10-09/I-shuffled-label-control.ipynb) | val \|ρ\| of 2000 random unit directions, PC1 and DiffMeans | not pre-registered | Weak criterion. Shuffled RFM reaches \|ρ\| ≈ 0.86 on `food`; PC1 0.94–0.97 and DiffMeans 0.95–0.98 on `food`. | [rho_baselines.csv](runs/2026-10-09/I-shuffled-label-control_rho_baselines.csv) |
| Ia | Does the RFM iteration itself concentrate the spectrum? | [Ia](runs/2026-10-09/Ia-shuffle-label-on-blocks.ipynb) | s(T), T=1..10, real labels and B=20 permutations, on all 108 cells (3 concepts × 6 blocks × 2 normalize × 3 σ). Gate: 18 stored shares to 1e-4. | H0 `s_shuf(T) = s_shuf(1)` refuted at (cell, T) if `abs(median Δ) > 0.05` and two-sided sign test p < 0.05 (972 tests) | H0 refuted in 972/972, from T=2 in every cell. Shuffled share grows by +0.15 to +0.59. At T=1 the real share is not above shuffled in 14/18 stored cells. Real labels reach s >= 0.999 in 71/108 cells; shuffled never reach 0.99. | [H0_tests.csv](runs/2026-10-09/Ia-shuffle-label-on-blocks_H0_tests.csv), [per_cell.csv](runs/2026-10-09/Ia-shuffle-label-on-blocks_per_cell.csv), [stored_hp.csv](runs/2026-10-09/Ia-shuffle-label-on-blocks_stored_hp.csv) |
| II | Was the PCA baseline wrong by construction? | [II](runs/2026-10-09/II-PCA-baseline.ipynb) | Best-block val \|ρ\| and test AUROC of uncentred PCA (sequential and 20 random pairings, worst case) vs DiffMeans. Gates: old Phase-8 numbers reproduced; centring identity holds in 18 cells. | Uncentred PCA "breaks even" with DM if, for all 3 concepts, AUROC >= DM − 0.005 and best-block \|ρ\| >= DM − 0.02 | Break-even not reached: fails on `valence` (AUROC 0.99 vs 1.000, \|ρ\| 0.667 vs 0.800) and on `env_social` \|ρ\| (0.882 vs 0.914). The old baseline was mis-specified: mean test AUROC of PCA goes from 0.879 to 0.998 after the fix. | [verdict.csv](runs/2026-10-09/II-PCA-baseline_verdict.csv), [headline.csv](runs/2026-10-09/II-PCA-baseline_headline.csv), [perblock.csv](runs/2026-10-09/II-PCA-baseline_perblock.csv), [bootstrap.csv](runs/2026-10-09/II-PCA-baseline_bootstrap.csv) |
| III, G1 | Is the blind judge valid for this concept? | [III](runs/2026-10-09/III-steering-metric.ipynb) | 5-fold CV AUROC of the TF-IDF + logistic regression judge g_c on GuppyLM dataset outputs | A concept with CV AUROC < 0.95 is not interpreted | Fails for `valence` (0.78; only 40+40 unique outputs). Passes for `food` (0.998) and `env_social` (≈1.000). G2 (36/36) and G3 (7/7) passed. | [judge_validity.csv](runs/2026-10-09/III-steering-metric_judge_validity.csv) |
| III, H1 | Does the steering direction move the blind judge both ways? | [III](runs/2026-10-09/III-steering-metric.ipynb) | g_c at ±r* vs r=0 on 7 prompts, r* ∈ {0.25, 0.5}, exact one-sided sign tests | Supported only if 7/7 prompts go up at +r* and 7/7 go down at −r* | Not supported for `food` (5/7 up) or `env_social` (7/7 up, 6/7 down). The one "supported" cell is `valence` at r=0.5, which G1 excludes. | [H1.csv](runs/2026-10-09/III-steering-metric_H1.csv) |
| III, H2 | Is any steered generation both on-concept and coherent? | [III](runs/2026-10-09/III-steering-metric.ipynb) | Coherent = PPL <= q95 and distinct-2 >= q05 of 100 unsteered generations; joint = g_c > 0 and coherent | "No coherent steering" refuted if, at some r > 0, >= 4/7 generations are joint | Not refuted: 0 coherent generations at any r ≠ 0 for every concept (lowest steered PPL ≈ 2559 vs threshold 2.25). Joint 0/7 everywhere. | [H2.csv](runs/2026-10-09/III-steering-metric_H2.csv), [per_r.csv](runs/2026-10-09/III-steering-metric_per_r.csv), [calibration.csv](runs/2026-10-09/III-steering-metric_calibration.csv) |
| III, H3 | Does the old probe agree with the blind judge? | [III](runs/2026-10-09/III-steering-metric.ipynb) | Spearman and binary agreement, probe vs g_c, 84 generations per concept | not pre-registered (descriptive) | Agree on `valence`/`env_social` (Spearman ≈ 0.81, agreement 99%). Not on `food` (Spearman 0.41, agreement 57%, 35 probe-yes/judge-no cases). | [H3.csv](runs/2026-10-09/III-steering-metric_H3.csv), [old_vs_new.csv](runs/2026-10-09/III-steering-metric_old_vs_new.csv) |
| III, duplication | Are the 7 prompts 7 observations? | [III](runs/2026-10-09/III-steering-metric.ipynb) | Distinct texts and mean bigram Jaccard across prompts, per (concept, r) | not pre-registered (added after seeing results) | No. Bigram Jaccard 0.71–1.00 at \|r\| >= 0.5, with 1 to 6 distinct texts out of 7. | [duplication.csv](runs/2026-10-09/III-steering-metric_duplication.csv) |
| IV, H3a | With n per class in {2, 4}, is RFM no better than DiffMeans? | [IV](runs/2026-10-09/IV-small-n.ipynb) | Δ_c(n) = E[AUROC(RFM) − AUROC(DM)] over R=200 balanced subsamples, B=10 000 bootstrap, primary block 3. Gate C0 (stops the run): at n=1 RFM = DM to 1e-8; every direction in the data span to 1e-6. | Refuted if some main concept at n ∈ {2, 4} has a lower 95% CI bound > 0 | Not refuted, in all 6 blocks. RFM is worse: Δ = −0.05 to −0.21, CIs entirely below 0. Even the oracle-best T stays below 0. | [verdicts.csv](runs/2026-10-09/IV-small-n_verdicts.csv), [summary.csv](runs/2026-10-09/IV-small-n_summary.csv), [draws.csv](runs/2026-10-09/IV-small-n_draws.csv) |
| IV, H-artigo | Does RFM beat DiffMeans at some n? | [IV](runs/2026-10-09/IV-small-n.ipynb) | Same Δ_c(n); Holm within each block over k=10 cells, α = 0.025, one-sided bootstrap p | Supported if Holm declares some cell with Δ > 0; refuted otherwise | Refuted at primary block 3 (min p = 0.0047 > 0.0025) and at block 5. Supported in blocks 0, 1, 2, 4, only at n=32. The only practically relevant win is `valence` block 0 (Δ = +0.077). | [verdicts.csv](runs/2026-10-09/IV-small-n_verdicts.csv), [summary.csv](runs/2026-10-09/IV-small-n_summary.csv) |
| IV, H3b | Do the curves cross (RFM loses at small n, wins at larger n)? | [IV](runs/2026-10-09/IV-small-n.ipynb) | Same Δ_c(n) and CIs | Supported if, in one concept, a cell has upper CI bound < 0 and a larger-n cell has lower CI bound > 0 | Supported in every block. Crossover at n ≈ 32. | [verdicts.csv](runs/2026-10-09/IV-small-n_verdicts.csv), [T_sensitivity.csv](runs/2026-10-09/IV-small-n_T_sensitivity.csv), [hp.csv](runs/2026-10-09/IV-small-n_hp.csv) |

**Check I.** The replication read s ≈ 1 as evidence that one direction carries the concept. Random labels already give large shares, up to 0.99, so a high share alone is not evidence. The real share still beats the null in 13 of 18 cells (Regime A) and 14 of 18 (Regime B). The refuted cells are in `valence` (blocks 0, 2, 5) and `env_social` (blocks 0–1); in 3 of them the real share is itself small (0.15–0.22, T=1, σ=100). Side check E shows that val |ρ| does not separate real from shuffled labels on `food`. Verdict on the page: weakened. Sources: SUMMARY_OF_TESTS.md, section "Check I"; [`reports/checks_I-IV/src.html`](reports/checks_I-IV/src.html), section `id="check-1"`.

**Check Ia.** This extends Check I to every cell and every T. The share grows under shuffled labels in all 972 tests, so the RFM iteration concentrates the spectrum by itself, and part of every reported s ≈ 1 comes from T, not from the concept. The real-vs-shuffled gap appears only after iterating: at T=1 the real share is not above the shuffled one in 14 of 18 stored cells. Real labels still end higher (s >= 0.999 in 71 of 108 cells). The Regime C curves of Check I agree with Ia (real curves identical, shuffled medians within Monte Carlo noise). Sources: SUMMARY_OF_TESTS.md, section "Check Ia"; report page, section `id="check-1a"`.

**Check II.** The replication's `pca_vector` centres the paired-difference matrix D before the SVD. Beaglehole et al. (App. H) do not. Since the mean row of D equals μ+ − μ−, centring removes exactly the DiffMeans direction. Without centring, PCA is almost parallel to DiffMeans and its mean test AUROC rises from 0.879 to 0.998. So "RFM beats PCA by a wide margin" rested on a mis-specified baseline. On test AUROC the gap closes. On val |ρ| it is absent on `food`, small on `env_social` (+0.06) and large on `valence` (+0.22, CI [0.10, 0.40]). Sources: SUMMARY_OF_TESTS.md, section "Check II"; report page, section `id="check-2"`.

**Check III.** The replication reported 100% steering success. That metric (probe > 0.5 or lexicon hit) reads the same AGOP basis the steering vector comes from, so it is partly circular. Re-scored with a judge that never sees activations, success is partial at best (3/7 for `food` at the best positive r), and no steered generation is coherent at any r ≠ 0, so "on-concept and coherent" is 0/7 everywhere. The steered texts are also near-copies across prompts, so the 7 prompts are not 7 independent observations. Sources: SUMMARY_OF_TESTS.md, section "Check III"; report page, section `id="check-3"`.

**Check IV.** The claim under test (H-artigo) is that RFM extracts a better direction than DiffMeans at some n. With few examples the opposite holds. At n=1 RFM equals DiffMeans by construction. At 2 <= n <= 8 it is worse in every block and concept, at n=16 it is worse or tied, and a small advantage appears only at n ≈ 32. At block 3, RFM also loses to LogReg at 2 <= n <= 32 on `valence` and `env_social`, and to uncentred PCA at n ∈ {2, 4} on all concepts. `food` is a ceiling control (RFM worse up to n=16, tied at n=32). The design, hypotheses and Amendment 1 are in [`IV-small-n_design.md`](IV-small-n_design.md). Sources: SUMMARY_OF_TESTS.md, section "Check IV"; report page, section `id="check-4"`.

## Caveats

**(a) AGOP mask ε.**

- The Check I outputs in `runs/2026-10-09` were produced with the absolute mask: `W[i,j]` set to 0 when `D[i,j] < 1e-12`. (Git history: the I–III rerun is commit `e4d1385`, 2026-10-09 11:47; the mask change is commit `f9fee8f`, 12:09 the same day.)
- Check IV and the Check Ia rerun use the relative mask now in [`../rfm_guppy/rfm.py`](../rfm_guppy/rfm.py): `W[i,j]` set to 0 when `D[i,j] <= 1e-12 · max D`. In `_agop_step` this is lines 77–78 (`eps = 1e-12 * D.max()`, `mask = D <= eps`; docstring line 63). In `agop_grad` it is lines 177–178 (docstring line 162). (Git history: IV is commit `4a75c5d` and the Ia rerun is commit `a9b198c`, 2026-10-16, both after `f9fee8f`.)
- Whether the two masks differ on these activations cannot be checked from files in this repo. [`IV-small-n_design.md`](IV-small-n_design.md), Amendment 1, records that Checks I, II and III were rerun with the relative mask, that every CSV matched `runs/2026-10-09/` byte for byte, and that the Check I NPZ files differed by at most 0. The outputs of that rerun are not stored in the repo.
- The three Ia CSVs in `runs/2026-10-09` (relative mask) are byte-identical to those in `runs/2026-10-02` (absolute mask), checked with `cmp`. (Conjecture) the mask change made no difference for Ia. The `_sT.npz` file exists only in the 10-09 run, so it cannot be compared.

**(b) Eight imprecise statements in SUMMARY_OF_TESTS.md.** The report page lists them with replacement sentences ([`reports/checks_I-IV/src.html`](reports/checks_I-IV/src.html), section `id="corrections"`). No conclusion changes. This README uses the corrected versions.

1. Check I, shuffled median: 0.43–0.76 holds only for the 15 cells with T*>1; with the three T*=1 cells it is 0.20–0.76.
2. Check I, shuffled |ρ| on `valence`/`env_social`: 0.1–0.5 holds for Regime A only; Regime B reaches 0.16–0.61 and 0.36–0.80.
3. Check Ia, shuffled maximum: it is 0.9893, so "never reach 0.99", not "never pass 0.989".
4. Check II, |cos(PCA_u, DM)| ≈ 0.95–0.99 holds from block 2 on in all concepts; `valence` block 1 is 0.76.
5. Check II, "the remaining gap is small": true on test AUROC, false on val |ρ| for `valence` (+0.22, CI [0.10, 0.40]); and "fixed PCA ≈ DM" conflicts with the failed break-even.
6. Check III, bigram Jaccard at |r| >= 0.5 is 0.71–1.00, not 0.85–1.00.
7. Check IV, "loses to LogReg at every n <= 32" and "to uncentred PCA at n <= 4" hold at block 3 only, with a tie on `food` at n=32.
8. One-line table, Check IV: "worse than DM at n <= 16" is exact at block 3; across blocks n=16 includes ties, and at n=1 the methods coincide by construction.

**(c) The κ side run.** [`kappa_side_run.py`](kappa_side_run.py) recomputes, on the stored cells (each concept of the run × 6 blocks, at the stored σ and normalize; 18 cells for the 3 default concepts), the AGOP eigenvalue trajectory for T=1..10, for the real labels and 20 permutations (seed 20261002, drawn in the same order as Check Ia). Per (cell, label set, t) it records λ1, λ2, λ3, the trace, s, and the top-2, top-3 and top-5 shares. It also reports the largest difference between its s and the stored Ia values in `Ia-shuffle-label-on-blocks_sT.npz` of `--source-run` (default `runs/2026-10-09/`, read only). It tested the toy model on the report page (section `id="check-1a"`), where each eigenvalue obeys λ ↦ cλ², so κ = λ1/λ2 becomes κ² per iteration and κ^(2^t) after t iterations. The script does not compute κ or the ratio below itself; both follow from the λ1 and λ2 columns.

Result, recorded in the owner's notes and not in any file in the repo: log κ_{t+1} / log κ_t pools to 1.01 on shuffled labels (κ ≈ 2.2, stable, while the top-5 share goes to 1) and to 1.26 on real labels. The κ^(2^t) claim fails and should not be repeated.

No `kappa.csv` is stored in the repo. The script writes it into a check run of its own, never into `runs/2026-10-09/` (pinned): `.venv/bin/python rfm_checks/kappa_side_run.py --run <run> [--source-run <run>]`, from the repo root, where `<run>` was made by `checkrun new` and supplies the concepts. It refuses an existing `kappa.csv` unless `RFM_FORCE=1` and appends a step to the run's `run.json`. With concepts other than the default three, `--source-run` must be a run whose Check Ia covered them in the same order, usually the same run after Ia; otherwise the script stops before computing. In the recipe test below (default concepts, gate against `runs/2026-10-09/`), it wrote 3 780 rows (18 cells × 21 label sets × 10 iterations) in 59 s, with gate max |s − stored| = 4.8e-15.

**(d) Check IV's failed first run and the C0 gate.** The first execution of Check IV stopped at the C0 gate: `C0 FAILED (residual): valence b3 n=1 r=0 RFM resid=1.00e+00`. The log is `runs/2026-10-09/IV-small-n_run_failedC0.log`. It is git-ignored (root `.gitignore` pattern `*.log`), so it exists only on the machine that ran it. Amendment 1 of [`IV-small-n_design.md`](IV-small-n_design.md) gives the cause: with n=1 the scale of M_t shrinks each iteration, the Mahalanobis distance fell below the absolute 1e-12 cut, all weights became 0, M_t became the zero matrix, and `eigh` returned an arbitrary basis vector outside the data span. The fix is the relative mask in caveat (a). No H3a, H-artigo or H3b result had been computed before the amendment. The final run passes C0: at n=1 RFM, DM, PCA and LogReg coincide (gap <= 1e-15), and every direction lies in the data span (residual <= 3e-12).

## Rerunning a check

### What each notebook reads and writes

Every notebook calls `load_check_run` (in [`checkrun.py`](checkrun.py)) in its setup cell. The run folder is `$RFM_RUN` if that is set, else the current directory; `nbconvert` runs a notebook in the folder that holds it. The folder must contain a `run.json` with a `config` block, written by `checkrun new`. Refused: `rfm_checks/` itself and the pinned runs in `PINNED_RUNS` (`rfm_guppy/paths.py`): `runs/2026-10-02/`, `runs/2026-10-09/` and `replication/runs/2026-08-27/`. Before computing anything, `load_check_run` checks that none of the notebook's outputs exist in the run folder, then appends one step to `run.json`: the notebook path, its concepts, the constants of its setup cell (such as `B`, `TMAX`, `LAM`, `SEED`, `FORCE`) and `RFM_FORCE`. The step is appended when the notebook starts, so a run that fails later also leaves a step. Every write goes through `out_file`, which refuses to overwrite an existing file unless `RFM_FORCE=1`.

The notebooks find the repo root by walking up from the current directory until they see `rfm_guppy/paths.py`. Inside the repo no `PYTHONPATH` is needed. Outside it, set `PYTHONPATH` to the repo root.

| Notebook | Reads | Writes | Cached results it can reload | Needs checkpoint |
|---|---|---|---|---|
| I | `CACHE_DIR`: `concept_<c>.json`, `<c>_{train,val}_activations.npy`, `<c>_vectors.npz` | Run folder: `I-shuffled-label-control_results.npz`, `_resultsB.npz`, `_resultsC.npz`, `_regimeA.csv`, `_regimeB.csv`, `_rho_baselines.csv`, `_share_vs_T.png` | `FORCE = True` (cell 1) recomputes. With `FORCE = False`, cells 5, 8 and 10 reload the three `.npz` files from the run folder if they exist. | No |
| Ia | `CACHE_DIR`: `concept_<c>.json` (train), `<c>_train_activations.npy`, `<c>_vectors.npz`. Cell 15 reads `I-shuffled-label-control_resultsC.npz` from the run folder for the Regime C cross-check; if absent it prints "skipped". | Run folder: `Ia-shuffle-label-on-blocks_sT.npz`, `_H0_tests.csv`, `_per_cell.csv`, `_stored_hp.csv`, `_share_vs_T.png` | `FORCE = True` (cell 1) recomputes. With `FORCE = False`, cell 3 reloads `_sT.npz` from the run folder if it exists. | No |
| II | `CACHE_DIR`: `concept_<c>.json`, `<c>_{train,val,test}_activations.npy`, `<c>_vectors.npz`. `config.replication_run` (default `replication/runs/2026-08-27/`): `baselines_monitoring.json`. | Run folder: `II-PCA-baseline_perblock.csv`, `_headline.csv`, `_verdict.csv`, `_bootstrap.csv` | None | No |
| III | `config.replication_run`: `steer_<c>.json`, `eval6_<c>.json`. `CKPT_DIR`: `tokenizer.json`, `config.json`, `pytorch_model.bin`. `GUPPYLM_DIR` (submodule). HF dataset `arman-bd/guppylm-60k-generic`, loaded with `HF_DATASETS_OFFLINE=1`, so it must already be in the local Hugging Face cache. | Run folder: `III-steering-metric_judge_validity.csv`, `_calibration.csv`, `_generations.csv`, `_per_r.csv`, `_H1.csv`, `_H2.csv`, `_H3.csv`, `_old_vs_new.csv`, `_duplication.csv` | None | Yes: gate G3 regenerates the 7 r=0 texts, the 100 calibration generations need the model, and perplexities are computed under it |
| IV | `CACHE_DIR`: `concept_<c>.json` (train, val, test), `<c>_{train,val,test}_activations.npy`, `<c>_vectors.npz` | Run folder: `IV-small-n_hp.csv`, `_draws.csv`, `_summary.csv`, `_T_sensitivity.csv`, `_verdicts.csv`, `_curves.png`, `_T_curves.png` | None | No |

Order: the only cross-notebook read is Ia reading Check I's `_resultsC.npz` from the run folder. To get the cross-check, run I before Ia in the same run folder. No other order is required. Runtime: the notebooks record ≈12 min for the Ia sweep and ≈45 min for IV with all 6 blocks (≈7 min with `RUN_SECONDARY = False`), both on a laptop CPU. In the recipe test below, on a laptop CPU (10 cores): I 532 s, Ia 344 s, II 5 s, III 11 s, IV 2 149 s with all 6 blocks, κ 59 s.

### Recipe

Every run starts with `checkrun new`, which creates the run folder, writes `run.json` with the run's configuration and copies the five notebooks into it (outputs cleared). Then each copy is executed in place in that folder. Nothing is copied or moved by hand.

```bash
# from the repo root
.venv/bin/python -m rfm_checks.checkrun new            # default: rfm_checks/runs/<today>/ (or <today>_2, ...)
RUN=rfm_checks/runs/2026-11-20                         # the folder it printed
cd "$RUN"
../../../.venv/bin/jupyter nbconvert --to notebook --execute --inplace I-shuffled-label-control.ipynb
../../../.venv/bin/jupyter nbconvert --to notebook --execute --inplace Ia-shuffle-label-on-blocks.ipynb
../../../.venv/bin/jupyter nbconvert --to notebook --execute --inplace II-PCA-baseline.ipynb
../../../.venv/bin/jupyter nbconvert --to notebook --execute --inplace III-steering-metric.ipynb
../../../.venv/bin/jupyter nbconvert --to notebook --execute --inplace IV-small-n.ipynb
cd ../../..
.venv/bin/python rfm_checks/kappa_side_run.py --run "$RUN"
```

`checkrun new` refuses a folder that exists and is not empty, and the pinned runs. Its options:

- `--run DIR`: the run folder. It may be outside the repo; then set `PYTHONPATH` to the repo root for `nbconvert`.
- `--concepts NAME ...`: the concepts (default `food valence env_social`). Each name must be registered in `rfm_guppy/concepts.py` (in `CONCEPTS` and `CONCEPT_LEXICONS`), and `new` refuses a concept whose files are missing from `data/cache/` or from the replication run.
- `--replication-run DIR`: the replication run that II (`baselines_monitoring.json`) and III (`steer_<c>.json`, `eval6_<c>.json`) read; default `replication/runs/2026-08-27/`. It is stored repo-relative in `run.json`.
- `--iv-main NAME NAME`: Check IV's MAIN set (see below).

`run.json` then has the form

```json
{"study": "rfm_checks", "created": "...",
 "config": {"concepts": ["food", "valence", "env_social"],
            "replication_run": "replication/runs/2026-08-27",
            "IV": {"concepts": ["valence", "env_social", "food"]}},
 "steps": []}
```

and each notebook, and `kappa_side_run.py`, appends one step to `steps`.

Environment variables:

- `RFM_RUN=<folder>` runs a notebook against that run folder instead of its current directory.
- `RFM_FORCE=1` allows a notebook (or `kappa_side_run.py`) to overwrite outputs that already exist in the run folder. Without it, rerunning a notebook in the same folder stops before computing, with `FileExistsError`. The pinned runs are refused even with `RFM_FORCE=1`. This is separate from the `FORCE` constant in I and Ia, which chooses between recomputing and reloading their cached `.npz` files.

**Adding a concept.** Register it in `rfm_guppy/concepts.py`, build its dataset and vectors (`python -m rfm_guppy.concepts --concepts NAME`, `replication/01_extract.py`), and produce its `steer_<c>.json`, `eval6_<c>.json` and `baselines_monitoring.json` entry in a replication run (`02_steer.py`, `03_monitor.py`, `04_baselines.py`; see [`../replication/README.md`](../replication/README.md)). Then `checkrun new --concepts food valence env_social NAME --replication-run <that run>`. Concept order fixes the seeds: Check Ia and κ draw permutations concept by concept, and Checks I and IV seed with the concept's index. So every check keeps its historical order for the default concepts (`food, valence, env_social`; for IV `valence, env_social, food`) and appends new concepts in the order given. The default three keep their 2026-10-09 seeds as long as all three are in the list; leaving one out shifts the seeds of those after it.

Places that refuse a new concept until something is added by hand:

- Check II, cell 3: gate (a) compares with the published best-block val |ρ| of the centred PCA in `PUB_RHO`, which holds only the three default concepts. Add the value from the `baselines_comparison.md` of the replication run.
- Check III, setup cell: gate G3 and the r=0 coherence count use `food`'s stored r=0 generations, so a run without `food` is refused.
- Check IV, setup cell: MAIN, below.
- `kappa_side_run.py`: `--source-run` must contain Check Ia cells for every concept.

Check III also changes for the old concepts when a concept is added: its calibration prompts come from the dataset categories that no concept of the run uses, so a new concept can change the held-out categories and therefore the coherence thresholds.

**Check IV MAIN.** H3a, H-artigo and H3b are about the MAIN concepts, pre-registered in [`IV-small-n_design.md`](IV-small-n_design.md) as `valence` and `env_social` for the concept list `valence, env_social, food`. That default applies only when IV's concept list is exactly this one. For any other list, `run.json` must contain `config.IV.main`, written by `checkrun new --iv-main NAME NAME`; otherwise the IV notebook stops in its setup cell, before computing anything, and `new` prints a note saying so. Choosing a new MAIN is a design amendment: record it in `IV-small-n_design.md` before the run. MAIN must name 2 concepts of IV's list, because the Holm family k = 10 = 2 concepts × 5 values of n is fixed in cells 13 and 15.

`nbconvert` in this environment (nbclient 0.11.0) has no per-cell timeout by default. If a gate fails (an `assert` in the notebook, such as C0 in IV), `nbconvert` exits with the traceback and does not save the executed notebook. Files written before the failure stay in the run folder, so a rerun there needs `RFM_FORCE=1`, or use a new run. Capture the output with `2>&1 | tee <name>.log` if you want a record like `IV-small-n_run_failedC0.log`.

Inputs: all checks need `data/cache/` (see [`../replication/README.md`](../replication/README.md) for how it is built). Check II also needs `baselines_monitoring.json` in the configured replication run (default `replication/runs/2026-08-27/`). Check III also needs the GuppyLM checkpoint in `data/checkpoints/guppylm-9M`, the files `steer_<c>.json` and `eval6_<c>.json` in the configured replication run, and the Hugging Face dataset in the local cache.

**Recipe test (default configuration, 2026-11-12).** `checkrun new --run <scratch folder outside the repo>`, then the five notebooks executed in order I, Ia, II, III, IV with `PYTHONPATH=<repo> .venv/bin/jupyter nbconvert --to notebook --execute --inplace`, then `kappa_side_run.py --run <scratch folder>`. All 24 CSVs were byte-identical to `runs/2026-10-09/` (`cmp`, 24/24); the four `.npz` files had equal arrays under every key; the five PNGs were byte-identical too. Gates passed as in the pinned run (Ia's Regime C cross-check: real curves differ by 0; III: G2 36/36, G3 7/7). `run.json` kept its `config` block and gained one step per notebook plus one for κ. The scratch folder is not in the repo.

### Rebuilding a report page

Each page folder holds `build_data.py`, `build.js`, `charts.js`, `src.html`, `data.json` and the built `index.html`.

```bash
cd rfm_checks/reports/checks_I-IV          # or checks_I-Ia_ptbr
../../../.venv/bin/python build_data.py    # reads the pinned run, writes data.json here
node build.js <katex-dir>                  # writes index.html here
```

- `build_data.py` reads the pinned run `runs/2026-10-09/` through `CHECKS_RUN_2026_10_09` in `rfm_guppy/paths.py`, not the latest run. Pointing a page at a new run needs a code change. It writes `data.json` to the current directory, so run it from the page folder.
- `build.js` takes one argument, the KaTeX package directory (the folder that contains `dist/katex.js`, `dist/katex.min.css` and `dist/fonts/`). It reads `src.html`, `data.json` and `charts.js` from the current directory and writes `index.html`. The built pages embed KaTeX 0.16.11, so use that version, for example from `npm install --prefix <dir> katex@0.16.11`, then `node build.js <dir>/node_modules/katex`.
- The two `build.js` files are byte-identical. Both load KaTeX by the absolute path given as the argument, so neither needs `NODE_PATH`.
- This rebuild was not tested in this step.

## Run history

| Run | What was run | Notes |
|---|---|---|
| [`runs/2026-10-02/`](runs/2026-10-02/) | Checks I, Ia, II, III | First run (commit `0aba57f`). One `.md` report per check. Absolute AGOP mask. No executed notebooks stored. |
| [`runs/2026-10-09/`](runs/2026-10-09/) | Checks I, II, III rerun; Check IV; Check Ia rerun | I–III rerun before the mask change (commit `e4d1385`). IV after it (commit `4a75c5d`), only in this run. Ia rerun added later into this folder (commit `a9b198c`, 2026-10-16), together with `SUMMARY_OF_TESTS.md`. Executed notebook copies for all five checks. All 19 CSV names shared with 2026-10-02 are byte-identical to the 2026-10-02 files (19/19, `cmp`); the two shared PNGs are identical too. |

Two things read from the pinned run: `kappa_side_run.py` gates against `runs/2026-10-09/Ia-shuffle-label-on-blocks_sT.npz` by default (`--source-run`) and writes `kappa.csv` only into its own `--run`, and both pages' `build_data.py` read `runs/2026-10-09/` through `CHECKS_RUN_2026_10_09`. The pinned runs (`PINNED_RUNS` in `rfm_guppy/paths.py`) cannot be used as a run folder; new runs under `runs/` are protected by the append-only rule.
