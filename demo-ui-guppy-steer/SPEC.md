# Spec — GuppyLM steering demo (side-by-side page)

*Synthesised 2026-10-20 from `PLAN_grilled.md` and a read of the code. `PLAN_grilled.md` holds the
mathematics and the pre-registered calibration; this file says what gets built and how it is
tested. Where the two disagree on the mathematics, `PLAN_grilled.md` wins, except for the points in
"Discrepancies found in the code" below, where the code wins.*

## Problem Statement

Students and members of the ACS group who study steering read that "a concept is a direction in
activation space and adding it changes behaviour", but they have no way to *see* it on a model they
can inspect end to end. The existing GuppyLM experiments only exist as notebooks and JSON files,
and Check III showed that the one steering regime ever run (all 6 blocks, all positions,
$|r| \ge 0.25$) gives zero coherent generations. So right now there is nothing to show except word
salad, and nobody knows whether a smaller, better-placed intervention gives fluent, on-concept text.

## Solution

A private, link-shared static web page. It runs GuppyLM entirely in the browser and shows two
answers to the same prompt side by side: the unsteered model ($r = 0$) and the model steered
along $v^{(c,m)}_\ell$ with strength $r$. The user controls the concept, the vector source
(RFM / DiffMeans / random), the block set, the position set, decoding and the seed. The page shows
where the coherent range of $r$ was measured to be, the effective dose $\rho$, the update rule with
its current numbers, and two scores that do not read the steered activations: a clean re-read and
an activation-blind judge.

Before the page is built, a pre-registered calibration on the Python side finds out whether such a
coherent range exists. The page is built either way, and its wording changes depending on the
result.

## User Stories

1. As a student, I want to type my own prompt and get two answers side by side, so that I can
   compare steered and unsteered behaviour on input I chose.
2. As a student, I want the left answer always to be the unsteered model, so that I have a fixed
   reference.
3. As a student, I want a slider over $r$ that is fine near 0 and coarse far from it, so that I can
   explore the small-$|r|$ range where coherent steering might happen.
4. As a student, I want generation to start when I release the slider, not while I drag it, so that
   the page stays responsive.
5. As a student, I want tokens to appear as they are generated in both columns, so that I don't
   stare at a blank page.
6. As a student, I want to choose the concept (`food`, `valence`, `env_social`), so that I can see
   that different directions push in different directions.
7. As a student, I want to switch the vector source between RFM, DiffMeans and a random direction,
   so that I can see whether the concept direction does anything a random direction doesn't.
8. As a student, I want to steer a single block or all blocks, so that I can see how the
   intervention accumulates in the residual stream.
9. As a student, I want to steer the prompt positions only, the generated positions only, or all of
   them, so that I can see where in the sequence the intervention matters.
10. As a student, I want to see the effective dose $\rho$ for my current setting, so that I can
    compare single-block and all-block settings on one scale.
11. As a student, I want to see the update rule written out with the current
    $\varepsilon_\ell$ and $\operatorname{med}_\ell$, so that I can connect the slider to the
    formula.
12. As a student, I want the slider track to shade the measured coherent range for my current
    $(c, m, S, P)$, so that I know where fluent text is expected.
13. As a student, I want the shading labelled as exploratory (7 prompts), so that I don't take it as
    a confirmed result.
14. As a student, I want a tick at $r^\star$ when my current setting is a confirmed cell, with
    $k/20$ and the McNemar $p$ in a tooltip, so that I can tell confirmed from exploratory claims.
15. As a student, I want a per-token strip under each answer showing the clean re-read projection,
    so that I can see where the text sits in concept space without reading the injected vector
    back.
16. As a student, I want the clean re-read to use the concept direction even when I steer with a
    random direction, so that the strip always measures the concept.
17. As a student, I want the activation-blind judge score $g_c$ for each answer, so that I have a
    second opinion that never looks at activations.
18. As a student, I want `valence` to carry a "judge not validated" badge and a greyed-out $g_c$, so
    that I don't read a score that failed its validity gate.
19. As a student, I want sampled decoding by default with a visible seed, so that what I see matches
    the experiments and I can reproduce it.
20. As a student, I want a greedy option, so that I can remove sampling noise from the comparison.
21. As a student, I want both columns to use the same seed, so that any difference comes from
    steering, not from the random stream.
22. As a student, I want a live token counter on the prompt box and Generate disabled above
    56 tokens, so that I can't overflow the 128-token context.
23. As a student, I want a "Check III setup" preset, so that I can see the regime that failed.
24. As a student, I want a "Best regime" preset, so that I can jump to the best setting the
    calibration found.
25. As a student, I want a "Random control" preset with the same $(S, P, r)$ as the best regime, so
    that I can check the result against a random direction.
26. As a student, I want each preset to show a one-line caption, so that I know what it shows.
27. As a student, I want honest wording when no cell was confirmed, so that the page does not
    overclaim.
28. As a presenter at the ACS group meeting, I want the page to work from a link without a server,
    so that I can open it on any machine.
29. As a presenter, I want the page to load the model in a few seconds and generate 64 tokens in
    about a second, so that a live demo doesn't stall.
30. As a presenter, I want the page private and shared by link, so that I control who sees it.
31. As the researcher, I want the calibration pre-registered with a falsification criterion, so that
    the demo's claims are backed by a test that could have failed.
32. As the researcher, I want the calibration written up whatever the result, so that a negative
    result is recorded, not lost.
33. As the researcher, I want the page's bands and ticks to be generated from the calibration
    output, so that the page and `CALIBRATION.md` cannot disagree.
34. As the researcher, I want to add a new concept by running the pipeline, the calibration and the
    export, then dropping in one file, so that the UI code never changes for a new concept.
35. As the researcher, I want the browser model to match PyTorch to $10^{-3}$ on logits and exactly
    on greedy text, so that what students see is what the Python experiments measured.
36. As the researcher, I want the Python steering wrapper to reproduce the existing
    `steering_hooks` output character for character when $P = P_{\text{all}}$, so that the new
    position masks don't silently change the Check III operation.
37. As the researcher, I want a precomputed static fallback page if the JS forward pass fails its
    gate, so that the demo still exists.

## Implementation Decisions

**Modules.**

- **Python reference generator (Seam A).** A small module in the demo folder, next to
  `rfm_guppy` without changing it. Public interface:
  `generate(prompt, model, tokenizer, vectors, eps, positions ∈ {all, prompt, gen},
  max_new_tokens, seed, temperature, top_k, greedy) → text`, plus a helper that builds `eps` and
  the vectors for a $(c, m, S, r)$. It uses its own position-masked forward hooks and its own
  decoding loop that mirrors `GuppyLM.generate` step for step (same ops, same RNG calls), with an
  argmax branch for greedy. It is the only source of truth: calibration, export and the JS golden
  fixtures all call it.
- **Calibration (step 1).** A script that freezes the prompt lists to a file first, then runs the
  exploration grid, the bands, the selection and the confirmation of §8.1 of the plan. Outputs:
  `calibration.json` (raw grid, bands, ticks) and `CALIBRATION.md`.
- **Export (step 2).** A script that writes: fp32 weights in 3 binary files of at most 15 MB each,
  plus a JSON manifest (tensor name → file, offset, shape); the tokenizer; `shared.json`
  (`med[6]`, `u[6×384]`); one `concept_<name>.json` per concept in the schema of §5 of the plan.
  Bands and ticks are read from `calibration.json`, never typed by hand.
- **JS engine (Seam B).** A DOM-free ES module. Interface:
  `load(bundleUrls)`, `tokenize(text)`, `logits(ids, {useCache})`,
  `generate(prompt, {concept, source, blocks, positions, r, seed, greedy, maxNewTokens, onToken})`,
  `judge(concept, text)`, `cleanReread(concept, source, prompt, y)`, `dose(concept, source, blocks, r)`.
  Inside: byte-level BPE tokenizer, hand-written forward pass with KV cache, a seeded PRNG, the
  steering term as one masked addition per block, the TF-IDF + logistic regression judge.
- **UI (step 5).** One HTML page that only calls the engine. Two columns, the controls of §7 of the
  plan, the formula panel, the band and tick on the slider track, the re-read strip, the presets.
  Published as a private Artifact (static page, weights as supporting files).

**The steering operation.** Exactly §3 of the plan:
$h_{\ell,t} \mapsto h_{\ell,t} + \mathbb 1[t \in P]\, r\,\operatorname{med}_\ell\, v^{(c,m)}_\ell$
for $\ell \in S$, with $P_{\text{prompt}} = \{t < T_0 - 1\}$ and $P_{\text{gen}} = \{t \ge T_0 - 1\}$.
In the Python reference the mask is applied with a select (`where`), not by multiplying by 0/1, so
masked-in positions are bitwise equal to `output + ε·v`.

**No KV cache in Python, KV cache in JS.** `GuppyLM.generate` recomputes the full sequence at every
step. Because attention is causal and the mask depends only on absolute position, recomputing the
prompt steered at every step gives the same states as computing it once and caching it. The JS
cache is therefore an optimisation that must reproduce the Python numbers, not a different
operation. The Python reference asserts $T_0 + \text{max\_new\_tokens} \le 128$, so the
`idx[:, -128:]` crop in `GuppyLM.generate` never shifts positions.

**Random directions.** `u` is drawn once with a fixed seed (Gaussian, normalised per block) and
stored in `shared.json`; the confirmation's 5 random controls use 5 further fixed seeds.

**DiffMeans vectors.** `rfm_guppy/baselines.diffmeans_vector`, oriented and unit-normalised as
there, computed per block from the cached training activations.

**Decoding.** Sampled, T = 0.7, top-$k$ = 50, seed 42 by default; greedy optional. The browser's
sampler cannot reproduce `torch.multinomial`, so sampled JS text is *not* expected to match Python;
only greedy text is compared. Calibration uses 60 new tokens (as in Check III); the page uses 64.

**Weights format.** fp32, row-major, little-endian, tied `lm_head` stored once.

## Testing Decisions

**What a good test is here.** Tests go through the two public seams only and compare outputs to a
reference. KV cache, BPE merges and the TF-IDF tokeniser are never tested directly; they are tested
by the engine's outputs being equal to Python's. Every gate in §8 of the plan is a test that runs
from the command line and prints PASS/FAIL.

**Seam A — Python reference.**
- With $P = P_{\text{all}}$: the text equals `generate_steered` character for character on the 7
  Check III prompts, for every concept and for several $r$ (including the stored Check III values).
- With $r = 0$: the text equals the unhooked model, and equals the stored $r = 0$ texts of
  `results/steer_<c>.json` (Check III's G3).
- $P_{\text{prompt}}$ and $P_{\text{gen}}$ partition the positions: the per-position deltas of the
  two runs add up to the delta of $P_{\text{all}}$ on one forward pass, and are disjoint.
- Export round-trip: reloaded tensors are identical to the checkpoint.

**Seam B — JS engine (Node, against golden fixtures written by Seam A).**
- Tokenizer: same ids as `tokenizers` on the 20 + 7 prompts.
- Logits on 20 prompts: $\max_i |z^{\text{JS}}_i - z^{\text{PT}}_i| < 10^{-3}$; cache and no-cache
  logits agree.
- Greedy generations identical to Python, unsteered on 20 prompts and steered on 5 cells covering
  all three position sets.
- $g_c$ equal to sklearn's to $10^{-6}$ on all fixture texts.

**UI.** Checked by hand in the browser; no automated tests.

**Prior art.** The repo has no test suite. The closest prior art is the gate checks in the
`revision_20261002` notebooks (Check III's G2 and G3: recompute and compare character for
character). The tests here follow that style as plain scripts.

## Out of Scope

- Any change to `rfm_guppy/` or to the GuppyLM checkpoint.
- Adding the calibration to `SUMMARY_OF_TESTS.md` or to the thesis.
- New concepts beyond the existing three (the page supports them, but none are added).
- A server, accounts, or any shared state on the page.
- Matching sampled (non-greedy) text between JS and Python.
- Using the existing ONNX export: it has no hook point at block outputs.
- Interpreting `valence` results (G1 failed).

## Discrepancies found in the code

These came up while reading the code for this spec. In each case the spec follows the code, because
the parity gates compare against it.

1. **Judge tokenisation.** The plan (§4) says `token_pattern = (?u)\b\w\w+\b`. The Check III judge
   uses `(?u)\b\w+\b` (single-character tokens are kept) and `sublinear_tf=True`. The JS judge must
   copy the latter, including the sublinear TF $1 + \log(\text{tf})$ and L2 normalisation.
2. **What $\operatorname{med}_\ell$ is computed on.** The plan (§3) says "all cached training
   activations". `measure_activation_norms` reads every `*_activations.npy`, i.e. the train, val
   and test splits of all three concepts. The values used in Check III come from that function, so
   the spec keeps it; the plan's wording should be corrected.

## Further Notes

- Steps, gates and hour estimates are those of §8 of the plan (≈ 22–27 h). Each step stops the
  chain if its gate fails; if the JS gate (step 3) fails, fallback A (precomputed static page) is
  built from the calibration outputs.
- Open minor items: preset captions, colour scale of the re-read strip.
