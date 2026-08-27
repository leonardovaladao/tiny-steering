# Steering a Fish

**Replicating the core mechanisms of *Toward Universal Steering and Monitoring of AI Models* on an 8.7M-parameter language model.**

This repository is a controlled, from-scratch replication of the *method* introduced in
[Beaglehole, Radhakrishnan, Boix-Adserà & Belkin (2025), *Toward Universal Steering and Monitoring of AI Models*](https://arxiv.org/abs/2502.03708) —
RFM concept-vector extraction, additive activation steering, and AGOP-eigenvector probing —
applied to [**GuppyLM**](https://github.com/arman-bd/guppylm), a ~8.7M-parameter chatbot that role-plays as a small, opinionated pet fish.

That is roughly **8,000× smaller** than the Llama-3.3-70B models the paper's headline results were produced on.

---

## Where the idea came from

The paper makes a strong and appealing claim about how language models store what they know:

1. **Concepts are linear directions.** A concept the model has learned — a mood, a topic, a stance — corresponds to a direction in the residual stream of each transformer block.
2. **Those directions can be extracted cheaply and supervisedly.** Using the **Recursive Feature Machine** (RFM; Radhakrishnan et al., *Science* 2024) — kernel ridge regression plus the **Average Gradient Outer Product** (AGOP) — the concept vector falls out as the top eigenvector of the learned AGOP matrix `M_T`. No text outputs are needed, only `(prompt, label)` pairs.
3. **Once you have the direction, you get two capabilities for free.**
   - **Steering:** add `ε·v_ℓ` to every block's output during the forward pass and the generations move toward (or, with `−ε`, away from) the concept.
   - **Monitoring:** project activations onto the top AGOP eigenvectors, train a small classifier, and detect whether the concept is active — reading the model's internals rather than its text.

On Llama-class models (8B–90B) the authors use this to expose jailbreaks, impose political stances, suppress deception, transfer concepts across languages, and build hallucination and toxicity monitors that outperform GPT-4o-as-judge. Those are remarkable results, but they are all reported at a scale where nothing can be inspected end-to-end, re-derived by hand, or re-run on a laptop.

That gap is where this project started. Reading the paper as a *method* rather than as a *result*, the natural question for someone who wants to genuinely understand the machinery is:

> **Is the mechanism itself doing the work, or is it borrowing its power from the fact that the model is enormous?**

The paper does not isolate this: its smallest models are still billions of parameters, and every concept it studies (refusal, deception, hallucination) is an emergent property that only exists at scale. If the pipeline were run on a model small enough to hold entirely in one's head, the mechanism would either survive on its own merits or it would not — and either answer is informative.

GuppyLM is what makes that experiment possible. It is deliberately tiny, fully open, and genuinely a real instruction-tuned transformer: 6 pre-norm blocks, hidden dimension 384, a 4,096-token vocabulary, ChatML prompt format, and a real residual stream `[B, T, 384]` at every block — which is exactly the object the paper's method reads from and writes to. It is small enough that every activation, every hook, and every eigenvector can be checked directly, and fast enough that the full pipeline runs on CPU in minutes.

Crucially, GuppyLM was trained on ~60k conversations across **60 labelled categories** (food, tank temperature, greetings, moods, filters, weather, …). Those categories are the *only* concepts it can plausibly represent — and they come with ground-truth labels for free. So they become our concepts.

---

## The main goal

> **Does linear concept extraction + additive steering + activation probing hold at fewer than 10M parameters, when restricted to concepts the model actually represents?**

The objective is a **mechanism replication**, not a safety replication. Concretely, the project aims to:

- **Re-implement RFM/AGOP from the mathematics in the paper**, not from the authors' code, and gate it on a synthetic recovery test before it ever touches real activations.
- **Build honest concept datasets** from GuppyLM's own training categories, with class balance and no prompt leakage across train/val/test.
- **Extract per-block concept vectors** and examine how linear the learned concepts actually are (the AGOP eigenvalue spectrum).
- **Steer generation** by adding those vectors to the residual stream, with the steering coefficient **recalibrated to GuppyLM's own activation scale** rather than copied from the paper's Llama-specific values.
- **Monitor** by probing the AGOP eigenbasis and measuring AUROC on held-out prompts.
- **Compare RFM against the paper's three baselines** — PCA, difference-in-means, and logistic regression — under identical conditions.

### In scope

RFM concept-vector extraction · additive activation steering · AGOP-eigenvector probing · the four-way baseline comparison.

### Deliberately out of scope

Anti-refusal and jailbreaks, deception, political stances (a fish chatbot has no such representations); cross-language transfer, coding, and reasoning (capabilities absent from a 4,096-token fish-domain model); hallucination and toxicity benchmarks such as FAVABENCH, HaluEval and ToxicChat (the tokenizer cannot even encode them); and the paper's **scaling** claim, which requires a model ladder and is a separate project.

Attempting any of these on GuppyLM would be measuring noise and presenting it as a finding.

### Working principles

The project was executed phase by phase against **pre-declared acceptance criteria**, written before any code, with a standing rule that a phase does not advance until its criteria pass. The guardrails that shaped the work:

- **A weak or negative result is a legitimate finding.** Nothing is tuned until success appears and then presented as typical.
- **Nothing is borrowed blindly from the paper's setup** — bandwidths and steering coefficients are recalibrated to this model's activation scale.
- **The implementation is gated on a synthetic test** before real concept vectors are trusted.
- **Everything is deterministic and cached**, so every phase is re-runnable without recomputation.

That commitment is on the record rather than asserted after the fact, and the two halves of it are kept deliberately separate:

- **[`plans/GUPPYLM_GUIDE.md`](plans/GUPPYLM_GUIDE.md)** — the **pre-registered plan**: the full method derivation, the nine phases, and the acceptance criteria each one had to meet. Written before a line of code, and frozen since.
- **[`plans/phase_plans/`](plans/phase_plans/)** — the **execution notes**, one file per phase: what was actually built, which criteria passed and with what evidence, and every decision taken under uncertainty — including the ones that were wrong and had to be reversed.

Reading them side by side shows what was intended versus what happened. The condensed version of the second is [`WORK_LOG.md`](WORK_LOG.md); the scientific write-up is [`results/REPORT.md`](results/REPORT.md).

---

## Getting started

```bash
# GuppyLM's own source lives in guppylm/ as a submodule — clone it too
git clone --recurse-submodules https://github.com/leonardovaladao/tiny-steering.git
cd tiny-steering
# (already cloned without the flag? run: git submodule update --init)

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# fetch the pretrained model into checkpoints/guppylm-9M/
hf download arman-bd/guppylm-9M --local-dir checkpoints/guppylm-9M

# run the pipeline in order — every stage caches, so re-runs skip inference
python experiments/01_extract.py     # concept datasets + RFM concept vectors
python experiments/02_steer.py       # steering sweep
python experiments/03_monitor.py     # monitoring probes + steering evaluation
python experiments/04_baselines.py   # PCA / DiffMeans / LogReg comparison
```

### Reading it rather than running it

Every experiment also exists as an executed notebook in [`notebooks/`](notebooks/), with the plots,
tables and generated text already rendered — so the whole replication can be read end to end
without installing anything or downloading the checkpoint:

| Notebook | Phases | What it shows |
|---|---|---|
| [`01_extract.ipynb`](notebooks/01_extract.ipynb) | 1–4 | the synthetic gate, AGOP convergence, activation-norm growth, val `\|ρ\|` per block, AGOP eigenvalue spectra |
| [`02_steer.ipynb`](notebooks/02_steer.ipynb) | 5 | ε calibration, the steering hook itself, hit-rate and collapse curves, side-by-side generations |
| [`03_monitor.ipynb`](notebooks/03_monitor.ipynb) | 6–7 | AGOP-projection scatter, per-block AUROC, ROC curves, probe-scored steering |
| [`04_baselines.ipynb`](notebooks/04_baselines.ipynb) | 8 | four-way `\|ρ\|` comparison, direction agreement, monitoring AUROC, a like-for-like feature-count control |

The scripts and the notebooks are two views of the same pipeline, not two pipelines: the notebooks
import the same `rfm_guppy` modules, recompute what is cheap (RFM, probes, baseline vectors) and
assert that the results match the committed artifacts, and load the expensive parts (the steering
sweeps) from `results/`. None of them writes to `rfm_guppy/cache/`, so opening one can never
change the numbers the report is built on. Keep `experiments/*.py` for headless or scripted runs.

Every experiment script imports the model definition with `from guppylm import GuppyLM, GuppyConfig`, so the submodule is required, not optional. It is pinned at upstream commit [`a30df30`](https://github.com/arman-bd/guppylm/commit/a30df3091cff73a259ce581dd8439271084bca40) — the exact revision every number in [`results/`](results/) was produced against. Neither the downloaded checkpoint (`checkpoints/`) nor the cached activations and vectors (`rfm_guppy/cache/`) are tracked here; both are regenerated by the commands above.

**Everything runs on CPU — no GPU is required.** GuppyLM is small enough (8.7M parameters, sequences ≤128 tokens) that the largest activation-extraction split takes well under a second, and a full 6-block RFM hyperparameter sweep takes seconds per concept.

### Dependencies

[`requirements.txt`](requirements.txt) pins the full set; the substantive ones are:

| Package | Used for |
|---|---|
| `torch` | GuppyLM inference; the `forward_hook` API that both reads activations and writes the steering perturbation |
| `tokenizers` | GuppyLM's BPE tokenizer |
| `numpy` | activations, AGOP matrices, eigendecomposition |
| `scipy` | `cdist` for Mahalanobis distances under `M`; Pearson correlation |
| `scikit-learn` | logistic-regression probes, `StandardScaler`, AUROC |
| `datasets` | loads `arman-bd/guppylm-60k-generic` to build the concept sets |
| `huggingface_hub` | downloads the `arman-bd/guppylm-9M` checkpoint |

`matplotlib`, `pandas` and the Jupyter packages are needed only by [`notebooks/`](notebooks/) — nothing in `experiments/` or `rfm_guppy/` imports them.

The environment this was developed and run on: Python 3.14, torch 2.12, numpy 2.5, scipy 1.18, scikit-learn 1.9, tokenizers 0.23, datasets 5.0, on macOS.

---

## References

- Beaglehole, Radhakrishnan, Boix-Adserà & Belkin. *Toward Universal Steering and Monitoring of AI Models.* [arXiv:2502.03708v2](https://arxiv.org/abs/2502.03708) (2025).
- Radhakrishnan, Beaglehole, Pandit & Belkin. *Mechanism for feature learning in neural networks and backpropagation-free machine learning models.* *Science* 383(6690) (2024).
- Reference implementation of the original method: [`dmbeaglehole/neural_controllers`](https://github.com/dmbeaglehole/neural_controllers).
- GuppyLM: [`arman-bd/guppylm`](https://github.com/arman-bd/guppylm) · model [`arman-bd/guppylm-9M`](https://huggingface.co/arman-bd/guppylm-9M) · dataset [`arman-bd/guppylm-60k-generic`](https://huggingface.co/datasets/arman-bd/guppylm-60k-generic).

---

*Master's research, Applied Mathematics, Universidade de São Paulo.*
