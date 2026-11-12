# replication: Beaglehole et al. on GuppyLM

## Purpose

This folder replicates the method of Beaglehole et al., *Toward Universal Steering and Monitoring of AI Models* ([arXiv:2502.03708](https://arxiv.org/abs/2502.03708)), on GuppyLM (8.73M parameters): RFM concept-vector extraction, additive steering, AGOP-eigenvector monitoring, and a comparison with PCA, DiffMeans and logistic regression. It was run phase by phase against the pre-registered plan in [`plans/GUPPYLM_GUIDE.md`](plans/GUPPYLM_GUIDE.md). Later controls on its claims are in [`../rfm_checks/README.md`](../rfm_checks/README.md).

## Layout

| Item | What it holds |
|---|---|
| [`01_extract.py`](01_extract.py) .. [`04_baselines.py`](04_baselines.py) | The pipeline scripts, one per stage (table below) |
| [`notebooks/`](notebooks/) | Executed notebooks, one per script, plus `nbcommon.py` (shared loaders). They recompute the cheap parts in memory, check them against the stored files, and write nothing to disk. |
| [`plans/`](plans/) | [`GUPPYLM_GUIDE.md`](plans/GUPPYLM_GUIDE.md), the pre-registered plan with acceptance criteria; [`phase_plans/`](plans/phase_plans/), the execution notes for each phase |
| [`runs/2026-08-27/`](runs/2026-08-27/) | Outputs of the scripts and the hand-written [`REPORT.md`](runs/2026-08-27/REPORT.md) |

## Phases, scripts, notebooks and outputs

Phases follow the headings of `plans/GUPPYLM_GUIDE.md` (sections 3 to 11). The guide still uses the old names `experiments/` and `results/`; they are now `replication/` and `replication/runs/2026-08-27/`.

| Phase | Script | Notebook | Output files |
|---|---|---|---|
| 0. Setup | none | none | none |
| 1. Activation extraction harness (`rfm_guppy/activations.py`) | used by `01_extract.py` | [`01_extract.ipynb`](notebooks/01_extract.ipynb) | none of its own |
| 2. Concept datasets (`rfm_guppy/concepts.py`) | `python -m rfm_guppy.concepts` (the module's `__main__`) | [`01_extract.ipynb`](notebooks/01_extract.ipynb) | `data/cache/concept_<c>.json` |
| 3. RFM implementation (`rfm_guppy/rfm.py`) | synthetic gate run at the start of `01_extract.py` | [`01_extract.ipynb`](notebooks/01_extract.ipynb) | none |
| 4. Extract and store concept vectors | [`01_extract.py`](01_extract.py) | [`01_extract.ipynb`](notebooks/01_extract.ipynb) | `data/cache/<c>_{train,val,test}_activations.npy`, `data/cache/<c>_vectors.npz` |
| 5. Steering | [`02_steer.py`](02_steer.py) | [`02_steer.ipynb`](notebooks/02_steer.ipynb) | [`steer_<c>.json`](runs/2026-08-27/steer_food.json), `steer_<c>.md` |
| 6. Steering evaluation | [`03_monitor.py`](03_monitor.py) | [`03_monitor.ipynb`](notebooks/03_monitor.ipynb) | [`eval6_<c>.json`](runs/2026-08-27/eval6_food.json), `eval6_<c>.md` |
| 7. Monitoring / probing | [`03_monitor.py`](03_monitor.py) | [`03_monitor.ipynb`](notebooks/03_monitor.ipynb) | [`monitor_auroc.json`](runs/2026-08-27/monitor_auroc.json), `data/cache/<c>_probe.pkl` |
| 8. Baselines and comparison | [`04_baselines.py`](04_baselines.py) | [`04_baselines.ipynb`](notebooks/04_baselines.ipynb) | [`baselines_monitoring.json`](runs/2026-08-27/baselines_monitoring.json), [`baselines_steering.json`](runs/2026-08-27/baselines_steering.json), [`baselines_comparison.md`](runs/2026-08-27/baselines_comparison.md) |
| 9. Report | none (written by hand) | none | [`REPORT.md`](runs/2026-08-27/REPORT.md) |

`<c>` is one of `food`, `valence`, `env_social`. Files without a `data/cache/` prefix are in `runs/2026-08-27/`.

## How to rerun

From the repo root, with the environment from `requirements.txt` in `.venv/`:

```bash
# checkpoint: every script loads it, so it must be in data/checkpoints/guppylm-9M
hf download arman-bd/guppylm-9M --local-dir data/checkpoints/guppylm-9M

.venv/bin/python -m rfm_guppy.concepts        # Phase 2: writes data/cache/concept_<c>.json
.venv/bin/python replication/01_extract.py    # Phases 1, 3, 4
.venv/bin/python replication/02_steer.py      # Phase 5
.venv/bin/python replication/03_monitor.py    # Phases 6, 7
.venv/bin/python replication/04_baselines.py  # Phase 8
```

- The `hf download` command is the one printed by `notebooks/nbcommon.py` when the checkpoint is missing.
- The scripts find the repo root themselves, so `PYTHONPATH=.` (shown in their docstrings) is not needed.
- `01_extract.py` reads `data/cache/concept_<c>.json` and does not create it. On an empty `data/cache/`, run `python -m rfm_guppy.concepts` first. The root README's "Getting started" list does not include this step.
- `rfm_guppy/concepts.py` loads `arman-bd/guppylm-60k-generic` from Hugging Face; if that fails, it falls back to `guppylm/data/train.jsonl`.
- Whether a fresh rebuild of the concept datasets gives the same splits as the stored ones is not recorded.

### Warning: a rerun overwrites the stored evidence

`RESULTS_DIR` in [`../rfm_guppy/paths.py`](../rfm_guppy/paths.py) is pinned to `replication/runs/2026-08-27/`. The scripts write there directly, with no new folder per run:

| Script | Writes to `runs/2026-08-27/` | Writes to `data/cache/` |
|---|---|---|
| `01_extract.py` | nothing | `<c>_vectors.npz` (always overwritten); `<c>_{split}_activations.npy` (only if missing) |
| `02_steer.py` | `steer_<c>.json`, `steer_<c>.md` (overwritten) | nothing |
| `03_monitor.py` | `eval6_<c>.json`, `eval6_<c>.md`, `monitor_auroc.json` (overwritten) | `<c>_probe.pkl` (always overwritten; `load_probe_if_cached` is defined but never called) |
| `04_baselines.py` | `baselines_monitoring.json`, `baselines_steering.json`, `baselines_comparison.md` (overwritten) | nothing |

Rerunning any script replaces the files that `REPORT.md`, the paper, the checks in `rfm_checks/` and the demo were built on. `rfm_guppy/concepts.py` also overwrites `data/cache/concept_<c>.json`. The checks gate against `data/cache/<c>_vectors.npz`, so a changed cache can break their reproduction gates.

Safe practice until there are per-run folders:

- `runs/2026-08-27/` is tracked by git. Commit or stash your work first; after a rerun, `git status` shows what changed, and the old files can be restored from git.
- `data/` is git-ignored. Copy it before a rerun, for example `cp -R data/cache data/cache.bak-$(date +%F)`.
- Better: copy `runs/2026-08-27/` to a new dated folder and change `RESULTS_DIR` to point at it before running. That is a code change and is not made here.

## Known issues found later

- **The Phase 8 steering table is a broken measurement, not a tie.** The English paper says: "The raw steering table reports 100\% steering success for all four methods, which reads like a four-way tie but is in fact a broken measurement" ([`../writing/paper_en/tex_files/03_replication.tex`](../writing/paper_en/tex_files/03_replication.tex), lines 438–440). The reasons given there are that the probe is built from the RFM eigenbasis and that success is "probe > 0.5 or lexicon hit".
- **The PCA baseline was mis-specified (Check II).** `pca_vector` centres the paired-difference matrix before the SVD, which removes exactly the DiffMeans direction. Without centring, the mean test AUROC of PCA goes from 0.879 to 0.998.
- **The steering metric is partly circular, and no steered text is coherent (Check III).** The old success metric reads the same AGOP basis the steering vector comes from. A blind text judge gives partial success at best, and there are 0 coherent generations at any r ≠ 0.

Details, criteria and evidence for these checks: [`../rfm_checks/README.md`](../rfm_checks/README.md).
