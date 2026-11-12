# replication: Beaglehole et al. on GuppyLM

## Purpose

This folder replicates the method of Beaglehole et al., *Toward Universal Steering and Monitoring of AI Models* ([arXiv:2502.03708](https://arxiv.org/abs/2502.03708)), on GuppyLM (8.73M parameters): RFM concept-vector extraction, additive steering, AGOP-eigenvector monitoring, and a comparison with PCA, DiffMeans and logistic regression. It was run phase by phase against the pre-registered plan in [`plans/GUPPYLM_GUIDE.md`](plans/GUPPYLM_GUIDE.md). Later controls on its claims are in [`../rfm_checks/README.md`](../rfm_checks/README.md).

## Layout

| Item | What it holds |
|---|---|
| [`01_extract.py`](01_extract.py) .. [`04_baselines.py`](04_baselines.py) | The pipeline scripts, one per stage (table below) |
| [`notebooks/`](notebooks/) | Executed notebooks, one per script, plus `nbcommon.py` (shared loaders). They recompute the cheap parts in memory, check them against the stored files, and write nothing to disk. |
| [`plans/`](plans/) | [`GUPPYLM_GUIDE.md`](plans/GUPPYLM_GUIDE.md), the pre-registered plan with acceptance criteria; [`phase_plans/`](plans/phase_plans/), the execution notes for each phase |
| [`runs/2026-08-27/`](runs/2026-08-27/) | Outputs of the scripts and the hand-written [`REPORT.md`](runs/2026-08-27/REPORT.md). Pinned and read-only: the scripts refuse to write here |
| `runs/<date>/` | New runs of the scripts, each with a `run.json` provenance record (see [How to rerun](#how-to-rerun)) |

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

`<c>` is one of the concepts in `CONCEPTS` of [`../rfm_guppy/concepts.py`](../rfm_guppy/concepts.py): `food`, `valence`, `env_social`. Files without a `data/cache/` prefix are in the run folder; for the pinned run that is `runs/2026-08-27/`.

## How to rerun

From the repo root, with the environment from `requirements.txt` in `.venv/`:

```bash
# checkpoint: every script loads it, so it must be in data/checkpoints/guppylm-9M
hf download arman-bd/guppylm-9M --local-dir data/checkpoints/guppylm-9M

RUN=replication/runs/$(date +%F)                         # a new run folder
.venv/bin/python -m rfm_guppy.concepts                   # Phase 2: writes data/cache/concept_<c>.json
.venv/bin/python replication/01_extract.py --run $RUN    # Phases 1, 3, 4 (writes data/cache/ only)
.venv/bin/python replication/02_steer.py --run $RUN      # Phase 5
.venv/bin/python replication/03_monitor.py --run $RUN    # Phases 6, 7
.venv/bin/python replication/04_baselines.py --run $RUN  # Phase 8
```

Each command takes `--concepts NAME [NAME ...]` to run on a subset of the registered concepts (default: all of them, in registry order); an unknown name stops the script with the list of valid ones. With the defaults the scripts reproduce `runs/2026-08-27/` byte for byte (checked on 2026-11-12 against an unchanged `data/cache/`). `--help` on each script lists its options.

- The `hf download` command is the one printed by `notebooks/nbcommon.py` when the checkpoint is missing.
- The scripts find the repo root themselves, so `PYTHONPATH=.` is not needed.
- `01_extract.py` reads `data/cache/concept_<c>.json` and does not create it. On an empty `data/cache/`, run `python -m rfm_guppy.concepts` first.
- `rfm_guppy/concepts.py` loads `arman-bd/guppylm-60k-generic` from Hugging Face; if that fails, it falls back to `guppylm/data/train.jsonl`.
- A rebuild of `concept_valence.json` (`python -m rfm_guppy.concepts --concepts valence`, 2026-11-12) was byte-identical to the stored file. Whether `food` and `env_social` rebuild identically is not recorded.

### Run folders are append-only

`02_steer.py`, `03_monitor.py` and `04_baselines.py` write to the run folder given by `--run DIR`. `02_steer.py` defaults to a new `replication/runs/<today>/` (or `<today>_2`, `_3`, ... if that folder is already in use); `03_monitor.py` and `04_baselines.py` require `--run`, because they read the previous phase's outputs from that folder (`steer_<c>.json` for 03; `monitor_auroc.json` and `eval6_<c>.json` for 04). `01_extract.py` writes only to `data/cache/`; its optional `--run` only records the step.

- A script never overwrites a file that already exists in the run folder. It checks before doing any work and stops with a `FileExistsError`; `--force` allows the overwrite.
- `runs/2026-08-27/` (`RESULTS_DIR` in [`../rfm_guppy/paths.py`](../rfm_guppy/paths.py)) and every existing folder under `rfm_checks/runs/` are refused as run folders, even with `--force`.
- Each invocation appends a step to `run.json` in the run folder: script, argv, start time, concepts, parameters, git commit, whether the working tree was dirty, Python and numpy/torch/scikit-learn versions.
- `monitor_auroc.json` and the three `baselines_*` files cover all the concepts of one invocation. Run 03 and 04 once per run folder with the full concept list, or use a new run folder for another subset.
- The ε calibration in `02_steer.py` and `04_baselines.py` takes the median over every `data/cache/*_activations.npy`, not only the concepts passed with `--concepts`. Adding a concept to the cache therefore changes ε for all concepts.

### Warning: a rerun overwrites `data/cache/`

| Script | Writes to the run folder | Writes to `data/cache/` (for the concepts passed) |
|---|---|---|
| `python -m rfm_guppy.concepts` | nothing | `concept_<c>.json` (always overwritten) |
| `01_extract.py` | `run.json` only, with `--run` | `<c>_vectors.npz` (always overwritten); `<c>_{split}_activations.npy` (only if missing) |
| `02_steer.py` | `steer_<c>.json`, `steer_<c>.md` | nothing |
| `03_monitor.py` | `eval6_<c>.json`, `eval6_<c>.md`, `monitor_auroc.json` | `<c>_probe.pkl` (always overwritten; `load_probe_if_cached` is defined but never called) |
| `04_baselines.py` | `baselines_monitoring.json`, `baselines_steering.json`, `baselines_comparison.md` | nothing |

`data/` is git-ignored, so the cache cannot be restored from git. The checks in `rfm_checks/` gate against `data/cache/<c>_vectors.npz`, so a changed cache can break their reproduction gates. Copy it before a rerun, for example `cp -R data/cache data/cache.bak-$(date +%F)`.

## Known issues found later

- **The Phase 8 steering table is a broken measurement, not a tie.** The English paper says: "The raw steering table reports 100\% steering success for all four methods, which reads like a four-way tie but is in fact a broken measurement" ([`../writing/paper_en/tex_files/03_replication.tex`](../writing/paper_en/tex_files/03_replication.tex), lines 438–440). The reasons given there are that the probe is built from the RFM eigenbasis and that success is "probe > 0.5 or lexicon hit".
- **The PCA baseline was mis-specified (Check II).** `pca_vector` centres the paired-difference matrix before the SVD, which removes exactly the DiffMeans direction. Without centring, the mean test AUROC of PCA goes from 0.879 to 0.998.
- **The steering metric is partly circular, and no steered text is coherent (Check III).** The old success metric reads the same AGOP basis the steering vector comes from. A blind text judge gives partial success at best, and there are 0 coherent generations at any r ≠ 0.

Details, criteria and evidence for these checks: [`../rfm_checks/README.md`](../rfm_checks/README.md).
