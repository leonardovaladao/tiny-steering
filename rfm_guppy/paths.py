"""Every location in the repository, defined here and nowhere else.

Scripts and notebooks import these constants instead of building paths themselves, so moving a
folder means changing one line in this file.
"""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root

GUPPYLM_DIR   = os.path.join(ROOT, "guppylm")                # git submodule; put on sys.path
NOTEBOOKS_DIR = os.path.join(ROOT, "notebooks")              # holds nbcommon.py
DATA_DIR      = os.path.join(ROOT, "data")                   # git-ignored; regenerable artifacts
CKPT_DIR      = os.path.join(DATA_DIR, "checkpoints", "guppylm-9M")  # pretrained GuppyLM-9M checkpoint
CACHE_DIR     = os.path.join(DATA_DIR, "cache")              # activations, datasets, vectors
RESULTS_DIR   = os.path.join(ROOT, "results")                # outputs of experiments/0*.py
CHECKS_DIR    = os.path.join(ROOT, "revision_20261002")      # revision checks I–IV

# Pinned run (not "latest"): the demo was calibrated on it and the report pages chart it.
CHECKS_RUN_2026_10_09 = os.path.join(CHECKS_DIR, "run_20261009")
