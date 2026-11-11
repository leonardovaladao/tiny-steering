"""Step 1, part 1: freeze the calibration prompt sets before anything is generated (§8.1).

Run from the project root:   HF_DATASETS_OFFLINE=1 .venv/bin/python demo-ui-guppy-steer/freeze_prompts.py

Writes demo-ui-guppy-steer/prompts_frozen.json with
  exploration   the 7 Check III prompts (from results/steer_food.json, r = 0)
  calibration   the 100 Check III calibration prompts, re-drawn exactly as in III-steering-metric
                (rng = default_rng(0) on the sorted held-out pool); used only for q95 / q05
  confirmation  20 prompts drawn with default_rng(CONFIRM_SEED) from the held-out pool, minus
                  - the 100 calibration prompts and the 7 exploration prompts,
                  - every input that also occurs in a category used by some concept (those rows
                    fit the vectors, via the exchanges, and the judge, via the outputs),
                  - prompts with T0 + 64 > 128 (the page generates 64 tokens).
                Disjointness is checked on exact strings and on a normalised form
                (lowercase, whitespace collapsed, trailing punctuation dropped).

The file stores the SHA-256 of its prompt lists; later steps refuse to run if it changes.
"""

import hashlib
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import steer_ref                                  # noqa: E402
from steer_ref import ROOT                        # noqa: E402
from rfm_guppy.paths import CHECKS_RUN_2026_10_09, RESULTS_DIR   # noqa: E402
from rfm_guppy.concepts import CONCEPTS           # noqa: E402

OUT = os.path.join(ROOT, "demo-ui-guppy-steer", "prompts_frozen.json")
CALIB_CSV = os.path.join(CHECKS_RUN_2026_10_09, "III-steering-metric_calibration.csv")
CONFIRM_SEED = 20261020        # date the calibration was pre-registered
N_CONFIRM = 20
PAGE_NEW_TOKENS = 64


def norm(s: str) -> str:
    return re.sub(r"[\s.?!]+$", "", re.sub(r"\s+", " ", s.strip().lower()))


ds = load_dataset("arman-bd/guppylm-60k-generic", split="train")
cats = sorted({r["category"] for r in ds})
used = set().union(*(set(CONCEPTS[c]["pos"]) | set(CONCEPTS[c]["neg"]) for c in CONCEPTS))
held_cats = sorted(set(cats) - used)

# ── calibration: exactly Check III's draw ────────────────────────────────────
pool = sorted({r["input"] for r in ds if r["category"] in held_cats})
calib = [str(p) for p in np.random.default_rng(0).choice(pool, size=100, replace=False)]
assert calib == list(pd.read_csv(CALIB_CSV).prompt), "Check III calibration draw does not reproduce"

# ── exploration: the 7 Check III prompts ─────────────────────────────────────
stored0 = json.load(open(os.path.join(RESULTS_DIR, "steer_food.json")))["results_per_r"]["0.0"]
explore = [o["prompt"] for o in stored0["outputs"]]
assert len(explore) == 7

# ── confirmation ─────────────────────────────────────────────────────────────
_, tok = steer_ref.load_model()
used_inputs = {r["input"] for r in ds if r["category"] in used}
banned = {norm(p) for p in calib + explore} | {norm(p) for p in used_inputs}
eligible = [p for p in pool
            if norm(p) not in banned
            and len(tok.encode(steer_ref.chatml(p)).ids) + PAGE_NEW_TOKENS <= 128]
confirm = [str(p) for p in np.random.default_rng(CONFIRM_SEED).choice(eligible, size=N_CONFIRM,
                                                                       replace=False)]

cat_of = {}
for r in ds:
    if r["input"] in confirm:
        cat_of.setdefault(r["input"], set()).add(r["category"])

# ── checks ───────────────────────────────────────────────────────────────────
N = lambda xs: {norm(x) for x in xs}            # noqa: E731
assert len(set(confirm)) == N_CONFIRM
assert not (N(confirm) & N(calib)) and not (N(confirm) & N(explore))
assert not (N(confirm) & N(used_inputs))
assert all(cat_of[p] <= set(held_cats) for p in confirm)

lists = dict(exploration=explore, calibration=calib, confirmation=confirm)
digest = hashlib.sha256(json.dumps(lists, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
frozen = dict(
    frozen_on="2026-10-27",
    sha256=digest,
    rules=dict(
        held_out_categories=held_cats,
        calibration="np.random.default_rng(0).choice(sorted held-out inputs, 100, replace=False)",
        confirmation=f"np.random.default_rng({CONFIRM_SEED}).choice(eligible, {N_CONFIRM}, "
                     "replace=False); eligible = held-out inputs not in calibration/exploration/"
                     "concept-category inputs (normalised match), T0 + 64 <= 128",
        n_pool=len(pool), n_eligible=len(eligible),
    ),
    confirmation_categories={p: sorted(cat_of[p]) for p in confirm},
    **lists,
)
with open(OUT, "w") as f:
    json.dump(frozen, f, indent=2, ensure_ascii=False)

print(f"pool {len(pool)}, eligible {len(eligible)}, held-out categories {len(held_cats)}")
print("confirmation prompts:")
for p in confirm:
    print(f"  [{', '.join(sorted(cat_of[p]))}] {p}")
print(f"sha256 {digest}\nwritten {OUT}")
