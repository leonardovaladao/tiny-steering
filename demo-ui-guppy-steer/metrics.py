"""Scores used by the calibration (§8.1 of PLAN_grilled.md), copied from Check III.

  judge g_c       TF-IDF (1–2-grams, (?u)\\b\\w+\\b, sublinear tf) + balanced LogReg, fit on the
                  deduplicated GuppyLM outputs of the concept's pos/neg categories;
                  g_c = 2 P(pos) − 1  (III-steering-metric, cells 5 and 9)
  cond_ppl        PPL_θ(y | x) under the unsteered model (cell 13)
  uncond_ppl      PPL_θ(y | ∅): y scored after the ChatML template with an empty user turn
  distinct2       unique bigrams / bigrams over whitespace words (cell 13)
  mean_jaccard    mean pairwise bigram Jaccard across texts (cell 28)
"""

import os
from itertools import combinations

import numpy as np
import torch

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import steer_ref                           # noqa: E402
from rfm_guppy.concepts import CONCEPTS    # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Activation-blind judge
# ─────────────────────────────────────────────────────────────────────────────

def make_judge():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True,
                        token_pattern=r"(?u)\b\w+\b"),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000),
    )


def fit_judges(concepts=steer_ref.CONCEPTS) -> dict:
    """Final Check III judges: fit on all unique outputs of the pos/neg categories."""
    from datasets import load_dataset
    ds = load_dataset("arman-bd/guppylm-60k-generic", split="train")
    by_cat = {}
    for r in ds:
        by_cat.setdefault(r["category"], set()).add(r["output"])
    judges = {}
    for c in concepts:
        pos = set().union(*(by_cat[k] for k in CONCEPTS[c]["pos"]))
        neg = set().union(*(by_cat[k] for k in CONCEPTS[c]["neg"]))
        both = pos & neg
        X = sorted(pos - both) + sorted(neg - both)
        y = np.array([1] * len(pos - both) + [0] * len(neg - both))
        judges[c] = make_judge().fit(X, y)
    return judges


def signed(judge, texts) -> np.ndarray:
    return 2 * judge.predict_proba(list(texts))[:, 1] - 1


# ─────────────────────────────────────────────────────────────────────────────
# Coherence
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _ppl_after(model, tok, prefix_ids, text):
    ids = (prefix_ids + tok.encode(text).ids)[: model.config.max_seq_len]
    if len(ids) - len(prefix_ids) < 1:
        return np.nan
    x = torch.tensor([ids])
    logits, _ = model(x)
    logp = torch.log_softmax(logits[0, :-1], dim=-1)
    lp = logp.gather(1, x[0, 1:, None]).squeeze(1)[len(prefix_ids) - 1:]
    return float(torch.exp(-lp.mean()))


def cond_ppl(model, tok, prompt, text):
    return _ppl_after(model, tok, tok.encode(steer_ref.chatml(prompt)).ids, text)


def uncond_ppl(model, tok, text):
    return _ppl_after(model, tok, tok.encode(steer_ref.chatml("")).ids, text)


def distinct2(text):
    w = text.lower().split()
    bg = list(zip(w, w[1:]))
    return len(set(bg)) / len(bg) if bg else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Prompt dependence
# ─────────────────────────────────────────────────────────────────────────────

def bigrams(t):
    w = t.lower().split()
    return set(zip(w, w[1:]))


def jacc(a, b):
    return len(a & b) / len(a | b) if (a | b) else 1.0


def mean_jaccard(texts):
    B = [bigrams(t) for t in texts]
    return float(np.mean([jacc(a, b) for a, b in combinations(B, 2)]))
