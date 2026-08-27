"""Phase 7: Activation probing for concept monitoring.

From the top-p AGOP eigenvectors stored by Phase 4, builds probe features
and trains binary classifiers. Two strategies mirror the paper:

  1. Per-block probe — one LR classifier per block (on its P projections);
     best block selected by val AUROC.
  2. Aggregate probe — one LR classifier on all blocks' projections concatenated
     into R^{L*P}.

Both are evaluated on the held-out test split by AUROC.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────────────────────────────────────
# Feature construction
# ─────────────────────────────────────────────────────────────────────────────

def build_probe_features(
    acts: np.ndarray,
    eigvecs_top3: np.ndarray,
    input_mean: np.ndarray,
    hp_normalize: np.ndarray | None = None,
) -> np.ndarray:
    """Project activations onto the top-P AGOP eigenvectors per block.

    Applies the same preprocessing (centering, optional L2-normalisation) that
    was used during RFM training to keep projections on the same scale.

    Args:
        acts:          [n, L, k]  last-token block activations
        eigvecs_top3:  [L, P, k]  top-P eigenvectors per block (rows = eigvecs)
        input_mean:    [L, k]     per-block training mean to subtract
        hp_normalize:  [L] int    1 = L2-normalise after centering (per block)

    Returns: [n, L, P] projection features
    """
    n, L, k = acts.shape
    P = eigvecs_top3.shape[1]
    features = np.zeros((n, L, P), dtype=np.float64)

    for ell in range(L):
        a_c = acts[:, ell, :].astype(np.float64) - input_mean[ell].astype(np.float64)
        if hp_normalize is not None and int(hp_normalize[ell]):
            norms = np.linalg.norm(a_c, axis=1, keepdims=True)
            a_c = a_c / np.maximum(norms, 1e-10)
        # eigvecs_top3[ell]: [P, k] — each row is one eigenvector
        # a_c @ eigvecs_top3[ell].T  =  [n, k] @ [k, P]  =  [n, P]
        features[:, ell, :] = a_c @ eigvecs_top3[ell].T

    return features


# ─────────────────────────────────────────────────────────────────────────────
# Probe training
# ─────────────────────────────────────────────────────────────────────────────

def _make_lr_pipeline(C: float) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=C, solver="liblinear", max_iter=2000, random_state=42
        )),
    ])


def _best_probe(X_tr: np.ndarray, y_tr: np.ndarray,
                X_va: np.ndarray, y_va: np.ndarray,
                C_values: list[float]) -> tuple[Pipeline, float]:
    """Grid-search over C; return (best_pipeline, best_val_auroc)."""
    best_auroc = -1.0
    best_pipe: Pipeline | None = None
    for C in C_values:
        pipe = _make_lr_pipeline(C)
        pipe.fit(X_tr, y_tr)
        probs = pipe.predict_proba(X_va)[:, 1]
        try:
            auroc = float(roc_auc_score(y_va, probs))
        except ValueError:
            auroc = 0.5
        if auroc > best_auroc:
            best_auroc = auroc
            best_pipe = pipe
    return best_pipe, best_auroc  # type: ignore[return-value]


def train_probes(
    features_train: np.ndarray,
    y_train: np.ndarray,
    features_val: np.ndarray,
    y_val: np.ndarray,
    C_values: list[float] | None = None,
) -> dict:
    """Train per-block and aggregate logistic-regression probes.

    Args:
        features_train: [n_train, L, P]
        y_train:        [n_train] binary {0,1} labels
        features_val:   [n_val,   L, P]
        y_val:          [n_val]   binary labels
        C_values: regularisation grid; default = [1000, 100, 1, 0.1]

    Returns dict with trained probes and val AUROCs.
    """
    if C_values is None:
        C_values = [1000.0, 100.0, 1.0, 0.1]

    n_train, L, P = features_train.shape
    n_val = features_val.shape[0]
    y_tr = np.asarray(y_train, dtype=int)
    y_va = np.asarray(y_val,   dtype=int)

    per_block_probes: list[Pipeline] = []
    per_block_val_auroc: list[float] = []

    for ell in range(L):
        probe, auroc = _best_probe(
            features_train[:, ell, :], y_tr,
            features_val[:,   ell, :], y_va,
            C_values,
        )
        per_block_probes.append(probe)
        per_block_val_auroc.append(auroc)
        print(f"    blk {ell}: val AUROC = {auroc:.4f}")

    best_block = int(np.argmax(per_block_val_auroc))

    # Aggregate probe on R^{L*P}
    X_agg_tr = features_train.reshape(n_train, L * P)
    X_agg_va = features_val.reshape(n_val,   L * P)
    agg_probe, agg_val_auroc = _best_probe(
        X_agg_tr, y_tr, X_agg_va, y_va, C_values
    )
    print(f"    aggregate ({L * P} features): val AUROC = {agg_val_auroc:.4f}")

    return {
        "per_block_probes":       per_block_probes,
        "per_block_val_auroc":    per_block_val_auroc,
        "best_block":             best_block,
        "best_block_val_auroc":   float(per_block_val_auroc[best_block]),
        "aggregate_probe":        agg_probe,
        "aggregate_val_auroc":    float(agg_val_auroc),
        "L": L,
        "P": P,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Probe evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_probes_on_test(
    probe_pkg: dict,
    features_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Evaluate trained probes on the held-out test split.

    Returns dict with AUROC scores and probability arrays.
    """
    n_test, L, P = features_test.shape
    y_te = np.asarray(y_test, dtype=int)
    best_block = probe_pkg["best_block"]

    X_block_te = features_test[:, best_block, :]
    block_probs = probe_pkg["per_block_probes"][best_block].predict_proba(X_block_te)[:, 1]
    try:
        block_auroc = float(roc_auc_score(y_te, block_probs))
    except ValueError:
        block_auroc = float("nan")

    X_agg_te = features_test.reshape(n_test, L * P)
    agg_probs = probe_pkg["aggregate_probe"].predict_proba(X_agg_te)[:, 1]
    try:
        agg_auroc = float(roc_auc_score(y_te, agg_probs))
    except ValueError:
        agg_auroc = float("nan")

    return {
        "best_block":            best_block,
        "best_block_test_auroc": block_auroc,
        "aggregate_test_auroc":  agg_auroc,
        "block_test_probs":      block_probs.tolist(),
        "agg_test_probs":        agg_probs.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Scoring helper (used by Phase 6)
# ─────────────────────────────────────────────────────────────────────────────

def predict_probe_probs(
    texts: list[str],
    model,
    tokenizer,
    probe_pkg: dict,
    eigvecs_top3: np.ndarray,
    input_mean: np.ndarray,
    hp_normalize: np.ndarray | None = None,
    format_chatml: bool = False,
) -> np.ndarray:
    """Run texts through GuppyLM, build probe features, return aggregate probs.

    Args:
        texts:         list of formatted text strings (ChatML or raw)
        model:         GuppyLM in eval mode
        tokenizer:     HuggingFace tokenizer
        probe_pkg:     returned by train_probes()
        eigvecs_top3:  [L, P, k]
        input_mean:    [L, k]
        hp_normalize:  [L] int (0/1)
        format_chatml: if True, wrap each text as a ChatML user message

    Returns: [n] float array of concept-positive probabilities
    """
    from rfm_guppy.activations import extract_activations

    acts = extract_activations(texts, model, tokenizer,
                               format_chatml=format_chatml)
    features = build_probe_features(acts, eigvecs_top3, input_mean, hp_normalize)
    n, L, P = features.shape
    X_agg = features.reshape(n, L * P)
    probs: np.ndarray = probe_pkg["aggregate_probe"].predict_proba(X_agg)[:, 1]
    return probs
