"""Phase 8: Baseline concept-vector extraction methods.

Three baselines from the paper (§1d of arXiv:2502.03708):
  - PCA:       top eigenvector of the pos-minus-neg difference matrix
  - DiffMeans: normalised mean(pos) − mean(neg)
  - LogReg:    normalised logistic-regression coefficient (C swept on val)

All methods:
  - operate per block (call independently for each block's [n, k] activations)
  - mean-centre training activations (store mean for reuse on val/test)
  - orient the result so positive projections → label-1 (Pearson sign flip)
  - return a unit vector

Monitoring helper:
  - baseline_monitoring_auroc(): project activations onto v_per_block, train
    an LR probe on the projections, evaluate AUROC on test.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _center(Z: np.ndarray, mean_vec: np.ndarray | None = None):
    """Subtract training mean; return (Z_c, mean_vec)."""
    if mean_vec is None:
        mean_vec = Z.mean(axis=0)
    return Z - mean_vec, mean_vec


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else v


def _orient(v: np.ndarray, Z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Flip v if Pearson(Z@v, y) < 0, to make positive projections = label-1."""
    proj = Z @ v
    if proj.std() < 1e-12 or np.std(y) < 1e-12:
        return v
    rho = float(np.corrcoef(proj, y)[0, 1])
    return (np.sign(rho) if rho != 0 else 1.0) * v


def _val_pearson(v: np.ndarray, Z_val_c: np.ndarray, y_val: np.ndarray) -> float:
    proj = Z_val_c @ v
    if proj.std() < 1e-12 or np.std(y_val) < 1e-12:
        return 0.0
    return abs(float(np.corrcoef(proj, y_val)[0, 1]))


# ─────────────────────────────────────────────────────────────────────────────
# Individual extraction functions (one block at a time)
# ─────────────────────────────────────────────────────────────────────────────

def pca_vector(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    Z_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Top eigenvector of the positive-minus-negative difference matrix.

    Pairs pos and neg samples (min(n_pos, n_neg) pairs, sequential),
    forms D = pos - neg, mean-centres D, takes the top right singular vector.

    Returns: (v [k], mean_vec [k], val_abs_rho float)
    """
    Z_c, mean_vec = _center(np.asarray(Z_train, dtype=np.float64))
    Z_val_c, _    = _center(np.asarray(Z_val, dtype=np.float64), mean_vec)
    y = np.asarray(y_train)
    y_val = np.asarray(y_val)

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_pairs = min(len(pos_idx), len(neg_idx))
    D = Z_c[pos_idx[:n_pairs]] - Z_c[neg_idx[:n_pairs]]  # [n_pairs, k]
    D -= D.mean(axis=0)

    # Top right singular vector of D (equivalent to top PC of D^T D)
    _, _, Vt = np.linalg.svd(D, full_matrices=False)
    v = _unit(Vt[0])
    v = _orient(v, Z_c, y)
    return v, mean_vec, _val_pearson(v, Z_val_c, y_val)


def diffmeans_vector(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    Z_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Normalised mean(pos) − mean(neg).

    Returns: (v [k], mean_vec [k], val_abs_rho float)
    """
    Z_c, mean_vec = _center(np.asarray(Z_train, dtype=np.float64))
    Z_val_c, _    = _center(np.asarray(Z_val, dtype=np.float64), mean_vec)
    y = np.asarray(y_train)
    y_val = np.asarray(y_val)

    pos_mean = Z_c[y == 1].mean(axis=0)
    neg_mean = Z_c[y == 0].mean(axis=0)
    v = _unit(pos_mean - neg_mean)
    v = _orient(v, Z_c, y)
    return v, mean_vec, _val_pearson(v, Z_val_c, y_val)


def logreg_vector(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    Z_val: np.ndarray,
    y_val: np.ndarray,
    C_values: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Normalised logistic-regression coefficient; C selected by val |Pearson|.

    Returns: (v [k], mean_vec [k], val_abs_rho float)
    """
    if C_values is None:
        C_values = [1000.0, 100.0, 1.0, 0.1]

    Z_c, mean_vec = _center(np.asarray(Z_train, dtype=np.float64))
    Z_val_c, _    = _center(np.asarray(Z_val, dtype=np.float64), mean_vec)
    y     = np.asarray(y_train, dtype=int)
    y_val = np.asarray(y_val, dtype=int)

    best_rho = -1.0
    best_v: np.ndarray | None = None

    for C in C_values:
        lr = LogisticRegression(C=C, solver="liblinear", max_iter=2000, random_state=42)
        lr.fit(Z_c, y)
        coef = lr.coef_[0]
        n = np.linalg.norm(coef)
        if n < 1e-10:
            continue
        v = coef / n
        v = _orient(v, Z_c, y)
        rho = _val_pearson(v, Z_val_c, y_val)
        if rho > best_rho:
            best_rho = rho
            best_v = v.copy()

    if best_v is None:
        best_v = np.zeros(Z_train.shape[1])
        best_v[0] = 1.0
        best_rho = 0.0

    return best_v, mean_vec, best_rho


# ─────────────────────────────────────────────────────────────────────────────
# Batch extraction over all blocks
# ─────────────────────────────────────────────────────────────────────────────

_METHOD_FN = {
    "pca":       pca_vector,
    "diffmeans": diffmeans_vector,
    "logreg":    logreg_vector,
}


def extract_all_baseline_vectors(
    acts_train: np.ndarray,
    y_train: np.ndarray,
    acts_val: np.ndarray,
    y_val: np.ndarray,
) -> dict[str, dict]:
    """Extract PCA / DiffMeans / LogReg vectors for every block.

    Args:
        acts_train: [n_train, L, k]
        y_train:    [n_train]
        acts_val:   [n_val,   L, k]
        y_val:      [n_val]

    Returns:
        {method_name: {"v_per_block": [L, k], "mean_per_block": [L, k], "val_abs_rho": [L]}}
    """
    _, L, k = acts_train.shape
    results: dict[str, dict] = {}

    for name, fn in _METHOD_FN.items():
        v_per_block   = np.zeros((L, k), dtype=np.float64)
        mean_per_block = np.zeros((L, k), dtype=np.float64)
        val_rho        = np.zeros(L, dtype=np.float64)

        for ell in range(L):
            Z_tr = acts_train[:, ell, :].astype(np.float64)
            Z_va = acts_val  [:, ell, :].astype(np.float64)
            v, mv, rho = fn(Z_tr, y_train, Z_va, y_val)
            v_per_block[ell]    = v
            mean_per_block[ell] = mv
            val_rho[ell]        = rho

        results[name] = {
            "v_per_block":    v_per_block,
            "mean_per_block": mean_per_block,
            "val_abs_rho":    val_rho,
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Monitoring: AUROC using baseline vectors as probe directions
# ─────────────────────────────────────────────────────────────────────────────

def _build_proj_features(
    acts: np.ndarray,
    v_per_block: np.ndarray,
    mean_per_block: np.ndarray,
) -> np.ndarray:
    """Project activations onto one vector per block → [n, L]."""
    n, L, _ = acts.shape
    feats = np.zeros((n, L), dtype=np.float64)
    for ell in range(L):
        a_c = acts[:, ell, :].astype(np.float64) - mean_per_block[ell]
        feats[:, ell] = a_c @ v_per_block[ell]
    return feats


def baseline_monitoring_auroc(
    acts_train: np.ndarray,
    y_train: np.ndarray,
    acts_val: np.ndarray,
    y_val: np.ndarray,
    acts_test: np.ndarray,
    y_test: np.ndarray,
    v_per_block: np.ndarray,
    mean_per_block: np.ndarray,
    C_values: list[float] | None = None,
) -> tuple[float, float]:
    """Train an LR probe on per-block projections; return (val_auroc, test_auroc).

    Features for each sample: [L] scalars = <a_ell − mean_ell, v_ell> for ell in 0..L-1.
    Trains on train features, selects C by val AUROC, returns test AUROC.
    """
    if C_values is None:
        C_values = [1000.0, 100.0, 1.0, 0.1]

    X_tr = _build_proj_features(acts_train, v_per_block, mean_per_block)
    X_va = _build_proj_features(acts_val,   v_per_block, mean_per_block)
    X_te = _build_proj_features(acts_test,  v_per_block, mean_per_block)
    y_tr = np.asarray(y_train, dtype=int)
    y_va = np.asarray(y_val,   dtype=int)
    y_te = np.asarray(y_test,  dtype=int)

    best_val_auroc = -1.0
    best_pipe: Pipeline | None = None

    for C in C_values:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=C, solver="liblinear", max_iter=2000, random_state=42)),
        ])
        pipe.fit(X_tr, y_tr)
        probs = pipe.predict_proba(X_va)[:, 1]
        try:
            auroc = float(roc_auc_score(y_va, probs))
        except ValueError:
            auroc = 0.5
        if auroc > best_val_auroc:
            best_val_auroc = auroc
            best_pipe = pipe

    assert best_pipe is not None
    te_probs = best_pipe.predict_proba(X_te)[:, 1]
    try:
        test_auroc = float(roc_auc_score(y_te, te_probs))
    except ValueError:
        test_auroc = float("nan")

    return best_val_auroc, test_auroc
