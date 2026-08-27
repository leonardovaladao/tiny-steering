"""Phase 3: RFM (Recursive Feature Machine) concept-vector extraction.

Implements the Mahalanobis-Laplace kernel ridge regression + Average Gradient
Outer Product (AGOP) iteration from Radhakrishnan et al. (Science, 2024),
applied to concept-vector extraction following Beaglehole et al. (arXiv:2502.03708).

Memory note: all intermediate [n,n,k] tensors are avoided by reformulating the
AGOP gradient computation as a matrix product (see agop_grad docstring).
Peak memory is O(n*k + n^2) rather than O(n^2*k).
"""

import numpy as np
from scipy.spatial.distance import cdist


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mahal_half(M: np.ndarray) -> np.ndarray:
    """Return M^{1/2}: the matrix such that d_M(a,b) = ||M_half @ (a-b)||_2.

    Uses eigendecomposition M = V D V^T  →  M_half = diag(sqrt(D_+)) @ V^T
    Negative eigenvalues (numerical noise) are clamped to 0.
    """
    evals, evecs = np.linalg.eigh(M)          # evals ascending, evecs columns
    sqrt_evals = np.sqrt(np.maximum(evals, 0.0))
    return (sqrt_evals[:, None] * evecs.T)    # [k, k]: rows are transformed axes


def _preprocess(
    Z: np.ndarray,
    center: bool,
    normalize: bool,
    mean_vec: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Centre (subtract training mean) and optionally L2-normalise rows.

    If mean_vec is supplied it is used directly (for applying to val/test);
    otherwise it is computed from Z (for training).
    Returns (Z_processed, mean_vec).
    """
    k = Z.shape[1]
    if mean_vec is None:
        mean_vec = Z.mean(axis=0) if center else np.zeros(k, dtype=np.float64)
    Z_c = Z - mean_vec
    if normalize:
        norms = np.linalg.norm(Z_c, axis=1, keepdims=True)
        Z_c = Z_c / np.maximum(norms, 1e-10)
    return Z_c, mean_vec


def _agop_step(
    Z_c: np.ndarray,
    y: np.ndarray,
    M: np.ndarray,
    L: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One RFM iteration: kernel ridge regression + AGOP update.

    Memory-efficient gradient formula (avoids the O(n^2 k) diff tensor):
        W[i,j] = α_j * K[i,j] / (L * D[i,j])   (0 when D[i,j] < eps)
        G[i]   = -( W_sum[i]*Z_c[i] − (W @ Z_c)[i] ) @ M

    Returns (M_new, alpha, K).
    """
    n = Z_c.shape[0]
    M_half = _mahal_half(M)
    Zm = Z_c @ M_half.T                        # [n, k] Mahalanobis-transformed
    D  = cdist(Zm, Zm, metric='euclidean')      # [n, n] Mahalanobis distances
    K  = np.exp(-D / L)                         # [n, n] kernel matrix

    alpha = np.linalg.solve(K + lam * np.eye(n), y)   # [n] dual coefficients

    # Build weight matrix W (contribution of each training point to each gradient)
    eps    = 1e-12
    safe_D = np.where(D < eps, 1.0, D)
    W      = np.where(D < eps, 0.0,
                      alpha[None, :] * K / (L * safe_D))   # [n, n]

    # Gradient: G[i] = -(W_sum[i]*Z_c[i] - (W@Z_c)[i]) @ M
    W_sum = W.sum(axis=1)                                  # [n]
    SZ    = W_sum[:, None] * Z_c - (W @ Z_c)              # [n, k]
    G     = -SZ @ M                                        # [n, k]

    # Mean-centre gradients then form AGOP
    G_c   = G - G.mean(axis=0)
    M_new = (1.0 / n) * (G_c.T @ G_c)                    # [k, k]
    return M_new, alpha, K


# ─────────────────────────────────────────────────────────────────────────────
# Public API required by the guide
# ─────────────────────────────────────────────────────────────────────────────

def laplace_kernel(
    Z1: np.ndarray,
    Z2: np.ndarray,
    M: np.ndarray,
    L: float,
) -> np.ndarray:
    """Mahalanobis-Laplace kernel matrix.

    K[i,j] = exp(-sqrt((Z1_i - Z2_j)^T M (Z1_i - Z2_j)) / L)

    Args:
        Z1: [n1, k]
        Z2: [n2, k]
        M:  [k, k] PSD matrix
        L:  bandwidth scalar
    Returns: [n1, n2]
    """
    M_half = _mahal_half(M)
    D = cdist(Z1 @ M_half.T, Z2 @ M_half.T, metric='euclidean')
    return np.exp(-D / L)


def rfm_fit(
    Z: np.ndarray,
    y: np.ndarray,
    L: float,
    lam: float = 1e-3,
    T: int = 5,
    center: bool = True,
    normalize: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Run T RFM iterations (kernel ridge regression + AGOP).

    Args:
        Z:         [n, k] training activations
        y:         [n] labels (real-valued or binary {0,1})
        L:         Laplace kernel bandwidth
        lam:       ridge regularisation (default 1e-3)
        T:         number of AGOP iterations
        center:    subtract training mean from Z
        normalize: L2-normalise rows after centering
    Returns:
        M_T:      [k, k] AGOP matrix after T iterations
        mean_vec: [k] training mean (for consistent preprocessing of new data)
    """
    Z = np.asarray(Z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    Z_c, mean_vec = _preprocess(Z, center=center, normalize=normalize)
    M = np.eye(Z_c.shape[1], dtype=np.float64)
    for _ in range(T):
        M, _, _ = _agop_step(Z_c, y, M, L, lam)
    return M, mean_vec


def agop_grad(
    alpha: np.ndarray,
    Z_train: np.ndarray,
    z_eval: np.ndarray,
    M: np.ndarray,
    L: float,
) -> np.ndarray:
    """Gradient of f_t(z) = Σ_i α_i K_M(a^(i), z) at eval points.

    Formula (memory-efficient, avoids [m,n,k] intermediate tensor):
        W[j,i] = α_i * K[j,i] / (L * D[j,i])    (0 when D[j,i] < eps)
        G[j]   = -( W_sum[j]*z_j − (W @ Z_train)[j] ) @ M

    Args:
        alpha:   [n] KRR dual coefficients
        Z_train: [n, k] training points (already preprocessed)
        z_eval:  [m, k] evaluation points (already preprocessed)
        M:       [k, k] current AGOP matrix
        L:       bandwidth
    Returns: [m, k]
    """
    M_half = _mahal_half(M)
    D = cdist(z_eval @ M_half.T, Z_train @ M_half.T, metric='euclidean')  # [m, n]
    K = np.exp(-D / L)

    eps    = 1e-12
    safe_D = np.where(D < eps, 1.0, D)
    W      = np.where(D < eps, 0.0, alpha[None, :] * K / (L * safe_D))  # [m, n]

    W_sum = W.sum(axis=1)                              # [m]
    SZ    = W_sum[:, None] * z_eval - (W @ Z_train)   # [m, k]
    return -SZ @ M                                     # [m, k]


def top_eigvectors(M: np.ndarray, p: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Top-p eigenvectors of symmetric M, descending eigenvalue order.

    Returns:
        V:     [k, p] eigenvector matrix (columns)
        evals: [p] eigenvalues (descending)
    """
    evals_all, evecs_all = np.linalg.eigh(M)           # ascending
    idx = np.argsort(evals_all)[::-1]                  # descending
    return evecs_all[:, idx[:p]], evals_all[idx[:p]]


def orient(v: np.ndarray, Z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return sign(Pearson(Z @ v, y)) * v so the positive direction encodes label=1."""
    proj = Z @ v
    if proj.std() < 1e-12 or np.std(y) < 1e-12:
        return v
    rho = np.corrcoef(proj, y)[0, 1]
    return (np.sign(rho) if rho != 0 else 1.0) * v


def extract_concept_vector(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    Z_val:   np.ndarray,
    y_val:   np.ndarray,
    sweep:   dict | None = None,
) -> tuple[np.ndarray, dict]:
    """Full RFM pipeline with HP selection on val set.

    Sweeps (L, T, normalize) and picks the combination maximising
    |Pearson(Z_val_processed @ v, y_val)|.  T is searched incrementally
    within each (L, normalize) combination to avoid redundant computation.

    Args:
        Z_train, y_train: training activations [n, k] and labels [n]
        Z_val,   y_val:   validation activations and labels
        sweep: optional dict with keys:
            L_values   (default [1, 10, 100])
            T_max      (default 10)
            normalize  (default [False, True])
            lam        (default 1e-3)
            p          (default 3, number of eigenvectors to keep)

    Returns:
        v:    [k] oriented unit concept vector for best HP
        info: dict with all results and metadata
    """
    if sweep is None:
        sweep = {}
    L_values  = sweep.get("L_values",  [1, 10, 100])
    T_max     = sweep.get("T_max",     10)
    norm_opts = sweep.get("normalize", [False, True])
    lam       = sweep.get("lam",       1e-3)
    p         = sweep.get("p",         3)

    Z_tr = np.asarray(Z_train, dtype=np.float64)
    Z_va = np.asarray(Z_val,   dtype=np.float64)
    y_tr = np.asarray(y_train, dtype=np.float64)
    y_va = np.asarray(y_val,   dtype=np.float64)
    n, k = Z_tr.shape

    best_abs_rho = -1.0
    best_info    = None
    all_results  = []

    for normalize in norm_opts:
        Z_c_tr, mean_vec = _preprocess(Z_tr, center=True, normalize=normalize)
        Z_c_va, _        = _preprocess(Z_va, center=True, normalize=normalize,
                                       mean_vec=mean_vec)

        for L in L_values:
            M = np.eye(k, dtype=np.float64)

            for T in range(1, T_max + 1):
                M, _, _ = _agop_step(Z_c_tr, y_tr, M, L, lam)

                # Evaluate top eigenvector on val
                evals_all, evecs_all = np.linalg.eigh(M)
                idx_desc  = np.argsort(evals_all)[::-1]
                evals_top = evals_all[idx_desc[:p]]
                V_top     = evecs_all[:, idx_desc[:p]]       # [k, p]
                v_raw     = V_top[:, 0]

                v_ori    = orient(v_raw, Z_c_tr, y_tr)
                val_proj = Z_c_va @ v_ori
                if val_proj.std() > 1e-12 and y_va.std() > 1e-12:
                    abs_rho = abs(float(np.corrcoef(val_proj, y_va)[0, 1]))
                else:
                    abs_rho = 0.0

                result = {"L": L, "T": T, "normalize": normalize,
                          "val_abs_rho": abs_rho}
                all_results.append(result)

                if abs_rho > best_abs_rho:
                    best_abs_rho  = abs_rho
                    evals_full    = evals_all[idx_desc]        # full spectrum
                    lam1_share    = (evals_full[0] / max(evals_full.sum(), 1e-12))
                    best_info = {
                        "v":         v_ori.copy(),
                        "V_top_p":   V_top.copy(),        # [k, p]
                        "M":         M.copy(),             # [k, k]
                        "mean_vec":  mean_vec.copy(),      # [k]
                        "eigenvalues": evals_top,          # [p] top eigenvalues
                        "eigenvalue_spectrum_full": evals_full,  # [k] descending
                        "lam1_share": float(lam1_share),   # λ1 / Σλ
                        "L": L, "T": T, "normalize": normalize,
                        "val_abs_rho": float(abs_rho),
                    }

    best_info["all_results"] = all_results
    return best_info["v"], best_info


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic recovery test (Phase 3 gate)
# ─────────────────────────────────────────────────────────────────────────────

def rfm_synthetic_test(
    n: int = 1000,
    k: int = 384,
    seed: int = 0,
    cos_threshold: float = 0.90,
) -> tuple[bool, float]:
    """Gate test: verify RFM recovers a known unit direction u in R^k.

    Generates Z ∈ R^{n×k} (standard normal) with regression labels y_i = Z_i @ u
    (the cleanest possible linear dependence on a single direction).  Binary labels
    require n >> k to converge past 0.9; regression labels converge at n=1000 with k=384.

    Returns: (passed: bool, cosine_value: float)
    """
    rng = np.random.default_rng(seed)
    u   = rng.standard_normal(k)
    u  /= np.linalg.norm(u)

    Z = rng.standard_normal((n, k))
    y = Z @ u      # regression: label = projection onto u (linear, no threshold noise)

    n_tr  = int(n * 0.8)
    Z_tr, y_tr = Z[:n_tr],  y[:n_tr]
    Z_va, y_va = Z[n_tr:],  y[n_tr:]

    v, info = extract_concept_vector(
        Z_tr, y_tr, Z_va, y_va,
        sweep={"L_values": [1, 10, 100], "T_max": 10,
               "normalize": [False, True], "lam": 1e-3, "p": 3},
    )
    cos_val = abs(float(np.dot(v, u)))
    passed  = cos_val >= cos_threshold
    print(
        f"[rfm] Synthetic test (n={n}, k={k}): |cos(v,u)| = {cos_val:.4f}  "
        f"(threshold {cos_threshold})  →  {'PASSED ✓' if passed else 'FAILED ✗'}\n"
        f"       Best HP: L={info['L']}, T={info['T']}, "
        f"normalize={info['normalize']}, "
        f"val_|ρ|={info['val_abs_rho']:.3f}, "
        f"λ1/Σλ={info['lam1_share']:.3f}"
    )
    return passed, cos_val


if __name__ == "__main__":
    passed, cos_val = rfm_synthetic_test()
    if not passed:
        print(
            "\n[rfm] WARNING: synthetic recovery test failed.\n"
            "      Do NOT proceed with real concept extraction until this passes.\n"
            "      Check the gradient formula and HP sweep."
        )
