"""Tyler's M-estimator for robust scatter matrix estimation.

Implements the fixed-point iteration for Tyler's (1987) distribution-free
scatter estimator, which is equivariant under affine transformations and
robust to heavy-tailed elliptical distributions.

Reference
---------
Tyler, D.E. (1987). A distribution-free M-estimator of multivariate scatter.
Annals of Statistics, 15(1), 234-251.
"""

import logging
import warnings
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def tyler_m_estimator(
    residuals: NDArray,
    max_iter: int = 500,
    tol: float = 1e-7,
    eps: float = 1e-6,
) -> dict:
    """Solve Tyler's M-estimator fixed-point equation for a residual matrix.

    The estimator V solves the implicit equation:

        V = (p / m_tilde) * sum_i  r_i r_i' / (r_i' V^{-1} r_i)

    subject to the trace normalisation tr(V) = p, where m_tilde is the number
    of non-zero residual rows.

    Algorithm
    ---------
    1. Discard exact-zero rows; let m_tilde = remaining row count.
    2. Initialise V_0 = I_p.
    3. Iterate:
       a. Compute V^{-1} (with optional ridge regularisation).
       b. Compute Mahalanobis distances d_i = r_i' V^{-1} r_i.
       c. Update V_new = (p / m_tilde) * sum_i r_i r_i' / d_i.
       d. Normalise: V_new *= p / tr(V_new).
       e. Check relative Frobenius convergence: ||V_new - V||_F / ||V||_F < tol.
    4. Apply ridge regularisation V <- (1 - eps) * V + eps * I_p if the
       minimum eigenvalue is below eps (numerical stability).

    Feasibility warning
    -------------------
    Tyler's estimator requires m_tilde > p (more observations than dimensions).
    If p >= m_tilde, a feasibility warning is recorded but iteration continues
    (may produce a degenerate result).

    Parameters
    ----------
    residuals : NDArray
        Residual matrix of shape (m, p). Rows are individual observation
        residuals (possibly from OLS or restricted OLS).
    max_iter : int, optional
        Maximum number of fixed-point iterations, by default 500.
    tol : float, optional
        Convergence tolerance on relative Frobenius change, by default 1e-7.
    eps : float, optional
        Ridge regularisation strength applied when numerically needed,
        by default 1e-6.

    Returns
    -------
    dict
        'V'              : NDArray (p, p) — Tyler scatter estimate, tr(V) = p.
        'converged'      : bool   — True if the iteration converged within max_iter.
        'n_iter'         : int    — Number of iterations performed.
        'eps_used'       : float  — Actual eps applied (0.0 if no regularisation needed).
        'failure_reason' : str or None — Reason for non-convergence / warnings.
    """
    residuals = np.asarray(residuals, dtype=float)
    m, p = residuals.shape

    # Remove exact-zero rows
    row_norms = np.linalg.norm(residuals, axis=1)
    mask = row_norms > 0.0
    R = residuals[mask]
    m_tilde = R.shape[0]

    failure_reason = None

    if m_tilde == 0:
        failure_reason = "All residual rows are zero; cannot compute Tyler estimator."
        logger.error(failure_reason)
        return {
            "V": np.eye(p),
            "converged": False,
            "n_iter": 0,
            "eps_used": 0.0,
            "failure_reason": failure_reason,
        }

    # Feasibility check: Tyler's estimator requires m_tilde > p
    if p >= m_tilde:
        msg = (
            f"Feasibility warning: p={p} >= m_tilde={m_tilde}. "
            "Tyler's estimator may be degenerate (tilde_c_n >= 1)."
        )
        logger.warning(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        failure_reason = msg

    # Initialise
    V = np.eye(p, dtype=float)
    eps_used = 0.0
    converged = False
    n_iter = 0
    I_p = np.eye(p, dtype=float)
    _check_interval = 20  # run eigvalsh only every this many iterations (O(p³) is expensive)

    for iteration in range(max_iter):
        # Regularise by trying Cholesky; fall back to ridge if not PD
        try:
            L = np.linalg.cholesky(V)
            # Solve V_inv @ R.T via two triangular solves (cheaper than explicit inv)
            # tmp[i,:] = V^{-1} r_i  →  solve L L' x = r_i'  for each row
            # scipy.linalg.cho_solve is shape-agnostic; use numpy equivalent
            tmp = np.linalg.solve(V, R.T).T  # (m_tilde, p)  — solves V x = R^T col-wise
            del L
        except np.linalg.LinAlgError:
            # V is not PD — apply ridge regularisation
            V = (1.0 - eps) * V + eps * I_p
            eps_used = eps
            try:
                tmp = np.linalg.solve(V, R.T).T
            except np.linalg.LinAlgError:
                failure_reason = f"V became singular at iteration {iteration}; stopping."
                logger.error(failure_reason)
                break

        # Mahalanobis distances d_i = r_i' V^{-1} r_i  (shape: m_tilde,)
        distances = np.einsum("ij,ij->i", tmp, R)  # element-wise dot

        # Guard against zero distances (should not occur after zero-row removal)
        distances = np.maximum(distances, 1e-15)

        # Fixed-point update: V_new = (p / m_tilde) * sum_i r_i r_i' / d_i
        weights = (p / m_tilde) / distances  # shape (m_tilde,)
        V_new = (R * weights[:, None]).T @ R  # (p, p)

        # Trace normalisation
        tr_V_new = np.trace(V_new)
        if tr_V_new <= 0:
            failure_reason = f"Trace of V_new <= 0 at iteration {iteration}; stopping."
            logger.error(failure_reason)
            break
        V_new *= p / tr_V_new

        # Convergence check (Frobenius relative change)
        norm_V = np.linalg.norm(V, "fro")
        rel_change = np.linalg.norm(V_new - V, "fro") / (norm_V + 1e-15)
        V = V_new
        n_iter = iteration + 1

        if rel_change < tol:
            converged = True
            break

        # Periodic eigenvalue check — O(p³) so do it sparingly
        if (iteration + 1) % _check_interval == 0:
            eigmin = np.linalg.eigvalsh(V).min()
            if eigmin < eps:
                V = (1.0 - eps) * V + eps * I_p
                eps_used = eps

    if not converged and failure_reason is None:
        failure_reason = (
            f"Did not converge within {max_iter} iterations "
            f"(last rel. change may exceed tol={tol})."
        )
        logger.warning(failure_reason)

    # Final regularisation pass to ensure PD
    eigmin_final = np.linalg.eigvalsh(V).min()
    if eigmin_final < eps:
        V = (1.0 - eps) * V + eps * I_p
        eps_used = eps

    # Enforce exact symmetry and trace normalisation
    V = (V + V.T) / 2.0
    V *= p / np.trace(V)

    return {
        "V": V,
        "converged": converged,
        "n_iter": n_iter,
        "eps_used": eps_used,
        "failure_reason": failure_reason,
    }
