"""Covariance/scatter matrix estimators for high-dimensional multivariate regression.

Provides four estimators:
- URE  : Unrestricted Residual Estimator (nonlinear shrinkage on S_u)
- RRE  : Restricted Residual Estimator  (nonlinear shrinkage on S_r)
- URE-T: Robust URE based on Tyler's M-estimator
- RRE-T: Robust RRE based on Tyler's M-estimator
- SSE+ : Positive-part Stein combination of URE and RRE (or their robust variants)
"""

import logging
import warnings
import numpy as np
from numpy.typing import NDArray
import scipy.stats

from .tyler import tyler_m_estimator
from .shrinkage import apply_nonlinear_shrinkage

logger = logging.getLogger(__name__)


def compute_ure_cov(S_u: NDArray, c_n: float) -> NDArray:
    """Unrestricted covariance-based estimator (URE).

    Applies analytic nonlinear shrinkage to the unrestricted scatter estimate
    S_u at aspect ratio c_n = p / (n - d).

    Parameters
    ----------
    S_u : NDArray
        Unrestricted scatter estimate of shape (p, p), symmetric PSD.
    c_n : float
        Aspect ratio p / (n - d).

    Returns
    -------
    NDArray
        Shrunk covariance estimate of shape (p, p).
    """
    return apply_nonlinear_shrinkage(S_u, c_n)


def compute_rre_cov(S_r: NDArray, tilde_c_n: float) -> NDArray:
    """Restricted covariance-based estimator (RRE).

    Applies analytic nonlinear shrinkage to the restricted scatter estimate
    S_r at aspect ratio tilde_c_n = p / (n - d + q).

    Parameters
    ----------
    S_r : NDArray
        Restricted scatter estimate of shape (p, p), symmetric PSD.
    tilde_c_n : float
        Restricted aspect ratio p / (n - d + q).

    Returns
    -------
    NDArray
        Shrunk covariance estimate of shape (p, p).
    """
    return apply_nonlinear_shrinkage(S_r, tilde_c_n)


def compute_ure_tyler(
    E_hat: NDArray,
    c_n: float,
    eps: float = 1e-6,
) -> dict:
    """Robust unrestricted estimator (URE-Tyler).

    Procedure:
    1. Run Tyler's M-estimator on the unrestricted OLS residuals E_hat.
    2. Compute the scale factor via the chi-squared median calibration:
           sigma_sq = median(d_i) / median_chi2(p)
       where d_i = r_i' V^{-1} r_i and median_chi2(p) = scipy.stats.chi2(p).median().
    3. Apply analytic nonlinear shrinkage to V at aspect ratio c_n.
    4. Return Sigma_hat = sigma_sq * shrunk_V.

    Parameters
    ----------
    E_hat : NDArray
        Unrestricted OLS residual matrix of shape (n, p).
    c_n : float
        Unrestricted aspect ratio p / (n - d).
    eps : float, optional
        Ridge regularisation strength for Tyler's iteration, by default 1e-6.

    Returns
    -------
    dict
        'Sigma_hat'      : NDArray (p, p) — final robust URE estimate.
        'scale'          : float   — calibration factor sigma_sq.
        'V'              : NDArray (p, p) — Tyler scatter (tr = p, pre-scaling).
        'converged'      : bool    — convergence of Tyler's iteration.
        'n_iter'         : int     — number of Tyler iterations.
        'eps_used'       : float   — regularisation actually applied.
        'failure_reason' : str or None.
    """
    tyler_result = tyler_m_estimator(E_hat, eps=eps)
    V = tyler_result["V"]
    p = V.shape[0]

    # Compute Mahalanobis distances for scale calibration
    residuals = np.asarray(E_hat, dtype=float)
    row_norms = np.linalg.norm(residuals, axis=1)
    mask = row_norms > 0.0
    R = residuals[mask]

    try:
        V_inv = np.linalg.inv(V)
    except np.linalg.LinAlgError:
        V_inv = np.linalg.pinv(V)

    tmp = R @ V_inv
    distances = np.einsum("ij,ij->i", tmp, R)  # r_i' V^{-1} r_i

    med_d = float(np.median(distances))
    med_chi2 = float(scipy.stats.chi2(p).median())
    sigma_sq = med_d / med_chi2 if med_chi2 > 0 else 1.0

    # Nonlinear shrinkage of Tyler scatter
    V_shrunk = apply_nonlinear_shrinkage(V, c_n)
    Sigma_hat = sigma_sq * V_shrunk

    return {
        "Sigma_hat": Sigma_hat,
        "scale": sigma_sq,
        "V": V,
        "converged": tyler_result["converged"],
        "n_iter": tyler_result["n_iter"],
        "eps_used": tyler_result["eps_used"],
        "failure_reason": tyler_result["failure_reason"],
    }


def compute_rre_tyler(
    E_hat_r: NDArray,
    tilde_c_n: float,
    eps: float = 1e-6,
) -> dict:
    """Robust restricted estimator (RRE-Tyler).

    Procedure:
    1. Run Tyler's M-estimator on the restricted OLS residuals E_hat_r.
    2. Calibrate scale:
           sigma_sq = median(d_i) / median_chi2(p)
    3. Apply analytic nonlinear shrinkage to V at aspect ratio tilde_c_n.
    4. Return Sigma_hat = sigma_sq * shrunk_V.

    Parameters
    ----------
    E_hat_r : NDArray
        Restricted OLS residual matrix of shape (n, p).
    tilde_c_n : float
        Restricted aspect ratio p / (n - d + q).
    eps : float, optional
        Ridge regularisation strength for Tyler's iteration, by default 1e-6.

    Returns
    -------
    dict
        'Sigma_hat'      : NDArray (p, p) — final robust RRE estimate.
        'scale'          : float   — calibration factor sigma_sq.
        'V'              : NDArray (p, p) — Tyler scatter (tr = p, pre-scaling).
        'converged'      : bool    — convergence of Tyler's iteration.
        'n_iter'         : int     — number of Tyler iterations.
        'eps_used'       : float   — regularisation actually applied.
        'failure_reason' : str or None.
    """
    tyler_result = tyler_m_estimator(E_hat_r, eps=eps)
    V = tyler_result["V"]
    p = V.shape[0]

    residuals = np.asarray(E_hat_r, dtype=float)
    row_norms = np.linalg.norm(residuals, axis=1)
    mask = row_norms > 0.0
    R = residuals[mask]

    try:
        V_inv = np.linalg.inv(V)
    except np.linalg.LinAlgError:
        V_inv = np.linalg.pinv(V)

    tmp = R @ V_inv
    distances = np.einsum("ij,ij->i", tmp, R)

    med_d = float(np.median(distances))
    med_chi2 = float(scipy.stats.chi2(p).median())
    sigma_sq = med_d / med_chi2 if med_chi2 > 0 else 1.0

    V_shrunk = apply_nonlinear_shrinkage(V, tilde_c_n)
    Sigma_hat = sigma_sq * V_shrunk

    return {
        "Sigma_hat": Sigma_hat,
        "scale": sigma_sq,
        "V": V,
        "converged": tyler_result["converged"],
        "n_iter": tyler_result["n_iter"],
        "eps_used": tyler_result["eps_used"],
        "failure_reason": tyler_result["failure_reason"],
    }


def align_eigenvectors(U_ref: NDArray, U_target: NDArray) -> NDArray:
    """Align sign conventions of eigenvector columns in U_target with U_ref.

    For each column k, if the inner product U_ref[:,k] · U_target[:,k] < 0,
    flip the sign of U_target[:,k].  This resolves the sign indeterminacy of
    eigenvectors before combining two spectral decompositions.

    Parameters
    ----------
    U_ref : NDArray
        Reference eigenvector matrix of shape (p, p).
    U_target : NDArray
        Target eigenvector matrix of shape (p, p) to be aligned.

    Returns
    -------
    NDArray
        Aligned eigenvector matrix of shape (p, p) with corrected column signs.
    """
    signs = np.sign(np.diag(U_target.T @ U_ref))  # shape (p,)
    # Replace zeros (orthogonal eigenvectors) with +1
    signs[signs == 0] = 1.0
    return U_target * signs[None, :]


def compute_sse_plus(
    Sigma_ure: NDArray,
    Sigma_rre: NDArray,
    T_n: float,
    n: int,
    d: int,
    q: int,
) -> dict:
    """Positive-part Stein combination of URE and RRE estimates (SSE+).

    Computes the convex combination:

        Sigma_hat = (1 - kappa_n) * Sigma_ure + kappa_n * Sigma_rre

    where the shrinkage weight is:

        kappa_n = min(1, max(0, (q - 2) / ((n - d) * T_n)))

    Special cases:
    - If q < 3, kappa_n = 0 and SSE+ reduces to URE.
    - If T_n <= 0, kappa_n = 0.

    Eigenvector signs are aligned between Sigma_ure and Sigma_rre before
    forming the convex combination to avoid sign-flip artefacts.

    Parameters
    ----------
    Sigma_ure : NDArray
        Unrestricted estimator of shape (p, p).
    Sigma_rre : NDArray
        Restricted estimator of shape (p, p).
    T_n : float
        Restriction test statistic (>= 0).
    n : int
        Sample size.
    d : int
        Number of predictors.
    q : int
        Number of restrictions.

    Returns
    -------
    dict
        'Sigma_hat' : NDArray (p, p) — SSE+ estimate.
        'kappa_n'   : float   — actual shrinkage weight used.
    """
    if q < 3 or T_n <= 0.0:
        kappa_n = 0.0
    else:
        kappa_n = min(1.0, max(0.0, (q - 2.0) / ((n - d) * T_n)))

    if kappa_n == 0.0:
        return {"Sigma_hat": Sigma_ure.copy(), "kappa_n": kappa_n}

    # Align eigenvector signs of Sigma_rre to Sigma_ure
    _, U_ure = np.linalg.eigh(Sigma_ure)
    eigvals_rre, U_rre = np.linalg.eigh(Sigma_rre)
    U_rre_aligned = align_eigenvectors(U_ure, U_rre)

    # Reconstruct Sigma_rre with aligned eigenvectors (same spectrum, fixed signs)
    Sigma_rre_aligned = U_rre_aligned @ np.diag(eigvals_rre) @ U_rre_aligned.T
    Sigma_rre_aligned = (Sigma_rre_aligned + Sigma_rre_aligned.T) / 2.0

    Sigma_hat = (1.0 - kappa_n) * Sigma_ure + kappa_n * Sigma_rre_aligned
    Sigma_hat = (Sigma_hat + Sigma_hat.T) / 2.0  # enforce symmetry

    return {"Sigma_hat": Sigma_hat, "kappa_n": kappa_n}
