"""Eigenvalue shrinkage estimators for covariance matrix estimation.

Implements analytic nonlinear shrinkage (custom, from scratch) and sklearn-based
linear shrinkage (LedoitWolf).

Nonlinear shrinkage reference
------------------------------
Ledoit, O. & Wolf, M. (2020). Analytical nonlinear shrinkage of large-dimensional
covariance matrices. Annals of Statistics, 48(5), 3043-3065.

The shrinkage formula used here approximates the Oracle Approximating Shrinkage
(OAS) estimator via the sample Stieltjes transform evaluated at each eigenvalue.
"""

import logging
import warnings
import numpy as np
from numpy.typing import NDArray
import scipy.stats
from sklearn.covariance import LedoitWolf

logger = logging.getLogger(__name__)


def marchenko_pastur_density(x: float, c: float) -> float:
    """Evaluate the Marchenko-Pastur limiting spectral density at x.

    The MP density for aspect ratio c (= p / n) is:

        f_{MP}(x) = sqrt((lambda_+ - x)(x - lambda_-)) / (2 * pi * c * x)

    on the support [lambda_-, lambda_+], where
        lambda_{+/-} = (1 +/- sqrt(c))^2.

    Parameters
    ----------
    x : float
        Point at which to evaluate the density.
    c : float
        Aspect ratio p / n.  Must be in (0, 1] for a proper density;
        values in (1, inf) correspond to the singular case.

    Returns
    -------
    float
        Density value at x, or 0.0 outside the support.
    """
    if c <= 0:
        raise ValueError(f"Aspect ratio c must be positive, got {c}.")
    lam_plus = (1.0 + np.sqrt(c)) ** 2
    lam_minus = (1.0 - np.sqrt(c)) ** 2
    if x <= lam_minus or x >= lam_plus:
        return 0.0
    return np.sqrt((lam_plus - x) * (x - lam_minus)) / (2.0 * np.pi * c * x)


def nonlinear_shrink_eigenvalues(eigenvalues: NDArray, c: float) -> NDArray:
    """Apply analytic nonlinear shrinkage to sample eigenvalues.

    Uses the Ledoit-Wolf (2020) Oracle-Approximating Shrinkage approach
    via the alternate Stieltjes transform m_alt(z) = (1/p) sum_j 1/(z - lam_j),
    evaluated at z = lam_k + ih with a small consistent bandwidth h.

    For each eigenvalue lam_k the shrunken eigenvalue is:

        lam_k^* = lam_k / |1 - c + c * lam_k * m_alt(lam_k + ih)|^2

    where Re[m_alt] = (1/p) sum_j (lam_k - lam_j) / ((lam_k - lam_j)^2 + h^2)
    and |Im[m_alt]| = (1/p) sum_j h / ((lam_k - lam_j)^2 + h^2).

    Both real and imaginary parts use the same bandwidth h (Silverman-scaled
    by 0.08), ensuring a consistent regularised Stieltjes transform.

    Parameters
    ----------
    eigenvalues : NDArray
        Sample eigenvalues of shape (p,), sorted in any order (will be sorted
        internally). Must all be non-negative.
    c : float
        Aspect ratio p / n, where n is the effective sample size.

    Returns
    -------
    NDArray
        Shrunken eigenvalues of shape (p,), in the same order as input.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    order = np.argsort(eigenvalues)
    inv_order = np.argsort(order)
    lam = eigenvalues[order]  # sorted ascending
    p = len(lam)

    # Bandwidth: 0.08 * min(std, iqr/1.34) * p^{-1/5}
    # Small bandwidth ensures Im_m is a good approximation to pi*f and
    # Re_m converges to the Cauchy principal value without sign inconsistency.
    std_lam = np.std(lam, ddof=1)
    iqr_lam = float(np.subtract(*np.percentile(lam, [75, 25])))
    bw = 0.08 * min(std_lam, iqr_lam / 1.34) * p ** (-0.2)
    bw = max(bw, 1e-8 * lam.max() if lam.max() > 0 else 1e-8)

    # Alternate Stieltjes transform at lam_k + i*bw:
    #   diffs[i,j] = lam_i - lam_j
    #   Re[m_alt(lam_k + i*bw)] = (1/p) sum_j (lam_k - lam_j) / ((lam_k-lam_j)^2 + bw^2)
    #   |Im[m_alt]|              = (1/p) sum_j bw / ((lam_k-lam_j)^2 + bw^2)
    diffs = lam[:, None] - lam[None, :]   # (p, p), diffs[i,j] = lam_i - lam_j
    diffs_sq = diffs ** 2 + bw ** 2       # (p, p)
    Re_m = np.mean(diffs / diffs_sq, axis=1)         # (p,)
    Im_m = np.mean(bw / diffs_sq, axis=1)            # (p,), positive

    # Denominator: |1 - c + c * lam_k * m_alt(lam_k + i*bw)|^2
    factor = 1.0 - c + c * lam * Re_m
    denom_sq = np.maximum(factor ** 2 + (c * lam * Im_m) ** 2, 1e-15)

    # Shrunken eigenvalues
    lam_star = lam / denom_sq

    # Ensure all shrunken eigenvalues are positive
    min_positive = np.min(lam[lam > 0]) * 1e-3 if np.any(lam > 0) else 1e-8
    lam_star = np.maximum(lam_star, min_positive)

    return lam_star[inv_order]


def apply_nonlinear_shrinkage(matrix: NDArray, c: float) -> NDArray:
    """Eigendecompose a matrix, shrink its eigenvalues, and reconstruct.

    Applies nonlinear shrinkage to the eigenvalues of a symmetric PSD matrix
    using :func:`nonlinear_shrink_eigenvalues`, then reconstructs the matrix
    via the spectral decomposition.  The output is symmetric and PSD.

    Parameters
    ----------
    matrix : NDArray
        Symmetric positive (semi)definite matrix of shape (p, p).
    c : float
        Aspect ratio p / n for the shrinkage formula.

    Returns
    -------
    NDArray
        Shrunk matrix of shape (p, p), symmetric PSD.
    """
    matrix = np.asarray(matrix, dtype=float)
    matrix = (matrix + matrix.T) / 2.0  # enforce exact symmetry

    eigvals, eigvecs = np.linalg.eigh(matrix)  # eigh guarantees real eigenvalues

    # Clip any small negative eigenvalues before shrinkage
    eigvals = np.maximum(eigvals, 0.0)

    shrunken = nonlinear_shrink_eigenvalues(eigvals, c)
    shrunken = np.maximum(shrunken, 0.0)  # enforce PSD

    result = eigvecs @ np.diag(shrunken) @ eigvecs.T
    result = (result + result.T) / 2.0  # enforce exact symmetry
    return result


def linear_ledoit_wolf_shrinkage(
    residuals: NDArray, assume_centered: bool = True
) -> NDArray:
    """Apply sklearn LedoitWolf linear shrinkage to residual rows.

    # Using sklearn LedoitWolf - Context7 confirmed.

    Fits the Ledoit-Wolf shrinkage estimator to the residual matrix and
    returns the resulting precision-adjusted covariance estimate.

    Parameters
    ----------
    residuals : NDArray
        Residual matrix of shape (m, p). Rows are individual observations.
    assume_centered : bool, optional
        If True, data are assumed centred (mean subtracted), by default True.
        Passed directly to :class:`sklearn.covariance.LedoitWolf`.

    Returns
    -------
    NDArray
        Linear Ledoit-Wolf shrinkage covariance estimate of shape (p, p).
    """
    lw = LedoitWolf(assume_centered=assume_centered)
    lw.fit(residuals)
    return lw.covariance_
