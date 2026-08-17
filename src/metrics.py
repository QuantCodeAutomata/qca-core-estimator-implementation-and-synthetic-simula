"""Evaluation metrics for covariance estimator performance.

Provides matrix loss functions, a held-out log-likelihood, and diagnostic
utilities for assessing the quality of covariance/scatter estimates.
"""

import logging
import math
import warnings
import numpy as np
from numpy.typing import NDArray
import scipy.stats
from scipy.special import gammaln

logger = logging.getLogger(__name__)


def weighted_frobenius_loss(Sigma_hat: NDArray, Sigma_true: NDArray) -> float:
    """Compute the weighted (Stein) Frobenius loss.

    Loss = tr((Sigma_true^{-1} @ Sigma_hat - I_p)^2)

    This equals ||Sigma_true^{-1/2} Sigma_hat Sigma_true^{-1/2} - I_p||_F^2
    and is the standard Stein loss for covariance matrices.  Results are
    scaled by 100 when reported in tables.

    Parameters
    ----------
    Sigma_hat : NDArray
        Estimated covariance matrix of shape (p, p).
    Sigma_true : NDArray
        True covariance matrix of shape (p, p), must be invertible.

    Returns
    -------
    float
        Weighted Frobenius loss (non-negative).
    """
    Sigma_true_inv = np.linalg.inv(Sigma_true)
    A = Sigma_true_inv @ Sigma_hat - np.eye(Sigma_hat.shape[0])
    return float(np.trace(A @ A))


def operator_norm_error(Sigma_hat: NDArray, Sigma_true: NDArray) -> float:
    """Compute the spectral (operator) norm error.

    Error = ||Sigma_true^{-1/2} Sigma_hat Sigma_true^{-1/2} - I_p||_2

    where ||·||_2 denotes the largest singular value (spectral norm).

    Parameters
    ----------
    Sigma_hat : NDArray
        Estimated covariance matrix of shape (p, p).
    Sigma_true : NDArray
        True covariance matrix of shape (p, p), must be positive definite.

    Returns
    -------
    float
        Operator norm error (non-negative).
    """
    # Compute Sigma_true^{-1/2} via eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(Sigma_true)
    eigvals = np.maximum(eigvals, 1e-15)
    Sigma_half_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

    M = Sigma_half_inv @ Sigma_hat @ Sigma_half_inv - np.eye(Sigma_hat.shape[0])
    return float(np.linalg.norm(M, ord=2))


def condition_number(Sigma_hat: NDArray) -> float:
    """Return the condition number of Sigma_hat.

    Defined as lambda_max / lambda_min.  Returns numpy.inf if the matrix is
    singular (minimum eigenvalue <= 0) or if a linear algebra error occurs.

    Parameters
    ----------
    Sigma_hat : NDArray
        Square matrix of shape (p, p).

    Returns
    -------
    float
        Condition number, or numpy.inf if the matrix is singular.
    """
    try:
        eigvals = np.linalg.eigvalsh(Sigma_hat)
        lam_min = eigvals.min()
        lam_max = eigvals.max()
        if lam_min <= 0.0:
            return float(np.inf)
        return float(lam_max / lam_min)
    except np.linalg.LinAlgError:
        return float(np.inf)


def is_psd(matrix: NDArray, tol: float = 1e-8) -> bool:
    """Check whether a matrix is positive semidefinite.

    A matrix is considered PSD if all eigenvalues are >= -tol (allowing for
    small numerical negativity due to floating-point arithmetic).

    Parameters
    ----------
    matrix : NDArray
        Square symmetric matrix of shape (p, p).
    tol : float, optional
        Numerical tolerance for negative eigenvalues, by default 1e-8.

    Returns
    -------
    bool
        True if all eigenvalues are >= -tol, False otherwise.
    """
    try:
        eigvals = np.linalg.eigvalsh(matrix)
        return bool(eigvals.min() >= -tol)
    except np.linalg.LinAlgError:
        return False


def multivariate_t_log_likelihood(
    Y_test: NDArray,
    mu: NDArray,
    Sigma_hat: NDArray,
    nu: float = 5.0,
) -> float:
    """Compute held-out multivariate-t log-likelihood for test observations.

    For each test observation y_i, the log-density of the multivariate-t
    distribution with location mu, scatter Sigma, and nu degrees of freedom is:

        log f(y_i) = gammaln((nu+p)/2) - gammaln(nu/2)
                     - (p/2)*log(nu*pi) - (1/2)*log|Sigma|
                     - ((nu+p)/2) * log(1 + mahal_i / nu)

    where mahal_i = (y_i - mu)' Sigma^{-1} (y_i - mu).

    Uses Cholesky decomposition for numerically stable log-determinant and
    quadratic form evaluation.

    Parameters
    ----------
    Y_test : NDArray
        Test observations of shape (m, p).
    mu : NDArray
        Location vector of shape (p,) or (1, p).
    Sigma_hat : NDArray
        Scatter matrix estimate of shape (p, p).
    nu : float, optional
        Degrees of freedom, by default 5.0.

    Returns
    -------
    float
        Mean log-likelihood over test observations, or -inf if Sigma_hat is
        not positive definite.
    """
    Y_test = np.asarray(Y_test, dtype=float)
    mu = np.asarray(mu, dtype=float).ravel()
    Sigma_hat = np.asarray(Sigma_hat, dtype=float)
    p = Sigma_hat.shape[0]
    m = Y_test.shape[0]

    # Attempt Cholesky decomposition
    try:
        L = np.linalg.cholesky(Sigma_hat)
    except np.linalg.LinAlgError:
        logger.warning("Sigma_hat is not positive definite; returning -inf log-likelihood.")
        return float("-inf")

    log_det = 2.0 * np.sum(np.log(np.diag(L)))

    # Mahalanobis distances via forward substitution: solve L z = (y - mu)
    centered = Y_test - mu[None, :]  # (m, p)
    z = np.linalg.solve(L, centered.T)  # (p, m)
    mahal = np.sum(z ** 2, axis=0)  # (m,)

    # Log-density for each observation
    const = (
        gammaln((nu + p) / 2.0)
        - gammaln(nu / 2.0)
        - 0.5 * p * np.log(nu * np.pi)
        - 0.5 * log_det
    )
    log_probs = const - 0.5 * (nu + p) * np.log(1.0 + mahal / nu)  # (m,)

    return float(np.mean(log_probs))


def trace_ratio(Sigma_hat: NDArray, p: int, n_minus_d: int) -> float:
    """Compute the estimated average trace ratio tr(Sigma_hat) / p.

    This diagnostic checks whether the estimator correctly captures the
    overall scale of the covariance matrix, normalised by dimension.

    Parameters
    ----------
    Sigma_hat : NDArray
        Estimated covariance matrix of shape (p, p).
    p : int
        Dimension of the matrix.
    n_minus_d : int
        Effective sample size n - d (unused in the ratio but kept for API
        consistency and future use).

    Returns
    -------
    float
        tr(Sigma_hat) / p.
    """
    return float(np.trace(Sigma_hat) / p)
