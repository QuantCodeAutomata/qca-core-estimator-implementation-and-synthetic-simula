"""Multivariate regression utilities for restricted and unrestricted OLS estimation.

Provides OLS, restricted OLS, and the restriction test statistic T_n used
to construct the positive-part Stein estimator.
"""

import logging
import warnings
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def compute_ols(X: NDArray, Y: NDArray) -> dict:
    """Compute ordinary least squares estimates for the multivariate linear model.

    Given Y = X @ B + E, computes:
    - G = (X'X)^{-1}                      (d x d Gram inverse)
    - P_X = X @ G @ X'                    (n x n hat / projection matrix)
    - B_hat = G @ X' @ Y                  (d x p OLS coefficient estimate)
    - E_hat = (I_n - P_X) @ Y            (n x p OLS residuals)
    - S_u = E_hat' @ E_hat / (n - d)      (p x p unrestricted scatter estimate)
    - c_n = p / (n - d)                   (aspect ratio for shrinkage)

    Numerical checks performed:
    - P_X is idempotent: ||P_X^2 - P_X||_F < 1e-8
    - rank(I_n - P_X) = n - d

    Parameters
    ----------
    X : NDArray
        Design matrix of shape (n, d).
    Y : NDArray
        Response matrix of shape (n, p).

    Returns
    -------
    dict
        Keys: 'G', 'P_X', 'B_hat', 'E_hat', 'S_u', 'c_n'.

    Raises
    ------
    ValueError
        If X does not have full column rank.
    """
    n, d = X.shape
    XtX = X.T @ X
    try:
        G = np.linalg.inv(XtX)
    except np.linalg.LinAlgError as exc:
        raise ValueError("X'X is singular; design matrix lacks full column rank.") from exc

    P_X = X @ G @ X.T

    # Idempotency check
    idempotency_err = np.linalg.norm(P_X @ P_X - P_X, "fro")
    if idempotency_err > 1e-8:
        logger.warning("P_X idempotency error: %.3e (threshold 1e-8)", idempotency_err)

    # Rank check for residual projector
    I_n = np.eye(n)
    M = I_n - P_X
    rank_M = np.linalg.matrix_rank(M)
    if rank_M != n - d:
        logger.warning("rank(I - P_X) = %d, expected %d", rank_M, n - d)

    B_hat = G @ X.T @ Y
    E_hat = M @ Y
    dof = n - d
    S_u = E_hat.T @ E_hat / dof
    c_n = Y.shape[1] / dof

    return {"G": G, "P_X": P_X, "B_hat": B_hat, "E_hat": E_hat, "S_u": S_u, "c_n": c_n}


def compute_restricted_ols(
    X: NDArray,
    Y: NDArray,
    R: NDArray,
    B_hat: NDArray,
    G: NDArray,
    P_X: NDArray,
) -> dict:
    """Compute restricted OLS estimates imposing the linear restriction R @ B = 0.

    Given existing unrestricted OLS results, the restricted estimator is:
    - Q = R @ G @ R'                           (q x q)
    - B_hat_r = B_hat - G @ R' @ Q^{-1} @ R @ B_hat
    - P_{X,R} = X @ G @ R' @ Q^{-1} @ R @ G @ X'
    - P_r = P_X - P_{X,R}                     (restricted projection matrix)
    - E_hat_r = (I_n - P_r) @ Y              (n x p restricted residuals)
    - S_r = E_hat_r' @ E_hat_r / (n - d + q)  (p x p restricted scatter estimate)
    - tilde_c_n = p / (n - d + q)              (restricted aspect ratio)

    Numerical check: rank(I_n - P_r) = n - d + q.

    Parameters
    ----------
    X : NDArray
        Design matrix of shape (n, d).
    Y : NDArray
        Response matrix of shape (n, p).
    R : NDArray
        Restriction matrix of shape (q, d).
    B_hat : NDArray
        Unrestricted OLS coefficient estimate of shape (d, p).
    G : NDArray
        Gram inverse (X'X)^{-1} of shape (d, d).
    P_X : NDArray
        OLS hat matrix of shape (n, n).

    Returns
    -------
    dict
        Keys: 'Q', 'B_hat_r', 'P_r', 'E_hat_r', 'S_r', 'tilde_c_n'.
    """
    n, d = X.shape
    p = Y.shape[1]
    q = R.shape[0]

    Q = R @ G @ R.T  # (q, q)
    Q_inv = np.linalg.inv(Q)

    GRt = G @ R.T  # (d, q)
    correction = GRt @ Q_inv @ R @ B_hat  # (d, p)
    B_hat_r = B_hat - correction

    P_XR = X @ GRt @ Q_inv @ GRt.T @ X.T  # (n, n)
    P_r = P_X - P_XR

    I_n = np.eye(n)
    M_r = I_n - P_r
    E_hat_r = M_r @ Y

    dof_r = n - d + q
    S_r = E_hat_r.T @ E_hat_r / dof_r
    tilde_c_n = p / dof_r

    # Rank check
    rank_Mr = np.linalg.matrix_rank(M_r)
    if rank_Mr != dof_r:
        logger.warning("rank(I - P_r) = %d, expected %d", rank_Mr, dof_r)

    return {
        "Q": Q,
        "B_hat_r": B_hat_r,
        "P_r": P_r,
        "E_hat_r": E_hat_r,
        "S_r": S_r,
        "tilde_c_n": tilde_c_n,
    }


def compute_restriction_stat(
    B_hat: NDArray, R: NDArray, G: NDArray, S_u: NDArray
) -> float:
    """Compute the restriction test statistic T_n.

    T_n = tr[B_hat' R' Q^{-1} R B_hat S_u^+] / (p * q)

    where Q = R @ G @ R' and S_u^+ is the Moore-Penrose pseudoinverse of S_u.
    T_n >= 0 is enforced by taking the absolute value to guard against
    negligible numerical negativity.

    Parameters
    ----------
    B_hat : NDArray
        Unrestricted OLS coefficient estimate of shape (d, p).
    R : NDArray
        Restriction matrix of shape (q, d).
    G : NDArray
        Gram inverse (X'X)^{-1} of shape (d, d).
    S_u : NDArray
        Unrestricted scatter estimate of shape (p, p).

    Returns
    -------
    float
        Test statistic T_n >= 0.
    """
    q = R.shape[0]
    p = S_u.shape[0]

    Q = R @ G @ R.T
    Q_inv = np.linalg.inv(Q)
    S_u_pinv = np.linalg.pinv(S_u)

    # Core quadratic form: (R B_hat)' Q^{-1} (R B_hat) @ S_u^+
    RB = R @ B_hat  # (q, p)
    A = Q_inv @ RB  # (q, p)
    T_n = np.trace(B_hat.T @ R.T @ A @ S_u_pinv) / (p * q)
    return float(abs(T_n))
