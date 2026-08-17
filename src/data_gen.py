"""Synthetic data generation for restricted robust covariance estimation experiments.

Supports Gaussian and heavy-tailed elliptical error distributions, multiple covariance
structures, and enforced linear restrictions on the coefficient matrix B.
"""

import logging
import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple
import scipy.stats

logger = logging.getLogger(__name__)


def generate_design_matrix(n: int, d: int, rng: np.random.Generator) -> NDArray:
    """Generate an n x d design matrix with iid standard Gaussian entries.

    Parameters
    ----------
    n : int
        Number of observations.
    d : int
        Number of predictors (columns).
    rng : np.random.Generator
        NumPy random generator for reproducibility.

    Returns
    -------
    NDArray
        Design matrix X of shape (n, d).

    Raises
    ------
    ValueError
        If the generated matrix does not have full column rank d.
    """
    X = rng.standard_normal((n, d))
    rank = np.linalg.matrix_rank(X)
    if rank != d:
        raise ValueError(f"Design matrix has rank {rank}, expected {d}. Increase n.")
    return X


def generate_restriction_matrix(q: int, d: int, rng: np.random.Generator) -> NDArray:
    """Generate a q x d restriction matrix R with orthonormal rows.

    Constructs R via QR decomposition of a random Gaussian matrix so that the
    rows of R form an orthonormal system, giving rank(R) = q exactly.

    Parameters
    ----------
    q : int
        Number of restrictions (rows).
    d : int
        Number of predictors (columns).
    rng : np.random.Generator
        NumPy random generator for reproducibility.

    Returns
    -------
    NDArray
        Restriction matrix R of shape (q, d) with orthonormal rows.

    Raises
    ------
    ValueError
        If q > d or if the resulting matrix does not have rank q.
    """
    if q > d:
        raise ValueError(f"q={q} cannot exceed d={d} for a full-rank restriction matrix.")
    A = rng.standard_normal((d, q))
    Q, _ = np.linalg.qr(A)
    R = Q[:, :q].T  # shape (q, d), orthonormal rows
    rank = np.linalg.matrix_rank(R)
    if rank != q:
        raise ValueError(f"Restriction matrix has rank {rank}, expected {q}.")
    return R


def generate_identity_covariance(p: int) -> NDArray:
    """Return the p x p identity matrix as a covariance structure.

    Parameters
    ----------
    p : int
        Dimension of the response (number of dependent variables).

    Returns
    -------
    NDArray
        Identity matrix of shape (p, p).
    """
    return np.eye(p)


def generate_banded_covariance(p: int, rho: float = 0.6, bandwidth: int = 10) -> NDArray:
    """Generate a banded covariance matrix.

    Entry (j, k) equals rho^|j-k| when |j-k| <= bandwidth, and 0 otherwise.
    The resulting matrix is symmetric positive definite for |rho| < 1.

    Parameters
    ----------
    p : int
        Dimension of the matrix.
    rho : float, optional
        Decay parameter, by default 0.6. Must satisfy |rho| < 1.
    bandwidth : int, optional
        Maximum lag with non-zero correlation, by default 10.

    Returns
    -------
    NDArray
        Banded covariance matrix of shape (p, p).
    """
    idx = np.arange(p)
    diff = np.abs(idx[:, None] - idx[None, :])
    Sigma = np.where(diff <= bandwidth, rho ** diff, 0.0)
    return Sigma


def generate_sparse_covariance(p: int, rng: np.random.Generator, seed: int = 0) -> NDArray:
    """Generate an approximately sparse symmetric positive definite covariance matrix.

    Implementation assumption (paper under-specified): starts from the identity,
    adds sparse symmetric off-diagonal entries drawn from Uniform([0.2, 0.6]) with
    random signs, then projects to the nearest SPD matrix via eigenvalue clipping
    (negative eigenvalues set to a small positive floor), and rescales so that
    tr(Sigma) = p.

    For each row i, floor(2 * sqrt(p)) off-diagonal column positions are chosen
    uniformly at random (without replacement) and symmetric values are assigned.

    Parameters
    ----------
    p : int
        Dimension of the matrix.
    rng : np.random.Generator
        NumPy random generator for reproducibility.
    seed : int, optional
        Unused legacy parameter kept for API compatibility, by default 0.

    Returns
    -------
    NDArray
        Approximately sparse SPD covariance matrix of shape (p, p).
    """
    Sigma = np.eye(p, dtype=float)
    k = int(np.floor(2 * np.sqrt(p)))

    for i in range(p):
        candidates = [j for j in range(p) if j != i]
        chosen = rng.choice(candidates, size=min(k, len(candidates)), replace=False)
        for j in chosen:
            if Sigma[i, j] == 0.0:  # avoid overwriting already-set symmetric entry
                val = rng.uniform(0.2, 0.6)
                sign = rng.choice([-1.0, 1.0])
                Sigma[i, j] = sign * val
                Sigma[j, i] = sign * val

    # Project to SPD via eigenvalue clipping
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    floor_val = max(1e-4, eigvals.max() * 1e-6)
    eigvals = np.maximum(eigvals, floor_val)
    Sigma = eigvecs @ np.diag(eigvals) @ eigvecs.T
    Sigma = (Sigma + Sigma.T) / 2  # enforce exact symmetry

    # Rescale so tr(Sigma) = p
    Sigma = Sigma * (p / np.trace(Sigma))
    return Sigma


def generate_coefficient_matrix(N_R: NDArray, p: int, rng: np.random.Generator) -> NDArray:
    """Generate the true coefficient matrix B satisfying R @ B = 0 exactly.

    Constructs B = N_R @ C where C has iid standard normal entries, so that
    B lies in the null space of R by construction.

    Parameters
    ----------
    N_R : NDArray
        Null space basis of R, shape (d, d-q). Columns span null(R).
    p : int
        Number of response variables (columns of B).
    rng : np.random.Generator
        NumPy random generator for reproducibility.

    Returns
    -------
    NDArray
        Coefficient matrix B of shape (d, p) satisfying R @ B = 0.
    """
    dim_null = N_R.shape[1]
    C = rng.standard_normal((dim_null, p))
    B = N_R @ C
    return B


def generate_elliptical_errors(
    n: int,
    Sigma: NDArray,
    error_law: str,
    nu: Optional[int],
    rng: np.random.Generator,
) -> NDArray:
    """Generate n x p elliptical error rows from a specified distribution.

    Each error vector e_i = sqrt(w_i) * z_i where z_i ~ N(0, Sigma).
    The mixing weight w_i controls the tail behaviour:

    - 'gaussian': w_i = 1 for all i (multivariate normal).
    - 't11', 't6', 't4': w_i = nu / chi2_i where chi2_i ~ chi2(nu),
      yielding multivariate-t with nu degrees of freedom.

    Parameters
    ----------
    n : int
        Number of observations.
    Sigma : NDArray
        True covariance/scatter matrix of shape (p, p).
    error_law : str
        Distribution family: 'gaussian', 't11', 't6', or 't4'.
    nu : Optional[int]
        Degrees of freedom for t-distributions; ignored for Gaussian.
    rng : np.random.Generator
        NumPy random generator for reproducibility.

    Returns
    -------
    NDArray
        Error matrix E of shape (n, p).

    Raises
    ------
    ValueError
        If error_law is not one of the supported values.
    """
    p = Sigma.shape[0]
    supported = {"gaussian", "t11", "t6", "t4"}
    if error_law not in supported:
        raise ValueError(f"error_law must be one of {supported}, got '{error_law}'.")

    # Cholesky factor for generating correlated normals
    L = np.linalg.cholesky(Sigma)
    Z = rng.standard_normal((n, p)) @ L.T  # shape (n, p), rows ~ N(0, Sigma)

    if error_law == "gaussian":
        return Z

    # t-distribution: mixing weight w_i = nu / chi2(nu), so e_i = sqrt(w_i) * z_i.
    # The scatter (shape) matrix is Sigma; the covariance is nu/(nu-2) * Sigma.
    # Tyler's M-estimator targets the shape matrix Sigma, and its scale is
    # recovered via chi-squared median calibration in estimators.compute_ure_tyler.
    chi2_samples = rng.chisquare(df=nu, size=n)  # shape (n,)
    w = nu / chi2_samples                         # shape (n,)
    E = np.sqrt(w)[:, None] * Z
    return E


def compute_null_space_basis(R: NDArray) -> NDArray:
    """Compute a basis for the null space of R using the economy SVD.

    The null space of R (of shape q x d) has dimension d - q when rank(R) = q.
    The last d-q right singular vectors (columns of V corresponding to zero or
    near-zero singular values) span null(R).

    Parameters
    ----------
    R : NDArray
        Restriction matrix of shape (q, d).

    Returns
    -------
    NDArray
        Null space basis N_R of shape (d, d-q). Columns are orthonormal
        and span null(R).
    """
    q, d = R.shape
    _, s, Vt = np.linalg.svd(R, full_matrices=True)
    # Singular values are sorted descending; last d-q correspond to null space
    N_R = Vt[q:, :].T  # shape (d, d-q)
    return N_R


def generate_data(
    n: int,
    d: int,
    p: int,
    q: int,
    cov_type: str,
    error_law: str,
    nu: Optional[int],
    rng: np.random.Generator,
) -> dict:
    """Generate a complete synthetic dataset for the restricted regression model.

    The model is Y = X @ B + E where:
    - X is the n x d design matrix,
    - B is the d x p coefficient matrix satisfying R @ B = 0,
    - R is the q x d restriction matrix,
    - E is the n x p error matrix with elliptical rows ~ (0, Sigma),
    - Sigma is the p x p true covariance matrix.

    Parameters
    ----------
    n : int
        Number of observations.
    d : int
        Number of predictors.
    p : int
        Number of response variables.
    q : int
        Number of linear restrictions.
    cov_type : str
        Covariance structure: 'identity', 'banded', or 'sparse'.
    error_law : str
        Error distribution: 'gaussian', 't11', 't6', or 't4'.
    nu : Optional[int]
        Degrees of freedom for t-distributions.
    rng : np.random.Generator
        NumPy random generator for reproducibility.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'X'        : NDArray (n, d), design matrix
        - 'B'        : NDArray (d, p), true coefficient matrix
        - 'R'        : NDArray (q, d), restriction matrix
        - 'N_R'      : NDArray (d, d-q), null space basis of R
        - 'Sigma'    : NDArray (p, p), true covariance matrix
        - 'E'        : NDArray (n, p), error matrix
        - 'Y'        : NDArray (n, p), response matrix
        - 'n'        : int
        - 'd'        : int
        - 'p'        : int
        - 'q'        : int
        - 'cov_type' : str
        - 'error_law': str
        - 'nu'       : Optional[int]

    Raises
    ------
    ValueError
        If cov_type is not one of 'identity', 'banded', 'sparse'.
    """
    supported_cov = {"identity", "banded", "sparse"}
    if cov_type not in supported_cov:
        raise ValueError(f"cov_type must be one of {supported_cov}, got '{cov_type}'.")

    X = generate_design_matrix(n, d, rng)
    R = generate_restriction_matrix(q, d, rng)
    N_R = compute_null_space_basis(R)
    B = generate_coefficient_matrix(N_R, p, rng)

    if cov_type == "identity":
        Sigma = generate_identity_covariance(p)
    elif cov_type == "banded":
        Sigma = generate_banded_covariance(p)
    else:  # sparse
        Sigma = generate_sparse_covariance(p, rng)

    E = generate_elliptical_errors(n, Sigma, error_law, nu, rng)
    Y = X @ B + E

    return {
        "X": X,
        "B": B,
        "R": R,
        "N_R": N_R,
        "Sigma": Sigma,
        "E": E,
        "Y": Y,
        "n": n,
        "d": d,
        "p": p,
        "q": q,
        "cov_type": cov_type,
        "error_law": error_law,
        "nu": nu,
    }
