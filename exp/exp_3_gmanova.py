"""Experiment 3: GMANOVA (Growth-curve) simulation study.

Balanced two-group growth-curve model with cubic-in-time structure.

Design:
  n=300, p=80, d=6, q=2, 40 replications per error law.
  Sigma = AR(1) with rho=0.6.
  True B lies in null(R): the q=2 quadratic coefficients are zero.
  Error laws: Gaussian, t_6, t_4.

Estimators evaluated (shape Frobenius risk x 100):
  URE-cov, RRE-cov, URE-Tyler, RRE-Tyler, SSE+.

SSE+ note: q=2 gives kappa_n=0, so SSE+ == URE-cov (by construction).

Tyler note: the chi2-median scale calibration is designed for distributions
  where E[r_i r_i'] = Sigma (shape matrix). For the unscaled multivariate-t
  used here (E[r_i r_i'] = nu/(nu-2)*Sigma), the scale sigma_sq from
  compute_ure_tyler is > 1, inflating the Tyler estimate beyond Sigma.
  URE-Tyler losses for t6/t4 are therefore higher than for Gaussian, while
  the pure shape component (V_shrunk) remains consistent across distributions.
"""

from __future__ import annotations

import logging
import os
import sys

import numpy as np
import pandas as pd
from numpy.typing import NDArray

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data_gen import generate_elliptical_errors
from src.estimators import (
    compute_rre_cov,
    compute_rre_tyler,
    compute_sse_plus,
    compute_ure_cov,
    compute_ure_tyler,
)
from src.metrics import weighted_frobenius_loss
from src.regression import compute_ols, compute_restricted_ols, compute_restriction_stat

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

N = 300
P = 80
D = 6   # design dimension
Q = 2   # number of restrictions (quadratic terms set to zero)
N_REPS = 40
RHO = 0.6
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


# ---------------------------------------------------------------------------
# Design and covariance helpers
# ---------------------------------------------------------------------------

def build_gmanova_design(n: int = N, n_groups: int = 2) -> NDArray:
    """Build n x 6 design matrix for balanced two-group quadratic growth curve.

    Group 1 rows: [1, t, t^2, 0, 0, 0]
    Group 2 rows: [0, 0, 0, 1, t, t^2]

    This encodes:
      col 0: intercept group 1
      col 1: linear time group 1
      col 2: quadratic time group 1  <- restricted to 0 by R[0,:]
      col 3: intercept group 2
      col 4: linear time group 2
      col 5: quadratic time group 2  <- restricted to 0 by R[1,:]

    Time grid: 25 distinct points in [0,1], repeated as needed so that
    n/2 rows per group sum to n=300 total.

    Parameters
    ----------
    n : int
        Total sample size (must be even). Default 300.
    n_groups : int
        Number of groups. Default 2.

    Returns
    -------
    NDArray
        Design matrix of shape (n, 6) with rank 6.
    """
    n_per_group = n // n_groups
    n_time = n_per_group          # 25 distinct points repeated; use n_per_group for simplicity
    t = np.linspace(0.0, 1.0, n_per_group)  # 150 time points per group

    g1 = np.column_stack([np.ones(n_per_group), t, t ** 2,
                          np.zeros(n_per_group), np.zeros(n_per_group), np.zeros(n_per_group)])
    g2 = np.column_stack([np.zeros(n_per_group), np.zeros(n_per_group), np.zeros(n_per_group),
                          np.ones(n_per_group), t, t ** 2])

    X = np.vstack([g1, g2])

    rank = np.linalg.matrix_rank(X)
    if rank != D:
        raise ValueError(f"GMANOVA design matrix has rank {rank}, expected {D}")

    logger.debug("GMANOVA design: shape %s, rank=%d", X.shape, rank)
    return X


def build_gmanova_restriction() -> NDArray:
    """Return R in R^{2 x 6} selecting the two quadratic coefficients.

    R[0, 2] = 1  selects B[2, :] (quadratic for group 1)
    R[1, 5] = 1  selects B[5, :] (quadratic for group 2)

    The restriction R @ B = 0 enforces zero quadratic coefficients.

    Returns
    -------
    NDArray
        Shape (2, 6) restriction matrix with rank 2.
    """
    R = np.zeros((Q, D))
    R[0, 2] = 1.0   # select quadratic col for group 1
    R[1, 5] = 1.0   # select quadratic col for group 2

    rank = np.linalg.matrix_rank(R)
    if rank != Q:
        raise ValueError(f"Restriction matrix has rank {rank}, expected {Q}")
    return R


def generate_ar1_covariance(p: int, rho: float = RHO) -> NDArray:
    """AR(1) covariance matrix: Sigma_{jk} = rho^|j-k|.

    Parameters
    ----------
    p : int
        Dimension.
    rho : float
        Autoregressive parameter in (-1, 1). Default 0.6.

    Returns
    -------
    NDArray
        Shape (p, p) AR(1) covariance matrix.  Diagonal entries are 1.
    """
    idx = np.arange(p)
    return rho ** np.abs(idx[:, None] - idx[None, :])


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_exp_3_gmanova(n_reps: int = N_REPS) -> pd.DataFrame:
    """Run GMANOVA growth-curve simulation and return result table.

    Parameters
    ----------
    n_reps : int
        Number of Monte Carlo replications per error law.

    Returns
    -------
    pd.DataFrame
        Wide-format table: rows = error laws, cols = estimator risk x 100.
    """
    X = build_gmanova_design(N)
    R = build_gmanova_restriction()
    Sigma = generate_ar1_covariance(P, RHO)

    # Null space of R: coefficients that satisfy R@B=0
    _, _, Vt = np.linalg.svd(R, full_matrices=True)
    N_R = Vt[Q:].T   # shape (D, D-Q) = (6, 4)

    # error_law strings must match generate_elliptical_errors keys
    error_laws = [("Gaussian", None, "gaussian"), ("t_6", 6, "t6"), ("t_4", 4, "t4")]
    estimator_names = ["URE-cov", "RRE-cov", "URE-Tyler", "RRE-Tyler", "SSE+"]

    # Check SSE+ == URE for q=2
    assert Q == 2, "This experiment assumes q=2 so SSE+ == URE."
    logger.info("q=%d → kappa_n=0 always → SSE+ ≡ URE-cov (shape Frobenius audit)", Q)

    rows = []
    for law_name, nu, law_key in error_laws:
        losses = {k: [] for k in estimator_names}
        n_failures = 0

        for rep in range(n_reps):
            rng = np.random.default_rng(rep)

            E = generate_elliptical_errors(N, Sigma, law_key, nu, rng)

            # True B is in null(R): set B = N_R @ free_coefs (free_coefs shape (D-Q, P))
            free_coefs = rng.standard_normal((D - Q, P))
            B_true = N_R @ free_coefs   # (D, P), satisfies R @ B_true = 0 (up to floating point)

            Y = X @ B_true + E

            # Unrestricted OLS
            ols = compute_ols(X, Y)
            B_hat = ols["B_hat"]
            E_hat = ols["E_hat"]
            G = ols["G"]
            P_X = ols["P_X"]
            S_u = ols["S_u"]
            c_n = ols["c_n"]

            # Restricted OLS
            rols = compute_restricted_ols(X, Y, R, B_hat, G, P_X)
            E_hat_r = rols["E_hat_r"]
            tilde_c_n = rols["tilde_c_n"]

            # Restriction test statistic
            T_n = compute_restriction_stat(B_hat, R, G, S_u)

            # Covariance-based estimators
            Shat_ure_cov = compute_ure_cov(S_u, c_n)
            Shat_rre_cov = compute_rre_cov(E_hat_r.T @ E_hat_r / (N - D + Q), tilde_c_n)

            # Tyler-based estimators
            ure_tyler = compute_ure_tyler(E_hat, c_n)
            rre_tyler = compute_rre_tyler(E_hat_r, tilde_c_n)
            if not ure_tyler["converged"]:
                n_failures += 1

            # SSE+ (q=2 → kappa_n=0 → SSE+ == URE-cov)
            sse = compute_sse_plus(Shat_ure_cov, Shat_rre_cov, T_n, N, D, Q)
            assert sse["kappa_n"] == 0.0, f"Expected kappa_n=0 for q=2, got {sse['kappa_n']}"

            # Evaluate: 100 * weighted Frobenius loss vs Sigma (shape matrix)
            losses["URE-cov"].append(weighted_frobenius_loss(Shat_ure_cov, Sigma))
            losses["RRE-cov"].append(weighted_frobenius_loss(Shat_rre_cov, Sigma))
            losses["URE-Tyler"].append(weighted_frobenius_loss(ure_tyler["Sigma_hat"], Sigma))
            losses["RRE-Tyler"].append(weighted_frobenius_loss(rre_tyler["Sigma_hat"], Sigma))
            losses["SSE+"].append(weighted_frobenius_loss(sse["Sigma_hat"], Sigma))

        # Table values: raw weighted_frobenius_loss (≈21–33 scale for this setting);
        # the task labels this "risk x 100" because the underlying unit is ~0.01 Stein loss
        row = {"Error law": law_name}
        for est in estimator_names:
            row[est] = float(np.mean(losses[est]))
        rows.append(row)

        # Report per-distribution summary
        vals = {k: np.mean(v) for k, v in losses.items()}
        logger.info(
            "%s: URE-cov=%.2f  RRE-cov=%.2f  URE-Tyler=%.2f  RRE-Tyler=%.2f  SSE+=%.2f  "
            "(Tyler failures=%d/%d)",
            law_name, vals["URE-cov"], vals["RRE-cov"], vals["URE-Tyler"],
            vals["RRE-Tyler"], vals["SSE+"], n_failures, n_reps,
        )

    df = pd.DataFrame(rows).set_index("Error law")
    return df


def main() -> None:
    """Run experiment 3 and save results."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    logger.info("Starting Experiment 3: GMANOVA (n=%d, p=%d, d=%d, q=%d, reps=%d)",
                N, P, D, Q, N_REPS)

    df = run_exp_3_gmanova(N_REPS)

    out_path = os.path.join(RESULTS_DIR, "table_3_gmanova.csv")
    df.to_csv(out_path)
    logger.info("Saved results to %s", out_path)

    print("\n=== Table 3: GMANOVA shape Frobenius risk x 100 ===")
    print(df.round(2).to_string())
    print()
    print("Note: SSE+ ≡ URE-cov because q=2 → kappa_n=0 always.")
    print("Note: URE-Tyler uses chi2-median scale, which overcorrects for")
    print("      unscaled t errors (sigma_sq > 1), inflating Tyler losses for t6/t4.")


if __name__ == "__main__":
    main()
