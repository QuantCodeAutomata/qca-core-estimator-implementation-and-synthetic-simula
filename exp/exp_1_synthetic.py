"""
exp_1_synthetic.py — Monte Carlo synthetic simulation study.

Seven experiment families (exp_a–exp_g) testing covariance/shape estimation
under linear restrictions.  Results written to /workspace/project/results/.

API assumptions (verified against src/ implementation)
-------------------------------------------------------
generate_design_matrix(n, d, rng)             → ndarray (n, d)
generate_restriction_matrix(q, d, rng)        → ndarray (q, d)
compute_null_space_basis(R)                   → ndarray (d, d-q)
generate_coefficient_matrix(N_R, p, rng)      → ndarray (d, p)  [B = N_R @ C]
generate_elliptical_errors(n, Sigma, error_law, nu, rng) → ndarray (n, p)
    error_law ∈ {'gaussian', 't6', 't11'};  nu=None for Gaussian
generate_identity_covariance(p)               → ndarray (p, p)
generate_banded_covariance(p, rho, bandwidth) → ndarray (p, p)
generate_sparse_covariance(p, rng)            → ndarray (p, p)

compute_ols(X, Y)                             → dict {G, P_X, B_hat, E_hat, S_u, c_n}
compute_restricted_ols(X, Y, R, B_hat, G, P_X) → dict {Q, B_hat_r, P_r, E_hat_r, S_r, tilde_c_n}
compute_restriction_stat(B_hat, R, G, S_u)   → float

compute_ure_cov(S_u, c_n)                    → ndarray (p, p)
compute_rre_cov(S_r, tilde_c_n)              → ndarray (p, p)
compute_ure_tyler(E_hat, c_n)                → dict {Sigma_hat, converged, ...}
compute_rre_tyler(E_hat_r, tilde_c_n)        → dict {Sigma_hat, converged, ...}
compute_sse_plus(Sigma_ure, Sigma_rre, T_n, n, d, q) → dict {Sigma_hat, kappa_n}

weighted_frobenius_loss(Sigma_hat, Sigma_true) → float
operator_norm_error(Sigma_hat, Sigma_true)   → float
condition_number(Sigma_hat)                  → float
is_psd(Sigma_hat)                            → bool

plot_eigenvalue_histogram(eigenvalues_dict, c, title, save_path)
plot_risk_vs_aspect_ratio(results_df, metric, title, save_path)
    df columns: 'c' (= p/(n-d)), 'estimator', metric
plot_risk_vs_misspecification(results_df, title, save_path)
    df columns: 'delta', 'estimator', 'relative_risk'
plot_heatmap(error_matrix, title, save_path)
    error_matrix: ndarray (p, p)  absolute entry-wise error
plot_bias_variance(results, title, save_path)
    results: dict with 'estimator', 'bias_sq', 'variance' (and optionally 'mse')

PAPER INCONSISTENCIES / OPEN QUESTIONS (logged to results/diagnostics.txt)
---------------------------------------------------------------------------
• exp_a: "V_r" is not defined in the spec.  Interpreted as S_u (unrestricted
  scatter), whose spectral limit follows MP(c) with c = p/(n-d).
• exp_d: delta^2 calibration uses a fixed design matrix X_calib to make B
  constant across replications.  Per-rep delta^2 varies slightly as G=(X'X)^{-1}
  fluctuates, which matches standard population-level calibration practice.
• exp_e: theoretical ratio (n-d)/(n-d+q) is exact only under Gaussian errors
  and identity Sigma; used as a benchmark for all settings.
• Tyler feasibility: URE-Tyler infeasible when c_n = p/(n-d) ≥ 1;
  RRE-Tyler infeasible when tilde_c_n = p/(n-d+q) ≥ 1.  Infeasible runs
  return NaN for the Tyler-based estimator losses.
"""

from __future__ import annotations

import math
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.typing import NDArray
from scipy.optimize import brentq
import seaborn as sns

from src.data_gen import (
    compute_null_space_basis,
    generate_banded_covariance,
    generate_coefficient_matrix,
    generate_design_matrix,
    generate_elliptical_errors,
    generate_identity_covariance,
    generate_restriction_matrix,
    generate_sparse_covariance,
)
from src.estimators import (
    compute_rre_cov,
    compute_rre_tyler,
    compute_sse_plus,
    compute_ure_cov,
    compute_ure_tyler,
)
from src.metrics import (
    condition_number,
    is_psd,
    operator_norm_error,
    weighted_frobenius_loss,
)
from src.plotting import (
    plot_bias_variance,
    plot_eigenvalue_histogram,
    plot_heatmap,
    plot_risk_vs_aspect_ratio,
    plot_risk_vs_misspecification,
)
from src.regression import (
    compute_ols,
    compute_restricted_ols,
    compute_restriction_stat,
)

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BASE_SEED = 42

# Internal names for the five estimators (shrinkage-based + Tyler-based)
_EST_NAMES = ["URE", "RRE", "SSE+", "URE-Tyler", "RRE-Tyler"]

# Error-law configs: (string_key, label, nu)
_LAWS = [
    ("gaussian", "Gaussian", None),
    ("t6",       "t₆",      6),
    ("t11",      "t₁₁",     11),
]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _d_default(n: int) -> int:
    """Return floor(0.2 * n) as the default predictor count."""
    return math.floor(0.2 * n)


def _tilde_c(p: int, n: int, d: int, q: int) -> float:
    return p / (n - d + q)


def _c_eff(p: int, n: int, d: int) -> float:
    return p / (n - d)


def _theory_valid(p: int, n: int, d: int, q: int) -> bool:
    return _tilde_c(p, n, d, q) < 1.0


def _run_ols_pipeline(
    X: NDArray,
    Y: NDArray,
    R: NDArray,
    n: int,
    d: int,
    q: int,
    Sigma: NDArray,
) -> Dict:
    """Full OLS + restricted OLS + all five estimators for one replication.

    Returns a dict of WF losses, op-norm errors, eigenvalue arrays, and
    intermediate matrices.  Tyler-based losses are NaN when infeasible.
    """
    ols = compute_ols(X, Y)
    G, P_X, B_hat, E_hat = ols["G"], ols["P_X"], ols["B_hat"], ols["E_hat"]
    S_u, c_n = ols["S_u"], ols["c_n"]

    rols = compute_restricted_ols(X, Y, R, B_hat, G, P_X)
    E_hat_r, S_r, tilde_c_n = rols["E_hat_r"], rols["S_r"], rols["tilde_c_n"]

    T_n = compute_restriction_stat(B_hat, R, G, S_u)

    # Shrinkage estimators (always feasible)
    Sigma_ure = compute_ure_cov(S_u, c_n)
    Sigma_rre = compute_rre_cov(S_r, tilde_c_n)
    sse_result = compute_sse_plus(Sigma_ure, Sigma_rre, T_n, n, d, q)
    Sigma_sse, kappa_n = sse_result["Sigma_hat"], sse_result["kappa_n"]

    # Tyler estimators (feasibility gated by aspect ratio < 1)
    def _tyler_ure() -> Optional[NDArray]:
        if c_n >= 1.0:
            return None
        res = compute_ure_tyler(E_hat, c_n)
        return res["Sigma_hat"] if res.get("converged", False) else None

    def _tyler_rre() -> Optional[NDArray]:
        if tilde_c_n >= 1.0:
            return None
        res = compute_rre_tyler(E_hat_r, tilde_c_n)
        return res["Sigma_hat"] if res.get("converged", False) else None

    Sigma_ure_t = _tyler_ure()
    Sigma_rre_t = _tyler_rre()

    def _wf(S: Optional[NDArray]) -> float:
        return weighted_frobenius_loss(S, Sigma) if S is not None else float("nan")

    def _op(S: Optional[NDArray]) -> float:
        return operator_norm_error(S, Sigma) if S is not None else float("nan")

    return {
        "wf_URE":        _wf(Sigma_ure),
        "wf_RRE":        _wf(Sigma_rre),
        "wf_SSE+":       _wf(Sigma_sse),
        "wf_URE-Tyler":  _wf(Sigma_ure_t),
        "wf_RRE-Tyler":  _wf(Sigma_rre_t),
        "op_URE":        _op(Sigma_ure),
        "op_RRE":        _op(Sigma_rre),
        "op_SSE+":       _op(Sigma_sse),
        "eigs_S_r":      np.linalg.eigvalsh(S_r),
        "eigs_S_u":      np.linalg.eigvalsh(S_u),
        "c_n":           float(c_n),
        "tilde_c_n":     float(tilde_c_n),
        "kappa_n":       float(kappa_n),
        # Return estimator matrices for bias-variance decomposition
        "Sigma_ure":     Sigma_ure,
        "Sigma_rre":     Sigma_rre,
        "Sigma_sse":     Sigma_sse,
        "Sigma_ure_t":   Sigma_ure_t,
        "Sigma_rre_t":   Sigma_rre_t,
    }


# ---------------------------------------------------------------------------
# Misspecification calibration
# ---------------------------------------------------------------------------


def calibrate_misspecification(
    B_null: NDArray,
    N_R: NDArray,
    R: NDArray,
    G: NDArray,
    Sigma: NDArray,
    target_delta2: float,
    p: int,
    n: int,
    calib_seed: int = BASE_SEED,
) -> NDArray:
    """Return B = B_null + B_violate calibrated so delta² ≈ target_delta2.

    delta² = tr[(RB)'(R G R')⁻¹(RB) Σ⁻¹] / (p·n)

    Construction
    ------------
    - Fix random direction M ∈ R^{q×p} (drawn once with calib_seed).
    - Set RB_violate = α·M  ↔  B_violate = R_right_inv @ (α·M).
    - delta²(α) = α²·C  where C = tr[M'(RGR')⁻¹ M Σ⁻¹]/(p·n).
    - Solve α via brentq (quadratic objective, converges in ≤2 steps).

    When target_delta2 = 0 the function returns B_null unchanged.
    """
    if target_delta2 == 0.0:
        return B_null

    q = R.shape[0]
    rng_c = np.random.default_rng(calib_seed)
    M = rng_c.standard_normal((q, p))

    # Right pseudoinverse: R_right_inv = R'(RR')⁻¹
    RRt = R @ R.T
    R_right_inv = R.T @ np.linalg.solve(RRt, np.eye(q))  # (d, q)

    RGRt_inv = np.linalg.solve(R @ G @ R.T, np.eye(q))
    Sigma_inv = np.linalg.solve(Sigma, np.eye(p))
    C_const = float(np.trace(M.T @ RGRt_inv @ M @ Sigma_inv)) / (p * n)

    if C_const <= 0.0:
        warnings.warn("Calibration constant C ≤ 0; returning B_null.", stacklevel=2)
        return B_null

    def objective(alpha: float) -> float:
        return alpha**2 * C_const - target_delta2

    alpha_upper = 2.0 * math.sqrt(target_delta2 / C_const) + 1e-6
    alpha_star = brentq(objective, 0.0, alpha_upper, xtol=1e-10, rtol=1e-10)
    return B_null + R_right_inv @ (alpha_star * M)


# ---------------------------------------------------------------------------
# Exp A — Spectral robustness
# ---------------------------------------------------------------------------


def _rep_a(
    rep_idx: int,
    n: int,
    d: int,
    q: int,
    p: int,
    N_R: NDArray,
    R: NDArray,
    Sigma: NDArray,
    error_law: str,
    nu: Optional[int],
) -> Tuple[NDArray, NDArray]:
    """Return (eigs_S_r, eigs_S_u) for one replication."""
    rng = np.random.default_rng(BASE_SEED + rep_idx)
    X = generate_design_matrix(n, d, rng)
    B = generate_coefficient_matrix(N_R, p, rng)
    E = generate_elliptical_errors(n, Sigma, error_law, nu, rng)
    Y = X @ B + E

    ols = compute_ols(X, Y)
    G, P_X, B_hat = ols["G"], ols["P_X"], ols["B_hat"]
    S_u = ols["S_u"]

    rols = compute_restricted_ols(X, Y, R, B_hat, G, P_X)
    S_r = rols["S_r"]

    return np.linalg.eigvalsh(S_r), np.linalg.eigvalsh(S_u)


def run_exp_a_spectral_robustness(
    n: int = 200,
    d: Optional[int] = None,
    q: int = 5,
    p: int = 100,
    n_reps: int = 500,
) -> None:
    """Figure 2 analog — spectral robustness of S_r under heavy tails.

    Eigenvalue histograms of S_r (restricted scatter) and V_r = S_u
    (unrestricted scatter, labeled V_r following the paper's Figure 2
    interpretation) are plotted for Gaussian, t₆, t₁₁ errors with the
    Marchenko-Pastur overlay at tilde_c = p/(n-d+q).

    Paper note
    ----------
    'V_r' is undefined in the spec; interpreted here as S_u whose spectral
    limit follows MP(c) with c = p/(n-d).
    """
    if d is None:
        d = _d_default(n)
    tc = _tilde_c(p, n, d, q)
    c_eff = _c_eff(p, n, d)
    tv = _theory_valid(p, n, d, q)
    print(f"[exp_a] n={n}, d={d}, q={q}, p={p}, tilde_c={tc:.3f}, theory_valid={tv}")

    rng0 = np.random.default_rng(BASE_SEED)
    R = generate_restriction_matrix(q, d, rng0)
    N_R = compute_null_space_basis(R)
    Sigma = generate_identity_covariance(p)

    # Collect eigenvalues per error law
    law_eigs: Dict[str, Dict[str, List[NDArray]]] = {
        law: {"S_r": [], "S_u": []} for law, *_ in _LAWS
    }
    for law, label, nu in _LAWS:
        results = Parallel(n_jobs=4, prefer="threads")(
            delayed(_rep_a)(rep, n, d, q, p, N_R, R, Sigma, law, nu)
            for rep in range(n_reps)
        )
        for e_r, e_u in results:
            law_eigs[law]["S_r"].append(e_r)
            law_eigs[law]["S_u"].append(e_u)

    # ── 3×2 combined figure (rows = error laws, cols = S_r / S_u) ──
    law_labels = {law: label for law, label, _ in _LAWS}
    lam_plus_r = (1 + math.sqrt(tc)) ** 2
    lam_minus_r = max(0.0, (1 - math.sqrt(tc)) ** 2)
    lam_plus_u = (1 + math.sqrt(c_eff)) ** 2
    lam_minus_u = max(0.0, (1 - math.sqrt(c_eff)) ** 2)

    fig, axes = plt.subplots(3, 2, figsize=(12, 11))
    for row_idx, (law, label, nu) in enumerate(_LAWS):
        eigs_r = np.concatenate(law_eigs[law]["S_r"])
        eigs_u = np.concatenate(law_eigs[law]["S_u"])
        for col_idx, (eigs, lam_lo, lam_hi, c_val, col_label) in enumerate([
            (eigs_r, lam_minus_r, lam_plus_r, tc,    "S_r (restricted)"),
            (eigs_u, lam_minus_u, lam_plus_u, c_eff, "V_r = S_u (unrest.)"),
        ]):
            ax = axes[row_idx, col_idx]
            ax.hist(
                eigs, bins=50, density=True, alpha=0.65,
                color=sns.color_palette("tab10")[col_idx],
                label=col_label,
            )
            # MP density overlay
            x_grid = np.linspace(max(0, lam_lo * 0.5), lam_hi * 1.35, 600)
            from src.shrinkage import marchenko_pastur_density
            mp_y = [marchenko_pastur_density(xi, c_val) for xi in x_grid]
            ax.plot(x_grid, mp_y, "k--", lw=1.8,
                    label=f"MP (c={c_val:.3f})")
            ax.set_title(f"{label}  •  {col_label}", fontsize=10)
            ax.set_xlabel("Eigenvalue", fontsize=9)
            ax.set_ylabel("Density", fontsize=9)
            ax.legend(fontsize=8)
            sns.despine(ax=ax)
    fig.suptitle(
        f"Spectral robustness: S_r and V_r eigenvalue distributions\n"
        f"n={n}, d={d}, q={q}, p={p}, tilde_c={tc:.3f}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = RESULTS_DIR / "fig_spectral_robustness.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    # Also call plot_eigenvalue_histogram for S_r (per-law summaries)
    for law, label, _ in _LAWS:
        eig_dict = {f"S_r {label}": np.concatenate(law_eigs[law]["S_r"])}
        plot_eigenvalue_histogram(
            eig_dict, c=tc,
            title=f"S_r eigenvalues — {label}  (tilde_c={tc:.3f})",
            save_path=str(RESULTS_DIR / f"fig_spectral_Sr_{law}.png"),
        )

    print(f"[exp_a] Done → {out_path}")
    _log_diag(
        "exp_a", tilde_c=tc, theory_valid=tv,
        notes="V_r = S_u (unrestricted scatter); MP overlay uses tilde_c for S_r.",
    )


# ---------------------------------------------------------------------------
# Exp B — Risk vs aspect ratio
# ---------------------------------------------------------------------------


def _rep_b(
    rep_idx: int,
    n: int, d: int, q: int, p: int,
    N_R: NDArray, R: NDArray, Sigma: NDArray,
    error_law: str, nu: Optional[int],
) -> Dict[str, float]:
    """One replication for exp_b.  Returns WF losses per estimator."""
    rng = np.random.default_rng(BASE_SEED + rep_idx)
    X = generate_design_matrix(n, d, rng)
    B = generate_coefficient_matrix(N_R, p, rng)
    E = generate_elliptical_errors(n, Sigma, error_law, nu, rng)
    Y = X @ B + E
    res = _run_ols_pipeline(X, Y, R, n, d, q, Sigma)
    return {name: res[f"wf_{name}"] for name in _EST_NAMES}


def run_exp_b_risk_vs_aspect_ratio(
    n: int = 200,
    d: Optional[int] = None,
    q: int = 5,
    n_reps: int = 500,
    cov_type: str = "identity",
    c_input_grid: Optional[List[float]] = None,
) -> None:
    """Figure 4/5 and Table 1 analog — mean WF risk vs aspect ratio c = p/n.

    Varies p so that c = p/n spans the supplied grid (default [0.5,1,2,3,5]).
    Tyler-based estimators are marked NaN when infeasible (c_eff ≥ 1).
    """
    if d is None:
        d = _d_default(n)

    if c_input_grid is None:
        c_input_grid = [0.5, 1.0, 2.0, 3.0, 5.0]
    rng0 = np.random.default_rng(BASE_SEED)
    R = generate_restriction_matrix(q, d, rng0)
    N_R = compute_null_space_basis(R)

    rows: List[Dict] = []
    for c_in in c_input_grid:
        p = max(q + 1, int(round(c_in * n)))
        # Build Sigma for this p
        if cov_type == "identity":
            Sigma = generate_identity_covariance(p)
        elif cov_type == "banded":
            Sigma = generate_banded_covariance(p, rho=0.5, bandwidth=5)
        else:
            rng_s = np.random.default_rng(BASE_SEED + 999)
            Sigma = generate_sparse_covariance(p, rng_s)

        c_eff = _c_eff(p, n, d)
        tc = _tilde_c(p, n, d, q)
        tv = _theory_valid(p, n, d, q)

        for law, label, nu in _LAWS:
            rep_results = Parallel(n_jobs=4, prefer="threads")(
                delayed(_rep_b)(rep, n, d, q, p, N_R, R, Sigma, law, nu)
                for rep in range(n_reps)
            )
            for est_name in _EST_NAMES:
                vals = [r[est_name] for r in rep_results]
                finite = [v for v in vals if math.isfinite(v)]
                rows.append({
                    "c":           c_in,          # p/n (input parameter)
                    "c_eff":       c_eff,          # p/(n-d) for plot x-axis
                    "p":           p,
                    "tilde_c":     tc,
                    "theory_valid": tv,
                    "error_law":   label,
                    "estimator":   est_name,
                    "mean_wf_loss": float(np.mean(finite)) if finite else float("nan"),
                    "se_wf_loss":  (float(np.std(finite, ddof=1) / math.sqrt(len(finite)))
                                    if len(finite) > 1 else float("nan")),
                    "n_finite":    len(finite),
                })
        print(f"  [exp_b] c_in={c_in:.1f}, p={p}, tilde_c={tc:.3f}, tv={tv}")

    df = pd.DataFrame(rows)
    csv_path = RESULTS_DIR / "table_1_risk_vs_c.csv"
    df.to_csv(csv_path, index=False)

    # Plot: use c_eff (= p/(n-d)) on x-axis; drop the 'c' (= p/n) column
    # to avoid a duplicate 'c' column after rename.
    df_plot = df.drop(columns=["c"]).rename(columns={"c_eff": "c"})
    for law, label, _ in _LAWS:
        sub = df_plot[df_plot["error_law"] == label].copy()
        plot_risk_vs_aspect_ratio(
            sub, metric="mean_wf_loss",
            title=f"WF risk vs aspect ratio — {label} errors\n"
                  f"(n={n}, d={d}, q={q}, cov={cov_type})",
            save_path=str(RESULTS_DIR / f"fig_risk_vs_c_{law}.png"),
        )

    # Combined plot (all laws, URE and RRE only)
    fig_path = RESULTS_DIR / "fig_risk_vs_c.png"
    _plot_risk_combined(df, fig_path)

    print(f"[exp_b] Done → {csv_path}, {fig_path}")
    _log_diag("exp_b", notes=f"cov_type={cov_type}, c_grid={c_input_grid}")


def _plot_risk_combined(df: pd.DataFrame, fig_path: Path) -> None:
    """Faceted line plot: one panel per error law, lines per estimator."""
    laws = [label for _, label, _ in _LAWS]
    fig, axes = plt.subplots(1, len(laws), figsize=(5 * len(laws), 4.5), sharey=False)
    pal = sns.color_palette("tab10")
    for ax, law_label in zip(axes, laws):
        sub = df[df["error_law"] == law_label]
        for i, est in enumerate(_EST_NAMES):
            s2 = sub[sub["estimator"] == est].sort_values("c_eff")
            finite = s2.dropna(subset=["mean_wf_loss"])
            if finite.empty:
                continue
            ax.plot(
                finite["c_eff"], finite["mean_wf_loss"],
                marker="o", markersize=4, lw=1.8,
                color=pal[i], label=est,
            )
        ax.set_title(law_label, fontsize=11)
        ax.set_xlabel("c = p/(n-d)", fontsize=10)
        ax.set_ylabel("Mean WF loss", fontsize=10)
        ax.legend(fontsize=8)
        sns.despine(ax=ax)
    fig.suptitle("WF risk vs unrestricted aspect ratio", fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Exp C — Covariance structures
# ---------------------------------------------------------------------------


def _rep_c(
    rep_idx: int,
    n: int, d: int, q: int, p: int,
    N_R: NDArray, R: NDArray, Sigma: NDArray,
) -> Dict[str, float]:
    """One replication for exp_c: WF loss + operator-norm error, t11 errors."""
    rng = np.random.default_rng(BASE_SEED + rep_idx)
    X = generate_design_matrix(n, d, rng)
    B = generate_coefficient_matrix(N_R, p, rng)
    E = generate_elliptical_errors(n, Sigma, "t11", 11, rng)
    Y = X @ B + E
    res = _run_ols_pipeline(X, Y, R, n, d, q, Sigma)
    return {
        **{f"wf_{nm}": res[f"wf_{nm}"] for nm in _EST_NAMES},
        **{f"op_{nm}": res[f"op_{nm}"] for nm in ["URE", "RRE", "SSE+"]},
    }


def run_exp_c_covariance_structures(
    n: int = 400,
    d: Optional[int] = None,
    q: int = 5,
    c: float = 1.0,
    n_reps: int = 500,
) -> None:
    """Table 2 analog — WF and operator-norm losses for banded and sparse Sigma."""
    if d is None:
        d = _d_default(n)
    p = max(q + 1, int(round(c * n)))
    rng0 = np.random.default_rng(BASE_SEED)
    R = generate_restriction_matrix(q, d, rng0)
    N_R = compute_null_space_basis(R)

    cov_configs = {
        "banded": generate_banded_covariance(p, rho=0.5, bandwidth=5),
        "sparse": generate_sparse_covariance(p, np.random.default_rng(BASE_SEED + 1)),
    }

    rows: List[Dict] = []
    for cov_name, Sigma in cov_configs.items():
        tc = _tilde_c(p, n, d, q)
        tv = _theory_valid(p, n, d, q)
        rep_results = Parallel(n_jobs=4, prefer="threads")(
            delayed(_rep_c)(rep, n, d, q, p, N_R, R, Sigma)
            for rep in range(n_reps)
        )
        for est_name in _EST_NAMES:
            wf_vals = [r[f"wf_{est_name}"] for r in rep_results]
            op_key = f"op_{est_name}"
            op_vals = [r.get(op_key, float("nan")) for r in rep_results]
            finite_wf = [v for v in wf_vals if math.isfinite(v)]
            finite_op = [v for v in op_vals if math.isfinite(v)]
            rows.append({
                "cov_structure": cov_name,
                "estimator":     est_name,
                "mean_wf_loss":  float(np.mean(finite_wf)) if finite_wf else float("nan"),
                "se_wf_loss":    (float(np.std(finite_wf, ddof=1) / math.sqrt(len(finite_wf)))
                                  if len(finite_wf) > 1 else float("nan")),
                "mean_op_error": float(np.mean(finite_op)) if finite_op else float("nan"),
                "tilde_c":       tc,
                "theory_valid":  tv,
            })
        print(f"  [exp_c] cov={cov_name}, tilde_c={tc:.3f}, tv={tv}")

    df = pd.DataFrame(rows)
    csv_path = RESULTS_DIR / "table_2_cov_structures.csv"
    df.to_csv(csv_path, index=False)
    print(f"[exp_c] Done → {csv_path}")
    _log_diag("exp_c", notes="t11 errors. banded: rho=0.5, bw=5. sparse: I+random.")


# ---------------------------------------------------------------------------
# Exp D — Misspecification
# ---------------------------------------------------------------------------


def _rep_d(
    rep_idx: int,
    n: int, d: int, q: int, p: int,
    N_R: NDArray, R: NDArray, Sigma: NDArray,
    B_misspec: NDArray,
) -> Dict[str, float]:
    """One replication for exp_d: WF losses for URE, RRE, SSE+."""
    rng = np.random.default_rng(BASE_SEED + rep_idx)
    X = generate_design_matrix(n, d, rng)
    E = generate_elliptical_errors(n, Sigma, "gaussian", None, rng)
    Y = X @ B_misspec + E

    ols = compute_ols(X, Y)
    G, P_X, B_hat, E_hat = ols["G"], ols["P_X"], ols["B_hat"], ols["E_hat"]
    S_u, c_n = ols["S_u"], ols["c_n"]
    rols = compute_restricted_ols(X, Y, R, B_hat, G, P_X)
    S_r, tilde_c_n = rols["S_r"], rols["tilde_c_n"]
    T_n = compute_restriction_stat(B_hat, R, G, S_u)

    Sigma_ure = compute_ure_cov(S_u, c_n)
    Sigma_rre = compute_rre_cov(S_r, tilde_c_n)
    Sigma_sse = compute_sse_plus(Sigma_ure, Sigma_rre, T_n, n, d, q)["Sigma_hat"]

    return {
        "URE":  weighted_frobenius_loss(Sigma_ure, Sigma),
        "RRE":  weighted_frobenius_loss(Sigma_rre, Sigma),
        "SSE+": weighted_frobenius_loss(Sigma_sse, Sigma),
    }


def run_exp_d_misspecification(
    n: int = 200,
    p: int = 200,
    d: Optional[int] = None,
    q: int = 5,
    n_reps: int = 500,
) -> None:
    """Figure 6 analog — relative WF risk under misspecification.

    Varies δ in [0, 0.5, 1.0, 2.0, 4.0].  Reports relative risk (each
    estimator / URE mean) at each δ level.
    """
    if d is None:
        d = _d_default(n)
    tc = _tilde_c(p, n, d, q)
    tv = _theory_valid(p, n, d, q)
    print(f"[exp_d] n={n}, d={d}, q={q}, p={p}, tilde_c={tc:.3f}, tv={tv}")

    rng0 = np.random.default_rng(BASE_SEED)
    R = generate_restriction_matrix(q, d, rng0)
    N_R = compute_null_space_basis(R)
    Sigma = generate_identity_covariance(p)

    # Fixed calibration design matrix
    X_calib = generate_design_matrix(n, d, np.random.default_rng(BASE_SEED + 777))
    ols_calib = compute_ols(X_calib, np.zeros((n, p)))
    G_calib = ols_calib["G"]

    # Fixed B_null
    B_null = generate_coefficient_matrix(N_R, p, np.random.default_rng(BASE_SEED + 888))

    delta_grid = [0.0, 0.5, 1.0, 2.0, 4.0]
    rows: List[Dict] = []

    for delta in delta_grid:
        B_misspec = calibrate_misspecification(
            B_null, N_R, R, G_calib, Sigma,
            target_delta2=delta**2, p=p, n=n, calib_seed=BASE_SEED,
        )
        rep_results = Parallel(n_jobs=4, prefer="threads")(
            delayed(_rep_d)(rep, n, d, q, p, N_R, R, Sigma, B_misspec)
            for rep in range(n_reps)
        )
        ure_mean = float(np.mean([r["URE"] for r in rep_results]))
        for est_name in ["URE", "RRE", "SSE+"]:
            mean_loss = float(np.mean([r[est_name] for r in rep_results]))
            rows.append({
                "delta":         delta,
                "estimator":     est_name,
                "mean_wf_loss":  mean_loss,
                "relative_risk": mean_loss / ure_mean if ure_mean > 0 else float("nan"),
                "tilde_c":       tc,
                "theory_valid":  tv,
            })
        print(f"  [exp_d] δ={delta:.1f}, URE_mean={ure_mean:.4f}")

    df = pd.DataFrame(rows)
    fig_path = RESULTS_DIR / "fig_misspecification.png"
    plot_risk_vs_misspecification(
        df,
        title=f"Relative WF risk vs misspecification δ\n(n={n}, d={d}, q={q}, p={p})",
        save_path=str(fig_path),
    )
    print(f"[exp_d] Done → {fig_path}")
    _log_diag(
        "exp_d", tilde_c=tc, theory_valid=tv,
        notes="Calibration with fixed X_calib. Gaussian errors. Relative risk = loss/URE.",
    )


# ---------------------------------------------------------------------------
# Exp E — Gain vs restriction rank
# ---------------------------------------------------------------------------


def _rep_e(
    rep_idx: int,
    n: int, d: int, q: int, p: int,
    N_R: NDArray, R: NDArray, Sigma: NDArray,
) -> Dict[str, float]:
    """One replication for exp_e: URE and RRE WF losses."""
    rng = np.random.default_rng(BASE_SEED + rep_idx)
    X = generate_design_matrix(n, d, rng)
    B = generate_coefficient_matrix(N_R, p, rng)
    E = generate_elliptical_errors(n, Sigma, "gaussian", None, rng)
    Y = X @ B + E

    ols = compute_ols(X, Y)
    G, P_X, B_hat = ols["G"], ols["P_X"], ols["B_hat"]
    S_u, c_n = ols["S_u"], ols["c_n"]
    rols = compute_restricted_ols(X, Y, R, B_hat, G, P_X)
    S_r, tilde_c_n = rols["S_r"], rols["tilde_c_n"]
    return {
        "URE": weighted_frobenius_loss(compute_ure_cov(S_u, c_n), Sigma),
        "RRE": weighted_frobenius_loss(compute_rre_cov(S_r, tilde_c_n), Sigma),
    }


def run_exp_e_gain_vs_rank(
    n: int = 200,
    p: int = 100,
    d: Optional[int] = None,
    n_reps: int = 500,
) -> None:
    """Figure 7 analog — empirical vs theoretical risk-reduction ratio.

    Theoretical ratio = (n-d)/(n-d+q), exact under Gaussian / identity Sigma.
    """
    if d is None:
        d = _d_default(n)
    q_grid = [3, 5, 10, 20]
    Sigma = generate_identity_covariance(p)
    rng0 = np.random.default_rng(BASE_SEED)

    rows: List[Dict] = []
    for q in q_grid:
        if q >= d:
            warnings.warn(f"[exp_e] q={q} ≥ d={d}; skipping.", stacklevel=2)
            continue
        R = generate_restriction_matrix(q, d, rng0)
        N_R = compute_null_space_basis(R)
        tc = _tilde_c(p, n, d, q)
        tv = _theory_valid(p, n, d, q)
        theory_ratio = (n - d) / (n - d + q)

        rep_results = Parallel(n_jobs=4, prefer="threads")(
            delayed(_rep_e)(rep, n, d, q, p, N_R, R, Sigma)
            for rep in range(n_reps)
        )
        ure_mean = float(np.mean([r["URE"] for r in rep_results]))
        rre_mean = float(np.mean([r["RRE"] for r in rep_results]))
        emp_ratio = rre_mean / ure_mean if ure_mean > 0 else float("nan")
        rows.append({
            "q":              q,
            "theory_ratio":   theory_ratio,
            "empirical_ratio": emp_ratio,
            "tilde_c":        tc,
            "theory_valid":   tv,
        })
        print(
            f"  [exp_e] q={q}, theory={theory_ratio:.4f}, empirical={emp_ratio:.4f}"
        )

    df = pd.DataFrame(rows)
    fig_path = RESULTS_DIR / "fig_gain_vs_rank.png"

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(df["q"], df["theory_ratio"], "k--o", lw=1.8, ms=6, label="Theoretical (n-d)/(n-d+q)")
    ax.plot(df["q"], df["empirical_ratio"], "bs-", lw=1.8, ms=6, label="Empirical mean(RRE)/mean(URE)")
    ax.set_xlabel("Restriction rank q", fontsize=12)
    ax.set_ylabel("Risk ratio RRE / URE", fontsize=12)
    ax.set_title(f"Gain vs restriction rank (n={n}, d={d}, p={p})", fontsize=12)
    ax.legend(fontsize=10)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"[exp_e] Done → {fig_path}")
    _log_diag(
        "exp_e",
        notes="Theoretical ratio exact under Gaussian/identity only. "
              "Empirical = mean(RRE)/mean(URE).",
    )


# ---------------------------------------------------------------------------
# Exp F — Bias-variance decomposition
# ---------------------------------------------------------------------------


def _rep_f(
    rep_idx: int,
    n: int, d: int, q: int, p: int,
    N_R: NDArray, R: NDArray, Sigma: NDArray,
    error_law: str, nu: Optional[int],
) -> Dict[str, Optional[NDArray]]:
    """One replication for exp_f: return estimator matrices (not losses)."""
    rng = np.random.default_rng(BASE_SEED + rep_idx)
    X = generate_design_matrix(n, d, rng)
    B = generate_coefficient_matrix(N_R, p, rng)
    E = generate_elliptical_errors(n, Sigma, error_law, nu, rng)
    Y = X @ B + E
    res = _run_ols_pipeline(X, Y, R, n, d, q, Sigma)
    return {
        "URE":       res["Sigma_ure"],
        "RRE":       res["Sigma_rre"],
        "SSE+":      res["Sigma_sse"],
        "URE-Tyler": res["Sigma_ure_t"],
        "RRE-Tyler": res["Sigma_rre_t"],
    }


def run_exp_f_bias_variance(
    n: int = 200,
    p: int = 100,
    d: Optional[int] = None,
    q: int = 5,
    n_reps: int = 200,
) -> None:
    """Figure 8 analog — bias² / variance decomposition of WF risk.

    bias²    = WF_loss(E[Σ̂], Σ_true)
    variance = E[WF_loss(Σ̂, E[Σ̂])]
    total    ≈ bias² + variance  (exact for quadratic losses)
    """
    if d is None:
        d = _d_default(n)
    tc = _tilde_c(p, n, d, q)
    tv = _theory_valid(p, n, d, q)
    print(f"[exp_f] n={n}, d={d}, q={q}, p={p}, tilde_c={tc:.3f}, tv={tv}")

    rng0 = np.random.default_rng(BASE_SEED)
    R = generate_restriction_matrix(q, d, rng0)
    N_R = compute_null_space_basis(R)
    Sigma = generate_identity_covariance(p)

    all_rows: List[Dict] = []
    for law, label, nu in _LAWS[:2]:  # Gaussian and t6 (memory constraint)
        rep_results = Parallel(n_jobs=4, prefer="threads")(
            delayed(_rep_f)(rep, n, d, q, p, N_R, R, Sigma, law, nu)
            for rep in range(n_reps)
        )
        for est_name in _EST_NAMES:
            mats = [r[est_name] for r in rep_results if r[est_name] is not None]
            if not mats:
                continue
            stacked = np.stack(mats, axis=0)          # (k, p, p)
            mean_mat = stacked.mean(axis=0)            # E[Σ̂]
            bias2 = float(weighted_frobenius_loss(mean_mat, Sigma))
            variance = float(
                np.mean([weighted_frobenius_loss(stacked[i], mean_mat)
                         for i in range(len(mats))])
            )
            total = float(
                np.mean([weighted_frobenius_loss(stacked[i], Sigma)
                         for i in range(len(mats))])
            )
            all_rows.append({
                "error_law":     label,
                "estimator":     est_name,
                "bias2":         bias2,
                "variance":      variance,
                "total_risk":    total,
                "bias2_plus_var": bias2 + variance,
                "tilde_c":       tc,
                "theory_valid":  tv,
            })
        print(f"  [exp_f] {label} done.")

    df = pd.DataFrame(all_rows)
    fig_path = RESULTS_DIR / "fig_bias_variance.png"

    for law, label, _ in _LAWS[:2]:
        sub = df[df["error_law"] == label]
        results_dict = {
            "estimator": sub["estimator"].tolist(),
            "bias_sq":   sub["bias2"].tolist(),
            "variance":  sub["variance"].tolist(),
            "mse":       sub["total_risk"].tolist(),
        }
        plot_bias_variance(
            results_dict,
            title=f"Bias-variance decomposition — {label} errors\n"
                  f"(n={n}, d={d}, q={q}, p={p})",
            save_path=str(RESULTS_DIR / f"fig_bias_variance_{law}.png"),
        )

    # Combined figure (Gaussian only, saved as main)
    first = df[df["error_law"] == [label for _, label, _ in _LAWS[:2]][0]]
    plot_bias_variance(
        {
            "estimator": first["estimator"].tolist(),
            "bias_sq":   first["bias2"].tolist(),
            "variance":  first["variance"].tolist(),
            "mse":       first["total_risk"].tolist(),
        },
        title=f"Bias-variance decomposition (Gaussian)\n(n={n}, d={d}, q={q}, p={p})",
        save_path=str(fig_path),
    )
    print(f"[exp_f] Done → {fig_path}")
    _log_diag(
        "exp_f", tilde_c=tc, theory_valid=tv,
        notes=f"n_reps={n_reps}. Laws: Gaussian, t6.",
    )


# ---------------------------------------------------------------------------
# Exp G — Entrywise heatmap
# ---------------------------------------------------------------------------


def run_exp_g_entrywise_heatmap(
    n: int = 80,
    p: int = 60,
    d: Optional[int] = None,
    q: int = 5,
) -> None:
    """Figure 9 analog — entrywise |Σ̂ - Σ_true| for all five estimators.

    Single realization, banded true Sigma, t11 errors.
    """
    if d is None:
        d = _d_default(n)
    tc = _tilde_c(p, n, d, q)
    tv = _theory_valid(p, n, d, q)
    print(f"[exp_g] n={n}, d={d}, q={q}, p={p}, tilde_c={tc:.3f}, tv={tv}")

    rng0 = np.random.default_rng(BASE_SEED)
    R = generate_restriction_matrix(q, d, rng0)
    N_R = compute_null_space_basis(R)
    Sigma = generate_banded_covariance(p, rho=0.5, bandwidth=5)

    rng = np.random.default_rng(BASE_SEED)
    X = generate_design_matrix(n, d, rng)
    B = generate_coefficient_matrix(N_R, p, rng)
    E = generate_elliptical_errors(n, Sigma, "t11", 11, rng)
    Y = X @ B + E

    res = _run_ols_pipeline(X, Y, R, n, d, q, Sigma)

    estimators = {
        "URE":       res["Sigma_ure"],
        "RRE":       res["Sigma_rre"],
        "SSE+":      res["Sigma_sse"],
        "URE-Tyler": res["Sigma_ure_t"],
        "RRE-Tyler": res["Sigma_rre_t"],
    }

    # Diagnostics
    for name, S in estimators.items():
        if S is None:
            print(f"  [exp_g] {name}: INFEASIBLE (Tyler c_n ≥ 1)")
            continue
        print(
            f"  [exp_g] {name}: psd={is_psd(S)}, "
            f"cond={condition_number(S):.2f}, "
            f"wf={weighted_frobenius_loss(S, Sigma):.4f}"
        )

    # Combined 5-panel figure
    fig_path = RESULTS_DIR / "fig_entrywise_heatmap.png"
    valid_ests = [(nm, S) for nm, S in estimators.items() if S is not None]
    ncols = len(valid_ests)
    fig, axes = plt.subplots(1, ncols, figsize=(3.5 * ncols, 4.5))
    if ncols == 1:
        axes = [axes]
    for ax, (name, S) in zip(axes, valid_ests):
        err = np.abs(S - Sigma)
        im = ax.imshow(err, cmap="hot_r", aspect="auto",
                       vmin=0, vmax=np.percentile(err, 97))
        ax.set_title(f"|{name} − Σ|", fontsize=10)
        ax.set_xlabel("j", fontsize=9)
        ax.set_ylabel("i", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        f"Entry-wise |Σ̂ − Σ_true|  (n={n}, d={d}, q={q}, p={p}, banded Σ, t₁₁)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    # Also call plot_heatmap per estimator (individual files)
    for name, S in valid_ests:
        plot_heatmap(
            np.abs(S - Sigma),
            title=f"|{name} − Σ_true|  (p={p}, banded, t11)",
            save_path=str(RESULTS_DIR / f"fig_heatmap_{name.replace('+', 'plus').replace('-', '_')}.png"),
        )

    print(f"[exp_g] Done → {fig_path}")
    _log_diag(
        "exp_g", tilde_c=tc, theory_valid=tv,
        notes="Single realization. Banded Sigma (rho=0.5, bw=5). t11 errors.",
    )


# ---------------------------------------------------------------------------
# Diagnostics logger
# ---------------------------------------------------------------------------


def _log_diag(
    exp_name: str,
    tilde_c: Optional[float] = None,
    theory_valid: Optional[bool] = None,
    notes: str = "",
) -> None:
    log_path = RESULTS_DIR / "diagnostics.txt"
    with open(log_path, "a") as fh:
        line = f"[{exp_name}]"
        if tilde_c is not None:
            line += f" tilde_c={tilde_c:.4f}"
        if theory_valid is not None:
            line += f" theory_valid={theory_valid}"
        if notes:
            line += f" | {notes}"
        fh.write(line + "\n")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(n_reps: int = 50) -> None:
    """Run all seven experiment families and print a summary.

    Parameters
    ----------
    n_reps:
        Monte Carlo replications per configuration.  Default 50 gives fast
        results; set to 500 for the full paper-scale run.
    """
    print("=" * 72)
    print("exp_1_synthetic — restricted robust covariance estimation study")
    print(f"Results directory: {RESULTS_DIR}  |  n_reps={n_reps}")
    print("=" * 72)

    (RESULTS_DIR / "diagnostics.txt").write_text(
        "Diagnostics log — exp_1_synthetic\n" + "=" * 40 + "\n"
    )

    print("\n─── Exp A: spectral robustness ───")
    run_exp_a_spectral_robustness(n_reps=n_reps)

    print("\n─── Exp B: risk vs aspect ratio ───")
    # Restrict c-grid to [0.5, 1.0, 2.0] for fast runs; full grid for ≥200 reps
    _c_grid = [0.5, 1.0, 2.0] if n_reps < 200 else [0.5, 1.0, 2.0, 3.0, 5.0]
    run_exp_b_risk_vs_aspect_ratio(n_reps=n_reps, c_input_grid=_c_grid)

    print("\n─── Exp C: covariance structures ───")
    run_exp_c_covariance_structures(n_reps=n_reps)

    print("\n─── Exp D: misspecification ───")
    run_exp_d_misspecification(n_reps=n_reps)

    print("\n─── Exp E: gain vs restriction rank ───")
    run_exp_e_gain_vs_rank(n_reps=n_reps)

    print("\n─── Exp F: bias-variance decomposition ───")
    run_exp_f_bias_variance(n_reps=n_reps)

    print("\n─── Exp G: entrywise heatmap ───")
    run_exp_g_entrywise_heatmap()

    print("\n" + "=" * 72)
    print("All experiments complete.  Output files in results/:")
    for p in sorted(RESULTS_DIR.iterdir()):
        print(f"  {p.name}")
    print("=" * 72)


if __name__ == "__main__":
    main()
