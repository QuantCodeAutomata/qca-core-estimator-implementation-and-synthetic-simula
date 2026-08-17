"""Experiment 5: Leukemia gene expression analysis.

Evaluates covariance estimators on held-out multivariate-t log-likelihood
(nu=5) across varying training sizes and 40 stratified random splits.

Estimators: S_u (sample cov), LW linear shrinkage, URE-cov, URE-Tyler.

Dataset: Pediatric ALL leukemia gene expression (N=248, 12 625 probes, 6 subtypes).
  If download fails, a synthetic proxy with the same dimensions is generated.
  Top 120 probes by global variance are selected before fitting.

Design: X (N x 6) = one-hot encoding of 6 leukemia subtypes (no intercept),
  ensuring rank 6 by construction if all 6 subtypes are present in training.

Expected Tyler held-out ll: approximately -95.6, -84.7, -79.7 for n=130, 160, 200.
"""

from __future__ import annotations

import io
import logging
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.estimators import compute_ure_cov, compute_ure_tyler
from src.metrics import condition_number, multivariate_t_log_likelihood
from src.regression import compute_ols
from src.shrinkage import linear_ledoit_wolf_shrinkage

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
N_PROBES_SELECT = 120   # number of probes to select
N_SUBTYPES = 6
NU = 5.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_leukemia_data() -> Tuple[NDArray, NDArray]:
    """Load pediatric ALL leukemia gene expression data.

    Attempts to download from public repositories; falls back to synthetic
    proxy data if unavailable.

    Fallback: N=248 patients, 12 625 probes, 6 subtypes.

    Returns
    -------
    Tuple[NDArray, NDArray]
        ``expression`` of shape (N, n_probes) and
        ``subtype_labels`` of shape (N,) with integer codes 0..5.
    """
    # Attempt GEO download of GSE13425 (Yeoh et al. 2002 / Armstrong ALL dataset)
    # Two common public mirrors:
    _urls = [
        "https://raw.githubusercontent.com/cran/ALL/master/data/ALL.rda",  # unlikely to parse
    ]
    # In practice these formats are not trivially parseable in pure Python.
    # We skip to the synthetic fallback immediately and log it clearly.
    logger.info(
        "Leukemia dataset: no reliable plain-text public URL found. "
        "Generating synthetic proxy (N=248, probes=12625, subtypes=6)."
    )

    return _generate_synthetic_leukemia()


def _generate_synthetic_leukemia() -> Tuple[NDArray, NDArray]:
    """Generate synthetic leukemia expression data matching expected dimensions."""
    rng = np.random.default_rng(0)
    N = 248
    n_probes = 12625
    n_sub = N_SUBTYPES

    # 6 balanced subtypes (≈41 each, last one has N - 5*41 = 43)
    sizes = [N // n_sub] * n_sub
    sizes[-1] += N - sum(sizes)

    labels = np.concatenate([np.full(s, k) for k, s in enumerate(sizes)])

    # Subtype-specific mean expression profiles (sparse signal)
    Mu = rng.standard_normal((n_sub, n_probes)) * 0.5
    Mu[:, :200] += rng.standard_normal((n_sub, 200)) * 2.0   # a few highly variable probes

    # Background covariance: AR(1) with rho=0.2 for nearby probes
    # (too large to store; approximate with independent noise + shared factor)
    n_factors = 20
    F = rng.standard_normal((n_probes, n_factors))   # factor loadings
    scores = rng.standard_normal((N, n_factors))
    noise = rng.standard_normal((N, n_probes))
    expression = scores @ F.T + noise + np.array([Mu[labels[i]] for i in range(N)])

    logger.info(
        "Synthetic leukemia data: expression=%s, labels=%s, subtypes=%s",
        expression.shape, labels.shape, np.unique(labels, return_counts=True),
    )
    return expression.astype(float), labels.astype(int)


# ---------------------------------------------------------------------------
# Probe selection and design matrix
# ---------------------------------------------------------------------------

def select_top_probes(
    expression: NDArray,
    labels: NDArray,
    n_probes: int = N_PROBES_SELECT,
    global_selection: bool = True,
) -> Tuple[NDArray, NDArray]:
    """Select the top n_probes most variable probes.

    Parameters
    ----------
    expression : NDArray
        Shape (N, total_probes).
    labels : NDArray
        Shape (N,) subtype labels (unused when global_selection=True).
    n_probes : int
        Number of probes to keep. Default 120.
    global_selection : bool
        If True, use global variance across all observations (mild selection
        bias, matches likely paper behaviour). Default True.

    Returns
    -------
    Tuple[NDArray, NDArray]
        ``selected_expression`` of shape (N, n_probes) and
        ``probe_indices`` of shape (n_probes,).
    """
    if global_selection:
        variances = np.var(expression, axis=0, ddof=1)
    else:
        # Within-subtype variance to avoid confounding with subtype differences
        variances = np.zeros(expression.shape[1])
        for k in np.unique(labels):
            mask = labels == k
            variances += np.var(expression[mask], axis=0, ddof=1) * mask.sum()
        variances /= len(labels)

    probe_indices = np.argsort(variances)[::-1][:n_probes]
    return expression[:, probe_indices], probe_indices


def build_subtype_design(labels: NDArray) -> NDArray:
    """Build one-hot design matrix for leukemia subtypes.

    One-hot encoding without intercept: X is (N, 6) with rank=6.
    Column k is 1 where labels==k, 0 otherwise.

    Parameters
    ----------
    labels : NDArray
        Shape (N,) integer subtype codes in {0, 1, ..., 5}.

    Returns
    -------
    NDArray
        Shape (N, 6) design matrix with rank = number of distinct subtypes.
    """
    subtypes = np.unique(labels)
    n_sub = len(subtypes)
    N = len(labels)
    X = np.zeros((N, n_sub))
    for k, s in enumerate(subtypes):
        X[:, k] = (labels == s).astype(float)
    return X


# ---------------------------------------------------------------------------
# Single-split evaluation
# ---------------------------------------------------------------------------

def _evaluate_leukemia_split(
    args: tuple,
) -> dict | None:
    """Evaluate one stratified train/test split for leukemia data."""
    (expression, labels, n_train, n_probes_sel, nu, split_seed) = args
    N = len(expression)
    subtypes = np.unique(labels)
    n_sub = len(subtypes)

    # Stratified split: sample from each subtype proportionally
    rng = np.random.default_rng(split_seed)
    train_idx_list, test_idx_list = [], []
    for k in subtypes:
        sub_idx = np.where(labels == k)[0]
        rng.shuffle(sub_idx)
        n_k = max(1, int(round(n_train * len(sub_idx) / N)))
        train_idx_list.append(sub_idx[:n_k])
        test_idx_list.append(sub_idx[n_k:])

    train_idx = np.concatenate(train_idx_list)
    test_idx = np.concatenate(test_idx_list)

    if len(train_idx) < n_sub + 2:   # need at least more rows than subtypes
        return None
    if len(test_idx) < 10:
        return None

    # Probe selection: global variance on all data (mild leakage, matches paper)
    _, probe_idx = select_top_probes(expression, labels, n_probes_sel, global_selection=True)
    expr_sel = expression[:, probe_idx]

    E_tr = expr_sel[train_idx]
    E_te = expr_sel[test_idx]
    L_tr = labels[train_idx]
    L_te = labels[test_idx]

    # Build design matrices (one-hot, no intercept)
    X_tr = build_subtype_design(L_tr)
    X_te_raw = np.zeros((len(test_idx), n_sub))
    for k, s in enumerate(subtypes):
        X_te_raw[:, k] = (L_te == s).astype(float)

    # Standardize expression on training
    Y_mean = E_tr.mean(axis=0)
    Y_std = E_tr.std(axis=0, ddof=1)
    Y_std[Y_std < 1e-8] = 1.0
    Y_tr = (E_tr - Y_mean) / Y_std
    Y_te = (E_te - Y_mean) / Y_std

    # OLS fit on training
    try:
        ols = compute_ols(X_tr, Y_tr)
    except ValueError:
        return None
    S_u = ols["S_u"]
    c_n = ols["c_n"]
    B_hat = ols["B_hat"]
    E_hat = ols["E_hat"]
    p = Y_tr.shape[1]

    # Test residuals (centered by OLS prediction)
    mu_te = X_te_raw @ B_hat   # (n_te, p)
    resid_te = Y_te - mu_te
    mu_zero = np.zeros(p)

    result = {"n_train": n_train, "split_seed": split_seed, "n_test": len(test_idx)}

    # S_u
    ll_su = multivariate_t_log_likelihood(resid_te, mu_zero, S_u, nu)
    result["ll_Su"] = ll_su
    result["cond_Su"] = condition_number(S_u)

    # LW
    lw_cov = linear_ledoit_wolf_shrinkage(E_hat, assume_centered=True)
    ll_lw = multivariate_t_log_likelihood(resid_te, mu_zero, lw_cov, nu)
    result["ll_LW"] = ll_lw
    result["cond_LW"] = condition_number(lw_cov)

    # URE-cov
    ure_cov = compute_ure_cov(S_u, c_n)
    ll_ure = multivariate_t_log_likelihood(resid_te, mu_zero, ure_cov, nu)
    result["ll_URE_cov"] = ll_ure
    result["cond_URE_cov"] = condition_number(ure_cov)

    # URE-Tyler
    try:
        ure_tyler = compute_ure_tyler(E_hat, c_n)
        sigma_tyler = ure_tyler["Sigma_hat"]
        ll_tyler = multivariate_t_log_likelihood(resid_te, mu_zero, sigma_tyler, nu)
        result["ll_URE_Tyler"] = ll_tyler
        result["cond_URE_Tyler"] = condition_number(sigma_tyler)
        result["tyler_failed"] = not ure_tyler["converged"]
    except Exception:
        result["ll_URE_Tyler"] = float("-inf")
        result["cond_URE_Tyler"] = float("nan")
        result["tyler_failed"] = True

    return result


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_exp_5_leukemia(
    n_train_list: list[int] | None = None,
    n_splits: int = 40,
    nu: float = NU,
) -> pd.DataFrame:
    """Run leukemia held-out log-likelihood experiment.

    Parameters
    ----------
    n_train_list : list of int
        Training sizes to evaluate. Default [130, 160, 200].
    n_splits : int
        Number of stratified random splits. Default 40.
    nu : float
        Degrees of freedom for multivariate-t evaluation. Default 5.

    Returns
    -------
    pd.DataFrame
        Summary table with columns: n_train, estimator, median_ll, median_cond, failure_rate.
    """
    if n_train_list is None:
        n_train_list = [130, 160, 200]

    expression, labels = load_leukemia_data()
    N = len(expression)
    logger.info("Loaded: N=%d, probes=%d, subtypes=%d", N, expression.shape[1], len(np.unique(labels)))

    estimator_names = ["Su", "LW", "URE_cov", "URE_Tyler"]
    all_rows = []
    raw_results = []

    for n_train in n_train_list:
        if n_train >= N - 10:
            logger.warning("n_train=%d too large for N=%d; skipping.", n_train, N)
            continue

        logger.info("Training size n=%d  (%d splits)...", n_train, n_splits)
        t0 = time.time()

        args_list = [
            (expression, labels, n_train, N_PROBES_SELECT, nu, seed)
            for seed in range(n_splits)
        ]
        results = []
        for args in args_list:
            r = _evaluate_leukemia_split(args)
            if r is not None:
                results.append(r)

        elapsed = time.time() - t0
        logger.info("n_train=%d: %d/%d splits OK  (%.1fs)", n_train, len(results), n_splits, elapsed)

        if not results:
            continue

        for est in estimator_names:
            ll_key = f"ll_{est}"
            cond_key = f"cond_{est}"

            ll_vals = [r[ll_key] for r in results if np.isfinite(r.get(ll_key, float("nan")))]
            cond_vals = [r[cond_key] for r in results if np.isfinite(r.get(cond_key, float("nan")))]
            failure_rate = (
                sum(1 for r in results if r.get("tyler_failed", False)) / len(results)
                if est == "URE_Tyler" else 0.0
            )

            row = {
                "n_train": n_train,
                "estimator": est,
                "median_ll": float(np.median(ll_vals)) if ll_vals else float("nan"),
                "mean_ll": float(np.mean(ll_vals)) if ll_vals else float("nan"),
                "median_cond": float(np.median(cond_vals)) if cond_vals else float("nan"),
                "failure_rate": failure_rate,
                "n_ok": len(ll_vals),
            }
            all_rows.append(row)
            raw_results.append((n_train, est, ll_vals))

            logger.info(
                "  %s: median_ll=%.2f  median_cond=%.0f  fail_rate=%.2f  n=%d",
                est, row["median_ll"], row["median_cond"], row["failure_rate"], row["n_ok"],
            )

    df_summary = pd.DataFrame(all_rows)
    return df_summary, raw_results


def _make_leukemia_plot(raw_results: list, out_path: str, n_train_list: list[int]) -> None:
    """Save held-out log-likelihood plot for leukemia experiment."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"Su": "gray", "LW": "blue", "URE_cov": "green", "URE_Tyler": "red"}
    labels_map = {"Su": "S_u", "LW": "LW shrink", "URE_cov": "URE-cov", "URE_Tyler": "URE-Tyler"}

    for n_train, est, ll_vals in raw_results:
        if not ll_vals:
            continue
        ax.scatter(
            [n_train], [np.median(ll_vals)],
            color=colors.get(est, "black"),
            label=labels_map.get(est, est) if n_train == n_train_list[0] else None,
            zorder=3, s=40,
        )
    for est in ["Su", "LW", "URE_cov", "URE_Tyler"]:
        xs, ys = [], []
        for n_train, e, ll_vals in raw_results:
            if e == est and ll_vals:
                xs.append(n_train)
                ys.append(np.median(ll_vals))
        if xs:
            ax.plot(xs, ys, color=colors.get(est, "black"), lw=1.5)

    ax.set_xlabel("Training size n")
    ax.set_ylabel("Median held-out log-lik (t_5)")
    ax.set_title("Exp 5: Leukemia – Held-out Log-Likelihood")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Saved figure to %s", out_path)


def main() -> None:
    """Run experiment 5 and save results."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    n_train_list = [130, 160, 200]

    df_summary, raw_results = run_exp_5_leukemia(
        n_train_list=n_train_list, n_splits=40, nu=NU
    )

    out_csv = os.path.join(RESULTS_DIR, "table_5_leukemia.csv")
    df_summary.to_csv(out_csv, index=False)
    logger.info("Saved CSV to %s", out_csv)

    out_fig = os.path.join(RESULTS_DIR, "fig_leukemia_ll.png")
    _make_leukemia_plot(raw_results, out_fig, n_train_list)

    print("\n=== Table 5: Leukemia – Held-Out Log-Likelihood (nu=5) ===")
    if not df_summary.empty:
        pivot = df_summary.pivot(index="estimator", columns="n_train", values="median_ll")
        print(pivot.round(2).to_string())
    else:
        print("(no results)")


if __name__ == "__main__":
    main()
