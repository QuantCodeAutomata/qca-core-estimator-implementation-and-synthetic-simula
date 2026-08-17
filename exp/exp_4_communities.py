"""Experiment 4: Communities and Crime real data analysis.

Evaluates covariance estimators on held-out multivariate-t log-likelihood
(nu=5) over varying training sizes and 60 random splits.

Estimators compared: S_u (sample cov), LW linear shrinkage, URE-cov, URE-Tyler.

Dataset: UCI Communities and Crime (2214 observations, up to 128 columns).
  URL: https://archive.ics.uci.edu/ml/machine-learning-databases/communities/
  If the download fails, a synthetic proxy is generated instead.

Design: X (N x 4) = intercept + 3 region dummies (4 US census regions,
  West as reference). Y = 102 selected numeric socioeconomic columns.
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

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data"
)
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
N_Y_COLS = 102   # number of response columns
N_X_COLS = 4     # intercept + 3 region dummies
N_TEST = 500
NU = 5.0

# US census region mapping (state number 1-50 from UCI communities encoding)
# Regions: 1=Northeast, 2=Midwest, 3=South, 4=West (reference)
_NORTHEAST_STATES = {9, 23, 25, 33, 44, 50, 34, 36, 42}   # CT ME MA NH RI VT NJ NY PA
_MIDWEST_STATES = {17, 18, 26, 39, 55, 19, 20, 27, 29, 31, 38, 46}
_SOUTH_STATES = {10, 11, 12, 13, 21, 22, 24, 28, 37, 45, 47, 48,
                 51, 54, 1, 5, 40}
# All others → West (reference)


# ---------------------------------------------------------------------------
# Data download and fallback
# ---------------------------------------------------------------------------

def _state_to_region(state_val: float) -> int:
    """Map a UCI state code (1-50) to a 4-level US census region (1-4)."""
    s = int(state_val)
    if s in _NORTHEAST_STATES:
        return 1
    if s in _MIDWEST_STATES:
        return 2
    if s in _SOUTH_STATES:
        return 3
    return 4   # West (reference / intercept-only)


def download_communities_crime() -> pd.DataFrame:
    """Download UCI Communities and Crime dataset.

    Returns
    -------
    pd.DataFrame
        Raw dataframe with 128 columns; missing values coded as '?'.
        If download fails, returns a synthetic proxy flagged with attribute
        ``df.attrs['synthetic'] = True``.
    """
    logger.info("Attempting download from %s", DATA_URL)
    try:
        req = urllib.request.urlopen(DATA_URL, timeout=30)
        raw = req.read().decode("utf-8")
        df = pd.read_csv(io.StringIO(raw), header=None, na_values="?")
        df.attrs["synthetic"] = False
        logger.info("Download succeeded: %d rows x %d cols", len(df), df.shape[1])
        return df
    except (urllib.error.URLError, OSError, Exception) as exc:
        logger.warning("Download failed (%s). Generating synthetic proxy.", exc)

    # Synthetic fallback: N=2214 rows, 128 cols
    rng = np.random.default_rng(42)
    n_syn = 2214
    n_cols = 128
    data = rng.standard_normal((n_syn, n_cols))
    # Simulate state column (col 0) as integers 1-50
    data[:, 0] = rng.integers(1, 51, size=n_syn).astype(float)
    # Introduce some missing as NaN
    mask = rng.random((n_syn, n_cols)) < 0.03
    data[mask] = np.nan
    df = pd.DataFrame(data)
    df.attrs["synthetic"] = True
    logger.info("Synthetic proxy created: %d rows x %d cols", len(df), df.shape[1])
    return df


def preprocess_communities(df: pd.DataFrame) -> Tuple[NDArray, NDArray]:
    """Preprocess Communities and Crime dataframe into design X and response Y.

    Steps:
    1. Drop non-numeric ID columns (cols 0-4 in UCI layout).
    2. Convert remaining columns to numeric; impute missing with column medians.
    3. Select the first N_Y_COLS=102 numeric columns as Y.
    4. Build X (N x 4): [1, northeast_dummy, midwest_dummy, south_dummy].

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe from :func:`download_communities_crime`.

    Returns
    -------
    Tuple[NDArray, NDArray]
        ``X`` of shape (N, 4) and ``Y`` of shape (N, 102).
    """
    synthetic = df.attrs.get("synthetic", False)

    if not synthetic:
        # UCI layout: col 0=state, col 1=county, col 2=community, col 3=name, col 4=fold
        state_col = df.iloc[:, 0].copy()
        numeric_df = df.iloc[:, 5:].copy()
    else:
        state_col = df.iloc[:, 0].copy()
        numeric_df = df.iloc[:, 5:].copy()

    # Force numeric, fill missing with column medians
    numeric_df = numeric_df.apply(pd.to_numeric, errors="coerce")
    for col in numeric_df.columns:
        median_val = numeric_df[col].median()
        if np.isnan(median_val):
            median_val = 0.0
        numeric_df[col] = numeric_df[col].fillna(median_val)

    # Remove all-zero / all-constant columns
    numeric_df = numeric_df.loc[:, numeric_df.std() > 1e-8]

    # Select first N_Y_COLS columns as response
    n_avail = numeric_df.shape[1]
    if n_avail < N_Y_COLS:
        logger.warning(
            "Only %d numeric columns available; requested %d.  Repeating last column.",
            n_avail, N_Y_COLS,
        )
        deficit = N_Y_COLS - n_avail
        extra = pd.concat(
            [numeric_df.iloc[:, [-1]]] * deficit, axis=1, ignore_index=True
        )
        numeric_df = pd.concat([numeric_df, extra], axis=1, ignore_index=True)

    Y = numeric_df.iloc[:, :N_Y_COLS].to_numpy(dtype=float)
    N = len(Y)

    # Build X from region coding
    state_vals = pd.to_numeric(state_col, errors="coerce").fillna(4).to_numpy()
    region = np.array([_state_to_region(s) for s in state_vals])

    X = np.zeros((N, N_X_COLS))
    X[:, 0] = 1.0                           # intercept
    X[:, 1] = (region == 1).astype(float)  # Northeast
    X[:, 2] = (region == 2).astype(float)  # Midwest
    X[:, 3] = (region == 3).astype(float)  # South

    rank = np.linalg.matrix_rank(X)
    if rank < N_X_COLS:
        logger.warning("Design matrix rank %d < %d; adding tiny jitter.", rank, N_X_COLS)
        rng_jitter = np.random.default_rng(0)
        X += rng_jitter.normal(0, 1e-6, X.shape)

    logger.info("Preprocessed: X=%s  Y=%s  rank(X)=%d", X.shape, Y.shape, rank)
    return X.astype(float), Y.astype(float)


# ---------------------------------------------------------------------------
# Single-split evaluation (called in parallel)
# ---------------------------------------------------------------------------

def _evaluate_split(
    args: tuple,
) -> dict | None:
    """Evaluate one train/test split.  Returns a result dict or None on failure."""
    (X_full, Y_full, n_train, n_test, nu, split_seed) = args
    N_full = len(X_full)

    rng = np.random.default_rng(split_seed)
    idx = rng.permutation(N_full)
    train_idx = idx[:n_train]
    test_idx = idx[n_train: n_train + n_test]
    if len(test_idx) < n_test:
        return None   # not enough data

    X_tr, Y_tr = X_full[train_idx], Y_full[train_idx]
    X_te, Y_te = X_full[test_idx], Y_full[test_idx]

    # Standardize Y on training only
    Y_mean = Y_tr.mean(axis=0)
    Y_std = Y_tr.std(axis=0, ddof=1)
    Y_std[Y_std < 1e-8] = 1.0
    Y_tr_std = (Y_tr - Y_mean) / Y_std
    Y_te_std = (Y_te - Y_mean) / Y_std

    # OLS on training
    try:
        ols = compute_ols(X_tr, Y_tr_std)
    except ValueError:
        return None
    S_u = ols["S_u"]
    c_n = ols["c_n"]
    B_hat = ols["B_hat"]
    E_hat = ols["E_hat"]

    # Test residuals (centered): subtract regression prediction
    mu_test = X_te @ B_hat   # (n_test, p)

    result = {"n_train": n_train, "split_seed": split_seed}

    # S_u (sample covariance of OLS residuals)
    ll_su = multivariate_t_log_likelihood(Y_te_std - mu_test, np.zeros(Y_tr_std.shape[1]), S_u, nu)
    result["ll_Su"] = ll_su
    result["cond_Su"] = condition_number(S_u)

    # LW linear shrinkage
    lw_cov = linear_ledoit_wolf_shrinkage(E_hat, assume_centered=True)
    ll_lw = multivariate_t_log_likelihood(Y_te_std - mu_test, np.zeros(Y_tr_std.shape[1]), lw_cov, nu)
    result["ll_LW"] = ll_lw
    result["cond_LW"] = condition_number(lw_cov)

    # URE-cov (nonlinear shrinkage of S_u)
    ure_cov = compute_ure_cov(S_u, c_n)
    ll_ure = multivariate_t_log_likelihood(Y_te_std - mu_test, np.zeros(Y_tr_std.shape[1]), ure_cov, nu)
    result["ll_URE_cov"] = ll_ure
    result["cond_URE_cov"] = condition_number(ure_cov)

    # URE-Tyler (robust nonlinear shrinkage via Tyler M-estimator)
    try:
        ure_tyler = compute_ure_tyler(E_hat, c_n)
        sigma_tyler = ure_tyler["Sigma_hat"]
        ll_tyler = multivariate_t_log_likelihood(
            Y_te_std - mu_test, np.zeros(Y_tr_std.shape[1]), sigma_tyler, nu
        )
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

def run_exp_4_communities(
    n_train_list: list[int] | None = None,
    n_test: int = N_TEST,
    n_splits: int = 60,
    nu: float = NU,
) -> pd.DataFrame:
    """Run Communities and Crime held-out log-likelihood experiment.

    Parameters
    ----------
    n_train_list : list of int
        Training sizes to evaluate. Default [120, 150, 200, 300].
    n_test : int
        Number of held-out test observations per split. Default 500.
    n_splits : int
        Number of random train/test splits per training size. Default 60.
    nu : float
        Degrees of freedom for multivariate-t evaluation. Default 5.

    Returns
    -------
    pd.DataFrame
        Summary table: rows = (n_train, estimator), cols = median_ll, cond, failure_rate.
    """
    if n_train_list is None:
        n_train_list = [120, 150, 200, 300]

    df_raw = download_communities_crime()
    X_full, Y_full = preprocess_communities(df_raw)
    N_full = len(X_full)
    logger.info("Dataset: N=%d, p=%d, d=%d", N_full, Y_full.shape[1], X_full.shape[1])

    estimator_names = ["Su", "LW", "URE_cov", "URE_Tyler"]
    all_rows = []
    raw_results = []

    for n_train in n_train_list:
        if n_train + n_test > N_full:
            logger.warning(
                "n_train=%d + n_test=%d > N=%d; skipping.", n_train, n_test, N_full
            )
            continue

        logger.info("Training size n=%d  (%d splits)...", n_train, n_splits)
        t0 = time.time()

        args_list = [
            (X_full, Y_full, n_train, n_test, nu, seed)
            for seed in range(n_splits)
        ]

        results = []
        for args in args_list:
            r = _evaluate_split(args)
            if r is not None:
                results.append(r)

        elapsed = time.time() - t0
        logger.info("n_train=%d: %d/%d splits OK  (%.1fs)", n_train, len(results), n_splits, elapsed)

        if not results:
            continue

        # Aggregate per estimator
        for est in estimator_names:
            ll_key = f"ll_{est}"
            cond_key = f"cond_{est}"
            fail_key = "tyler_failed" if est == "URE_Tyler" else None

            ll_vals = [r[ll_key] for r in results if np.isfinite(r.get(ll_key, float("nan")))]
            cond_vals = [r[cond_key] for r in results if np.isfinite(r.get(cond_key, float("nan")))]
            failure_rate = (
                sum(1 for r in results if r.get("tyler_failed", False)) / len(results)
                if fail_key else 0.0
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


def _make_ll_plot(raw_results: list, out_path: str, n_train_list: list[int]) -> None:
    """Save held-out log-likelihood plot."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"Su": "gray", "LW": "blue", "URE_cov": "green", "URE_Tyler": "red"}
    labels = {"Su": "S_u", "LW": "LW shrink", "URE_cov": "URE-cov", "URE_Tyler": "URE-Tyler"}

    for n_train, est, ll_vals in raw_results:
        if not ll_vals:
            continue
        ax.scatter([n_train], [np.median(ll_vals)], color=colors.get(est, "black"),
                   label=labels.get(est, est) if n_train == n_train_list[0] else None,
                   zorder=3, s=40)
    # Connect medians per estimator
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
    ax.set_title("Exp 4: Communities & Crime – Held-out Log-Likelihood")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Saved figure to %s", out_path)


def main() -> None:
    """Run experiment 4 and save results."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    n_train_list = [120, 150, 200, 300]

    df_summary, raw_results = run_exp_4_communities(
        n_train_list=n_train_list, n_test=N_TEST, n_splits=60, nu=NU
    )

    out_csv = os.path.join(RESULTS_DIR, "table_4_communities.csv")
    df_summary.to_csv(out_csv, index=False)
    logger.info("Saved CSV to %s", out_csv)

    out_fig = os.path.join(RESULTS_DIR, "fig_communities_ll.png")
    _make_ll_plot(raw_results, out_fig, n_train_list)

    print("\n=== Table 4: Communities & Crime – Held-Out Log-Likelihood (nu=5) ===")
    if not df_summary.empty:
        pivot = df_summary.pivot(index="estimator", columns="n_train", values="median_ll")
        print(pivot.round(2).to_string())
    else:
        print("(no results)")


if __name__ == "__main__":
    main()
