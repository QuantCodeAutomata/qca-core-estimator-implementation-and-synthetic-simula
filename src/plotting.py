"""Visualization utilities for restricted robust covariance estimation experiments.

All plotting functions save figures to disk (save_path) and close the figure
to avoid memory leaks when called in batch simulation loops.
"""

import logging
import os
import numpy as np
from numpy.typing import NDArray
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless / batch runs
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import pandas as pd

from .shrinkage import marchenko_pastur_density

logger = logging.getLogger(__name__)

# Consistent colour palette across all plots
_PALETTE = sns.color_palette("tab10")
_ESTIMATOR_COLORS = {
    "URE": _PALETTE[0],
    "RRE": _PALETTE[1],
    "SSE+": _PALETTE[2],
    "URE-T": _PALETTE[3],
    "RRE-T": _PALETTE[4],
}


def _ensure_dir(path: str) -> None:
    """Create parent directories of path if they do not exist."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def plot_eigenvalue_histogram(
    eigenvalues_dict: dict,
    c: float,
    title: str,
    save_path: str,
) -> None:
    """Plot empirical eigenvalue histograms with Marchenko-Pastur overlay.

    For each entry in eigenvalues_dict, draws a normalised histogram of the
    provided eigenvalues.  The theoretical Marchenko-Pastur density for the
    given aspect ratio c is overlaid as a black dashed curve.

    Parameters
    ----------
    eigenvalues_dict : dict
        Mapping {label: array_of_eigenvalues} where each value is a 1-D array.
    c : float
        Aspect ratio p / n for the Marchenko-Pastur density overlay.
    title : str
        Figure title.
    save_path : str
        Absolute or relative path at which the PNG is saved.
    """
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(8, 5))

    for idx, (label, eigvals) in enumerate(eigenvalues_dict.items()):
        color = list(_ESTIMATOR_COLORS.values())[idx % len(_ESTIMATOR_COLORS)]
        ax.hist(
            eigvals,
            bins=min(60, max(10, len(eigvals) // 5)),
            density=True,
            alpha=0.45,
            color=color,
            label=label,
        )

    # Marchenko-Pastur overlay
    lam_plus = (1.0 + np.sqrt(c)) ** 2
    lam_minus = max(0.0, (1.0 - np.sqrt(c)) ** 2)
    x_grid = np.linspace(max(0.0, lam_minus * 0.5), lam_plus * 1.3, 500)
    mp_vals = np.array([marchenko_pastur_density(xi, c) for xi in x_grid])
    ax.plot(x_grid, mp_vals, "k--", linewidth=1.8, label=f"MP density (c={c:.2f})")

    ax.set_xlabel("Eigenvalue", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved eigenvalue histogram to %s", save_path)


def plot_risk_vs_aspect_ratio(
    results_df: pd.DataFrame,
    metric: str,
    title: str,
    save_path: str,
) -> None:
    """Plot estimator risk (metric) versus aspect ratio c.

    Expects results_df to contain columns 'c', 'estimator', and the column
    named by metric.  One line is drawn per unique 'estimator' value.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results table with at least columns 'c', 'estimator', and metric.
    metric : str
        Column name of the loss / risk measure to plot on the y-axis.
    title : str
        Figure title.
    save_path : str
        Path at which the PNG is saved.
    """
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(8, 5))

    for estimator, grp in results_df.groupby("estimator"):
        grp_sorted = grp.sort_values("c")
        color = _ESTIMATOR_COLORS.get(estimator, None)
        ax.plot(
            grp_sorted["c"],
            grp_sorted[metric],
            marker="o",
            markersize=4,
            linewidth=1.8,
            label=estimator,
            color=color,
        )

    ax.set_xlabel("Aspect ratio  c = p / (n - d)", fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved risk-vs-aspect-ratio plot to %s", save_path)


def plot_risk_vs_misspecification(
    results_df: pd.DataFrame,
    title: str,
    save_path: str,
) -> None:
    """Plot relative risk versus misspecification strength delta.

    Expects results_df to contain columns 'delta', 'estimator', and
    'relative_risk'.  One line per estimator (URE, RRE, SSE+) is drawn.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results with columns 'delta', 'estimator', 'relative_risk'.
    title : str
        Figure title.
    save_path : str
        Path at which the PNG is saved.
    """
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(8, 5))

    for estimator, grp in results_df.groupby("estimator"):
        grp_sorted = grp.sort_values("delta")
        color = _ESTIMATOR_COLORS.get(estimator, None)
        ax.plot(
            grp_sorted["delta"],
            grp_sorted["relative_risk"],
            marker="s",
            markersize=4,
            linewidth=1.8,
            label=estimator,
            color=color,
        )

    ax.axhline(y=1.0, color="grey", linestyle=":", linewidth=1.2, label="Baseline = 1")
    ax.set_xlabel("Misspecification strength  δ", fontsize=12)
    ax.set_ylabel("Relative risk", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved risk-vs-misspecification plot to %s", save_path)


def plot_heatmap(
    error_matrix: NDArray,
    title: str,
    save_path: str,
) -> None:
    """Plot an absolute entry-wise error heatmap |Sigma_hat - Sigma_true|.

    Parameters
    ----------
    error_matrix : NDArray
        Absolute error matrix |Sigma_hat - Sigma_true| of shape (p, p).
    title : str
        Figure title.
    save_path : str
        Path at which the PNG is saved.
    """
    _ensure_dir(save_path)
    p = error_matrix.shape[0]
    figsize = max(5, min(14, p // 10 + 5))
    fig, ax = plt.subplots(figsize=(figsize, figsize))

    # Subsample for very large p to keep the plot readable
    max_display = 100
    if p > max_display:
        idx = np.linspace(0, p - 1, max_display, dtype=int)
        display_mat = error_matrix[np.ix_(idx, idx)]
    else:
        display_mat = error_matrix

    im = ax.imshow(display_mat, aspect="auto", cmap="hot_r", interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Column index", fontsize=11)
    ax.set_ylabel("Row index", fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved heatmap to %s", save_path)


def plot_bias_variance(
    results: dict,
    title: str,
    save_path: str,
) -> None:
    """Plot bias-variance decomposition across estimators or settings.

    Expects results to be a dict with keys 'estimator', 'bias_sq', 'variance',
    and optionally 'mse'.  Draws a grouped bar chart with bias^2 and variance
    stacked.

    Parameters
    ----------
    results : dict
        Dictionary with list-valued keys: 'estimator' (str labels),
        'bias_sq' (floats), 'variance' (floats).  An optional 'mse' key
        may be provided for a total-MSE overlay.
    title : str
        Figure title.
    save_path : str
        Path at which the PNG is saved.
    """
    _ensure_dir(save_path)

    estimators = results["estimator"]
    bias_sq = np.asarray(results["bias_sq"], dtype=float)
    variance = np.asarray(results["variance"], dtype=float)
    x = np.arange(len(estimators))
    width = 0.55

    fig, ax = plt.subplots(figsize=(max(6, len(estimators) * 1.4), 5))

    bars_bias = ax.bar(x, bias_sq, width, label="Bias²", color=_PALETTE[0], alpha=0.85)
    bars_var = ax.bar(
        x, variance, width, bottom=bias_sq, label="Variance", color=_PALETTE[1], alpha=0.85
    )

    if "mse" in results:
        mse = np.asarray(results["mse"], dtype=float)
        ax.plot(x, mse, "D--", color="black", markersize=6, linewidth=1.5, label="MSE")

    ax.set_xticks(x)
    ax.set_xticklabels(estimators, rotation=20, ha="right", fontsize=11)
    ax.set_ylabel("Loss decomposition", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved bias-variance decomposition plot to %s", save_path)
