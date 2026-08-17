"""
exp_2_selection.py — Post-selection scale bias in restricted covariance estimation.

Tests how data-driven selection of a restriction matrix from a candidate pool
inflates or deflates the trace of the resulting estimator, and evaluates a
simple trace-correction strategy.

All results are written to /workspace/project/results/.

Experiment design
-----------------
A pool of M_candidates restriction matrices is constructed:
  candidates[0]  — oracle: true B lies exactly in null(R_oracle)
  candidates[1:] — random restrictions, generally misspecified

Five estimators per replication:
  1. URE                  — unrestricted (no restriction used)
  2. Oracle-restricted    — RRE with oracle R  (ground-truth knowledge)
  3. Naive-selected       — RRE with data-selected R  (full data for both
                             selection and estimation → selection bias)
  4. Split-selected       — selection on first n1 = n//2 obs; estimation on
                             remaining n2 obs (avoids reuse bias; smaller n_eff)
  5. Corrected-selected   — naive-selected rescaled so its trace matches URE:
                             Σ_corr = (τ_u / τ_r) · Σ_naive_sel
                             τ_u = tr(URE) / p
                             τ_r = tr(Σ_naive_sel) / p

Ambiguities / implementation assumptions
-----------------------------------------
  • SmallestStatSelector: selects the candidate R with the smallest
    compute_restriction_stat() value.  Smaller T_n → restriction more
    consistent with the data (lower evidence against RB = 0).
    Documented as an "implementation assumption"; the spec prescribes no
    specific selection rule.
  • Trace correction: normalises Σ̂_naive_sel to match the URE trace.  It
    does NOT adjust eigenvector structure or spectral shape.
  • Split-sample: effective sample size halved to n2 ≈ n/2 for estimation.
  • All true B are generated in null(R_oracle) so oracle restriction is valid.
"""

from __future__ import annotations

import math
import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from numpy.typing import NDArray

from src.data_gen import (
    compute_null_space_basis,
    generate_coefficient_matrix,
    generate_design_matrix,
    generate_elliptical_errors,
    generate_identity_covariance,
    generate_restriction_matrix,
)
from src.estimators import (
    compute_rre_cov,
    compute_ure_cov,
)
from src.metrics import weighted_frobenius_loss
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


def _d_default(n: int) -> int:
    return math.floor(0.2 * n)


# ---------------------------------------------------------------------------
# Base class and concrete selector
# ---------------------------------------------------------------------------


class RestrictionSelector(ABC):
    """Base class for data-driven restriction-matrix selection.

    Concrete subclasses implement :meth:`select`, which chooses one
    restriction matrix from a candidate pool and returns it with a scores
    dictionary.  By convention, **lower score ↔ preferred candidate**.
    """

    @abstractmethod
    def select(
        self,
        X: NDArray,
        Y: NDArray,
        candidates: List[NDArray],
        config: Optional[Dict] = None,
    ) -> Tuple[NDArray, Dict[str, float]]:
        """Choose one restriction matrix from *candidates*.

        Parameters
        ----------
        X : ndarray (n, d)
        Y : ndarray (n, p)
        candidates : list of ndarray, each (q, d)
        config : optional extra parameters

        Returns
        -------
        R_selected : ndarray (q, d) — the chosen restriction matrix.
        scores : dict {str(candidate_index): score}.
            Lower score ↔ preferred.
        """


class SmallestStatSelector(RestrictionSelector):
    """Select the candidate with the smallest restriction test statistic T_n.

    Implementation assumption (fallback rule)
    -----------------------------------------
    compute_restriction_stat(B_hat, R, G, S_u) is the Wald-type statistic for
    testing R B = 0.  Larger T_n → stronger evidence against the restriction.
    Selecting argmin T_n therefore picks the restriction most consistent with
    the observed B_hat.

    Under the oracle restriction (true B ∈ null(R)), T_n is approximately
    chi-squared scaled; under wrong restrictions T_n has a positive
    non-centrality term proportional to ‖RB‖².  Hence the oracle is selected
    with high probability in large samples.

    Tie-breaking: first candidate by index (degenerate case only).
    """

    def select(
        self,
        X: NDArray,
        Y: NDArray,
        candidates: List[NDArray],
        config: Optional[Dict] = None,
    ) -> Tuple[NDArray, Dict[str, float]]:
        ols = compute_ols(X, Y)
        B_hat, G, S_u = ols["B_hat"], ols["G"], ols["S_u"]
        scores: Dict[str, float] = {}
        for idx, R in enumerate(candidates):
            scores[str(idx)] = float(compute_restriction_stat(B_hat, R, G, S_u))
        best_str = min(scores, key=lambda k: scores[k])
        best_idx = int(best_str)
        return candidates[best_idx], scores


# ---------------------------------------------------------------------------
# Per-replication worker
# ---------------------------------------------------------------------------


def _rep_2(
    rep_idx: int,
    n: int,
    d: int,
    q: int,
    p: int,
    candidates: List[NDArray],
    oracle_idx: int,
    N_R_oracle: NDArray,
    Sigma: NDArray,
    selector: RestrictionSelector,
) -> Dict:
    """One replication for exp_2.

    Returns scalar metrics for all five estimators and selection diagnostics.

    Split-sample note
    -----------------
    n1 = n // 2 observations are used for restriction selection.
    n2 = n - n1 observations are used for estimation (split-selected only).
    The effective sample for the split estimator is n2 < n, so its absolute
    WF loss will typically exceed the naive estimator's even without selection
    bias — what matters is the trace ratio (scale bias).
    """
    rng = np.random.default_rng(BASE_SEED + rep_idx)

    # ── Generate full dataset ──
    X = generate_design_matrix(n, d, rng)
    B = generate_coefficient_matrix(N_R_oracle, p, rng)   # B ∈ null(R_oracle)
    E = generate_elliptical_errors(n, Sigma, "gaussian", None, rng)
    Y = X @ B + E

    # ── Split indices ──
    n1 = n // 2
    X1, Y1 = X[:n1], Y[:n1]
    X2, Y2 = X[n1:], Y[n1:]

    # ─── 1. URE (full data) ───
    ols_full = compute_ols(X, Y)
    G_f, P_X_f = ols_full["G"], ols_full["P_X"]
    B_hat_f, S_u_f, c_n = ols_full["B_hat"], ols_full["S_u"], ols_full["c_n"]
    Sigma_ure = compute_ure_cov(S_u_f, c_n)
    tau_u = float(np.trace(Sigma_ure)) / p        # mean eigenvalue of URE

    # ─── 2. Oracle-restricted (full data) ───
    R_oracle = candidates[oracle_idx]
    rols_oracle = compute_restricted_ols(X, Y, R_oracle, B_hat_f, G_f, P_X_f)
    Sigma_oracle = compute_rre_cov(rols_oracle["S_r"], rols_oracle["tilde_c_n"])

    # ─── 3. Naive-selected (full data for selection AND estimation) ───
    R_naive, scores_naive = selector.select(X, Y, candidates)
    sel_id_naive = _find_idx(candidates, R_naive)
    rols_naive = compute_restricted_ols(X, Y, R_naive, B_hat_f, G_f, P_X_f)
    Sigma_naive = compute_rre_cov(rols_naive["S_r"], rols_naive["tilde_c_n"])

    # ─── 4. Split-sample-selected ───
    R_split, scores_split = selector.select(X1, Y1, candidates)
    sel_id_split = _find_idx(candidates, R_split)
    ols2 = compute_ols(X2, Y2)
    rols_split = compute_restricted_ols(
        X2, Y2, R_split, ols2["B_hat"], ols2["G"], ols2["P_X"]
    )
    Sigma_split = compute_rre_cov(rols_split["S_r"], rols_split["tilde_c_n"])

    # ─── 5. Corrected-selected (trace correction on naive-selected) ───
    tau_r = float(np.trace(Sigma_naive)) / p
    scale = (tau_u / tau_r) if tau_r > 1e-15 else 1.0
    Sigma_corrected = scale * Sigma_naive

    # ─── Metrics ───
    def _tr(S: NDArray) -> float:
        return float(np.trace(S)) / p

    def _wf(S: NDArray) -> float:
        return weighted_frobenius_loss(S, Sigma)

    wf_ure  = _wf(Sigma_ure)
    wf_ora  = _wf(Sigma_oracle)
    wf_nav  = _wf(Sigma_naive)
    wf_spl  = _wf(Sigma_split)
    wf_cor  = _wf(Sigma_corrected)

    return {
        # Selection diagnostics
        "sel_id_naive":          sel_id_naive,
        "sel_id_split":          sel_id_split,
        "oracle_hit_naive":      int(sel_id_naive == oracle_idx),
        "oracle_hit_split":      int(sel_id_split == oracle_idx),
        "score_oracle_naive":    scores_naive.get(str(oracle_idx), float("nan")),
        "score_best_naive":      min(scores_naive.values()),
        # Trace ratios: tr(Σ̂)/p (should ≈ 1 for identity Sigma)
        "trace_ure":             _tr(Sigma_ure),
        "trace_oracle":          _tr(Sigma_oracle),
        "trace_naive":           _tr(Sigma_naive),
        "trace_split":           _tr(Sigma_split),
        "trace_corrected":       _tr(Sigma_corrected),
        # WF losses
        "wf_ure":                wf_ure,
        "wf_oracle":             wf_ora,
        "wf_naive":              wf_nav,
        "wf_split":              wf_spl,
        "wf_corrected":          wf_cor,
        # Relative risks (each / URE)
        "rel_risk_oracle":       wf_ora / wf_ure if wf_ure > 0 else float("nan"),
        "rel_risk_naive":        wf_nav / wf_ure if wf_ure > 0 else float("nan"),
        "rel_risk_split":        wf_spl / wf_ure if wf_ure > 0 else float("nan"),
        "rel_risk_corrected":    wf_cor / wf_ure if wf_ure > 0 else float("nan"),
    }


def _find_idx(candidates: List[NDArray], R: NDArray) -> int:
    """Return list index of the array equal to R (by object identity or value)."""
    for i, C in enumerate(candidates):
        if C is R or np.array_equal(C, R):
            return i
    return -1


# ---------------------------------------------------------------------------
# Main experiment function
# ---------------------------------------------------------------------------


def run_exp_2_selection(
    n: int = 200,
    p: int = 100,
    d: Optional[int] = None,
    q: int = 5,
    M_candidates: int = 10,
    n_reps: int = 500,
) -> None:
    """Post-selection inference experiment.

    Parameters
    ----------
    n            : sample size
    p            : response dimension
    d            : predictor count; default floor(0.2*n)
    q            : restriction rank (rows of each R)
    M_candidates : total candidate pool size (1 oracle + M-1 random)
    n_reps       : Monte Carlo replications

    Outputs
    -------
    results/fig_selection_trace_bias.png
    results/table_selection_risk.csv
    results/diagnostics_exp2.txt
    """
    if d is None:
        d = _d_default(n)

    tilde_c = p / (n - d + q)
    theory_valid = tilde_c < 1.0
    print(
        f"[exp_2] n={n}, d={d}, q={q}, p={p}, M={M_candidates}, "
        f"tilde_c={tilde_c:.3f}, theory_valid={theory_valid}, n_reps={n_reps}"
    )

    # ── Build candidate pool ──
    rng0 = np.random.default_rng(BASE_SEED)
    oracle_idx = 0
    R_oracle = generate_restriction_matrix(q, d, rng0)
    candidates: List[NDArray] = [R_oracle]
    for _ in range(M_candidates - 1):
        candidates.append(generate_restriction_matrix(q, d, rng0))

    N_R_oracle = compute_null_space_basis(R_oracle)
    Sigma = generate_identity_covariance(p)
    selector = SmallestStatSelector()

    # ── Parallel replications ──
    rep_results = Parallel(n_jobs=4, prefer="threads")(
        delayed(_rep_2)(
            rep, n, d, q, p, candidates, oracle_idx, N_R_oracle, Sigma, selector
        )
        for rep in range(n_reps)
    )
    df_raw = pd.DataFrame(rep_results)

    # ── Summary table ──
    est_specs = [
        ("URE",                "wf_ure",       "trace_ure"),
        ("Oracle-restricted",  "wf_oracle",    "trace_oracle"),
        ("Naive-selected",     "wf_naive",     "trace_naive"),
        ("Split-selected",     "wf_split",     "trace_split"),
        ("Corrected-selected", "wf_corrected", "trace_corrected"),
    ]
    summary_rows: List[Dict] = []
    for est_name, wf_col, tr_col in est_specs:
        wf_vals = df_raw[wf_col].values.astype(float)
        tr_vals = df_raw[tr_col].values.astype(float)
        summary_rows.append({
            "estimator":        est_name,
            "mean_wf_loss":     float(np.nanmean(wf_vals)),
            "se_wf_loss":       float(np.nanstd(wf_vals, ddof=1) / math.sqrt(n_reps)),
            "mean_trace_ratio": float(np.nanmean(tr_vals)),
            "se_trace_ratio":   float(np.nanstd(tr_vals, ddof=1) / math.sqrt(n_reps)),
        })
    df_summary = pd.DataFrame(summary_rows)
    ure_mean = df_summary.loc[df_summary["estimator"] == "URE", "mean_wf_loss"].iloc[0]
    df_summary["rel_risk"] = df_summary["mean_wf_loss"] / ure_mean

    csv_path = RESULTS_DIR / "table_selection_risk.csv"
    df_summary.to_csv(csv_path, index=False)

    # ── Selection frequency ──
    naive_oracle_rate = float(df_raw["oracle_hit_naive"].mean())
    split_oracle_rate = float(df_raw["oracle_hit_split"].mean())
    print(
        f"  [exp_2] Oracle selection rate — naive: {naive_oracle_rate:.3f}, "
        f"split: {split_oracle_rate:.3f}"
    )
    print(df_summary.to_string(index=False))

    # ── Trace bias figure ──
    fig_path = RESULTS_DIR / "fig_selection_trace_bias.png"
    _plot_trace_bias(df_raw, est_specs, fig_path)

    print(f"[exp_2] Done.  Table → {csv_path}  Plot → {fig_path}")
    _write_diagnostics(
        tilde_c=tilde_c, theory_valid=theory_valid,
        n=n, d=d, q=q, p=p, M=M_candidates,
        naive_oracle_rate=naive_oracle_rate,
        split_oracle_rate=split_oracle_rate,
    )


# ---------------------------------------------------------------------------
# Trace-bias violin plot
# ---------------------------------------------------------------------------


def _plot_trace_bias(
    df_raw: pd.DataFrame,
    est_specs: List[Tuple[str, str, str]],
    out_path: Path,
) -> None:
    """Violin plot of tr(Σ̂)/p per estimator across all replications.

    A dashed red line marks 1.0 (the true trace ratio for identity Sigma).
    Estimators centred above 1 are trace-inflated; those below are deflated.
    """
    records: List[Dict] = []
    for est_name, _, tr_col in est_specs:
        for v in df_raw[tr_col].values:
            records.append({"estimator": est_name, "trace_ratio": float(v)})
    df_long = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(11, 5))
    sns.violinplot(
        data=df_long,
        x="estimator",
        y="trace_ratio",
        hue="estimator",
        palette="Set2",
        inner="quartile",
        legend=False,
        ax=ax,
    )
    ax.axhline(1.0, color="red", linestyle="--", lw=1.8,
               label="True  tr(Σ)/p = 1")
    ax.set_title(
        "Trace ratio  tr($\\hat{\\Sigma}$)/p  across estimators\n"
        "(post-selection; identity Σ, Gaussian errors)"
    )
    ax.set_xlabel("Estimator")
    ax.set_ylabel("tr($\\hat{\\Sigma}$) / p")
    ax.legend(fontsize=10)
    plt.xticks(rotation=15, ha="right")
    sns.despine(ax=ax)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Diagnostics log
# ---------------------------------------------------------------------------


def _write_diagnostics(**kw) -> None:
    log_path = RESULTS_DIR / "diagnostics_exp2.txt"
    lines = [
        "exp_2_selection — diagnostics and ambiguity log",
        "=" * 60,
        "",
        "Design parameters:",
        f"  n={kw['n']}, d={kw['d']}, q={kw['q']}, p={kw['p']}, "
        f"M_candidates={kw['M']}",
        f"  tilde_c={kw['tilde_c']:.4f}, theory_valid={kw['theory_valid']}",
        "",
        "Selection frequency (fraction of reps oracle R chosen):",
        f"  Naive-selected:  {kw['naive_oracle_rate']:.3f}",
        f"  Split-selected:  {kw['split_oracle_rate']:.3f}",
        "",
        "Ambiguities and implementation assumptions:",
        "",
        "  1. SmallestStatSelector selects argmin T_n over candidates.",
        "     T_n = compute_restriction_stat(B_hat, R, G, S_u).",
        "     Smaller T_n → restriction more plausible given data.",
        "     This is a FALLBACK RULE documented as an implementation",
        "     assumption; no selection rule is prescribed in the spec.",
        "",
        "  2. Trace correction:",
        "       Sigma_corr = (tau_u / tau_r) * Sigma_naive_sel",
        "       tau_u = tr(URE) / p    (mean eigenvalue of URE)",
        "       tau_r = tr(Sigma_naive_sel) / p",
        "     Corrects trace inflation/deflation but not eigenvectors.",
        "",
        "  3. Split-sample splits n obs into n1=n//2 (selection) and",
        "     n2=n-n1 (estimation).  n_eff for estimation is halved.",
        "",
        "  4. Sigma = I_p throughout.  True trace ratio = 1.0.",
        "     Deviation from 1.0 indicates scale bias.",
        "",
        "  5. Oracle restriction = candidates[0].  All true B are",
        "     generated in null(R_oracle) via B = N_R_oracle @ C.",
        "     Remaining M-1 candidates are independently random.",
    ]
    log_path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run exp_2 and summarize results."""
    print("=" * 70)
    print("exp_2_selection — post-selection scale bias experiment")
    print(f"Results directory: {RESULTS_DIR}")
    print("=" * 70)
    run_exp_2_selection()
    print("=" * 70)
    print("Output files (exp_2):")
    for fp in sorted(RESULTS_DIR.iterdir()):
        if any(kw in fp.name for kw in ("selection", "exp2")):
            print(f"  {fp.name}")
    print("=" * 70)


if __name__ == "__main__":
    main()
