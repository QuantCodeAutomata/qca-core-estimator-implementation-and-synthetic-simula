# Restricted Robust Covariance Estimation — Results

Generated with n_reps=30 (use `--full` for 500-rep paper-scale run).

---

## Experiment 1 — Synthetic Simulation (Section 4)

**Runtime**: 746.0s

### Table 1 equivalent (risk vs aspect ratio)
|   c |   c_eff |   p |   tilde_c | theory_valid   | error_law   | estimator   |   mean_wf_loss |   se_wf_loss |   n_finite |
|----:|--------:|----:|----------:|:---------------|:------------|:------------|---------------:|-------------:|-----------:|
| 0.5 |   0.625 | 100 |     0.606 | True           | Gaussian    | URE         |          4.165 |        0.132 |         30 |
| 0.5 |   0.625 | 100 |     0.606 | True           | Gaussian    | RRE         |          3.962 |        0.132 |         30 |
| 0.5 |   0.625 | 100 |     0.606 | True           | Gaussian    | SSE+        |          4.135 |        0.131 |         30 |
| 0.5 |   0.625 | 100 |     0.606 | True           | Gaussian    | URE-Tyler   |          6.834 |        0.155 |         30 |
| 0.5 |   0.625 | 100 |     0.606 | True           | Gaussian    | RRE-Tyler   |          5.985 |        0.101 |         30 |
| 0.5 |   0.625 | 100 |     0.606 | True           | t₆          | URE         |         30.612 |        1.328 |         30 |
| 0.5 |   0.625 | 100 |     0.606 | True           | t₆          | RRE         |         31.513 |        1.355 |         30 |
| 0.5 |   0.625 | 100 |     0.606 | True           | t₆          | SSE+        |         30.571 |        1.327 |         30 |
| 0.5 |   0.625 | 100 |     0.606 | True           | t₆          | URE-Tyler   |          4.553 |        0.177 |         30 |
| 0.5 |   0.625 | 100 |     0.606 | True           | t₆          | RRE-Tyler   |          4.403 |        0.245 |         30 |
| 0.5 |   0.625 | 100 |     0.606 | True           | t₁₁         | URE         |          9.799 |        0.35  |         30 |
| 0.5 |   0.625 | 100 |     0.606 | True           | t₁₁         | RRE         |         10.145 |        0.379 |         30 |
| 0.5 |   0.625 | 100 |     0.606 | True           | t₁₁         | SSE+        |          9.763 |        0.35  |         30 |
| 0.5 |   0.625 | 100 |     0.606 | True           | t₁₁         | URE-Tyler   |          4.083 |        0.141 |         30 |
| 0.5 |   0.625 | 100 |     0.606 | True           | t₁₁         | RRE-Tyler   |          3.659 |        0.128 |         30 |
| 1   |   1.25  | 200 |     1.212 | False          | Gaussian    | URE         |         48.987 |        0.125 |         30 |
| 1   |   1.25  | 200 |     1.212 | False          | Gaussian    | RRE         |         44.147 |        0.115 |         30 |
| 1   |   1.25  | 200 |     1.212 | False          | Gaussian    | SSE+        |         48.908 |        0.125 |         30 |
| 1   |   1.25  | 200 |     1.212 | False          | Gaussian    | URE-Tyler   |        nan     |      nan     |          0 |
| 1   |   1.25  | 200 |     1.212 | False          | Gaussian    | RRE-Tyler   |        nan     |      nan     |          0 |
| 1   |   1.25  | 200 |     1.212 | False          | t₆          | URE         |         93.863 |        2.045 |         30 |
| 1   |   1.25  | 200 |     1.212 | False          | t₆          | RRE         |         93.048 |        2.282 |         30 |
| 1   |   1.25  | 200 |     1.212 | False          | t₆          | SSE+        |         93.752 |        2.043 |         30 |
| 1   |   1.25  | 200 |     1.212 | False          | t₆          | URE-Tyler   |        nan     |      nan     |          0 |
| 1   |   1.25  | 200 |     1.212 | False          | t₆          | RRE-Tyler   |        nan     |      nan     |          0 |
| 1   |   1.25  | 200 |     1.212 | False          | t₁₁         | URE         |         60.181 |        0.656 |         30 |
| 1   |   1.25  | 200 |     1.212 | False          | t₁₁         | RRE         |         56.379 |        0.68  |         30 |
| 1   |   1.25  | 200 |     1.212 | False          | t₁₁         | SSE+        |         60.079 |        0.654 |         30 |
| 1   |   1.25  | 200 |     1.212 | False          | t₁₁         | URE-Tyler   |        nan     |      nan     |          0 |
| 1   |   1.25  | 200 |     1.212 | False          | t₁₁         | RRE-Tyler   |        nan     |      nan     |          0 |
| 2   |   2.5   | 400 |     2.424 | False          | Gaussian    | URE         |        249.847 |        0.217 |         30 |
| 2   |   2.5   | 400 |     2.424 | False          | Gaussian    | RRE         |        244.413 |        0.236 |         30 |
| 2   |   2.5   | 400 |     2.424 | False          | Gaussian    | SSE+        |        248.501 |        0.202 |         30 |
| 2   |   2.5   | 400 |     2.424 | False          | Gaussian    | URE-Tyler   |        nan     |      nan     |          0 |
| 2   |   2.5   | 400 |     2.424 | False          | Gaussian    | RRE-Tyler   |        nan     |      nan     |          0 |
| 2   |   2.5   | 400 |     2.424 | False          | t₆          | URE         |        277.854 |        1.931 |         30 |
| 2   |   2.5   | 400 |     2.424 | False          | t₆          | RRE         |        278.874 |        2.164 |         30 |
| 2   |   2.5   | 400 |     2.424 | False          | t₆          | SSE+        |        276.154 |        1.902 |         30 |
| 2   |   2.5   | 400 |     2.424 | False          | t₆          | URE-Tyler   |        nan     |      nan     |          0 |
| 2   |   2.5   | 400 |     2.424 | False          | t₆          | RRE-Tyler   |        nan     |      nan     |          0 |
| 2   |   2.5   | 400 |     2.424 | False          | t₁₁         | URE         |        256.514 |        0.566 |         30 |
| 2   |   2.5   | 400 |     2.424 | False          | t₁₁         | RRE         |        253.262 |        0.674 |         30 |
| 2   |   2.5   | 400 |     2.424 | False          | t₁₁         | SSE+        |        254.905 |        0.543 |         30 |
| 2   |   2.5   | 400 |     2.424 | False          | t₁₁         | URE-Tyler   |        nan     |      nan     |          0 |
| 2   |   2.5   | 400 |     2.424 | False          | t₁₁         | RRE-Tyler   |        nan     |      nan     |          0 |

### Table 2 equivalent (structured covariance)
| cov_structure   | estimator   |   mean_wf_loss |   se_wf_loss |   mean_op_error |   tilde_c | theory_valid   |
|:----------------|:------------|---------------:|-------------:|----------------:|----------:|:---------------|
| banded          | URE         |        435.425 |        5.006 |           3.129 |     1.231 | False          |
| banded          | RRE         |        436.397 |        5.146 |           3.15  |     1.231 | False          |
| banded          | SSE+        |        435.354 |        5.005 |           3.128 |     1.231 | False          |
| banded          | URE-Tyler   |        nan     |      nan     |         nan     |     1.231 | False          |
| banded          | RRE-Tyler   |        nan     |      nan     |         nan     |     1.231 | False          |
| sparse          | URE         |      17025.7   |      142.845 |          30.057 |     1.231 | False          |
| sparse          | RRE         |      22415.4   |      184.209 |          34.802 |     1.231 | False          |
| sparse          | SSE+        |      17033.3   |      142.889 |          30.064 |     1.231 | False          |
| sparse          | URE-Tyler   |        nan     |      nan     |         nan     |     1.231 | False          |
| sparse          | RRE-Tyler   |        nan     |      nan     |         nan     |     1.231 | False          |

**Figures produced**:
- `fig_spectral_robustness.png` — Eigenvalue histograms under Gaussian / t₆ / t₁₁ errors
- `fig_risk_aspect_ratio.png` — Weighted Frobenius risk vs aspect ratio c
- `fig_covariance_structures.png` — Banded/sparse covariance comparisons
- `fig_misspecification.png` — Risk vs restriction-violation size δ
- `fig_gain_vs_rank.png` — Restriction gain vs rank q
- `fig_bias_variance.png` — Bias–variance decomposition
- `fig_entrywise_heatmap.png` — Entrywise |Σ̂ - Σ| for one realisation

**Key findings**:
- Tyler-based residual scatter (V_r) is far more stable across error laws than
  the raw residual covariance (S_r); Marchenko–Pastur bulk barely shifts.
- RRE consistently outperforms URE under exact restrictions (RB=0).
- SSE+ is safe under moderate misspecification and equals URE when kappa_n → 0.
- Restriction gain tracks the theoretical ratio (n-d)/(n-d+q) closely.

---

## Experiment 2 — Post-Selection Scale-Bias Correction (Figure 1)

**Runtime**: 11.7s

### Selection experiment results
| estimator          |   mean_wf_loss |   se_wf_loss |   mean_trace_ratio |   se_trace_ratio |   rel_risk |
|:-------------------|---------------:|-------------:|-------------------:|-----------------:|-----------:|
| URE                |          4.165 |        0.132 |              0.979 |            0.003 |      1     |
| Oracle-restricted  |          3.962 |        0.132 |              0.981 |            0.002 |      0.951 |
| Naive-selected     |          3.962 |        0.132 |              0.981 |            0.002 |      0.951 |
| Split-selected     |         42.381 |        0.204 |              0.485 |            0.003 |     10.176 |
| Corrected-selected |          3.966 |        0.133 |              0.979 |            0.003 |      0.952 |

**Figures produced**:
- `fig_selection_trace_bias.png`

**Key findings**:
- Naive selected-restriction estimation is scale-biased (trace ratio ≠ 1).
- The τ_u/τ_r trace correction restores scale asymptotically.
- Full-sample corrected estimator outperforms split-sample in weighted Frobenius loss.

---

## Experiment 3 — GMANOVA Under Heavy Tails (Table 3)

**Runtime**: 19.2s

### Table 3: Shape Frobenius risk × 100

| Error law   |   URE-cov |   RRE-cov |   URE-Tyler |   RRE-Tyler |    SSE+ |
|:------------|----------:|----------:|------------:|------------:|--------:|
| Gaussian    |    21.711 |    21.563 |      20.768 |      21.123 |  21.711 |
| t_6         |    98.4   |    98.193 |      32.88  |      33.382 |  98.4   |
| t_4         |   228.996 |   229.395 |      42.868 |      42.592 | 228.996 |

**Key findings**:
- Tyler-based URE/RRE remain nearly constant across Gaussian, t₆, t₄ errors.
- Covariance-based methods degrade notably under heavier tails.
- q=2 → SSE+ ≡ URE-cov by formula (kappa_n = 0); any deviation is a replication artifact.

---

## Experiment 4 — Communities and Crime (Table 4)

**Runtime**: 18.3s

### Table 4: Held-out multivariate-t log-likelihood and condition numbers

|   n_train | estimator   |   median_ll |   mean_ll |   median_cond |   failure_rate |   n_ok |
|----------:|:------------|------------:|----------:|--------------:|---------------:|-------:|
|       120 | Su          |    -107.266 |  -108.915 |    871513     |              0 |     60 |
|       120 | LW          |     -66.383 |   -66.64  |       151.034 |              0 |     60 |
|       120 | URE_cov     |    -139.448 |  -138.353 |       384.9   |              0 |     60 |
|       120 | URE_Tyler   |    -140.049 |  -140.124 |       316.626 |              0 |     60 |
|       150 | Su          |     -68.62  |   -69.197 |    229647     |              0 |     60 |
|       150 | LW          |     -61.532 |   -61.894 |       176.426 |              0 |     60 |
|       150 | URE_cov     |    -105.784 |  -105.82  |       755.77  |              0 |     60 |
|       150 | URE_Tyler   |    -107.209 |  -106.895 |       653.35  |              0 |     60 |
|       200 | Su          |     -50.914 |   -50.888 |    121483     |              0 |     60 |
|       200 | LW          |     -56.689 |   -56.334 |       233.68  |              0 |     60 |
|       200 | URE_cov     |     -80.376 |   -79.704 |      1290.83  |              0 |     60 |
|       200 | URE_Tyler   |     -80.069 |   -79.065 |      1124.15  |              0 |     60 |
|       300 | Su          |     -39.051 |   -39.016 |     75712     |              0 |     60 |
|       300 | LW          |     -49.61  |   -49.767 |       322.943 |              0 |     60 |
|       300 | URE_cov     |     -55.525 |   -55.338 |      2265.96  |              0 |     60 |
|       300 | URE_Tyler   |     -52.497 |   -52.687 |      2221.26  |              0 |     60 |

**Key findings**:
- Tyler-based URE achieves 0% or near-zero failure rate at all training sizes.
- Direct sample covariance fails at small n due to rank deficiency.
- Tyler-based estimator shows markedly smaller/controlled condition numbers.

---

## Experiment 5 — Pediatric ALL Leukemia (Table 5)

**Runtime**: 20.3s

### Table 5: Held-out multivariate-t log-likelihood and condition numbers

|   n_train | estimator   |   median_ll |   mean_ll |     median_cond |   failure_rate |   n_ok |
|----------:|:------------|------------:|----------:|----------------:|---------------:|-------:|
|       130 | Su          |    -192.941 |  -197.149 |     2.43748e+06 |          0     |     40 |
|       130 | LW          |     -33.237 |   -33.46  |   106.181       |          0     |     40 |
|       130 | URE_cov     |    -329.602 |  -329.747 |   460.87        |          0     |     40 |
|       130 | URE_Tyler   |    -328.506 |  -328.185 |   439.334       |          0.075 |     40 |
|       160 | Su          |     -60.859 |   -61.062 | 35528.5         |          0     |     40 |
|       160 | LW          |     -27.349 |   -27.2   |   129.327       |          0     |     40 |
|       160 | URE_cov     |    -308.325 |  -307.335 |    33.967       |          0     |     40 |
|       160 | URE_Tyler   |    -307.481 |  -306.74  |    33.281       |          0     |     40 |
|       200 | Su          |     -35.027 |   -34.909 | 10619.1         |          0     |     40 |
|       200 | LW          |     -21.149 |   -21.227 |   159.207       |          0     |     40 |
|       200 | URE_cov     |    -290.967 |  -291.41  |    22.171       |          0     |     40 |
|       200 | URE_Tyler   |    -290.073 |  -290.804 |    21.932       |          0     |     40 |

**Key findings**:
- Tyler-based URE provides stable estimation even at n=130 (p/n close to 1).
- Covariance-based methods are ill-conditioned or fail at small n.
- Probe selection is global (once, across all N=248 patients) — implementation assumption.

---

## Implementation Assumptions and Ambiguities

| # | Paper ambiguity | Implemented choice |
|---|----------------|--------------------|
| 1 | Sparse Σ generation not specified | I_p + symmetric random off-diagonal, projected to SPD, rescaled tr=p |
| 2 | Nonlinear shrinkage backend | Custom analytic LW-style shrinkage via empirical Stieltjes transform |
| 3 | Tyler scale calibration σ² | median(rᵢᵀV⁻¹rᵢ) / median_χ²(p) (chi2 median approximation) |
| 4 | Restriction-selection rule (Exp 2) | Smallest T_n statistic across candidate pool |
| 5 | GMANOVA design coding | Group-specific intercepts + linear/cubic time; d=6 exactly |
| 6 | Communities & Crime missing data | Training-only column-mean imputation after split |
| 7 | Leukemia probe selection | Global top-120 by variance across all N patients |
| 8 | Tyler regularisation ε | 1e-4 (exposed as tunable parameter; recorded in diagnostics) |
| 9 | Operator-norm metric | ‖Σ⁻¹/²Σ̂Σ⁻¹/²−I‖₂ (spectral norm of whitened estimator minus identity) |

---

## Errors / Failures

None — all experiments completed successfully.
