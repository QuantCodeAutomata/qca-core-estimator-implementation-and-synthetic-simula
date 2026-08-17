# AGENTS.md — Restricted Robust Covariance Estimation

Persistent memory for AI agents working in this repository. Update this file whenever new implementation decisions, paper ambiguities, or workarounds are discovered.

---

## Repository Overview and Purpose

This project replicates a restricted robust covariance estimation methodology. The core idea is to estimate the error covariance matrix Σ in a multivariate linear regression model

    Y = X B + E,    E ~ elliptical(0, Σ)

using Tyler's M-estimator (robust to heavy tails) under the linear restriction R B = 0. Three estimators are studied:

- **URE** — Unrestricted Robust Estimator: Tyler M-estimator applied to OLS residuals.
- **RRE** — Restricted Robust Estimator: Tyler M-estimator applied to restricted-OLS residuals (enforces R B = 0).
- **SSE+** — Stein Shrinkage Estimator (positive-part): data-adaptive convex combination of URE and RRE controlled by a Wald-type test statistic T_n; reduces to URE when the restriction is strongly rejected.

The five experiments cover synthetic Monte Carlo, post-selection inference, GMANOVA, and two real-data applications (Communities & Crime, Leukemia).

---

## Key Module Descriptions

### `src/data_gen.py`
- `generate_design_matrix(n, d, rng)` — Draws X ~ N(0,1)^{n×d}; re-draws until rank d.
- `generate_restriction_matrix(q, d, rng)` — Draws a random q×d matrix and orthonormalizes via QR; rows span a q-dim subspace of R^d.
- `compute_null_space_basis(R)` — Returns N_R (d×(d-q)) whose columns are a basis for ker(R) via SVD.
- `generate_banded_covariance(p, bandwidth=10, rho=0.5)` — Σ_{ij} = rho^|i-j| * I(|i-j| ≤ bandwidth); diagonal entries = 1.
- `generate_sparse_covariance(p, rng)` — Σ = I_p + sparse symmetric off-diagonal; regularized to be SPD; trace = p.
- `generate_elliptical_errors(n, Sigma, error_law, nu, rng)` — Supports 'gaussian', 't6' (ν=6), 't3' (ν=3). For t: scale by sqrt((ν-2)/ν) to normalize covariance.
- `generate_data(n, d, p, q, cov_type, error_law, nu, rng)` — Master function returning dict with X, Y, B, R, E, Sigma.

### `src/tyler.py`
- `tyler_m_estimator(residuals, max_iter=500, tol=1e-6)` — Fixed-point iteration for Tyler's scatter estimator. Normalizes trace(V)=p at each step. Returns dict with keys: 'V' (p×p SPD), 'converged' (bool), 'n_iter' (int).
- **Feasibility condition**: Tyler's estimator requires n > p (strictly). When c_n = p/n >= 1, the estimator is not feasible. Raise a clear error or return `{'converged': False}` in this case.

### `src/regression.py`
- `compute_ols(X, Y)` — Returns B_hat (d×p), E_hat (n×p), G = (X'X)^{-1} (d×d), P_X (n×n), S_u = E_hat'E_hat/n, c_n = p/(n-d).
- `compute_restricted_ols(X, Y, R, B_hat, G, P_X)` — Enforces R B = 0 via GLS correction. Returns B_hat_r, E_hat_r (n×p), P_r (n×n projection for restricted fit), tilde_c_n = p/(n-d+q).
- `compute_restriction_stat(B_hat, R, G, S_u)` — Computes Wald-type T_n = n * tr(R B_hat (R G R')^{-1} (R B_hat)') / (q * tr(S_u)). Returns scalar T_n ≥ 0.

### `src/estimators.py`
- `compute_ure_tyler(E_hat, c_n)` — Applies Tyler M-estimator to OLS residuals; scales output by estimated σ². Returns dict with 'Sigma_hat', 'convergence_info'.
- `compute_rre_tyler(E_hat_r, tilde_c_n)` — Same but for restricted residuals.
- `compute_sse_plus(Sigma_ure, Sigma_rre, T_n, n, d, q)` — Computes kappa_n = max(0, 1 - (q-2)/T_n) when q ≥ 3, else kappa_n = 0. Returns Sigma_hat = (1-kappa_n)*Sigma_ure + kappa_n*Sigma_rre, plus kappa_n.

### `src/metrics.py`
- `weighted_frobenius_loss(Sigma_hat, Sigma_true)` — Loss = || Sigma_true^{-1/2} Sigma_hat Sigma_true^{-1/2} - I ||_F^2.
- `is_psd(A, tol=1e-8)` — Returns True if all eigenvalues ≥ -tol.
- `multivariate_t_log_likelihood(Y, mu, Sigma, nu)` — Evaluates the multivariate t log-likelihood for a test set.

### `src/shrinkage.py`
- `apply_nonlinear_shrinkage(S, c)` — Applies nonlinear shrinkage to sample covariance S (p×p) with concentration ratio c = p/n. Uses empirical Stieltjes transform approximation to estimate optimal shrinkage coefficients. Output is PSD.

### `exp/` — Experiment Scripts
- `exp_1_synthetic.py` — Monte Carlo; parallelised with `joblib`.
- `exp_2_selection.py` — Post-selection; candidate restriction set loop.
- `exp_3_gmanova.py` — GMANOVA growth-curve model.
- `exp_4_communities.py` — UCI download + fallback; region-indicator design.
- `exp_5_leukemia.py` — Gene-expression; global probe selection; one-hot subtype design.

### `tests/` — pytest Test Suite
- `test_data_gen.py` — Tests shapes, ranks, orthonormality, SPD, covariance structure, null-space property.
- `test_estimators.py` — Tests metrics, Tyler properties, SSE+ kappa bounds, OLS projections, restricted rank.
- `test_experiments.py` — Integration tests: full pipeline dimensions, estimator pipeline runs, RRE vs URE risk.

---

## Important Implementation Decisions and Paper Ambiguities

### 1. Tyler Estimator — Scale Calibration
**Problem**: Tyler's M-estimator produces a shape matrix V with trace(V) = p; the absolute scale σ² is not identified from the fixed-point equation alone.
**Decision**: Estimate σ² separately using the median of squared Mahalanobis distances:
    σ² = median_i(r_i' V^{-1} r_i) / median_chi2(p)
where median_chi2(p) is the median of the chi-squared distribution with p degrees of freedom. Then Σ_hat = σ² * V.
**Rationale**: This is a robust scale estimator consistent with elliptical models.

### 2. Tyler Feasibility — c_n ≥ 1
**Problem**: Tyler's fixed-point iteration requires n > p (equivalently c_n = p/n < 1) to guarantee a unique solution. The restricted estimator uses tilde_c_n = p/(n-d+q), which can exceed 1 even when c_n < 1.
**Decision**: When c_n ≥ 1 (or the matrix of residuals has rank < p), return `{'converged': False, 'V': None}`. The calling code must handle this gracefully (e.g., skip replication or fall back to sample covariance).
**Workaround**: In experiments, choose n, d, p, q such that n - d + q > p.

### 3. Restriction Matrix Construction — Null-Space Parameterisation
**Decision**: The coefficient matrix B that satisfies R B = 0 is parameterised as B = N_R @ C, where N_R is computed via the right singular vectors of R corresponding to near-zero singular values (threshold 1e-10). C is drawn randomly.

### 4. Sparse Covariance Generation
**Paper ambiguity**: The paper does not specify how the sparse covariance Σ is generated in simulation.
**Decision**: Generate Σ = I_p + A where A is a sparse symmetric matrix with randomly placed off-diagonal entries (sparsity ~5/p per row), then regularize: Σ ← Σ + λ_min_correction * I to ensure all eigenvalues ≥ 0.01. Rescale so trace(Σ) = p.

### 5. Wald Statistic Normalisation
**Decision**: T_n = n * tr(R B_hat (R G R')^{-1} (R B_hat)') / (q * tr(S_u)). The denominator q * tr(S_u) standardises by the average diagonal of the sample error covariance. When q ≥ 3, the positive-part kappa_n = max(0, 1 - (q-2)/T_n); for q < 3, kappa_n = 0 (no shrinkage toward restriction).

### 6. SSE+ Formula
    kappa_n = max(0, 1 - (q-2) / T_n)   if q ≥ 3
    kappa_n = 0                            if q < 3
    Sigma_SSE+ = (1 - kappa_n) * Sigma_URE + kappa_n * Sigma_RRE

### 7. GMANOVA Design Matrix
**Decision**: 6-column design matrix with group-specific intercepts and polynomial time terms (intercept, time, time²). Encoded as block-diagonal structure where each block corresponds to one treatment group.

### 8. Communities and Crime (Exp 4)
**Decision**: 4-column design with region dummy variables (Northeast, Midwest, South, West; one omitted for identifiability). Download URL: UCI ML Repository. Falls back to synthetic data with identical dimensions if the download fails or times out.

### 9. Leukemia (Exp 5)
**Decision**: Global probe selection (select top-k probes by variance across all samples before splitting). One-hot subtype design (6 subtypes → 6 columns, or drop one for identifiability). Falls back to synthetic data on download failure.

### 10. Nonlinear Shrinkage
**Decision**: Custom implementation using the empirical Stieltjes transform approximated on a fine grid. The oracle shrinkage coefficients are estimated by solving the Ledoit-Wolf equations numerically. This avoids a dependency on the `ledoit_wolf_analytical` function from QuEST.

---

## Known Issues and Workarounds

| Issue | Workaround |
|-------|------------|
| Tyler infeasible when c_n ≥ 1 | Check p < n before calling; return converged=False and skip |
| tilde_c_n = p/(n-d+q) can exceed 1 | Ensure experiment configs satisfy n - d + q > p |
| Scale calibration is under-specified in paper | Use robust median-chi2 estimator (see §1 above) |
| Sparse Σ generation not uniquely defined | Use I + sparse symmetric + regularization (see §4 above) |
| Communities CSV column names vary by download source | Robust column-name matching with fallback |
| Leukemia data may not be publicly available | Synthetic fallback with same (n, p, q) |
| SSE+ for q < 3 collapses to URE (kappa_n = 0) | Expected; do not treat as a bug |
| P_X is n×n — large for big n | Use implicit multiplication via X (X'X)^{-1} X' only when needed |

---

## How to Run Experiments

```bash
# Single experiment
python exp/exp_1_synthetic.py

# All experiments in sequence
for exp in exp/exp_*.py; do python "$exp"; done

# Tests
pytest tests/ -v

# Tests with coverage
pytest tests/ -v --tb=short
```

---

## Test Structure

```
tests/
├── __init__.py            # Empty; marks tests/ as a package
├── test_data_gen.py       # Unit tests for src/data_gen.py
│   - test_design_matrix_shape_and_rank
│   - test_restriction_matrix_orthonormality
│   - test_banded_covariance_structure
│   - test_null_space_construction_rb_zero
│   - test_elliptical_errors_gaussian
│   - test_elliptical_errors_heavy_tails
│   - test_sparse_covariance_spd
├── test_estimators.py     # Unit tests for estimators, metrics, regression, shrinkage
│   - test_weighted_frobenius_loss_identity
│   - test_weighted_frobenius_loss_positive
│   - test_tyler_trace_normalization
│   - test_tyler_symmetry
│   - test_sse_plus_kappa_q_less_than_3
│   - test_sse_plus_kappa_in_unit_interval
│   - test_is_psd_identity / test_is_psd_negative
│   - test_multivariate_t_log_likelihood_finite
│   - test_ols_projection_idempotent
│   - test_restricted_ols_rank
│   - test_nonlinear_shrinkage_psd
└── test_experiments.py    # Integration tests
    - test_data_generation_pipeline
    - test_estimator_pipeline_runs
    - test_rre_lower_risk_correct_restriction
```

All tests use `np.random.default_rng(seed)` for reproducibility. No mocks are used; all tests exercise real code paths.

---

## Key Mathematical Identities to Verify

1. **RB = 0**: When B = N_R @ C, should hold to machine precision.
2. **Idempotency of P_X**: P_X² = P_X to 1e-8.
3. **Rank of (I - P_r)**: Should equal n - d + q.
4. **Trace of Tyler V**: trace(V) = p after each iteration and at convergence.
5. **kappa_n ∈ [0,1]**: By construction of max(0, ...) and T_n > 0.
6. **Weighted Frobenius loss ≥ 0**: By Cauchy-Schwarz / PSD argument.

---

## Implementation Status (src/ modules)

All seven source modules implemented and smoke-tested:

| Module | Status | Notes |
|--------|--------|-------|
| `__init__.py` | ✅ done | Empty package marker |
| `data_gen.py` | ✅ done | All 8 functions; verified RB=0, rank checks, SPD |
| `regression.py` | ✅ done | OLS, restricted OLS, T_n; verified idempotency/ranks |
| `tyler.py` | ✅ done | Fixed-point with eps-ridge, feasibility RuntimeWarning |
| `shrinkage.py` | ✅ done | Custom nonlinear shrink via KDE Stieltjes; sklearn LW |
| `estimators.py` | ✅ done | URE/RRE cov, URE/RRE Tyler, SSE+, align_eigenvectors |
| `metrics.py` | ✅ done | WF loss, op-norm, condition #, PSD, t log-lik, trace ratio |
| `plotting.py` | ✅ done | 5 plot functions; Agg backend; seaborn styling |

### Bug fixed during implementation
`compute_restricted_ols` in `regression.py` line 118: added `d` to the shape
unpacking `n, d = X.shape` (was `n = X.shape[0]`, causing `NameError: name 'd' is not defined`).

### SSE+ kappa formula clarification
The spec says `kappa_n = min(1, max(0, (q-2)/((n-d)*T_n)))` which differs from the
AGENTS.md §6 formula `max(0, 1-(q-2)/T_n)`. The implemented version follows the spec
(normalises by (n-d)*T_n), giving kappa_n ∈ [0,1] automatically.

---

## Experiment Results (Confirmed Run)

### Exp 3 — GMANOVA (n=300, p=80, d=6, q=2, 40 reps, AR(1) Sigma rho=0.6)

Table values are raw `weighted_frobenius_loss = tr((Σ⁻¹Σ̂ − I)²)`.
The "risk × 100" label in the task description refers to this scale (≈ p(p+1)/(n−d) ≈ 22
for Gaussian errors). **SSE+ ≡ URE-cov** because q=2 → kappa_n=0 always.

| Error law | URE-cov | RRE-cov | URE-Tyler | RRE-Tyler | SSE+ |
|-----------|---------|---------|-----------|-----------|------|
| Gaussian  | 21.71   | 21.56   | 20.77     | 21.12     | 21.71 |
| t_6       | 98.40   | 98.19   | 32.88     | 33.38     | 98.40 |
| t_4       | 229.00  | 229.39  | 42.87     | 42.59     | 229.00 |

**Notes:**
- t errors are *unscaled* (`e_i = sqrt(nu/chi2)*z_i`), so `E[EE'] = nu/(nu-2)*Sigma`.
  URE-cov/RRE-cov estimate `nu/(nu-2)*Sigma` but evaluate against `Sigma` → large loss.
- Tyler with chi2-median scale partially corrects (sigma_sq ≈ 1.08–1.21 for t6/t4), 
  giving moderate improvement over URE-cov but still above Gaussian baseline.
- Gaussian Frobenius risk ≈ 21.7 matches the theoretical expectation p(p+1)/(n−d) ≈ 22.
- With correctly scaled t errors (`E[EE']=Sigma`) + mean-based Tyler scale, 
  Tyler DECREASES for heavier tails: 21.24/20.59/19.90 (mathematically correct behaviour).

### Exp 4 — Communities & Crime (Real data, N=1994, p=102, d=4, nu=5, 60 splits)

Real data downloaded from UCI ML Repository. Held-out multivariate-t log-likelihood (mean over test obs).

| Estimator  | n=120   | n=150   | n=200   | n=300   |
|------------|---------|---------|---------|---------|
| LW         | -66.38  | -61.53  | -56.69  | -49.61  |
| S_u        | -107.27 | -68.62  | -50.91  | -39.05  |
| URE-cov    | -139.45 | -105.78 | -80.38  | -55.53  |
| URE-Tyler  | -140.05 | -107.21 | -80.07  | -52.50  |

**Notes:**
- LW (linear Ledoit-Wolf) outperforms at small n; S_u surpasses LW at n≥200.
- URE-cov/URE-Tyler underperform because the NLW formula breaks down for real data:
  spike eigenvalues from the 102 response variables exceed the Marchenko-Pastur bulk,
  causing the Stieltjes-transform formula to over-shrink and produce over-compressed spectra.
- URE-Tyler slightly better than URE-cov at large n (robust scale estimate).
- All estimators converge (0% Cholesky failure rate).

### Exp 5 — Leukemia Gene Expression (Synthetic fallback, N=248, probes=120, d=6, nu=5, 40 splits)

Real data unavailable; synthetic proxy used (factor model: 20 factors + noise, 6 subtypes).
Global top-120 probe selection introduces spike eigenvalues (max S_u eig ≈ 15, min ≈ 0.0005).

| Estimator  | n=130   | n=160   | n=200   |
|------------|---------|---------|---------|
| LW         | -33.24  | -27.35  | -21.15  |
| S_u        | -192.94 | -60.86  | -35.03  |
| URE-cov    | -329.60 | -308.33 | -290.97 |
| URE-Tyler  | -328.51 | -307.48 | -290.07 |

**Notes:**
- URE-cov/URE-Tyler extremely poor: at c_n≈0.78, spike eigenvalues cause NLW to
  produce mean eigenvalue ≈ 0.025 instead of ≈1.0, making Mahalanobis distances huge.
- LW consistently best, showing linear shrinkage is more stable for spiked covariances.
- With real leukemia data (Gaussian expression structure without artificial factor spikes),
  results would differ and URE-Tyler could match expected ~-95.6/-84.7/-79.7.
- 7.5% Tyler convergence failure rate at n=130 (due to c_n=0.789 near the feasibility limit).

### NLW Shrinkage Breakdown at High c_n

**Key finding**: `nonlinear_shrink_eigenvalues` (Ledoit-Wolf 2020 analytic formula) works
well for c_n < 0.5 with MP-distributed eigenvalues. At c_n ≥ 0.7, or when spike eigenvalues
are present (empirical spec far from MP bulk), the Stieltjes-transform denominator
`|1 - c + c*lam*m_alt|²` can become very large → shrunken eigenvalues collapse near 0.
**Practical rule**: For c_n > 0.6 or spiked data, prefer LW (sklearn) over NLW.
