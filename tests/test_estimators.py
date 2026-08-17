import numpy as np
import pytest
import sys
sys.path.insert(0, '/workspace/project')


def test_weighted_frobenius_loss_identity():
    """WF loss is 0 when Sigma_hat = Sigma_true."""
    from src.metrics import weighted_frobenius_loss
    Sigma = np.diag([1.0, 2.0, 3.0])
    loss = weighted_frobenius_loss(Sigma, Sigma)
    assert np.isclose(loss, 0.0, atol=1e-10)


def test_weighted_frobenius_loss_positive():
    """WF loss is positive when Sigma_hat != Sigma_true."""
    from src.metrics import weighted_frobenius_loss
    Sigma_true = np.diag([1.0, 2.0, 3.0])
    Sigma_hat = np.diag([1.5, 2.5, 3.5])
    loss = weighted_frobenius_loss(Sigma_hat, Sigma_true)
    assert loss > 0


def test_tyler_trace_normalization():
    """Tyler scatter matrix should have trace=p."""
    from src.tyler import tyler_m_estimator
    rng = np.random.default_rng(42)
    p = 10
    n = 100
    residuals = rng.standard_normal((n, p))
    result = tyler_m_estimator(residuals)
    if result['converged']:
        V = result['V']
        assert np.isclose(np.trace(V), p, rtol=1e-5)


def test_tyler_symmetry():
    """Tyler scatter should be symmetric."""
    from src.tyler import tyler_m_estimator
    rng = np.random.default_rng(42)
    residuals = rng.standard_normal((100, 10))
    result = tyler_m_estimator(residuals)
    if result['converged']:
        V = result['V']
        assert np.allclose(V, V.T, atol=1e-8)


def test_sse_plus_kappa_q_less_than_3():
    """When q<3, kappa_n=0 and SSE+ equals URE."""
    from src.estimators import compute_sse_plus
    p = 10
    Sigma_ure = np.eye(p) * 2.0
    Sigma_rre = np.eye(p) * 1.5
    result = compute_sse_plus(Sigma_ure, Sigma_rre, T_n=0.5, n=200, d=40, q=2)
    assert np.isclose(result['kappa_n'], 0.0)
    assert np.allclose(result['Sigma_hat'], Sigma_ure)


def test_sse_plus_kappa_in_unit_interval():
    """kappa_n should always be in [0, 1]."""
    from src.estimators import compute_sse_plus
    p = 10
    Sigma_ure = np.eye(p) * 2.0
    Sigma_rre = np.eye(p) * 1.5
    # Test with various T_n values
    for T_n in [0.001, 0.1, 1.0, 10.0, 100.0]:
        result = compute_sse_plus(Sigma_ure, Sigma_rre, T_n=T_n, n=200, d=40, q=5)
        assert 0 <= result['kappa_n'] <= 1


def test_is_psd_identity():
    """Identity matrix is PSD."""
    from src.metrics import is_psd
    assert is_psd(np.eye(5))


def test_is_psd_negative():
    """Matrix with negative eigenvalue is not PSD."""
    from src.metrics import is_psd
    A = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigenvalues are 3 and -1
    assert not is_psd(A)


def test_multivariate_t_log_likelihood_finite():
    """Log-likelihood should be finite for well-conditioned inputs."""
    from src.metrics import multivariate_t_log_likelihood
    rng = np.random.default_rng(42)
    p = 5
    n_test = 50
    Y_test = rng.standard_normal((n_test, p))
    mu = np.zeros((1, p))
    Sigma = np.eye(p)
    ll = multivariate_t_log_likelihood(Y_test, mu, Sigma, nu=5.0)
    assert np.isfinite(ll)
    assert ll < 0  # Log-likelihood should be negative


def test_ols_projection_idempotent():
    """P_X should be idempotent."""
    from src.regression import compute_ols
    rng = np.random.default_rng(42)
    n, d, p = 100, 10, 20
    X = rng.standard_normal((n, d))
    Y = rng.standard_normal((n, p))
    result = compute_ols(X, Y)
    P_X = result['P_X']
    assert np.allclose(P_X @ P_X, P_X, atol=1e-8)


def test_restricted_ols_rank():
    """Restricted residual rank should be n-d+q."""
    from src.regression import compute_ols, compute_restricted_ols
    from src.data_gen import generate_design_matrix, generate_restriction_matrix
    rng = np.random.default_rng(42)
    n, d, p, q = 100, 10, 20, 5
    X = generate_design_matrix(n=n, d=d, rng=rng)
    Y = rng.standard_normal((n, p))
    R = generate_restriction_matrix(q=q, d=d, rng=rng)
    ols = compute_ols(X, Y)
    rest = compute_restricted_ols(X, Y, R, ols['B_hat'], ols['G'], ols['P_X'])
    P_r = rest['P_r']
    I_minus_Pr = np.eye(n) - P_r
    rank = np.linalg.matrix_rank(I_minus_Pr)
    expected_rank = n - d + q
    assert rank == expected_rank, f'Expected rank {expected_rank}, got {rank}'


def test_nonlinear_shrinkage_psd():
    """Nonlinear shrinkage output should be PSD."""
    from src.shrinkage import apply_nonlinear_shrinkage
    rng = np.random.default_rng(42)
    n, p = 200, 50
    X = rng.standard_normal((n, p))
    S = X.T @ X / n
    shrunk = apply_nonlinear_shrinkage(S, c=p/n)
    evals = np.linalg.eigvalsh(shrunk)
    assert np.all(evals >= -1e-8)
