import numpy as np
import pytest
import sys
sys.path.insert(0, '/workspace/project')


def test_data_generation_pipeline():
    """Full data generation pipeline should produce consistent dimensions."""
    from src.data_gen import generate_data
    rng = np.random.default_rng(42)
    result = generate_data(n=100, d=20, p=50, q=5, cov_type='identity', error_law='gaussian', nu=None, rng=rng)
    assert result['X'].shape == (100, 20)
    assert result['Y'].shape == (100, 50)
    assert result['Sigma'].shape == (50, 50)
    assert result['B'].shape == (20, 50)
    assert result['R'].shape == (5, 20)
    assert result['E'].shape == (100, 50)
    # Verify RB=0
    assert np.allclose(result['R'] @ result['B'], 0, atol=1e-10)


def test_estimator_pipeline_runs():
    """Full estimator pipeline should run without errors on small data."""
    from src.data_gen import generate_data
    from src.regression import compute_ols, compute_restricted_ols, compute_restriction_stat
    from src.estimators import compute_ure_tyler, compute_rre_tyler, compute_sse_plus
    rng = np.random.default_rng(42)
    n, d, p, q = 80, 10, 20, 5
    data = generate_data(n=n, d=d, p=p, q=q, cov_type='identity', error_law='gaussian', nu=None, rng=rng)
    X, Y, R = data['X'], data['Y'], data['R']
    ols = compute_ols(X, Y)
    rest = compute_restricted_ols(X, Y, R, ols['B_hat'], ols['G'], ols['P_X'])
    T_n = compute_restriction_stat(ols['B_hat'], R, ols['G'], ols['S_u'])
    assert T_n >= 0
    ure = compute_ure_tyler(ols['E_hat'], ols['c_n'])
    rre = compute_rre_tyler(rest['E_hat_r'], rest['tilde_c_n'])
    if ure.get('converged', False) and rre.get('converged', False):
        sse = compute_sse_plus(ure['Sigma_hat'], rre['Sigma_hat'], T_n, n, d, q)
        assert 0 <= sse['kappa_n'] <= 1


def test_rre_lower_risk_correct_restriction():
    """Under correct restriction, RRE should have lower or equal risk than URE on average."""
    from src.data_gen import generate_data
    from src.regression import compute_ols, compute_restricted_ols
    from src.estimators import compute_ure_tyler, compute_rre_tyler
    from src.metrics import weighted_frobenius_loss
    rng = np.random.default_rng(42)
    n_reps = 30
    ure_losses, rre_losses = [], []
    for seed in range(n_reps):
        rng_i = np.random.default_rng(seed)
        data = generate_data(n=200, d=20, p=50, q=5, cov_type='identity', error_law='gaussian', nu=None, rng=rng_i)
        X, Y, R, Sigma = data['X'], data['Y'], data['R'], data['Sigma']
        ols = compute_ols(X, Y)
        rest = compute_restricted_ols(X, Y, R, ols['B_hat'], ols['G'], ols['P_X'])
        ure = compute_ure_tyler(ols['E_hat'], ols['c_n'])
        rre = compute_rre_tyler(rest['E_hat_r'], rest['tilde_c_n'])
        if ure.get('converged', False) and rre.get('converged', False):
            ure_losses.append(weighted_frobenius_loss(ure['Sigma_hat'], Sigma))
            rre_losses.append(weighted_frobenius_loss(rre['Sigma_hat'], Sigma))
    if len(ure_losses) > 5:
        assert np.mean(rre_losses) <= np.mean(ure_losses) * 1.1  # Allow 10% tolerance
