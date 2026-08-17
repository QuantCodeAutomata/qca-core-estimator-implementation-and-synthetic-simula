import numpy as np
import pytest
import sys
sys.path.insert(0, '/workspace/project')
from src.data_gen import (
    generate_design_matrix,
    generate_restriction_matrix,
    generate_banded_covariance,
    compute_null_space_basis,
    generate_elliptical_errors,
    generate_sparse_covariance,
)


def test_design_matrix_shape_and_rank():
    """Test design matrix has correct shape and full column rank."""
    rng = np.random.default_rng(42)
    X = generate_design_matrix(n=100, d=20, rng=rng)
    assert X.shape == (100, 20)
    assert np.linalg.matrix_rank(X) == 20


def test_restriction_matrix_orthonormality():
    """Test restriction matrix rows are orthonormal and full row rank."""
    rng = np.random.default_rng(42)
    R = generate_restriction_matrix(q=5, d=20, rng=rng)
    assert R.shape == (5, 20)
    assert np.linalg.matrix_rank(R) == 5
    RRT = R @ R.T
    assert np.allclose(RRT, np.eye(5), atol=1e-10)


def test_banded_covariance_structure():
    """Test banded covariance has correct entries and is SPD."""
    Sigma = generate_banded_covariance(p=50)
    assert Sigma.shape == (50, 50)
    # Check diagonal is 1.0
    assert np.allclose(np.diag(Sigma), 1.0)
    # Check bandwidth
    assert Sigma[0, 11] == 0.0  # beyond bandwidth
    assert Sigma[0, 10] > 0.0  # within bandwidth
    # Check SPD
    evals = np.linalg.eigvalsh(Sigma)
    assert np.all(evals > 0)


def test_null_space_construction_rb_zero():
    """Test that B=N_R@C satisfies RB=0 exactly."""
    rng = np.random.default_rng(42)
    R = generate_restriction_matrix(q=5, d=20, rng=rng)
    N_R = compute_null_space_basis(R)
    C = rng.standard_normal((N_R.shape[1], 30))
    B = N_R @ C
    RB = R @ B
    assert np.allclose(RB, 0, atol=1e-10), f'||RB||_F = {np.linalg.norm(RB)}'


def test_elliptical_errors_gaussian():
    """Test Gaussian elliptical errors have correct covariance structure."""
    rng = np.random.default_rng(42)
    Sigma = np.eye(10)
    E = generate_elliptical_errors(n=10000, Sigma=Sigma, error_law='gaussian', nu=None, rng=rng)
    assert E.shape == (10000, 10)
    # Sample covariance should be close to Sigma
    S = E.T @ E / E.shape[0]
    assert np.allclose(S, Sigma, atol=0.1)


def test_elliptical_errors_heavy_tails():
    """Test that t-distributed errors have heavier tails than Gaussian (higher kurtosis)."""
    rng = np.random.default_rng(42)
    Sigma = np.eye(5)
    E_gauss = generate_elliptical_errors(n=5000, Sigma=Sigma, error_law='gaussian', nu=None, rng=rng)
    E_t6 = generate_elliptical_errors(n=5000, Sigma=Sigma, error_law='t6', nu=6, rng=rng)
    # t6 should have heavier tails: higher variance of row norms
    var_gauss = np.var(np.linalg.norm(E_gauss, axis=1))
    var_t6 = np.var(np.linalg.norm(E_t6, axis=1))
    assert var_t6 > var_gauss


def test_sparse_covariance_spd():
    """Test sparse covariance is SPD and has correct trace."""
    rng = np.random.default_rng(42)
    p = 30
    Sigma = generate_sparse_covariance(p=p, rng=rng)
    assert Sigma.shape == (p, p)
    evals = np.linalg.eigvalsh(Sigma)
    assert np.all(evals > 0), 'Sparse covariance not SPD'
    assert np.isclose(np.trace(Sigma), p, rtol=0.01)
