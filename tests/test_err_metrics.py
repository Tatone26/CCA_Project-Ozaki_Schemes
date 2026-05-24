import numpy as np
import pytest
from src.err_metrics import backward_error, forward_error, forward_bound, average_relative_error

@pytest.fixture
def sample_matrices():
    # Simple matrices where A @ B is easy to track
    A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    B = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
    return A, B

def test_zero_error_when_perfect(sample_matrices):
    """If AB is exactly the product, all error metrics should be 0."""
    A, B = sample_matrices
    # Compute 'computed' product perfectly
    AB = A @ B 
    
    assert forward_error(A, B, AB) == pytest.approx(0.0, abs=1e-15)
    assert backward_error(A, B, AB) == pytest.approx(0.0, abs=1e-15)
    assert average_relative_error(A, B, AB) == pytest.approx(0.0, abs=1e-15)

def test_forward_error_scaling(sample_matrices):
    """Test that forward error responds linearly to a manual perturbation."""
    A, B = sample_matrices
    C = A @ B
    # Introduce a 1% error
    AB_perturbed = C * 1.01
    
    err = forward_error(A, B, AB_perturbed)
    # The relative error should be exactly 0.01
    assert err == pytest.approx(0.01, rel=1e-5)

def test_average_relative_error_logic():
    """Test average relative error with a matrix containing different scales."""
    A = np.array([[1.0, 0.0], [0.0, 1.0]])
    B = np.array([[1.0, 0.0], [0.0, 1e-10]])
    C = A @ B # [[1.0, 0.0], [0.0, 1e-10]]
    
    # Introduce a flat error of 1e-5 to all elements
    # For the first element, relative error is 1e-5 / 1.0 = 1e-5
    # For the last element, relative error is 1e-5 / 1e-10 = 100,000 (10^5)
    AB_perturbed = C + 1e-5
    
    avg_rel = average_relative_error(A, B, AB_perturbed)
    
    # Average relative error will be dominated by the 10^5 error on the small entry
    assert avg_rel > 1000 

def test_forward_bound_value(sample_matrices):
    """Check that the theoretical bound is a positive finite number."""
    A, B = sample_matrices
    bound = forward_bound(A, B)
    assert bound > 0
    assert np.isfinite(bound)

def test_metrics_with_phi_matrices():
    """
    Test metrics using a phi matrix to ensure dynamic range 
    doesn't crash the calculation.
    """
    from src.utils import random_phi_matrix
    A = random_phi_matrix(4, 4, 1.0, np.float64, seed=1)
    B = random_phi_matrix(4, 4, 1.0, np.float64, seed=2)
    AB = A @ B # Standard float64 multiplication
    
    f_err = forward_error(A, B, AB)
    b_err = backward_error(A, B, AB)
    
    assert f_err >= 0
    assert b_err >= 0