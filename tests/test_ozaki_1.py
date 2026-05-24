import numpy as np
import numpy.typing as npt
import pytest
from src.ozaki_1 import (
    split_matrix_dtype, 
    split_matrix_dtype_trunc, 
    sum_with_scaling, 
    get_scaling_factors, 
    multiply_lists_dtype,
    full_ozaki_1
)
from src.baseline import get_exact_product
from src.err_metrics import forward_bound, forward_error, backward_error

# --- 1. RECONSTRUCTION TESTS ---

# forcing only legal parameters, as there is no checks in the functions
@pytest.mark.parametrize("output_type, d", [
    (np.int8, 4),   # Legal: 4 bits < 8 bits
    (np.int8, 7),   # Legal: 7 bits fits in signed int8
    (np.int16, 8),  # Legal
    (np.int16, 15), # Legal
    (np.int32, 16), # Legal
    (np.int32, 31), # Legal
])
def test_splitting_reconstruction_dtype(d, output_type):
    """Verifies that A = sum(S_i * scale_i) holds for different integer types."""
    # Create random matrix
    scale = 100.0
    A = np.random.uniform(-scale, scale, (5, 5))
    
    # Calculate optimal sigma (power of 2 >= max(abs(A)))
    sigma = 2.0**np.frexp(np.max(np.abs(A)))[1]

    # Split
    splits = split_matrix_dtype(A, sigma, d, output_type)
    
    # Verify dtypes
    for s in splits:
        assert s.dtype == np.dtype(output_type)

    # Reconstruct
    scalings = [sigma * 2.0**(-d * i) for i in range(len(splits))]
    reconstructed = sum_with_scaling(splits, scalings)
    
    # Check reconstruction (allowing for float64 precision limits)
    np.testing.assert_allclose(reconstructed, A, atol=1e-12, err_msg=f"Reconstruction failed for {output_type} with d={d}")

# --- 2. TRUNCATION TESTS ---

def test_truncation_logic():
    """Checks that 's' parameter effectively limits the number of splits."""
    A = np.random.uniform(100, 200, (4, 4))
    sigma = 256.0
    d = 4
    s_limit = 3
    
    splits, scalings = split_matrix_dtype_trunc(A, np.float64(sigma), d, s_limit, np.int16)
    
    assert len(splits) <= s_limit
    assert len(scalings) == len(splits)
    
    # If s is very small, we expect the reconstruction to be an approximation
    reconstructed = sum_with_scaling(splits, scalings)
    # The error should be roughly related to the last sigma dropped
    error = np.max(np.abs(A - reconstructed))
    assert error > 0, "Truncation should have lost some information"

# --- 3. MULTIPLICATION & SCALING MATH ---

def test_scaling_factors_consistency():
    """Verifies get_scaling_factors matches the mathematical expectation."""
    sigma_a, sigma_b = 16.0, 32.0
    d = 8
    lena, lenb = 2, 2
    
    factors = get_scaling_factors(lena, lenb, np.float64(sigma_a), np.float64(sigma_b), d)
    
    # Expected: 
    # [sa*sb*2^0, sa*sb*2^-d, sa*sb*2^-d, sa*sb*2^-2d]
    expected = [
        sigma_a * sigma_b,           # i=0, j=0
        sigma_a * sigma_b * 2**-d,    # i=0, j=1
        sigma_a * sigma_b * 2**-d,    # i=1, j=0
        sigma_a * sigma_b * 2**(-2*d) # i=1, j=1
    ]
    
    np.testing.assert_allclose(factors, expected)

def test_complete_chain_multiplication():
    """Tests the full process: Split -> Multiply Lists -> Scaled Sum."""
    A = np.array([[1.5, 2.3], [0.1, 4.4]], dtype=np.float64)
    B = np.array([[0.5, 1.2], [3.3, 0.2]], dtype=np.float64)
    
    d = 16
    split_type = np.int32
    store_type = np.int64 # Larger type for products to avoid overflow
    
    sigma_a = 8.0
    sigma_b = 8.0
    
    # 1. Split
    LA = split_matrix_dtype(A, np.float64(sigma_a), d, split_type)
    LB = split_matrix_dtype(B, np.float64(sigma_b), d, split_type)
    
    # 2. Multiply
    products = multiply_lists_dtype(LA, LB, store_type)
    
    # 3. Scaling factors
    factors = get_scaling_factors(len(LA), len(LB), np.float64(sigma_a), np.float64(sigma_b), d)
    
    # 4. Sum
    result = sum_with_scaling(products, factors)
    
    # 5. Baseline check
    expected = A @ B
    np.testing.assert_allclose(result, expected, atol=1e-10)

# --- 4. EDGE CASES ---

def test_null_matrix_splitting_dtype():
    """Ensures a null matrix returns a single zero matrix of the correct dtype."""
    A = np.zeros((3, 3))
    out_type = np.int8
    splits = split_matrix_dtype(A, np.float64(1.0), 8, out_type)
    
    assert len(splits) == 1
    assert splits[0].dtype == np.dtype(out_type)
    np.testing.assert_equal(splits[0], 0)

def test_sum_with_scaling_mismatch():
    """Checks that providing mismatched list lengths raises ValueError."""
    L = [np.ones((2, 2))]
    S = [np.float64(1.0), np.float64(2.0)] # Mismatch
    with pytest.raises(ValueError, match="Not same number of matrices"):
        sum_with_scaling(L, S)

# --- 5. Stability tests ---

def check_back_and_forth_error(A: npt.NDArray, B: npt.NDArray, split_type: npt.DTypeLike, d:int, s:int, store_type: npt.DTypeLike):
    '''A quick way to check for stability.'''
    our_C = full_ozaki_1(A, B, split_type, d, store_type)
    n = A.shape[0]
    u = 2.0**(-53)
    
    bck_error = backward_error(A, B, our_C)
    fw_error = forward_error(A, B, our_C)
    bn_forward = forward_bound(A, B)

    assert bck_error <= n * u, f"Backward error trop élevée : {bck_error} > {n*u}"
    assert fw_error <= bn_forward, (
        f"Forward error dépasse la borne de stabilité : {fw_error} > {bn_forward}. "
    )
    return bck_error