import numpy as np
import pytest
import warnings
from src.utils import random_matrix, random_phi_matrix, correct_matrix, random_mean_var_matrix

# --- 1. CORE PROPERTIES (REPRODUCIBILITY & METADATA) ---

def test_random_matrix_reproducibility():
    """Same seed must produce the exact same matrix."""
    args = (5, 5, 0, 10, np.float64)
    assert np.array_equal(random_matrix(*args, seed=1), random_matrix(*args, seed=1))
    assert not np.array_equal(random_matrix(*args, seed=1), random_matrix(*args, seed=2))

def test_random_phi_matrix_reproducibility():
    """Same seed must produce the exact same Ozaki matrix."""
    args = (5, 5, 1.0, np.float64)
    assert np.array_equal(random_phi_matrix(*args, seed=1), random_phi_matrix(*args, seed=1))

@pytest.mark.parametrize("m, n", [(0, 0), (1, 10), (10, 1)])
def test_dimensions(m, n):
    """Checks if m and n are strictly respected, including empty matrices."""
    res = random_matrix(m, n, 0, 1, np.float64, seed=42)
    assert res.shape == (m, n)
    assert res.size == m * n

def test_random_mean_var_matrix_reproducibility():
    """Same seed must produce the exact same normal distribution."""
    args = (5, 5, np.float64, 0.0, 1.0)
    assert np.array_equal(
        random_mean_var_matrix(*args, seed=1), 
        random_mean_var_matrix(*args, seed=1)
    )
    assert not np.array_equal(
        random_mean_var_matrix(*args, seed=1), 
        random_mean_var_matrix(*args, seed=2)
    )

@pytest.mark.parametrize("m, n", [(0, 0), (1, 10), (10, 1)])
def test_random_mean_var_dimensions(m, n):
    """Checks if m and n are strictly respected, including empty matrices."""
    res = random_mean_var_matrix(m, n, np.float64, 0.0, 1.0, seed=42)
    assert res.shape == (m, n)
    assert res.size == m * n

# --- 2. DEGENERATE & EXTREME CASES ---

def test_random_matrix_constant_output():
    """Force the matrix to contain only one specific value via range."""
    # Integers: [7, 8) -> only 7
    res_int = random_matrix(5, 5, 7, 8, np.int32, seed=42)
    assert np.all(res_int == 7)
    # Floats: [pi, pi] -> only pi
    res_flt = random_matrix(5, 5, np.pi, np.pi, np.float64, seed=42)
    assert np.all(res_flt == np.pi)

def test_random_phi_matrix_int_truncation():
    """Casting Ozaki matrix to int. At low phi, everything should be 0."""
    # U is in (-0.5, 0.5), so int(U) is always 0.
    res = random_phi_matrix(10, 10, 0.0, np.int32, seed=42)
    assert np.all(res == 0)

def test_random_phi_matrix_overflow_raises_error():
    """
    Ensures that if phi is too high for float16, 
    the function raises an OverflowError instead of returning 'inf'.
    """
    with pytest.raises(OverflowError) as excinfo:
        # phi=50 is guaranteed to produce values > 65504
        random_phi_matrix(10, 10, 50.0, np.float16, seed=42)
    
    assert "exceeds limits" in str(excinfo.value)

def test_random_matrix_out_of_bounds_raises():
    """
    If I ask for numbers up to 1000 in an int8 matrix (max 127),
    it should blow up immediately.
    """
    with pytest.raises(ValueError) as excinfo:
        random_matrix(5, 5, 0, 1000, np.int8)
    assert "outside the representable range" in str(excinfo.value)

def test_random_matrix_float_precision_safety():
    """
    Ensures float16 doesn't get values it can't handle.
    """
    with pytest.raises(ValueError):
        # 1e6 is way above float16's ~65504
        random_matrix(2, 2, 0, 1e6, np.float16)

def test_random_matrix_negative_unsigned():
    """
    If I ask for -10 in a uint8 (unsigned), it should raise an error
    instead of wrapping around to 246.
    """
    with pytest.raises(ValueError):
        random_matrix(5, 5, -10, 10, np.uint8)

def test_random_mean_var_constant_output():
    """If variance is 0, the matrix must contain exactly the mean value."""
    mean_val = 42.0
    res = random_mean_var_matrix(5, 5, np.float64, mean=mean_val, var=0.0, seed=42)
    assert np.all(res == mean_val)

def test_random_mean_var_overflow_raises_error():
    """
    Ensures that if the mean/variance combination is too high for float16, 
    the function raises an OverflowError.
    """
    with pytest.raises(OverflowError) as excinfo:
        # Mean of 1e6 is way above float16's limit (~65504)
        random_mean_var_matrix(10, 10, np.float16, mean=1e6, var=1.0, seed=42)
    assert "exceeds limits" in str(excinfo.value)

def test_random_mean_var_out_of_bounds_raises():
    """
    If the distribution parameters significantly exceed the representable 
    range of an integer type (e.g., int8 max 127).
    """
    with pytest.raises(ValueError) as excinfo:
        # A mean of 200 is impossible to represent in int8
        random_mean_var_matrix(5, 5, np.int8, mean=200.0, var=1.0, seed=42)
    assert "outside the representable range" in str(excinfo.value)

def test_random_mean_var_negative_unsigned():
    """
    Ensures that requesting a distribution that produces negative values 
    for a unsigned type (uint) raises an error.
    """
    with pytest.raises(ValueError):
        # Normal distribution with mean -10 will definitely hit negative values
        random_mean_var_matrix(5, 5, np.uint8, mean=-10.0, var=1.0, seed=42)

# --- 3. STATISTICAL VALIDATION ---

def test_random_matrix_distribution():
    """Standard Uniform distribution check: Mean (a+b)/2, Var (b-a)^2/12."""
    a, b = -10.0, 10.0
    res = random_matrix(1000, 1000, a, b, np.float64, seed=42)
    assert np.mean(res) == pytest.approx(0.0, abs=1e-2)
    assert np.var(res) == pytest.approx(400/12, rel=1e-2)

def test_random_phi_distribution_stable():
    """
    Ozaki distribution check.
    Theory: Var = (1/12) * exp(2 * phi^2)
    """
    phi = 1.0 # phi=1 is statistically stable for 1M samples
    m, n = 1000, 1000
    res = random_phi_matrix(m, n, phi, np.float64, seed=42)
    
    expected_var = (1/12) * np.exp(2)
    assert np.mean(res) == pytest.approx(0.0, abs=1e-2)
    assert np.var(res) == pytest.approx(expected_var, rel=0.05)

def test_random_mean_var_distribution():
    """
    Statistical validation for Normal Distribution.
    Expected Mean: mean, Expected Variance: var.
    """
    target_mean = 5.0
    target_var = 9.0
    m, n = 1000, 1000 # 1M samples for statistical stability
    
    res = random_mean_var_matrix(m, n, np.float64, mean=target_mean, var=target_var, seed=42)
    
    # Absolute tolerance for mean, relative for variance
    assert np.mean(res) == pytest.approx(target_mean, abs=1e-2)
    assert np.var(res) == pytest.approx(target_var, rel=1e-2)

# --- 4. DTYPE ROBUSTNESS ---

@pytest.mark.parametrize("dtype", [np.uint8, np.complex128, np.float32])
def test_dtype_consistency(dtype):
    """Ensures the function returns the exact requested numpy dtype."""
    res = random_matrix(2, 2, 0, 5, dtype, seed=42)
    assert res.dtype == np.dtype(dtype)

@pytest.mark.parametrize("dtype", [np.uint8, np.float64, np.float32, np.int8]) # DOES NOT WORK WITH COMPLEX128
def test_random_mean_var_dtype_conversion(dtype):
    """Ensures the function returns the exact requested numpy dtype, without any issue"""
    m, n = 100, 100
    res = random_mean_var_matrix(m, n, dtype, mean=10.0, var=1.0)
    
    assert res.dtype == dtype
    assert not np.isnan(res).any()


# ---- Undefined behaviors ---
def test_undefined_behaviors_catching():
    A = np.array(["a", None])
    assert not correct_matrix(A)
    B = np.array([1, 2, 3, 4])
    assert correct_matrix(B)
    C = np.array([1, 2, None, 3])
    assert not correct_matrix(C)
    D = np.array([np.inf, 2, 3, 4])
    assert not correct_matrix(D)
    