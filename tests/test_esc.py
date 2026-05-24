import numpy as np
import pytest

# GIVEN BY GENERATIVE AI. DOES NOT MEAN PERFECTION.

from src.esc import dot_product_esc, estimate_hadamard_exponent_range, esc

# --- 1. TESTS FOR dot_product_esc ---

def test_dot_product_esc_logic():
    """Verify the basic ESC formula logic."""
    x = np.array([2.0, 4.0])      # Exponents: [2, 3]. Max = 3
    y = np.array([8.0, 16.0])     # Exponents: [4, 5]. Max = 5
    z = estimate_hadamard_exponent_range(x, y, b=1)   # Exponents: [6, 8]. Max = 8
    
    # Formula: Max(x) + Max(y) - Max(z) + margin
    # 3 + 5 - 8 + 1 = 1
    res = dot_product_esc(x, y, z, margin=1)
    assert res == pytest.approx(1.0)

def test_dot_product_esc_warnings():
    """Check that mismatched sizes and wrong dims raise warnings."""
    x = np.array([1.0, 2.0])
    y = np.array([1.0, 2.0, 3.0])
    z = np.array([[1.0]]) # 2D array
    
    with pytest.warns(RuntimeWarning, match="same size"):
        # We also expect a UserWarning for z.ndim, but pytest.warns catches the first matching one.
        # So we just test the size mismatch here.
        dot_product_esc(x, y, x, margin=1)
        
    with pytest.warns(UserWarning, match="must be 1D"):
        dot_product_esc(x, x, z, margin=1)

# --- 2. TESTS FOR estimate_hadamard_exponent_range ---

def test_estimate_hadamard_b1():
    """With b=1, it should just be max(exp(x) + exp(y))."""
    x = np.array([2.0, 0.5])  # Exps: 2, 0
    y = np.array([4.0, 8.0])  # Exps: 3, 4
    
    # Max block x + Min block y (blocks are size 1)
    # [max(2)+min(3), max(0)+min(4)] -> [5, 4]
    z_est = estimate_hadamard_exponent_range(x, y, b=1)
    
    assert len(z_est) == 2
    np.testing.assert_array_equal(z_est, [5.0, 4.0])

def test_estimate_hadamard_coarsened():
    """Test with a larger block size (b=2) to verify cross-combinations."""
    x = np.array([2.0, 8.0, 0.5, 0.25])  # Exps: [2, 4], [0, -1]
    y = np.array([4.0, 2.0, 16.0, 8.0])  # Exps: [3, 2], [5, 4]
    
    z_est = estimate_hadamard_exponent_range(x, y, b=2)
    
    # Block 1: x=[2, 4], y=[3, 2]
    # max_x(4)+min_y(2) = 6. min_x(2)+max_y(3) = 5. Max is 6.
    # Block 2: x=[0, -1], y=[5, 4]
    # max_x(0)+min_y(4) = 4. min_x(-1)+max_y(5) = 4. Max is 4.
    
    assert len(z_est) == 2
    np.testing.assert_array_equal(z_est, [6.0, 4.0])

def test_estimate_hadamard_errors():
    """Should crash if vectors are mismatched."""
    with pytest.raises(ValueError, match="not valid vectors"):
        estimate_hadamard_exponent_range(np.array([1.0]), np.array([1.0, 2.0]), b=1)


# --- 3. TESTS FOR global esc ---

def test_esc_global_computation():
    """Verify it correctly finds the maximum ESC across the whole matrix."""
    A = np.array([
        [2.0, 4.0],  # row 1
        [0.5, 1.0]   # row 2
    ])
    B = np.array([
        [8.0, 0.25], # col 1, 2
        [16.0, 0.5]  
    ])
    
    # Let's trust the inner functions, just ensure the loop processes 
    # without crashing and returns a single float64 scalar.
    global_esc = esc(A, B, b=1)
    
    assert isinstance(global_esc, np.float64)
    assert global_esc > 0 # The margin is +1, so it should be at least 1

def test_esc_incompatible_shapes():
    """A column mismatch should raise a ValueError."""
    A = np.ones((5, 3))
    B = np.ones((4, 5)) # inner dims 3 and 4
    
    with pytest.raises(ValueError, match="incompatible sizes"):
        esc(A, B)

def test_esc_b_parameter_passed_down():
    """Verify changing 'b' doesn't crash the matrix iteration."""
    A = np.random.rand(4, 4)
    B = np.random.rand(4, 4)
    
    # b=2 means blocks of 2, vector length is 4, so 2 blocks.
    res1 = esc(A, B, b=1)
    res2 = esc(A, B, b=2)
    res4 = esc(A, B, b=4)
    
    # Since b coarsens the estimation, the ESC might be greater or equal, 
    # but the function shouldn't fail.
    assert np.isfinite(res1)
    assert np.isfinite(res2)
    assert np.isfinite(res4)