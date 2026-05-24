'''
This file contains the basic functionalities of the Ozaki-2 scheme (Modular/CRT), 
where the input is scaled to integers and matrix products are evaluated in modular fields.

It also contains the full schemes algorithms and some examples, which you can see 
by running this file directly.

For simplicity, our INPUT_TYPE is always float64. 

This file can be run to see examples.
'''

import math
import numpy as np
import ml_dtypes
import numpy.typing as npt
try:
    import src.utils as utils
except ModuleNotFoundError:
    import utils as utils


def get_diagonal_scales(A: npt.NDArray[np.float64], B: npt.NDArray[np.float64], target_bits: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    '''
    Computes the diagonal scaling factors (powers of 2) for rows of A and columns of B
    to extract 'target_bits' of information into the integer domain.
    '''
    # take exponents of the matrices
    _, exp_A = np.frexp(np.abs(A))
    _, exp_B = np.frexp(np.abs(B))

    # find maximum exponents, line-by-line and columns-by-columns
    # this gives vectors
    max_A : npt.NDArray[np.float64] = np.max(exp_A, axis=1, keepdims=True)
    max_B : npt.NDArray[np.float64] = np.max(exp_B, axis=0, keepdims=True)
    
    # deduce scaling factors
    scale_A = 2.0**(target_bits - max_A)
    scale_B = 2.0**(target_bits - max_B)
    
    return scale_A, scale_B

def scale_to_integer(A: npt.NDArray[np.float64], scale: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    '''Applies the previously computed scaling factors. 
    Returns a float64 that represent a integer. 
    The mantissa bits remain unchanged.'''
    return np.rint(A * scale)

def apply_modulo(A_scaled: npt.NDArray[np.float64],  m : int, compute_type: npt.DTypeLike) -> npt.NDArray:
    '''Compute (A_scaled mod m), and stores it in compute_type. 
    Since A_scaled and is supposed to already be an integer, we shouldn't lose any information.'''
    A_m = np.mod(np.rint(A_scaled), m).astype(compute_type)
    return A_m

def modular_multiply(A_m: npt.NDArray, B_m: npt.NDArray, m:int, store_type: npt.DTypeLike) -> npt.NDArray:
    '''Taking the already reduced via modulo A and B array (they are in compute_type),
    this computes (A * B) mod m in store_type.'''
    # we cast A_m and B_m before doing the calculation to be safe. 
    # a true tensor core would do the computation directly.
    C_m = np.matmul(A_m.astype(store_type), B_m.astype(store_type), dtype=store_type)
    return np.mod(C_m, m)

def crt_reconstruction(results: list[npt.NDArray], moduli: list[int]) -> npt.NDArray:
    '''From a list of C_m given in store_type, 
    computes the CRT reconstruction in multi-precision.
    Uses python <object> type as our multi-precision type.
    Returns an <object> array.'''

    if len(results) != len(moduli):
        raise ValueError("Number of results must match number of moduli.")

    M : int = math.prod(moduli)
    C_int = np.zeros_like(results[0], dtype=object)

    for C_m, m in zip(results, moduli):
        M_i : int = M // m
        y_i : int = pow(M_i, -1, m) 
        term : int = M_i * y_i
        
        C_m_pos = np.mod(C_m.astype(object), m)
        
        # Accumulate the term safely using pure big-ints
        C_int += (C_m_pos * term)

    # Apply global modulo
    C_int = np.mod(C_int, M)

    # Symmetric modulo to handle negative values safely
    half_M = M // 2
    mask = C_int > half_M
    C_int[mask] -= M

    return C_int

def descale_matrix(C_int: npt.NDArray[np.object_], scale_A: npt.NDArray[np.float64], scale_B: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    '''Recreates the final matrix by rescaling it and putting it back into float64.'''
    # calculate inverse of scaling vectors
    inv_scale_A : npt.NDArray[np.float64] = 1.0 / scale_A  # shape: (M, 1)
    inv_scale_B : npt.NDArray[np.float64] = 1.0 / scale_B  # shape: (1, N)

    C_scaled : npt.NDArray[np.object_] = inv_scale_A * C_int * inv_scale_B # shape (M, M)

    # cast back from object to float64
    C_float : npt.NDArray[np.float64] = C_scaled.astype(float).astype(np.float64)
    return C_float

def get_moduli_universal(dtype, count, q=100):
    '''Tries to get a good moduli list. It may not be perfect, but it is pretty good. 
    Refer to the paper for the absolute perfect list.'''
    dt = np.dtype(dtype)
    
    if np.issubdtype(dt, np.integer):
        bits = dt.itemsize * 8
        m_max = 2 ** (bits - 1)
    else:
        info = utils.get_dtype_info(dtype)
        u_inv = 2 ** info['nmant']
            
        m_max = math.isqrt((4 * u_inv) // q)

    moduli = []
    c = m_max
    
    while len(moduli) < count and c >= 2:
        # Check if 'c' shares any common factors with the moduli we already found
        # (very slow)
        if all(math.gcd(c, m) == 1 for m in moduli):
            moduli.append(c)
        c -= 1
        
    return moduli

def full_ozaki_2(
    A: npt.NDArray[np.float64], 
    B: npt.NDArray[np.float64], 
    target_bits: int, 
    nb_moduli: int|None,
    compute_type: npt.DTypeLike, 
    store_type: npt.DTypeLike,
) -> npt.NDArray[np.float64]:
    '''Does A * B using the ozaki 2 scheme, following the parameters. '''
    scale_A, scale_B = get_diagonal_scales(A, B, target_bits) # float64
    A_int : npt.NDArray[np.float64] = scale_to_integer(A, scale_A)
    B_int : npt.NDArray[np.float64] = scale_to_integer(B, scale_B)

    q = A.shape[1] # inner product size
    assert(q == B.shape[0])

    # get number of moduli necessary for perfect product 
    # (i think)
    if nb_moduli is None:
        dt = np.dtype(compute_type)
        if np.issubdtype(dt, np.integer):
            effective_bits = dt.itemsize * 8 - 1
        else:
            info = utils.get_dtype_info(compute_type)
            u_inv = 2 ** info['nmant']
            
            m_max = math.isqrt((4 * u_inv) // q)
            effective_bits = math.floor(math.log2(m_max))
            
        nb_moduli = int(np.ceil((np.log2(q) + 2 * target_bits + 1) / (effective_bits - 0.5)))

    moduli = get_moduli_universal(compute_type, nb_moduli, q)

    # compute modular matrices products
    modular_results = []
    for m in moduli:
        A_m : npt.NDArray = apply_modulo(A_int, m, compute_type) # compute_type
        B_m : npt.NDArray = apply_modulo(B_int, m, compute_type) # compute_type
        C_m : npt.NDArray = modular_multiply(A_m, B_m, m, store_type) # store_type
        modular_results.append(C_m)

    # CRT reconstruction
    C_large_int : npt.NDArray[np.object_]= crt_reconstruction(modular_results, moduli)

    # get back float64
    C_float64 : npt.NDArray[np.float64] = descale_matrix(C_large_int, scale_A, scale_B)
    return C_float64

def get_accurate_scales(
    a: npt.NDArray[np.float64], 
    b: npt.NDArray[np.float64], 
    moduli: list[int],
    compute_type: npt.DTypeLike
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    '''
    Implements true row-wise (A) and column-wise (B) scaling factors
    using a float32 scout GEMM to guarantee 2 * c_max < M globally.
    '''
    compute_type = np.float32

    _, q_dim = a.shape
    M = math.prod(moduli)
    
    # row-wise max for A cColumn-wise max for B
    max_a = np.max(np.abs(a), axis=1, keepdims=True)  # Shape: (p, 1)
    max_b = np.max(np.abs(b), axis=0, keepdims=True)  # Shape: (1, r)
    
    # Avoid zero-division rules
    max_a[max_a == 0] = 1.0
    max_b[max_b == 0] = 1.0
    
    # Find base floating point exponents for every unique line
    ufp_a = 2.0 ** np.floor(np.log2(max_a))
    ufp_b = 2.0 ** np.floor(np.log2(max_b))
    
    is_fp8 = "float8" in str(np.dtype(compute_type))
    base_bits = 4 if is_fp8 else 7
    # map every row/column up to the maximum precision boundary
    mu_prime = (2.0 ** base_bits) / ufp_a  # Vector shape: (p, 1)
    nu_prime = (2.0 ** base_bits) / ufp_b  # Vector shape: (1, r)
    
    # create proxy matrices with real signs to accurately capture cancellation
    a_bar = np.rint(a * mu_prime).astype(np.float32)
    b_bar = np.rint(b * nu_prime).astype(np.float32)
    
    # scout GEMM: Evaluate cross-element accumulation & sign cancellation
    c_bar_prime = np.matmul(a_bar, b_bar)
        
    # extract the absolute maximum value produced ANYWHERE in the output matrix
    c_proxy = np.max(np.abs(c_bar_prime))
    if c_proxy == 0:
        c_proxy = 1.0

    # apply float32 machine epsilon safety margin
    epsilon = q_dim * (2.0 ** -24)
    c_max = c_proxy * (1.0 + epsilon)

    # calculate the global remaining bit budget available, using arbitrary precision computation
    # and log2
    log2_M = M.bit_length() - 1
    total_bit_budget = math.floor(log2_M - 1 - math.log2(c_max))
    
    # split the remaining bit safety buffer evenly 
    k_A_adjust = np.floor(total_bit_budget / 2.0)
    k_B_adjust = total_bit_budget - k_A_adjust
    
    # compute final distinct scaling vectors
    d_diag_values = mu_prime * (2.0 ** k_A_adjust) # Shape: (p, 1)
    e_diag_values = nu_prime * (2.0 ** k_B_adjust) # Shape: (1, r)
    
    return d_diag_values, e_diag_values

def full_ozaki_2_acc(
    a: npt.NDArray[np.float64], 
    b: npt.NDArray[np.float64], 
    target_bits: int, 
    nb_moduli: int|None,
    compute_type: npt.DTypeLike, 
    store_type: npt.DTypeLike,
) -> npt.NDArray[np.float64]:
    '''
    Does a * b using the accurate ozaki 2 scheme with a scout gemm, allowing for more precision
    As in the paper, the scouting is done via int8 x int8 -> float32 (after some maths)
    '''
    q = a.shape[1]
    assert(q == b.shape[0])

    if nb_moduli is None:
        dt = np.dtype(compute_type)
        if np.issubdtype(dt, np.integer):
            effective_bits = dt.itemsize * 8 - 1
        else:
            info = utils.get_dtype_info(compute_type)
            u_inv = 2 ** info['nmant']
            
            m_max = math.isqrt((4 * u_inv) // q)
            effective_bits = math.floor(math.log2(m_max))
            
        nb_moduli = int(np.ceil((np.log2(q) + 2 * target_bits + 1) / (effective_bits - 0.5)))

    moduli = get_moduli_universal(compute_type, nb_moduli, q)

    scale_a, scale_b = get_accurate_scales(a, b, moduli, compute_type)
    
    a_int : npt.NDArray[np.float64] = scale_to_integer(a, scale_a)
    b_int : npt.NDArray[np.float64] = scale_to_integer(b, scale_b)

    modular_results = []
    for m in moduli:
        a_m : npt.NDArray = apply_modulo(a_int, m, compute_type)
        b_m : npt.NDArray = apply_modulo(b_int, m, compute_type)
        c_m : npt.NDArray = modular_multiply(a_m, b_m, m, store_type)
        modular_results.append(c_m)

    c_large_int : npt.NDArray[np.object_] = crt_reconstruction(modular_results, moduli)
    
    c_float64 : npt.NDArray[np.float64] = descale_matrix(c_large_int, scale_a, scale_b)
    return c_float64


def _crt_example():
    print("CRT Reconstruction example :")
    moduli = [1009, 1013, 1019] # Coprime
    
    True_C = np.array([[-1500000, 2000000], [0, -500000]], dtype=object)
    print("Original large integer matrix:")
    print(True_C)
    
    results = [np.mod(True_C, m) for m in moduli]
    
    Reconstructed_C = crt_reconstruction(results, moduli)
    print("\nReconstituted matrix via CRT:")
    print(Reconstructed_C)

def _multiplication_example():
    print("\nMultiplication example :")
    scale = 10.0**2
    A: npt.NDArray[np.float64] = np.random.random((3, 3)) * scale
    B: npt.NDArray[np.float64] = np.random.random((3, 3)) * scale

    try:
        import baseline
        Res_exact = baseline.get_exact_product(A, B, 256)
        utils.print_matrix(Res_exact, "Exact multiplication")
    except ImportError:
        Res_exact = A @ B
        utils.print_matrix(Res_exact, "Basic float64 multiplication (Fallback Exact)")

    Res_base = A @ B
    utils.print_matrix(Res_base, "Basic multiplication")
    print("Absolute error : ", np.sum(np.abs((Res_exact - Res_base))))

    target_bits = 22
    nb_moduli = 4
    compute_type: npt.DTypeLike = np.int32
    mult_store_type: npt.DTypeLike = np.int64

    Res_ozaki2 = full_ozaki_2(A, B, target_bits, nb_moduli, compute_type, mult_store_type)

    utils.print_matrix(Res_ozaki2, "Multiplication via Ozaki-2 (CRT)")
    print("Absolute error : ", np.sum(np.abs((Res_exact - Res_ozaki2))))

def _proof_of_work():
    print("\nOther multiplication example, using int8, int32 and full auto.")
    scale = 10.0**3
    A: npt.NDArray[np.float64] = np.random.random((3, 3)) * scale
    B: npt.NDArray[np.float64] = np.random.random((3, 3)) * scale

    try:
        import baseline
        Res_exact = baseline.get_exact_product(A, B, 256)
        utils.print_matrix(Res_exact, "Exact multiplication")
    except ImportError:
        Res_exact = A @ B
        utils.print_matrix(Res_exact, "Basic float64 multiplication (Fallback Exact)")

    Res_base = A @ B
    utils.print_matrix(Res_base, "Basic multiplication")
    print("Absolute error : ", np.sum(np.abs((Res_exact - Res_base))))

    target_bits = 53 # same as fp64 !
    compute_type: npt.DTypeLike = np.int8
    store_type: npt.DTypeLike = np.int32

    print("Ozaki 2 multiplications, using int8 x int8 -> int32")

    Res_ozaki2 = full_ozaki_2(A, B, target_bits, None, compute_type, store_type)

    utils.print_matrix(Res_ozaki2, "Multiplication via Ozaki-2 (CRT)")
    print("Absolute error : ", np.sum(np.abs((Res_exact - Res_ozaki2))))

    Res_ozaki2_acc = full_ozaki_2_acc(A, B, target_bits, None, compute_type, store_type)

    utils.print_matrix(Res_ozaki2_acc, "Multiplication via Ozaki-2 accurate (CRT) - +1 fp16 GEMM")
    print("Absolute error : ", np.sum(np.abs((Res_exact - Res_ozaki2_acc))))


if __name__ == "__main__":
    _crt_example()
    _multiplication_example()
    _proof_of_work()