'''
This file contains the specific functions needed for the last version, which is an "hybrid" method that 
manages to use FP8 as input. This is sadly NOT the case here. Since it does NOT work well, there was
not a lot of effort put in explaining the code. Sorry. 

A lot of functions come from the ozaki_2.py file. This one is just a modification of some steps.

You can also run this file to get some examples.
'''
try:
    import src.ozaki_2 as o2
except ModuleNotFoundError:
    import ozaki_2 as o2

try:
    import src.utils as utils
except ModuleNotFoundError:
    import utils

import numpy as np
import numpy.typing as npt
import math
import ml_dtypes

def apply_sym_modulo(A_scaled: npt.NDArray[np.float64], m: int, compute_type: npt.DTypeLike) -> npt.NDArray:
    """
    Computes (A_scaled mod m) using signed 64-bit integers to avoid 
    float64 remainder grid quantization errors on massive exponents.
    """
    # Cast safely to int64 to isolate individual integer bits precisely
    A_int64 = np.rint(A_scaled).astype(np.int64)
    shift = m // 2
    
    # Evaluate the symmetric modulo via true integer equations
    A_m = (A_int64 + shift) % m - shift
    return A_m.astype(compute_type)

def get_hybrid_moduli(count: int, max_square: int = 1089, max_regular: int = 511) -> list[int]:
    moduli = np.zeros(count, dtype=int)
    c = 0

    HARD_LIMIT = 389 # we never need more than 30 moduli, so we don't look at smaller number than that...
    # this allowed use to get the same list of moduli than described in the paper

    def is_coprime_with_all(n, current_moduli):
        return not np.any(np.gcd(n, current_moduli[current_moduli != 0]) != 1)
        
    start_root = math.isqrt(max_square)
    for i in range(start_root, 1, -1):
        candidate = i**2
        if candidate < HARD_LIMIT:
            break

        if is_coprime_with_all(candidate, moduli):
            moduli[c] = candidate
            c += 1
            if c == count:
                break
                
    candidate = max_regular
    while c < count and candidate >= HARD_LIMIT:
        if is_coprime_with_all(candidate, moduli):
            moduli[c] = candidate
            c += 1
        candidate -= 1
        
    moduli_list = moduli.tolist()
    moduli_list.sort(reverse=True)
    return moduli_list[:c]

def hybrid_modular_multiply(a_m: npt.NDArray, b_m: npt.NDArray, m: int, compute_type: npt.DTypeLike, store_type: npt.DTypeLike) -> npt.NDArray:
    is_square = math.isqrt(m)**2 == m
    
    if is_square:
        s = math.isqrt(m)

        a_1_raw = np.round(a_m / s)
        b_1_raw = np.round(b_m / s)
        
        a_2_raw = a_m - s * a_1_raw
        b_2_raw = b_m - s * b_1_raw
        
        a_1 = a_1_raw.astype(compute_type)
        a_2 = a_2_raw.astype(compute_type)
        b_1 = b_1_raw.astype(compute_type)
        b_2 = b_2_raw.astype(compute_type)
        
        c_1 = np.matmul(a_1.astype(store_type), b_2.astype(store_type), dtype=store_type)
        c_2 = np.matmul(a_2.astype(store_type), b_1.astype(store_type), dtype=store_type)
        c_3 = np.matmul(a_2.astype(store_type), b_2.astype(store_type), dtype=store_type)
        
        c_m_acc = s * (c_1.astype(np.float64) + c_2.astype(np.float64)) + c_3.astype(np.float64)
    else:
        s = 16

        a_m_int = np.rint(a_m).astype(np.int64)
        b_m_int = np.rint(b_m).astype(np.int64)
        
        a_1_raw = np.sign(a_m_int) * ((np.abs(a_m_int) + (s - 1)) // s)
        b_1_raw = np.sign(b_m_int) * ((np.abs(b_m_int) + (s - 1)) // s)
        
        a_2_raw = a_m_int - s * a_1_raw
        b_2_raw = b_m_int - s * b_1_raw
        
        # Cast base terms to FP8
        a_1 = a_1_raw.astype(compute_type)
        a_2 = a_2_raw.astype(compute_type)
        b_1 = b_1_raw.astype(compute_type)
        b_2 = b_2_raw.astype(compute_type)
        
        a_3 = (a_1_raw + a_2_raw).astype(compute_type)
        b_3 = (b_1_raw + b_2_raw).astype(compute_type)
        
        c_1 = np.matmul(a_1.astype(store_type), b_1.astype(store_type), dtype=store_type)
        c_2 = np.matmul(a_2.astype(store_type), b_2.astype(store_type), dtype=store_type)
        c_3 = np.matmul(a_3.astype(store_type), b_3.astype(store_type), dtype=store_type)
        
        c_1_f64 = c_1.astype(np.float64)
        c_2_f64 = c_2.astype(np.float64)
        c_3_f64 = c_3.astype(np.float64)
        
        cross_term = c_3_f64 - c_1_f64 - c_2_f64
        c_m_acc = (s**2) * c_1_f64 + c_2_f64 + s * cross_term
        
    return apply_sym_modulo(c_m_acc, m, store_type)

def full_hybrid_ozaki_2(
    A: npt.NDArray[np.float64], 
    B: npt.NDArray[np.float64], 
    target_bits: int, 
    nb_moduli: int | None,
    compute_type: npt.DTypeLike, 
    store_type: npt.DTypeLike
) -> npt.NDArray[np.float64]:
    
    scale_A, scale_B = o2.get_diagonal_scales(A, B, target_bits) 
    A_int = o2.scale_to_integer(A, scale_A)
    B_int = o2.scale_to_integer(B, scale_B)

    q = A.shape[1]
    assert(q == B.shape[0])

    if nb_moduli is None:
        dt = np.dtype(compute_type)
        if np.issubdtype(dt, np.integer):
            info = utils.get_dtype_info(compute_type)
            effective_bits = info['bits'] - 1
        else:
            effective_bits = 9
            
        nb_moduli = int(np.ceil((np.log2(q) + 2 * target_bits + 1) / (effective_bits - 0.5)))

    moduli = get_hybrid_moduli(nb_moduli)

    modular_results = []
    for m in moduli:
        A_m = apply_sym_modulo(A_int, m, np.float64)
        B_m = apply_sym_modulo(B_int, m, np.float64)
        C_m = hybrid_modular_multiply(A_m, B_m, m, compute_type, store_type)
        modular_results.append(C_m)

    C_large_int = o2.crt_reconstruction(modular_results, moduli)
    C_float64 = o2.descale_matrix(C_large_int, scale_A, scale_B)
    
    return C_float64


def full_hybrid_ozaki_2_acc(
    a: npt.NDArray[np.float64], 
    b: npt.NDArray[np.float64], 
    target_bits: int,
    nb_moduli: int | None, 
    compute_type: npt.DTypeLike,
    store_type: npt.DTypeLike,
) -> npt.NDArray[np.float64]:

    q = a.shape[1]
    assert(q == b.shape[0])

    if nb_moduli is None:
        dt = np.dtype(compute_type)
        if np.issubdtype(dt, np.integer):
            info = utils.get_dtype_info(compute_type)
            effective_bits = info['bits'] - 1
        else:
            effective_bits = 9
            
        nb_moduli = int(np.ceil((np.log2(q) + 2 * target_bits + 1) / (effective_bits - 0.5)))

    moduli = get_hybrid_moduli(nb_moduli)
    
    scale_a, scale_b = o2.get_accurate_scales(a, b, moduli, compute_type)
    
    # Cap scaling factors to protect int64 upper limits during remainder tracking
    max_a_val = np.max(np.abs(a) * scale_a)
    max_b_val = np.max(np.abs(b) * scale_b)
    if max_a_val > 2.0**60 or max_b_val > 2.0**60:
        excess_bits_a = max(0, math.ceil(math.log2(max_a_val) - 60))
        excess_bits_b = max(0, math.ceil(math.log2(max_b_val) - 60))
        drop_bits = max(excess_bits_a, excess_bits_b)
        scale_a /= (2.0 ** drop_bits)
        scale_b /= (2.0 ** drop_bits)
    
    a_int = o2.scale_to_integer(a, scale_a)
    b_int = o2.scale_to_integer(b, scale_b)

    modular_results = []
    for m in moduli:
        a_m = apply_sym_modulo(a_int, m, np.float64)
        b_m = apply_sym_modulo(b_int, m, np.float64)
        
        c_m = hybrid_modular_multiply(a_m, b_m, m, compute_type, store_type)
        modular_results.append(c_m)
        
    c_large_int = o2.crt_reconstruction(modular_results, moduli)
    c_float64 = o2.descale_matrix(c_large_int, scale_a, scale_b)
    
    return c_float64

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

    print("Ozaki 2 hybrid multiplications, using int8 x int8 -> int32")

    Res_ozaki2_hybrid = full_hybrid_ozaki_2(A, B, target_bits, None, compute_type, store_type)

    utils.print_matrix(Res_ozaki2_hybrid, "Multiplication via Ozaki-2 hybrid")
    print("Absolute error : ", np.sum(np.abs((Res_exact - Res_ozaki2_hybrid))))

    Res_ozaki2_hybrid_acc = full_hybrid_ozaki_2_acc(A, B, target_bits, None, compute_type, store_type)

    utils.print_matrix(Res_ozaki2_hybrid_acc, "Multiplication via Ozaki-2 hybrid accurate")
    print("Absolute error : ", np.sum(np.abs((Res_exact - Res_ozaki2_hybrid_acc))))

def _hybrid_fp8_test():
    print("\nfp8 hybrid test :")
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

    compute_type: npt.DTypeLike = ml_dtypes.float8_e4m3fn
    mult_store_type: npt.DTypeLike = np.float32

    Res_ozaki2 = full_hybrid_ozaki_2(A, B, 53, None, compute_type, mult_store_type)

    utils.print_matrix(Res_ozaki2, "Multiplication via Ozaki-2 Hyb Fast")
    print("Absolute error : ", np.sum(np.abs((Res_exact - Res_ozaki2))))

    Res_ozaki2_acc = full_hybrid_ozaki_2_acc(A, B, 53, None, compute_type, mult_store_type)

    utils.print_matrix(Res_ozaki2_acc, "Multiplication via Ozaki-2 Hyb Accurate")
    print("Absolute error : ", np.sum(np.abs((Res_exact - Res_ozaki2_acc))))

    Res_ozaki2_acc17 = full_hybrid_ozaki_2_acc(A, B, 53, 17, compute_type, mult_store_type)

    utils.print_matrix(Res_ozaki2_acc, "Multiplication via Ozaki-2 Hyb Accurate 17 moduli")
    print("Absolute error : ", np.sum(np.abs((Res_exact - Res_ozaki2_acc))))

def _test_moduli():
    test = get_hybrid_moduli(100)
    test.sort()
    test = list(reversed(test))
    print(test)
    print(len(test))

def _test_hybrid_at_moduli_count(nb_moduli: int):
    print(f"\n--- Running Isolated Test for {nb_moduli} Moduli ---")
    
    # 1. Generate random matrix entries replicating your notebook scales
    scale = 10.0**2
    np.random.seed(42)  # Fixed seed for perfect test reproducibility
    A = np.random.random((3, 3)) * scale
    B = np.random.random((3, 3)) * scale

    # 2. Compute exact ground truth matrix product
    Res_exact = A @ B

    # 3. Setup your exact low-precision type parameters
    compute_type = ml_dtypes.float8_e4m3fn
    mult_store_type = np.float32

    # 4. Run the emulation routines 
    try:
        # Testing fast mode
        Res_ozaki2 = full_hybrid_ozaki_2(A, B, 53, nb_moduli, compute_type, mult_store_type)
        err_fast = np.sum(np.abs(Res_exact - Res_ozaki2))
        print(f"Fast Mode ({nb_moduli} moduli) Absolute error : {err_fast}")
    except Exception as e:
        print(f"Fast Mode failed with exception: {e}")

    try:
        # Testing accurate mode
        Res_ozaki2_acc = full_hybrid_ozaki_2_acc(A, B, 53, nb_moduli, compute_type, mult_store_type)
        err_acc = np.sum(np.abs(Res_exact - Res_ozaki2_acc))
        print(f"Accurate Mode ({nb_moduli} moduli) Absolute error : {err_acc}")
    except Exception as e:
        print(f"Accurate Mode failed with exception: {e}")

if __name__ == "__main__":
    _proof_of_work()
    _hybrid_fp8_test()
    _test_moduli()
    # Test 12 moduli (where it works well)
    _test_hybrid_at_moduli_count(12)
    
    # Test 13 moduli (where the error jumps up)
    _test_hybrid_at_moduli_count(13)
    
    # Test 17 moduli (to observe continuous error accumulation)
    _test_hybrid_at_moduli_count(17)

