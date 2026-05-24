'''
This file contains the basic functionalities of the Ozaki-1 scheme, where the input is in float64 and the output is a chosen integer type.

You can find all functions as well as the full algorithms. 

This file can be run to see some examples.
'''

import numpy as np
import numpy.typing as npt
try:
    import src.utils as utils
except ModuleNotFoundError:
    import utils as utils

try:
    import src.esc as esc
except ModuleNotFoundError:
    import esc 

def split_matrix_dtype(A: npt.NDArray[np.float64], sigma: np.float64, d: int, output_type: npt.DTypeLike) -> list[npt.NDArray]:
    '''
    Split the given matrix in k lower-precision matrices, following the Ozaki-1 method. It doesn't work if d isn't big enough. 
    
    :param A: Matrix to split
    :type A: npt.NDArray[np.float64]
    :param sigma: scaling factor of the first matrix (a power of two >= max(A))
    :type sigma: np.float64
    :param d: the 'size' of each lower-precision matrix (how many bits of information we keep / extract in each). 
    Should be equivalent to the output_type, or at least be set in respect to the output_type of the matrix multiplication.
    :type d: int
    :return: The list of lower-precision matrices
    :param output_type: the output type of the matrix. This function is primarly made for integers as output.
    :type output_type: npt.DTypeLike
    :rtype: list[NDArray]
    '''
    
    # Algorithm
    remainder: npt.NDArray[np.float64] = np.copy(A)
    result_matrices: list[npt.NDArray] = [] # arrays of output_type
    csigma = sigma

    while np.any(remainder):    # while there is information to get
        if csigma < np.finfo(np.float64).tiny:  # underflow protection (and anti-infinite loop)
            result_matrices.append(remainder)
            break

        scaled_remainder = (remainder / csigma).round(decimals=0) # get d bits of information in the integer part of the remainder
        # indefinite result if d > size(output_type)...

        """ if not np.isfinite(scaled_remainder).all(): # maybe some strange case to take care of here
            print("Found NaN or Inf in scaled_remainder!")
            print(f"Max value: {np.max(scaled_remainder)}")
            print(f"Min value: {np.min(scaled_remainder)}") """

        result_matrices.append(scaled_remainder.astype(output_type)) # store these

        remainder = remainder - scaled_remainder * csigma
        csigma *= 2.0**(-d)
    
    if len(result_matrices) == 0:   # special case for null matrix
        return [np.zeros_like(A, dtype=output_type)]
    
    return result_matrices

def split_matrix_dtype_trunc(
    A: npt.NDArray[np.float64], 
    sigma: np.float64, 
    d: int, 
    s: int, 
    output_type: npt.DTypeLike
) -> tuple[list[npt.NDArray], list[np.float64]]:
    '''
    Split the given matrix in up to 's' lower-precision matrices, following the Ozaki-1 method.
    Truncates the rest (loss of information) if 's' is reached before remainder is null.
    This is the true version of Ozaki-1.
    
    :param A: Matrix to split
    :type A: npt.NDArray[np.float64]
    :param sigma: scaling factor of the first matrix (a power of two >= max(A))
    :type sigma: np.float64
    :param d: how many bits of information we keep / extract in each split.
    :type d: int
    :param s: maximum number of splits to perform (truncation parameter).
    :type s: int
    :param output_type: the output type of the matrix (e.g., np.int8).
    :type output_type: npt.DTypeLike
    :return: A tuple containing the list of split matrices and the list of their respective scaling factors.
    :rtype: tuple[list[NDArray], list[np.float64]]
    '''
    
    remainder: npt.NDArray[np.float64] = np.copy(A)
    result_matrices: list[npt.NDArray] = []
    scalings: list[np.float64] = []
    
    csigma: np.float64 = sigma

    for _ in range(s):
        if not np.any(remainder):    
            break
        if csigma < np.finfo(np.float64).tiny:  
            break

        scaled_remainder = np.rint(remainder / csigma) 
        # storing matrix and scaling factor
        result_matrices.append(scaled_remainder.astype(output_type))
        scalings.append(csigma)

        remainder = remainder - scaled_remainder * csigma
        
        csigma = np.ldexp(csigma, -d) 
    
    # null matrix
    if len(result_matrices) == 0:   
        return [np.zeros_like(A, dtype=output_type)], [sigma]
    
    return result_matrices, scalings

def multiply_dtype(A: npt.NDArray, B: npt.NDArray, output_type: npt.DTypeLike) -> npt.NDArray:
    '''
    A * B with the result being stored as output_type

    Warning : if output_type cannot store a scalar product directly, OVERFLOW will happen.
    
    :param A: matrix
    :type A: npt.NDArray
    :param B: matrix
    :type B: npt.NDArray
    :param output_type: output type. Should be bigger than dtype of A and B to not lose information
    :type output_type: npt.DTypeLike
    :return: A*B
    :rtype: list[NDArray[Any]]
    '''
    return np.matmul(A.astype(output_type), B.astype(output_type), dtype=output_type)


def multiply_lists_dtype(LA: list[npt.NDArray], LB: list[npt.NDArray], output_type: npt.DTypeLike) -> list[npt.NDArray]:
    '''
    Multiplies all pairs of matrix in A and B, with output type defined.
    It is too much, but truncation and removal of unnecessary products is done in the full_ozaki_1 function.
    
    :param LA: list of matrix
    :type LA: list[npt.NDArray]
    :param LB: list of matrix
    :type LB: list[npt.NDArray]
    :param output_type: computation and output type
    :type output_type: npt.DTypeLike
    :return: a list of matrix, with order [A1.B1...A1.Bm, A2.B1, ... An.Bm]
    :rtype: list[NDArray[Any]]
    '''
    res: list[npt.NDArray] = []

    for A in LA:
        for B in LB:
            res.append(multiply_dtype(A, B, output_type))

    return res

def get_scaling_factors(lena: int, lenb: int, sigma_a: np.float64, sigma_b: np.float64, d: int) -> list[np.float64]:
    '''
    Returns the scaling factors corresponding to the multiplication in the multiply_list function.
    
    :param lena: size of list A
    :type lena: int
    :param lenb: size of list B
    :type lenb: int
    :param sigma_a: sigma, like in split_matrix
    :type sigma_a: np.float64
    :param sigma_b: sigma, like in split_matrix
    :type sigma_b: np.float64
    :param d: d, like in split_matrix
    :type d: int
    :return: the list of factors (size lena * lenb)
    :rtype: list[float64]
    '''
    # using log2 to make sure there is no overflow...
    return [2.0**(np.log2(sigma_a) + np.log2(sigma_b) - d*(i + j)) 
        for i in range(lena) for j in range(lenb)]

def sum_with_scaling(L: list[npt.NDArray], scaling: list[np.float64]) -> npt.NDArray[np.float64]:
    '''
    Sums a list of matrix as a list of float64. Each will be multiplied by the scaling found at the corresponding index in 'scaling'.
    
    Ordered from smallest scaling to biggest, to limit absorption.
    
    :param L: list of matrices
    :type L: list[npt.NDArray]
    :param scaling: the scaling factors
    :type scaling: list[np.float64]
    :return: Description
    :rtype: NDArray[float64]
    '''
    if len(L) != len(scaling):
        raise ValueError(f"Not same number of matrices ({len(L)}) and scaling factors ({len(scaling)}).")

    res : npt.NDArray[np.float64] = np.zeros_like(L[0], dtype=np.float64)

    for A, S in sorted(zip(L, scaling), key=lambda x: x[1]): # iterating from smallest scaling to biggest
        res += A.astype(np.float64) * S

    return res

def s_to_d(s: int, tau: int=53) -> int:
    '''
    Given s the number of splits and tau the wanted precision to extract (ex: 53 for f64), returns d the depth for each matrix (the number of bits per matrix)

    Be warned that it can result in some precision loss if used incorrectly.
    
    :param s: number of splits
    :type s: int
    :param tau: wanted precision
    :type tau: int
    :return: depth for each split matrix
    :rtype: int
    '''
    return np.ceil(tau / s)

def full_ozaki_1(A: npt.NDArray, B: npt.NDArray, split_type: npt.DTypeLike, d:int, store_type: npt.DTypeLike) -> npt.NDArray:
    '''
    Does the product A x B using the ozaki-1 scheme. Very basic. Perfect in all parameters. Just proving that it works.
    '''
    sigma_A, sigma_B = 2**(np.frexp(np.max(np.abs(A))))[1], 2**(np.frexp(np.max(np.abs(B))))[1] # this is the best way to choose sigma as to not lose any information.
    ozaki1_split_A = split_matrix_dtype(A, sigma_A, d, split_type)
    ozaki1_split_B = split_matrix_dtype(B, sigma_B, d, split_type)
    # using dynamic splitting
    ozaki1_product = multiply_lists_dtype(ozaki1_split_A, ozaki1_split_B, store_type)
    ozaki1_AB = sum_with_scaling(ozaki1_product, get_scaling_factors(len(ozaki1_split_A), len(ozaki1_split_B), sigma_A, sigma_B, d))
    return ozaki1_AB

def full_ozaki_1_trunc(A: npt.NDArray, B: npt.NDArray, split_type: npt.DTypeLike, d:int, store_type: npt.DTypeLike, s: int) -> npt.NDArray:
    '''
    Does A*B using Ozaki-1 scheme.
    '''
    # get the raw max exponent
    exp_A = np.frexp(np.max(np.abs(A)))[1]
    exp_B = np.frexp(np.max(np.abs(B)))[1]
    
    # shift the ceiling down by (d - 1). 
    # This makes the max value equal to 2^(d-1) (e.g., 64), which fits perfectly in int8.
    sigma_A = 2.0**(exp_A - d + 1)
    sigma_B = 2.0**(exp_B - d + 1) 
    
    # split the matrices independantly, with the specified number of slices. Better since not dynamic.
    ozaki1_split_A, scalings_A = split_matrix_dtype_trunc(A, sigma_A, d, s, split_type)
    ozaki1_split_B, scalings_B = split_matrix_dtype_trunc(B, sigma_B, d, s, split_type)

    # do all the small-type matrices product
    ozaki1_product = multiply_lists_dtype(ozaki1_split_A, ozaki1_split_B, store_type)

    # reconstitute final matrix by summing with right scalings
    global_scalings = [s_a * s_b for s_a in scalings_A for s_b in scalings_B]
    ozaki1_AB = sum_with_scaling(ozaki1_product, global_scalings)
    return ozaki1_AB

def full_ozaki_1_fast(A: npt.NDArray, B: npt.NDArray, split_type: npt.DTypeLike, d:int, store_type: npt.DTypeLike, s: int) -> npt.NDArray:
    '''
    Corrected Ozaki-1: Initial sigma is shifted so the first slice fully utilizes the d bits.
    Filters out the absorbed cross-terms (where i + j >= s) after computing them, 
    without modifying the underlying helper functions.
    '''
    exp_A = np.frexp(np.max(np.abs(A)))[1]
    exp_B = np.frexp(np.max(np.abs(B)))[1]
    
    sigma_A = 2.0**(exp_A - d + 1)
    sigma_B = 2.0**(exp_B - d + 1) 
    
    ozaki1_split_A, scalings_A = split_matrix_dtype_trunc(A, sigma_A, d, s, split_type)
    ozaki1_split_B, scalings_B = split_matrix_dtype_trunc(B, sigma_B, d, s, split_type)
    
    # we do all products anyway... Even if it is catastrophically not optimised. That way we are sure there are no issues there.
    ozaki1_product_full = multiply_lists_dtype(ozaki1_split_A, ozaki1_split_B, store_type)

    global_scalings = [s_a * s_b for s_a in scalings_A for s_b in scalings_B]

    # here we remove the unnecessary products. 
    ozaki1_product_trunc = []
    scaling_factors_trunc = []
    lenb = len(ozaki1_split_B)
    for index, (prod, scale) in enumerate(zip(ozaki1_product_full, global_scalings)):
        i = index // lenb
        j = index % lenb
        if i + j < s: # or <= ? this can be chosen to dictate how "fast" (and inaccurate) this method is 
            ozaki1_product_trunc.append(prod)
            scaling_factors_trunc.append(scale)
            
    ozaki1_AB = sum_with_scaling(ozaki1_product_trunc, scaling_factors_trunc)
    
    return ozaki1_AB

def full_ozaki_1_auto_esc(
    A: npt.NDArray, 
    B: npt.NDArray, 
    split_type: npt.DTypeLike, 
    d: int, 
    store_type: npt.DTypeLike, 
    target_precision: int = 53, # Default to 53 for FP64-like mantissa
    b: int = 1
) -> npt.NDArray:
    '''
    Ozaki-1 scheme where the number of slices 's' is automatically calculated 
    using the block-coarsened ESC to ensure target_precision bits are preserved.
    '''
    # use ESC to get the necessary number of slices
    calculated_esc = esc.esc(A, B, b=b)
    s = esc.esc_to_slices(calculated_esc, d, target_precision)
        
    # Sigma calculation (Peak of the bit ladder)
    sigma_A = 2.0**(np.frexp(np.max(np.abs(A)))[1])
    sigma_B = 2.0**(np.frexp(np.max(np.abs(B)))[1])
    
    # Splitting
    ozaki1_split_A, scalings_A = split_matrix_dtype_trunc(A, sigma_A, d, s, split_type)
    ozaki1_split_B, scalings_B = split_matrix_dtype_trunc(B, sigma_B, d, s, split_type)
    
    # Computation (The O(s^2) part)
    ozaki1_product = multiply_lists_dtype(ozaki1_split_A, ozaki1_split_B, store_type)
    
    # Re-scaling and Summation
    global_scalings = [s_a * s_b for s_a in scalings_A for s_b in scalings_B]
    ozaki1_AB = sum_with_scaling(ozaki1_product, global_scalings)
    
    return ozaki1_AB

def _split_exemple():
    print("Splitting exemple :")
    A: npt.NDArray[np.float64] = np.array([[np.pi, np.pi * 10], [np.pi / 10, np.pi * 100]], dtype=np.float64)
    d: int = 8
    print(f"d : {d}")
    sigma: np.float64 = 2**(np.frexp(np.max(np.abs(A)))[1])
    utils.print_matrix(A, "A")
    result = split_matrix_dtype(A, sigma, d, np.int8)
    print(f"\nNumber of matrices after split : {len(result)}")
    for i, B in enumerate(result):
        utils.print_matrix(B, f"A({i})")
    print("\nReconstructing original matrix:")
    sum_result = sum_with_scaling(result, [sigma * 2.0**((-d) * i) for i in range(len(result))])
    utils.print_matrix(sum_result, "Reconstituated matrix")
    error = A - sum_result
    utils.print_matrix(error, "Error lost in reconstitution")

def _mutliplication_exemple():
    print("Multiplication example :")
    scale = 10.0**2
    A: npt.NDArray[np.float64] = np.random.random((3, 3)) * scale
    B: npt.NDArray[np.float64] = np.random.random((3, 3)) * scale

    import baseline
    Res_exact = baseline.get_exact_product(A, B, 256)
    utils.print_matrix(Res_exact, "Exact multiplication")

    Res_base = A @ B
    utils.print_matrix(Res_base, "Basic multiplication")
    print("Absolute error : ", np.sum(np.abs((Res_exact - Res_base))))

    sigma = 2.0**np.frexp(scale)[1]
    compute_type: npt.DTypeLike = np.int8
    mult_store_type: npt.DTypeLike = np.int32
    d = 7
    A_s = split_matrix_dtype(A, sigma, d, compute_type)
    B_s = split_matrix_dtype(B, sigma, d, compute_type)
    Res = multiply_lists_dtype(A_s, B_s, mult_store_type)
    scaling_factors = get_scaling_factors(len(A_s), len(B_s), sigma, sigma, d)
    Res_ozaki = sum_with_scaling(Res, scaling_factors)

    utils.print_matrix(Res_ozaki, "Multiplication via int8")
    print("Absolute error : ", np.sum(np.abs((Res_exact - Res_ozaki))))

def _proof_of_work():
    print("Working proof for different version (int8) :")
    scale = 10.0**2
    A: npt.NDArray[np.float64] = np.random.random((3, 3)) * scale
    B: npt.NDArray[np.float64] = np.random.random((3, 3)) * scale

    import baseline
    Res_exact = baseline.get_exact_product(A, B, 256)
    utils.print_matrix(Res_exact, "Exact multiplication")

    Res_base = A @ B
    utils.print_matrix(Res_base, "Basic multiplication")
    print("Absolute error : ", np.sum(np.abs((Res_exact - Res_base))))

    Res_fast = full_ozaki_1_fast(A, B, np.int8, 7, np.int32, 8)
    utils.print_matrix(Res_fast, "Fast Ozaki 1 (d=7, s=8)")
    print("Absolute error : ", np.sum(np.abs((Res_exact - Res_fast))))

    Res_acc = full_ozaki_1_trunc(A, B, np.int8, 7, np.int32, 8)
    utils.print_matrix(Res_acc, "Acc Ozaki 1 (d=7, s=8)")
    print("Absolute error : ", np.sum(np.abs((Res_exact - Res_acc))))

    Res_auto = full_ozaki_1(A, B, np.int8, 7, np.int32)
    utils.print_matrix(Res_auto, "Full Ozaki 1 (d=7, s=auto)")
    print("Absolute error : ", np.sum(np.abs((Res_exact - Res_auto))))


if __name__ == "__main__":
    _split_exemple()
    _mutliplication_exemple()
    _proof_of_work()