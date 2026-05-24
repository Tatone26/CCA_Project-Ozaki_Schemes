'''
This file contains the basic functionalities of the Ozaki-1 scheme, but all in float64 (input and output)
It is useful to see how the algorithm is functioning. 
These functions are parameter-driven, allowing for easy implementation of optimisations and measurements. 
'''

import numpy as np
import numpy.typing as npt
try:
    import src.utils as utils
except ModuleNotFoundError:
    import utils as utils

def split_matrix(A: npt.NDArray[np.float64], sigma: np.float64, d: int) -> list[npt.NDArray[np.float64]]:
    '''
    Split the given matrix in k lower-precision matrices, following the Ozaki-1 method proposed in 2012.
    
    :param A: Matrix to split
    :type A: npt.NDArray[np.float64]
    :param sigma: the power of two at which to start splitting. Approximatively the max of the matrix.
    :type sigma: np.float64
    :param d: the 'size' of each lower-precision matrix (how many bits we keep in each). Used to make sure we don't lose any information in following computations.
    :type d: int
    :return: The list of lower-precision matrices
    :rtype: list[NDArray[float64]]
    '''
    
    # Protection. May not be necessary later.
    if np.frexp(sigma)[0] > 0.5 or sigma < np.max(A):
        raise(ValueError("Invalid value for sigma"))
    if 2 * d + np.log2(np.max(np.shape(A))) > np.finfo(np.float64).nmant + 1: # adding implicit bit
        raise(ValueError("Invalid value for d"))    # a bigger d will lose information
    
    # Algorithm
    remainder: npt.NDArray[np.float64] = np.copy(A)
    result_matrices: list[npt.NDArray[np.float64]] = [] 
    csigma = sigma  # will decrease in correspondance with the parameter d 

    while np.any(remainder):    # while there is information to get
        if csigma < np.finfo(np.float64).tiny:  # underflow protection (and anti-infinite loop)
            result_matrices.append(remainder)
            break

        A_k = (remainder / csigma).round(decimals=0) * csigma
        result_matrices.append(A_k)

        remainder = remainder - A_k
        csigma *= 2.0**(-d)
    
    if len(result_matrices) == 0:   # special case for null matrix
        return [np.zeros_like(A)]
    
    return result_matrices


def full_products(split_A: list[npt.NDArray[np.float64]], split_B: list[npt.NDArray[np.float64]]) -> list[npt.NDArray[np.float64]]:
    '''
    Does the product of all matrices in split_A with all matrices in split_B. 

    Matrix multiplication is done via Numpy, calling optimized BLAS code.
    
    Precision may be lost, but not if split_A and split_B are computed correctly.
    
    :param split_A: Matrices.
    :type split_A: list[npt.NDArray[np.float64]]
    :param split_B: Matrices.
    :type split_B: list[npt.NDArray[np.float64]]
    :return: A list containing all the computed product matrices.
    :rtype: list[NDArray[float64]]
    '''
    computed_products: list[npt.NDArray[np.float64]] = []

    for A in split_A:
        for B in split_B:
            R = A @ B   # matrix multiplication via Numpy
            computed_products.append(R)

    return computed_products


def perfect_summation(L : list[npt.NDArray[np.float64]]) -> npt.NDArray[np.float64]:
    '''
    Sums the list of matrices while losing as little information as possible (none would be great)

    Warning : some precision is lost here because of accumulation in double
    
    :param L: A list of matrices
    :type L: list[npt.NDArray[np.float64]]
    :return: The sum of the matrices
    :rtype: NDArray[float64]
    '''
    Res: npt.NDArray[np.float64] = np.zeros_like(L[0], dtype=np.float64)

    # For now : we consider L to be a list computed with the above function. In those, smaller values are at the end of the list.
    # NOT PERFECT.
    for x in reversed(L):
        Res += x

    return Res


def _split_exemple():
    print("Splitting exemple :")
    A: npt.NDArray[np.float64] = np.array([[np.pi, np.pi * 10], [np.pi / 10, np.pi * 100]], dtype=np.float64)
    sigma: np.float64 = 2.0**np.frexp(np.max(np.abs(A)))[1]
    d: int = 20
    print(f"Sigma : 2^{np.log2(sigma)} ; d : {d}")
    utils.print_matrix(A, "A")
    result = split_matrix(A, sigma, d)
    print(f"\nNumber of matrices after split : {len(result)}")
    for i, B in enumerate(result):
        utils.print_matrix(B, f"A({i})")
    print("\nReconstructing original matrix:")
    sum_result = np.sum(result, axis = 0)
    utils.print_matrix(sum_result, "Reconstructed sum")
    utils.print_matrix(A - sum_result, "Error")
    


if __name__ == "__main__":
    _split_exemple()