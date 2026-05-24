'''
This file contains implementations of some forms of error measurements, like forward/backward error, relative error etc.

It is used expensively during this project. 

All functions need an exact result of the product so they can all receive an optional "C" parameter.
'''

import numpy as np
import numpy.typing as npt

try :
    from baseline import get_exact_product
except ModuleNotFoundError:
    from src.baseline import get_exact_product

def backward_error(A: npt.NDArray, B: npt.NDArray, AB: npt.NDArray, C: npt.NDArray|None = None) -> np.float64:
    '''
    Computes the Frobenius norm-based backward error of the product AB.
    Measured relative to the product of the absolute values of the inputs.
    
    :param A: First input matrix
    :type A: npt.NDArray
    :param B: Second input matrix
    :type B: npt.NDArray
    :param AB: Computed product of A and B to evaluate
    :type AB: npt.NDArray
    :param C: Pre-computed multi-precision result
    :type C: npt.NDArray
    :return: The backward error value
    :rtype: np.float64
    '''
    if C is None:
        C = get_exact_product(A, B, 256)
    # denominator = || |A| @ |B| ||_F
    denom_backward = np.linalg.norm(get_exact_product(np.abs(A), np.abs(B), 256), ord='fro')
    backward_err = np.linalg.norm(AB - C, ord='fro') / denom_backward
    return np.float64(backward_err)

def forward_bound(A: npt.NDArray, B: npt.NDArray) -> np.float64:
    '''
    Computes the theoretical forward error bound for standard float64 multiplication.
    The bound is defined as $n . u . cond(A, B)$, where $u = 2^{-53}$.
    
    :param A: First input matrix
    :type A: npt.NDArray
    :param B: Second input matrix
    :type B: npt.NDArray
    :return: The theoretical upper bound for the forward error
    :rtype: np.float64
    '''
    u = 2.0**(-53)

    # Using standard @ here as an approximation for the bound's denominator logic
    denom_backward = np.linalg.norm(np.abs(A) @ np.abs(B), ord='fro')
    C = get_exact_product(A, B, 256)
    cond_prod = denom_backward / np.linalg.norm(C, ord='fro')

    # Bound formula: n * u * condition_number
    f_bound = A.shape[0] * u * cond_prod
    return f_bound

import numpy as np
import numpy.typing as npt

def average_relative_error(A: npt.NDArray, B: npt.NDArray, AB: npt.NDArray, C: npt.NDArray | None = None) -> np.float64:
    '''
    Computes the element-wise relative error averaged over all valid entries.
    Useful for checking accuracy on high-dynamic range matrices.
    
    :param A: First input matrix
    :type A: npt.NDArray
    :param B: Second input matrix
    :type B: npt.NDArray
    :param AB: Computed product of A and B to evaluate
    :type AB: npt.NDArray
    :param C: Pre-computed multi-precision result. If None, it will be computed.
    :type C: npt.NDArray | None
    :return: Average of abs(AB - C) / abs(C) for non-zero elements
    :rtype: np.float64
    '''
    if C is None:
        C = get_exact_product(A, B, 256)

    # Find where the exact result is non-zero
    mask = (C != 0)
    
    # Edge case: The exact matrix is entirely zeros
    if not np.any(mask):
        return np.float64(0.0) if np.all(AB == 0) else np.float64(np.inf)

    # Compute error strictly on non-zero elements to avoid skewing the mean
    rel_err = np.abs(AB[mask] - C[mask]) / np.abs(C[mask])
    
    # If the exact result is 0, but our computed result is NOT 0, the relative error is infinite
    if np.any((C == 0) & (AB != 0)):
        return np.float64(np.inf)
        
    return np.float64(np.mean(rel_err))

def max_relative_error(A: npt.NDArray, B: npt.NDArray, AB: npt.NDArray, C: npt.NDArray | None = None) -> np.float64:
    '''
    Computes the maximum element-wise relative error.
    Highlights the "worst" entry in the computed product.
    
    :param A: First input matrix
    :type A: npt.NDArray
    :param B: Second input matrix
    :type B: npt.NDArray
    :param AB: Computed product of A and B to evaluate
    :type AB: npt.NDArray
    :param C: Pre-computed multi-precision result. If None, it will be computed.
    :type C: npt.NDArray | None
    :return: Maximum of abs(AB - C) / abs(C)
    :rtype: np.float64
    '''
    if C is None:
        C = get_exact_product(A, B, 256)
    
    # Find where the exact result is non-zero
    mask = (C != 0)
    
    # Edge case: The exact matrix is entirely zeros
    if not np.any(mask):
        return np.float64(0.0) if np.all(AB == 0) else np.float64(np.inf)

    # Compute error strictly on non-zero elements
    rel_err = np.abs(AB[mask] - C[mask]) / np.abs(C[mask])
    
    # If the exact result is 0, but our computed result is NOT 0, the relative error is infinite
    if np.any((C == 0) & (AB != 0)):
        return np.float64(np.inf)
        
    return np.float64(np.max(rel_err))


def max_norm_relative_error(A: npt.NDArray, B: npt.NDArray, AB: npt.NDArray, C: npt.NDArray | None = None) -> np.float64:
    '''
    Computes the maximum norm-based relative error.
    
    :param A: First input matrix
    :type A: npt.NDArray
    :param B: Second input matrix
    :type B: npt.NDArray
    :param AB: Computed product of A and B to evaluate
    :type AB: npt.NDArray
    :param C: Pre-computed multi-precision result. If None, it will be computed.
    :type C: npt.NDArray | None
    :return: Maximum of abs(AB - C) / abs(C)
    :rtype: np.float64
    '''
    if C is None: 
        C = get_exact_product(A, B, 256)

    max_C = np.max(np.abs(C))
    if max_C == 0: 
        return np.float64(0.0) if np.all(AB == 0) else np.float64('inf')
    
    return np.float64(np.max(np.abs(AB - C)) / max_C)


def forward_error(A: npt.NDArray, B: npt.NDArray, AB: npt.NDArray, C: npt.NDArray | None = None) -> np.float64:
    '''
    Computes the standard forwar (Norm Relative) Error of the matrix product, Frobenius Norm.
    This is the standard metric used in BLAS/LAPACK and HPC papers.
    
    :param A: First input matrix
    :type A: npt.NDArray
    :param B: Second input matrix
    :type B: npt.NDArray
    :param AB: Computed product of A and B to evaluate
    :type AB: npt.NDArray
    :param C: Pre-computed multi-precision result.
    :type C: npt.NDArray | None
    :return: ||AB - C||_F / ||C||_F
    :rtype: np.float64
    '''
    if C is None:
        C = get_exact_product(A, B, 256)
        
    # Calculate the Frobenius norm of the error matrix
    norm_error = np.linalg.norm(AB - C, ord='fro')
    
    # Calculate the Frobenius norm of the exact matrix
    norm_C = np.linalg.norm(C, ord='fro')
    
    if norm_C == 0:
        return np.float64(0.0) if norm_error == 0 else np.float64(np.inf)
        
    return np.float64(norm_error / norm_C)

# just some renaming
def norm_relative_error(A: npt.NDArray, B: npt.NDArray, AB: npt.NDArray, C: npt.NDArray | None = None) -> np.float64:
    return forward_error(A, B, AB, C)