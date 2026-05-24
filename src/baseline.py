"""
This module provides methods for computing the exact solution of matrix products thanks to multi-precision.
The functions implemented here are intended to serve as a baseline for testing and validating other,
potentially less precise or approximate, matrix multiplication algorithms.

Due to the focus on precision, these methods may be significantly slower than optimized or approximate alternatives.
The speed should not be used for comparison since we do not know how the module we used has been compiled or optimised.

We use the "mpmath" module to simulate multi-precision. As for everything else in this project, it uses numpy and float64 for input and output of matrices.
"""

import numpy as np
import numpy.typing as npt
from mpmath import mp
from typing import Any
try:
    import src.utils as utils
except ModuleNotFoundError:
    import utils as utils

MPMat = Any

def _to_mp_matrix(np_matrix: npt.NDArray[np.float64]) -> MPMat :
    return mp.matrix(np_matrix.tolist())

def _to_np_matrix(mp_matrix: MPMat) -> npt.NDArray[np.float64]:
    return np.array(mp_matrix.tolist(), dtype=np.float64)

def get_exact_product(A : npt.NDArray[np.float64], B : npt.NDArray[np.float64], prec: int) -> npt.NDArray[np.float64]:
    '''
    Does A x B using multi-precision. Result is stored as float64.
    
    :param A: Matrix
    :type A: npt.NDArray[np.float64]
    :param B: Matrix
    :type B: npt.NDArray[np.float64]
    :param prec: Precision, in bits, to use for computation (ex: 256)
    :type prec: int
    :return: The product of the two matrices, with values stored as float64
    :rtype: NDArray[float64]
    '''
    mp.prec = prec

    A_mp: MPMat = _to_mp_matrix(A)
    B_mp: MPMat = _to_mp_matrix(B)

    Res_mp: MPMat = A_mp * B_mp

    return _to_np_matrix(Res_mp)

def get_exact_sum(L : list[npt.NDArray[np.float64]], prec: int) -> npt.NDArray[np.float64]:
    '''
    Does A + B using multi-precision. Result is stored as float64.
    
    :param L: List of matrices to sum
    :type L: list[npt.NDArray[np.float64]]
    :param prec: Precision, in bits, to use for computation (ex: 256)
    :type prec: int
    :return: The sum of matrices, with values stored as float64
    :rtype: NDArray[float64]
    '''
    mp.prec = prec

    Res_mp: MPMat = _to_mp_matrix(np.zeros_like(L[0]))
    for x in L:
        x_mp = _to_mp_matrix(x)
        Res_mp += x_mp
    
    return _to_np_matrix(Res_mp)

def _exemple():
    print("Baseline multi-precision. \nExemple :")
    x = 2.0**27
    A = np.array([[x + 1.0, x], [0.0, 0.0]], dtype=np.float64)
    B = np.array([[x - 1.0, 0.0], [-x, 0.0]], dtype=np.float64)
    utils.print_matrix(A, "A")
    utils.print_matrix(B, "B")
    Res = A @ B
    Res_mp = get_exact_product(A, B, 256)
    print("\nResult of multiplication :")
    utils.print_matrix(Res, "Normal precision")
    utils.print_matrix(Res_mp, f"{mp.prec} precision")

if __name__ == '__main__':
    _exemple()

