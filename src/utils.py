'''
This file contains some functions to make our life easier with matrices. 
Typically : printing methods, matrix creation, etc.
'''

import numpy as np
import numpy.typing as npt
import math

def print_matrix(M: npt.NDArray[np.float64], name: str="Matrix") -> None:
    '''
    Prints a given matrix to terminal
    Shows power of two or near numbers as 2^x
    
    :param m: Matrix to print
    :type m: npt.NDArray[np.float64]
    :param name: Name of the matrix, will be printed above
    :type name: str
    '''

    m = None

    if M.shape[1] > 10 or M.shape[0] > 10:
        print(f"/!\\ Matrix {name} is too large : {M.shape}. Upper-left 3x3 values :") 
        m = M[:min(M.shape[0], 3), :min(M.shape[1], 3)]
    else: 
        m = M


    def format_val(x: np.float64) -> str:
        s = f"{x:.16g}" # 16 decimal numbers are the maximum in float64

        if (abs(x) < 1024):
            return s
        
        # looking for potential power of 2 (or near power of 2)
        def check_pwr2(val: np.float64) -> str | None:
            if val == 0: return None
            # frexp => val as m * 2^e, with m in [0.5, 1[
            m, e = np.frexp(val)
            # if mantissa == +-0.5, then it is a power of 2
            if np.abs(m) == 0.5:
                # adujsting to get the 1 * 2^e' 
                pwr = e - 1
                sign = "-" if val < 0 else ""
                return f"{sign}2^{pwr}"
            return None

        # power of 2 ?
        pwr_direct = check_pwr2(x)
        if pwr_direct:
            return f"{pwr_direct}"
        
        # power of 2 + 1 ?
        pwr_plus = check_pwr2(x - 1.0)
        if pwr_plus:
            return f"{pwr_plus} + 1"
            
        # power of 2 - 1 ?
        pwr_minus = check_pwr2(x + 1.0)
        if pwr_minus:
            return f"{pwr_minus} - 1"
        
        return s

    # converting all numbers
    str_matrix = [[format_val(val) for val in row] for row in m.tolist()]
    
    # finding width of every column
    num_cols = m.shape[1]
    col_widths = [
        max(len(str_matrix[i][j]) for i in range(m.shape[0])) 
        for j in range(num_cols)
    ]

    # printing
    header = f" {name} "
    total_width = sum(col_widths) + (len(col_widths) * 3) + 2
    print(f"{header:=^{max(total_width, len(header)+4)}}")
    
    for row_strs in str_matrix:
        line = " | ".join(s.rjust(width) for s, width in zip(row_strs, col_widths)) # rjust justifies the text to the right
        print(f" | {line} |")
    
    print("=" * max(total_width, len(header)+4))

from typing import Any
def random_matrix(m: int, n: int, a, b, dtype: npt.DTypeLike, seed: int|None = None) -> npt.NDArray[Any]:
    '''
    Creates a random matrix.
    
    :param m: number of lines
    :type m: int
    :param n: number of columns
    :type n: int
    :param a: minimum for coefficients
    :param b: maximum for coefficients (EXCLUDED)
    :param dtype: type of coefficients
    :type dtype: npt.DTypeLike
    :return: a randomly chosen matrix
    :rtype: NDArray[Any]
    '''

    rng = np.random.default_rng() if seed is None else np.random.default_rng(seed)

    # checking for correct representability
    info = np.finfo(dtype) if np.dtype(dtype).kind in ('f', 'c') else np.iinfo(dtype)
    if a < float(info.min) or b > float(info.max):
        raise ValueError(
            f"Bounds [{a}, {b}] are outside the representable range of {dtype} "
            f"([{info.min}, {info.max}])."
        )

    # we can create float or integer matrices, so need both cases
    if np.dtype(dtype).kind in ('i', 'u'):
        return rng.integers(low=a, high=b, size=(m, n), dtype=dtype)
    else:
        return rng.uniform(low=a, high=b, size=(m, n)).astype(dtype)
    

def random_phi_matrix(m: int, n: int, phi: float, dtype: npt.DTypeLike, seed: int|None = None) -> npt.NDArray[Any]:
    '''
    Creates a random matrix following the method Ozaki et al. proposed in arXiv:2306.11975v4, 4.2

    :param m: number of lines
    :type m: int
    :param n: number of columns
    :type n: int
    :param phi: control of the exponent distribution range
    :type phi: np.float64
    :param dtype: type of coefficients
    :type dtype: npt.DTypeLike
    :return: a random matrix following the formula
    :rtype: NDArray[Any]
    '''
    
    rng = np.random.default_rng() if seed is None else np.random.default_rng(seed)

    res = np.zeros((m, n), dtype=dtype)

    info = np.finfo(dtype) if np.dtype(dtype).kind == 'f' else np.iinfo(dtype)

    for i in range(m):
        # get random elements
        uniform = rng.uniform(-0.5, 0.5, n)
        normal = rng.normal(0, 1, n)
        for j in range(n):
            # apply formula
            val = uniform[j] * np.exp(phi * normal[j])
            # checking for overflow before casting
            if val > info.max or val < info.min:
                raise OverflowError(
                    f"Value {val:.2e} exceeds limits of {dtype} ({info.min:.2e} to {info.max:.2e}). "
                    f"Lower your 'phi' or use a larger 'dtype'.")
            # storing
            res[i, j] = np.astype(val, dtype)
        
    return res


def random_mean_var_matrix(m: int, n: int, dtype: npt.DTypeLike, mean:float=0.0, var:float=1.0, seed: int|None = None) -> npt.NDArray[Any]:
    '''
    Creates a random matrix following a normal distribution with specified mean and variance.

    :param m: number of lines
    :type m: int
    :param n: number of columns
    :type n: int
    :param dtype: type of coefficients
    :type dtype: npt.DTypeLike
    :param mean: mean of distribution
    :type mean: float
    :param var: variance of distribution
    :type var: float
    :param seed: if specified, used as seed for random generation
    :type seed: int|None
    :return: a random matrix following the specified normal distribution
    :rtype: NDArray[Any]
    '''
    rng = np.random.default_rng() if seed is None else np.random.default_rng(seed)

    A = rng.normal(loc=mean, scale=np.sqrt(var), size=(m, n))

    info = np.finfo(dtype) if np.dtype(dtype).kind == 'f' else np.iinfo(dtype)
    
    if np.any(A > info.max) or np.any(A < info.min):
        if np.issubdtype(dtype, np.floating):
            raise OverflowError(f"Value exceeds limits of {dtype}")
        raise ValueError(f"Value outside the representable range of {dtype}")
        
    return A.astype(dtype)

def correct_matrix(A: npt.NDArray) -> bool:
    '''Matrices that return false here are not supported by our algorithms.'''
    if not np.issubdtype(A.dtype, np.number):
        return False

    if (np.isnan(A) | np.isinf(A)).any():
        return False
    
    return True

def get_dtype_info(dtype) -> dict:
    """
    Custom wrapper to safely extract precision depth metadata.
    Handles native NumPy integers/floats and custom ml_dtypes (FP8 variations).
    Returns a dict with 'bits' and 'nmant'.
    """
    dt = np.dtype(dtype)
    
    # Check for custom ml_dtypes
    if dt.name.startswith('float8'):
        # Extract mantissa bits based on the FP8 variant standard
        if 'e4m3' in dt.name:
            nmant = 3
        elif 'e5m2' in dt.name:
            nmant = 2
        else:
            nmant = 3  # Fallback safe approximation
        return {'bits': 8, 'nmant': nmant}
        
    # Check for standard integers
    if np.issubdtype(dt, np.integer):
        bits = dt.itemsize * 8
        return {'bits': bits, 'nmant': bits - 1} # for consistency in bit-depth math
        
    # Check for standard floating points
    try:
        f_info = np.finfo(dt)
        return {'bits': dt.itemsize * 8, 'nmant': f_info.nmant}
    except ValueError:
        # Extreme fallback if something else custom arrives
        u_inv = 2 ** dt.alignment if hasattr(dt, 'alignment') else 8
        return {'bits': dt.itemsize * 8, 'nmant': int(math.log2(u_inv))}