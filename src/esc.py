'''
This file contains the implementation of the ESC (Exponent Span Capacity) method,
used to determine the number of slices needed in the Ozaki-1 algorithm to
get to wanted matrix product precision.
'''


import numpy as np
import numpy.typing as npt
import warnings

def dot_product_esc(x: npt.NDArray, y: npt.NDArray, z: npt.NDArray, margin: int=1) -> np.float64:
    '''
    Computes the ESC for a given dot product, with x and y the input vectors and z the true - or estimated - output hadamar product vector

    The computations done here can be considered free, as they work independently on the three vectors. 

    :param x: first input vector
    :type x: npt.NDArray
    :param y: second input vector
    :type y: npt.NDArray
    :param z: estimated hadamar product output vector ; may not be of the same size as the others. Is a vector of EXPONENTS
    :type z: npt.NDArray
    :param margin: added to the final ESC sum to make equation true. Should be 1, or maybe 3...
    :type margin: int
    :return: the ESC value
    :rtype: np.float64
    '''
    if x.ndim != 1 or y.ndim != 1 or z.ndim != 1:
        warnings.warn(f"The three parameters must be 1D vectors. Got dims x={x.ndim}, y={y.ndim}, z={z.ndim}.", stacklevel=2)
    if x.size != y.size:
        warnings.warn(f"Vectors x and y should have same size but got x={x.size}, y={y.size}.", RuntimeWarning, stacklevel=2)

    exp_x = np.max(np.float64(np.frexp(x)[1] if x.size > 0 else 0.0))
    exp_y = np.max(np.float64(np.frexp(y)[1] if y.size > 0 else 0.0))

    #exp_z = np.max(np.float64(np.frexp(z)[1] if z.size > 0 else 0.0)) # this would be the code if z was given as an array of values instead of an array of exponents
    exp_z = np.max(z) 
    
    return np.float64(exp_x + exp_y - exp_z + margin)

def estimate_hadamard_exponent_range(x: npt.NDArray, y: npt.NDArray, b: int=1) -> npt.NDArray:
    '''
    Estimate the hadamar product range of x and y by computing block by block. 
    Returns a vector corresponding to the maximal possible exponent for each block, with max(exp(block_x)) + min(exp(block_y)) or the opposite.
    If b isn't set, it will default to 1, meaning true (not useful) hadamard product.

    :param x: Input vector x
    :type x: npt.NDArray
    :param y: Input vector y
    :type y: npt.NDArray
    :param b: size of block
    :type b: int
    :return: Estimated exponent range for each block hadamard product
    :rtype: npt.NDArray
    '''
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
        raise ValueError(f"x and y are not valid vectors ; they may not be 1D vectors (x={x.ndim}, y={y.ndim}) or not of the same size (x={x.size}, y={y.size})")

    _, exps_x = np.frexp(x)
    _, exps_y = np.frexp(y)

    maxs_x = [np.max(exps_x[i : i + b]) for i in range(0, x.size, b)]
    mins_x = [np.min(exps_x[i : i + b]) for i in range(0, x.size, b)]
    
    maxs_y = [np.max(exps_y[i : i + b]) for i in range(0, y.size, b)]
    mins_y = [np.min(exps_y[i : i + b]) for i in range(0, y.size, b)]

    z = [max(maxs_x[i] + mins_y[i], mins_x[i] + maxs_y[i]) for i in range(len(maxs_x))]
    
    return np.array(z, dtype=np.float64)


def esc(A: npt.NDArray, B: npt.NDArray, b: int=1, margin: int=3) -> np.float64:
    """
    Compute the exponential synchronization criterion (ESC).

    :param A: First input matrix
    :type A: npt.NDArray
    :param B: Second input matrix
    :type B: npt.NDArray
    :param b: Block size of computation. Defaults to 1, meaning non-coarsened version.
    :type b: int
    :return: The maximum ESC value of all dot products.
    :rtype: np.float64
    """
    ''''''
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"The A * B product is impossible as A and B have incompatible sizes : A={A.shape}, B={B.shape}")
    
    res = []
    # for every scalar product do ESC estimation
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            # Get worst case scenario for each block
            z: npt.NDArray = estimate_hadamard_exponent_range(A[i, :], B[:, j], b)
            # Calculate ESC based on max(z), max(A), max(B)
            rb: np.float64 = dot_product_esc(A[i, :], B[:, j], z, margin=margin)
            res.append(rb)
    # return max ESC found
    return np.float64(max(res))


def esc_to_slices(esc: np.float64, d: int, u: int) -> int:
    '''Returns the number of slices corresponding to the previously calculated ESC value.'''
    return int(np.ceil(float(u + esc) / float(d)))