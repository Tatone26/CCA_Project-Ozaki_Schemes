import numpy as np
import pytest
import mpmath as mp

from src.ozaki_1_f64 import split_matrix, full_products, perfect_summation
from src.baseline import get_exact_product

@pytest.mark.parametrize("d", [2, 4, 7, 10, 20, 26])
@pytest.mark.parametrize("scale", [1.0, 16.0, 1024.0, 4096.0, 1e-10, 1e-25])
def test_splitting_reconstruction(d, scale):
    '''
    Verifying that the ozaki-1 split does not lose any information.
    
    :param d: size (in bits) of sub-matrix
    :param scale: to test different values for sigma
    '''

    # A is a complex, full matrix.
    A = np.array([[np.pi, np.e], [np.sqrt(2), 1.0/7.0]], dtype=np.float64) * scale
    sigma = 2.0**np.frexp(np.max(np.abs(A)))[1]

    splits = split_matrix(A, sigma, d)

    reconstructed = np.sum(splits, axis = 0)
    np.testing.assert_array_equal(reconstructed, A, err_msg=f"Reconstruction failed with d={d} and scale={scale}")

def test_sigma_value_error():
    '''
    Making sure that we don't let a wrong sigma get through
    '''
    A = np.array([[100.0]])
    sigma = np.float64(64.0) # too small
    with pytest.raises(ValueError):
        split_matrix(A, sigma, 4)
    sigma = np.float64(62.0) # not a power of two
    with pytest.raises(ValueError):
        split_matrix(A, sigma, 4)


def test_multiplication():
    '''Checking if there is no loss of information during products'''
    scale : np.float64 = np.float64(2.0**10)
    A = np.array(np.random.random((5, 5)), dtype=np.float64) * scale
    B = np.array(np.random.random((5, 5)), dtype=np.float64) * scale

    sigma = scale # bigger than any number in A or B
    d = 25 # max without losing precision (fl(26.5 - log2(5)) = 25)
    A_s = split_matrix(A, sigma, d)
    B_s = split_matrix(A, sigma, d)

    perfect_products = [get_exact_product(i, j, 256) for i in A_s for j in B_s]
    auto_products = full_products(A_s, B_s)

    for (x, y) in zip(perfect_products, auto_products):
        np.testing.assert_array_equal(x, y)
    

def test_null_matrix_splitting():
    '''
    Making sure a null matrix does not cause any problem
    '''
    A = np.zeros((3, 3)) # size doesn't change anything
    splits = split_matrix(A, np.float64(2.0**2), 1)
    assert len(splits) == 1
    np.testing.assert_equal(splits[0], A, err_msg="null matrix modified by splitting.")

