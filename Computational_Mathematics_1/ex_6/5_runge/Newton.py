"""
author        : Moritz Mossböck             | 11820925  | moritz.mossboeck@student.tugraz.at
file          : Newton.py                   | UTF-8
target version: python 3.10.9               | 64-bit
course        : Computational Mathematics 1 | MAT.208UB
"""

# imports
from numpy import ndarray, flip, float64, array

def compute_coeffs(y_data : ndarray, x_data : ndarray) -> ndarray:
    """
    Compute the coefficients for newton-interpolation using divided differences


    Parameters:
    -----------
    y_data : ndarray
        samples of the unknown function
    x_data : ndarray
        sample-points (times) for the passed samples

    Returns:
    --------
    ndarray:
        the coefficients for the newton-interpolation formula

    Raises:
    -------
    ValueError
        if `x_data` is not sorted in ascending order
    ValueError
        if `x_data` and `y_data` are not of the same length
    """

    for i,x in enumerate(x_data[:-1]):
        if x > x_data[i+1]:
            raise ValueError('x-nodes must be sorted in increasing order')

    if len(y_data) != len(x_data):
        raise ValueError('x-nodes and y-values must have same dimension')

    n = len(x_data) - 1
    c = array(y_data).astype(float64)
    for k in range(1,n+1):
        for i in reversed(range(k,n+1)):
            c[i] = (c[i] - c[i-1]) / (x_data[i] - x_data[i-k])

    return c

def horner(coeffs : ndarray, x : ndarray, x_data : ndarray) -> ndarray:
    """
    Adapted horner's method for evaluating the newton interpolant

    Parameters:
    -----------
    coeffs : ndarray
        array of coefficients for newton interpolation
    x : ndarray
        x-values to evaluate the interpolant on
    x_data : ndarray
        x-coordinates of the samples

    Returns:
    --------
    ndarray
        the interpolant evaluated on `x`

    Raises:
    -------
    ValueError
        if `x_data` and `coeffs` are of different lengths

    """
    if len(coeffs) != len(x_data):
        raise ValueError('mismatched number of coefficients and nodes')

    p = coeffs[-1]
    for (c,xk) in zip(flip(coeffs[:-1]), flip(x_data[:-1])):
        p = c + (x - xk) * p
    return p