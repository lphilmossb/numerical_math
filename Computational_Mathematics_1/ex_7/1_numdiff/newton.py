"""
author        : Moritz Mossböck             | 11820925  | moritz.mossboeck@student.tugraz.at
file          : Newton.py                   | UTF-8
target version: python 3.10.9               | 64-bit
course        : Computational Mathematics 1 | MAT.208UB
pylint-score  : 10 / 10

Provide functions for computing newton-interpolant coefficients and evaluate interpolant
over axis.
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

    if not all(x_data[:-1] < x_data[1:]):
        raise ValueError('x_data must be sorted in ascending order')

    if len(y_data) != len(x_data):
        raise ValueError('x-nodes and y-values must have same dimension')

    length = len(x_data) - 1
    coeffs = array(y_data).astype(float64)
    for k in range(1,length+1):
        for i in reversed(range(k,length+1)):
            coeffs[i] = (coeffs[i] - coeffs[i-1]) / (x_data[i] - x_data[i-k])

    return coeffs

def horner(coeffs : ndarray, axis : ndarray, x_data : ndarray) -> ndarray:
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

    rval = coeffs[-1]
    for (coeff, node) in zip(flip(coeffs[:-1]), flip(x_data[:-1])):
        rval = coeff + (axis - node) * rval
    return rval
