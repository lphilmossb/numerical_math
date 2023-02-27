"""
author        : Moritz Mossböck             | 11820925  | moritz.mossboeck@student.tugraz.at
file          : numdiff.py                  | UTF-8
target version: python 3.10.9               | 64-bit
course        : Computational Mathematics 1 | MAT.208UB
pylint-score  : 10 / 10

Provide functions for approximating the first and second derivative of a function, whose
values are given in an array `y_data` sampled at `x_data`.
"""

# imports
from typing import Optional, Union
from numpy import where, isclose, ndarray, average, linspace
from newton import compute_coeffs, horner


def dfdx(y_data: ndarray, x_data: ndarray,
         x_values: Union[ndarray, float]) -> Union[ndarray, float]:
    """
    Compute the second order forward finite difference approximation of f'

    Parameters:
    -----------
    y_data : ndarray
        the sampled values of the unknown function
    x_data : ndarray
        the sample-points
    x_values : Union[ndarray, float]
        x-value(s) to approximate first derivative in

    Returns:
    --------
    Union[ndarray, float]
        the approximation of the first derivative

    Notes:
    ------
    The sample points in `x_data` are assumed to be equidistant.
    The second order finite difference approximation is given by:

        f'(x) = 1/(2h) * (4 * f(x + h) - 3 * f(x) - f(x + 2h))
    """
    x_index = where(isclose(x_data[:-2], x_values))[0][0]
    step = x_data[1] - x_data[0]
    return 1.0 / (2*step) * (4 * y_data[x_index+1] - 3*y_data[x_index] - y_data[x_index + 2])


def df2dx(y_data: ndarray, x_data: ndarray,
          x_values: Union[ndarray, float]) -> Union[ndarray, float]:
    """
    Compute the second order forward finite difference approximation of f''

    Parameters:
    -----------
    y_data : ndarray
        the sampled values of the unknown function
    x_data : ndarray
        the sample-points
    x_values : Union[ndarray, float]
        x-value(s) to approximate second derivative in

    Returns:
    --------
    Union[ndarray, float]
        the approximation of the second derivative

    Notes:
    ------
    The sample points in `x_data` are assumed to be equidistant.
    The second order finite difference approximation is given by:

        f''(x) = 1/(h**2) * (f(x + 2h) - 2 * f(x + h) + f(x))
    """
    step = x_data[1] - x_data[0]
    x_index = where(isclose(x_data[1:-1], x_values+step))[0][0]
    return 1.0 / (step**2) * (y_data[x_index + 2] - 2* y_data[x_index + 1] + y_data[x_index])


def inter_dfdx(y_data: ndarray, x_data: ndarray, x_values: Union[ndarray, float],
               order: Optional[int] = 1) -> Union[ndarray, float]:
    """
    Approximate the derivative of a function via Newton-Interpolation

    Using the function values from `y_data` and the sample-points `x_data`, we interpolate the
    function with newton-interpolation, and approximate f', or f'' respectively, by computing the
    second order finite difference approximations given the y-values from the interpolant.

    Parameters:
    -----------
    y_data : ndarray
        sampled values of the unknown function
    x_data : ndarray
        sample points
    x_values : Union[ndarray, float]
        x-value(s) to approximate the derivative in
    order : int, default = 1
        order of the derivative, choose between first and second (`order == 2`)

    Returns:
    --------
    Union[ndarray, float]
        approximation of the derivative

    Notes:
    ------
    In contrast to `dfdx` and `df2dx`, the sample-points need not be equidistant, as the
    finite difference approximation is computed with new-sample data based on the interpolant.
    """
    newton_coeffs = compute_coeffs(y_data, x_data)
    step = average(x_data[1:] - x_data[:-1])
    length = 1
    x_new = None
    if isinstance(x_values, list):
        length = len(x_values)
        step = x_values[1] - x_values[0]
        x_new = linspace(x_values[0], x_values[-1] + 2*step, length + 2)
    else:
        x_new = linspace(x_values, x_values + 2*step, 3)

    interpolant = horner(newton_coeffs, x_new, x_data)
    der = None
    if order == 2:
        der = df2dx(interpolant, x_new, x_new)
    else:
        der = dfdx(interpolant, x_new, x_new)
    return der
