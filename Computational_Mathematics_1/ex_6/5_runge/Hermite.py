"""
author        : Moritz Mossböck             | 11820925  | moritz.mossboeck@student.tugraz.at
file          : Hermite.py                  | UTF-8
target version: python 3.10.9               | 64-bit
course        : Computational Mathematics 1 | MAT.208UB
"""

# imports
from numpy import zeros, full_like, ndarray, float64, max, min, linspace
from matplotlib.axes import Axes
from matplotlib.lines import Line2D


def Hermite(x_data : ndarray, y_data : ndarray, y_prime : ndarray) -> ndarray:
    """
    Perform Hermite-Interpolation with respect to first derivatives

    Hermite interpolation is similar to newton's method, however instead of simply interpolation the values,
    we also restrain the interpolant to certain values of the derivative of the interpolant. Further 
    explanation can be found in the accompaniying submission.

    Parameters:
    -----------
    x_data : ndarray
        x-coordinates for the samples
    y_data : ndarray
        samples of the unknown function
    y_prime : ndarray
        samples of the derivative of the unknown function

    Returns:
    --------
    ndarray
        coefficients for the hermite polynomial

    Raises:
    -------
    ValueError
        if `x_data` is not sorted in ascending order
    ValueError
        if `x_data`, `y_data` or `y_prime` are of different length
    """

    if not all(x_data[:-1] < x_data[1:]):
        raise ValueError('x_data must be sorted in ascending order')
    
    if x_data.shape != y_data.shape:
        raise ValueError('x_data and y_data must be of same shape')

    n = len(x_data)

    xcol = zeros(2 * n, dtype=float64)
    col1 = zeros(2 * n, dtype=float64)
    col2 = zeros(2 * n - 1, dtype=float64)

    # setup for divided differences
    for i in range(0, 2*n-1, 2):
        col1[i] = y_data[int(i / 2)]
        xcol[i] = x_data[int(i/2)]
        col1[i+1] = y_data[int(i / 2)]
        xcol[i+1] = x_data[int(i/2)]
        col2[i] = y_prime[int(i / 2)]

    for i in range(1, 2*n-2,2):
        col2[i] = (col1[i+1] - col1[i]) / (xcol[i+1] - xcol[i])

    c = zeros(2*n)
    c[0] = col1[0]
    c[1:] = col2
    # regular divided differences, start with second column
    for k in range(2, 2*n):
        for i in reversed(range(k,2*n)):
            c[i] = (c[i] - c[i-1]) / (xcol[i] - xcol[i-k])

    return c, xcol

def Hermiteval(c : ndarray, xcol : ndarray, x : ndarray) -> ndarray:
    """
    Evaluate hermite-polynomial with coefficients `c`

    Parameters:
    -----------
    c : ndarray
        coefficients for the hermite polynomial
    xcol : ndarray
        column of x-nodes produced by divided differences for hermite interpolation
    x : ndarray
        values to evaluate interpolant on

    Returns:
    --------
    ndarray
        evaluation of interpolant on `x`
    
    """

    value = full_like(x, c[0])
    pfactor = (x - xcol[0])
    for k, (c,xk) in enumerate(zip(c[1:],xcol[1:])):
        value = value + c * pfactor
        pfactor = pfactor * (x - xk)
    return value


def HermiteInterp(x_data : ndarray, y_data : ndarray, y_prime : ndarray, x: ndarray, ax : Axes) -> Line2D:
    """
    Interpolate samples and plot the interpolant, print the value of H in `x`

    Parameters:
    -----------
    x_data : ndarray
        x-coordinates for the samples
    y_data : ndarray
        samples of the unknown funtion
    y_prime : ndarray
        samples of the derivative of the unknown function
    x : ndarray
        values to evaluata interpolant in
    ax : Axes
        axes-instance to plot on
    
    Returns:
    --------
    Line2D:
        the return value of the `ax.plot` call, in order to allow further modifications to the plot
    """
    c, xcol = Hermite(x_data, y_data, y_prime)
    axis = linspace(min(x_data), max(x_data), 1000)
    p = Hermiteval(c, xcol, axis)
    print(f'p({x}) = {Hermiteval(c, xcol, x)}')
    return ax.plot(axis, p)
    