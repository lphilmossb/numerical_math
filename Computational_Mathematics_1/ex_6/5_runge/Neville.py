"""
author        : Moritz Mossböck             | 11820925  | moritz.mossboeck@student.tugraz.at
file          : Neville.py                  | UTF-8
target version: python 3.10.9               | 64-bit
course        : Computational Mathematics 1 | MAT.208UB
"""

# imports
from numpy import ndarray, float64, array, sign, all

def Neville(x_data: ndarray, y_data: ndarray, x: ndarray) -> ndarray:
    """
    Perform neville-interpolation (iteratively)

    Parameters:
    -----------
    x_data : ndarray
        x-coordinates for the samples
    y_data : ndarray
        samples of the unknown function
    x : ndarray
        x-values to evaluate the interpolant on

    Returns:
    --------
    ndarray:
        the interpolant evaluated on `x`

    Raises:
    -------
    ValueError
        if `x_data` is not sorted in ascending order
    ValueError
        if `y_data` and `x_data` are of different shape
    """
    x_data = array(x_data)
    y_data = array(y_data)

    if not all(x_data[:-1] < x_data[1:]):
        raise ValueError('x_data must be sorted in ascending order')
    
    if x_data.shape != y_data.shape:
        raise ValueError('x_data and y_data must be of same shape')

    n = len(x_data)
    p = array(y_data, dtype=object)  # i'm unhappy about this :/

    for k in range(1, n):
        for i in range(n-k):
            p[i] = ((x - x_data[i+k]) * p[i] + (x_data[i] - x) * p[i+1]) / (x_data[i] - x_data[i+k])
    return p[0]


def InverseNeville(x_data: ndarray, y_data: ndarray, i: int, k: int) -> float64:
    """
    Inverse interpolation using neville's method for finding roots

    Inverse interpolation is the process of finding x-values where the unknown function f equals a desired 
    y-value. This function specifically searches roots in a single interval given by two adjacent x-nodes, 
    `xl` (the „left“ x-node) and `xr` (the „right“ x-node), between which a sign change occurs in the samples of 
    `y_data`.

    The following algorithm is based on the bisection method. However instead of using f, which is unknown, we 
    linearly interpolate it using `Neville()` between `xl` and `xr` and solve the linear equation. Then with the 
    found x-value, we compute the y-value using all available nodes.

    Parameters:
    -----------
    x_data : ndarray
        x-coordinates for the samples
    y_data : ndarray
        samples of the unknown function
    i : int
        index of the left x-value (i.e. before the sign change)
    k : int
        number of iterations

    Returns:
    --------
    float
        the found x-coordinate for the root

    Raises:
    -------
    ValueError
        if no sign-change occurs between `xl` and `xr`

    """

    if sign(y_data[i]) == sign(y_data[i+1]) or sign(y_data[i]) == 0 or sign(y_data[i+1]) == 0:
        raise ValueError('specified interval does not contain zero of f')

    yl = y_data[i]
    yr = y_data[i+1]

    xl = x_data[i]
    xr = x_data[i+1]
    xz = (xl*yr - xr*yl) / (yr - yl)

    for l in range(k):
        xz = (xl*yr - xr*yl) / (yr - yl)
        yz = Neville(x_data, y_data, xz)

        if yz > 0: # das some if-statements
            if yl > 0:
                yl = yz
                xl = xz
            elif yr > 0 :
                yr = yz
                xr = xz
        elif yz < 0.0:
            if yl < 0.0:
                yl = yz
                xl = xz
            elif yr < 0.0:
                yr = yz
                xr = xz

    return xz


