"""
author             : Moritz Mossböck
file               : util.py
development-version: python 3.10.9 (64-bit)

Provide commonly used utility functions for multiple areas
"""

from typing import Tuple
from numpy import ndarray, pad

def adapt_coefficients(array1: ndarray, array2: ndarray) -> Tuple[ndarray, ndarray]:
    """
    Bring two arrays to same length and pad the shorter one with zeros

    Parameters:
    -----------
    array1 : ndarray
        first array to check / pad
    array2 : ndarray
        second array to check / pad

    Returns:
    --------
    Tuple[ndarray, ndarray]
        the adapted arrays
    """
    len1 = len(array1)
    len2 = len(array2)
    rval1 = pad(array1, (0, abs(len1 - max(len1, len2))), 'constant')
    rval2 = pad(array2, (0, abs(len2 - max(len1, len2))), 'constant')
    return rval1, rval2


def get_sign(number: float) -> str:
    """
    Get a string representation of the sign of the value

    Parameters:
    -----------
    number : float
        the number to get the sign of

    Returns:
    --------
    str
        sign of `number`
    """
    rval = ''
    if number > 0:
        rval = '+'
    elif number < 0:
        rval = '-'
    elif number == 0:
        rval = ''
    return rval


def polystring(coeffs: ndarray) -> str:
    """
    Get a string representation of the polynomial with coefficients from `coeffs`

    Parameters:
    -----------
    coeffs : ndarray
        the coefficients of the polynomial in the order [c0, c1, ..., cn]

    Returns:
    --------
    str
        string of the polynomial expression in the form c0 + c1 x + c2 x^2 + ... + cn x^n
    """
    rval = ''
    sign = '+'
    for i, val in enumerate(coeffs):
        if val != 0:
            sign = get_sign(val)
            val = abs(val)
            if i == 0:
                if sign == '-':
                    rval += f'{sign}{val} '
                else:
                    rval += f'{val} '
                continue
            if i == 1:
                if val == 1:
                    rval += f'{sign} x '
                else:
                    rval += f'{sign} {val}x '
            if i == len(coeffs) - 1:
                if val == 1:
                    rval += f'{sign} x^{i}'
                else:
                    rval += f'{sign} {val}x^{i}'
            elif i > 1 and i < len(coeffs) - 1:
                if val == 1:
                    rval += f'{sign} x^{i} '
                else:
                    rval += f'{sign} {val}x^{i} '
    return rval
