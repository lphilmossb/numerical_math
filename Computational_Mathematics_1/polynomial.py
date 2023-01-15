"""
author             : Moritz Mossböck
file               : Polynomial.py
development-version: python 3.10.9 (64-bit)


Provides the `Polynomial` class
"""

# imports
from __future__ import annotations # allow type hints for class in class-methods
from typing import Iterator
from numpy import ndarray, array, float64
from numpy import trim_zeros, flip, full_like, zeros, any as npany, eye, allclose
from util import polystring, adapt_coefficients


class Polynomial:
    """
    Simple class for handling polynomials

    Attributes:
    -----------
    _coeffs : ndarray
        coefficients of the polynomial in the order [c0, c1, ..., cn]


    Examples:
    ---------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> c = [1, 0, 1]
    >>> poly = Polynomial(c)
    >>> x = np.linspace(-1, 1, 500)
    >>> y = poly(x)
    >>> plt.plot(x, y, label=poly.latex_string())
    >>> plt.legend()
    >>> plt.show()
    """

    def __init__(self, coefficients: list) -> None:
        """
        Construct a `Polynomial` instance

        Notes:
        ------

        The passed list / ndarray `coefficients` may contain trailing zeros from automatic
        coefficient setup. To avoid unnecessary computations, trailing zeros are
        trimmed from `coefficients`.
        """
        self._coeffs = trim_zeros(array(coefficients, dtype=float64), 'b')
        self._deg = len(self._coeffs)

    def __call__(self, axis: ndarray) -> ndarray:
        """
        Evaluate the polynomial defined by the instance coefficients using horner's method

        Due to the fact, that the iteration variable `xk` is of the same shape as `axis`
        vectorized operation is guaranteed. Additionally even evaluation in matrices is
        possible.

        Parameters:
        -----------
        axis : ndarray
            x-values to evaluate over

        Returns:
        --------
        ndarray
            polynomial evaluated on `axis`

        Raises:
        -------
        ValueError
            if `axis` is `None` or any `axis[i] is None`

        Examples:
        ---------
        >>> import numpy as np
        >>> c = [2, 1, 1]
        >>> poly = Polynomial(c)
        >>> x = np.eye(3)
        >>> poly(x)
        [[4. 0. 0.]
         [0. 4. 0.]
         [0. 0. 4.]]
        """

        if axis is None or npany(axis is None):
            raise ValueError('passed x-values is/contains None')

        xk = full_like(axis, self._coeffs[-1])
        for c in flip(self._coeffs[:-1]):
            xk = axis * xk + c * eye(axis.shape[0])
        return xk

    def __str__(self) -> str:
        """
        Get a simple string representation of the represented polynomial

        The resulting string is typically of the following form:

            c0 + c1 x + c2 x^2 + ... + cn x^n

        Notice that special cases such as ck = 1 or ck = 0 are taken into account

        See Also:
        ---------
        polystring : return the string representation of a polynomial from an array
        """
        return polystring(self._coeffs)

    def __iter__(self) -> Iterator[float64]:
        """
        Iterate over the coefficient array of the instance

        Returns:
        --------
        Iterator[float64]
            iterator-object for `self._coeffs`
        """
        for c in self._coeffs:
            yield c

    def __add__(self, other: Polynomial) -> Polynomial:
        """
        Create a new polynomial representing the sum of two other polynomials

        Since any two polynomials can be added by just adding the coefficients, we can create
        the sum-polynomial very easily.

        Parameters:
        -----------
        other : Polynomial
            summand for the sum

        Returns:
        --------
        Polynomial
            sum of `self` and `other`

        """
        c1, c2 = adapt_coefficients(self._coeffs, other._coeffs)
        return Polynomial(c1 + c2)

    def __sub__(self, other: Polynomial) -> Polynomial:
        """
        Create a new polynomial representing the difference of two other polynomials

        Very similar to `__add__()`, except that we form the difference of the coefficient vectors

        Parameters:
        -----------
        other : Polynomial
            subtrahend of the difference

        Returns:
        --------
        Polynomial
            difference of `self` and `other`

        """
        c1, c2 = adapt_coefficients(self._coeffs, other._coeffs)
        return Polynomial(c1 - c2)

    def __mul__(self, other: Polynomial) -> Polynomial:
        """
        Create a new polynomial representing the product of two other polynomials

        The product of two polynomials of degree n and m is of degree `n + m - 1`.

        Parameters:
        -----------
        other : Polynomial
            factor for the product

        Returns:
        --------
        Polynomial
            product of `self` and `other`

        """
        deg = len(self._coeffs) + len(other._coeffs) - 1
        ck = zeros(deg)
        for i, ak in enumerate(self._coeffs):
            for j, bk in enumerate(other._coeffs):
                ck[i + j] += ak * bk
        return Polynomial(ck)

    def __neg__(self) -> Polynomial:
        """
        Get negative polynomial by multiplying coefficients with -1
        """
        return Polynomial(-self._coeffs)

    def __eq__(self, other: Polynomial) -> bool:
        """
        Check for equality via coefficients

        Parameters:
        -----------
        other : Polynomial
            polynomial to check equality for

        Returns:
        --------
        bool
            `True` if the coefficients of both polynomials are sufficiently close
        """
        coeffs_1, coeffs_2 = adapt_coefficients(self._coeffs, other._coeffs)
        return allclose(coeffs_1, coeffs_2)

    def latex_string(self) -> str:
        """
        Get a texifyd version of `__str__`

        Returns:
        --------
        str
            latex-formatted string representation of the polynomial
        """
        rval = str(self).replace('*', '').replace('  +  ', '+').replace('  -  ', '-')
        return rf'${rval}$'
