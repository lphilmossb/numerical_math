"""
author        : Moritz Mossböck             | 11820925  | moritz.mossboeck@student.tugraz.at
file          : ncote.py                    | UTF-8
target version: python 3.10.9               | 64-bit
course        : Computational Mathematics 1 | MAT.208UB
pylint-score  : 9.59 / 10 (too few public class methos for `NCotes` and `SummedNCotes`)

Provide classes for numerical integration with newton-cotes quadrature
"""

from typing import Optional, Callable
from numpy import array, sum as nsum, multiply

class NCotes:
    """
    Integrate any function over [`start`, `stop`] using newton-cotes quadrature

    The `__weights_i` „static“ arrays store the integration weights, while `__factors`
    contains constant factors factored from the quadrature rule.

    Attributes:
    -----------
    __weights_1 : ndarray, static
        weights for newton-cotes quadrature of order 2
    __weights_2 : ndarray, static
        weights for newton-cotes quadrature of order 3
    __weights_3 : ndarray, static
        weights for newton-cotes quadrature of order 4
    __weights_4 : ndarray, static
        weights for newton-cotes quadrature of order 5
    __weights : ndarray, static
        collection of weights for quick-access without if-statement
    __factors : ndarray, static
        common factors for the quadrature rules
    _a : float
        start point of integration interval
    _b : float
        end point of integration interval
    _n : int
        quadrature order
    _h : float
        step size between nodes
    _steps : ndarray
        x-nodes used for quadrature
    _weights : ndarray
        weights used for quadrature rule
    _factor : float
        common factor from quadrature rule
    """

    __weights_1 = [1.0, 1.0]
    __weights_2 = [1.0, 4.0, 1.0]
    __weights_3 = [3.0, 9.0, 9.0, 3.0]
    __weights_4 = [14.0, 64.0, 24.0, 64.0, 14.0]
    __weights = [__weights_1, __weights_2, __weights_3, __weights_4]
    __factors = [2.0, 3.0, 8.0, 45.0]

    def __init__(self, start : float, stop : float, order : Optional[int] = 1) -> None:
        self._a = start
        self._b = stop
        self._n = order
        self._h = (stop-start) / order
        self._steps = array([start + self._h * k for k in range(order+1)])
        self._weights = NCotes.__weights[order-1]
        self._factor = 1.0 / NCotes.__factors[order-1]

    def integrate(self, func : Callable[[float], float]) -> float:
        """
        Integrate a callable function over the instance interval

        Parameters:
        -----------
        func : Callable[[float], float]
            function to integrate

        Returns:
        --------
        float
            quadrature of `func`
        """
        return self._h * self._factor * nsum(multiply(func(self._steps), self._weights))


class SummedNCotes:
    """
    Class for handling composite newton-cotes quadratures, similar to `NCotes`

    Attributes:
    -----------
    _a : float
        start of integration interval
    _b : float
        stop of integration interval
    _n : int
        quadrature order
    _num_int : int
        number of part-intervals
    _h : float
        size of part-intervals
    _steps : ndarray
        x-nodes for quadrature
    """

    def __init__(self, start : float, stop : float, order : Optional[int] = 1,
                       num_int : Optional[int] = 1) -> None:
        self._a = start
        self._b = stop
        self._n = order
        self._num_int = num_int
        self._h = (stop-start) / num_int
        self._steps = array([start + self._h * k for k in range(num_int+1)])

    def integrate(self, func : Callable[[float], float]) -> float:
        """
        Compute composite newton-cotes quadrature of `funct`

        Parameters:
        -----------
        func : Callable[[float], float]
            function to approximate integral of

        Returns:
        --------
        float
            composite newton-cotes quadrature of `func`
        """
        rval = 0.0
        fdata = func(self._steps)

        if self._n == 1:
            rval = 0.5 * self._h * ( fdata[0] + fdata[-1] + 2.0 * sum(fdata[1:-1]) )
        elif self._n == 2:
            psum_1 = 2 * nsum(fdata[1::2])
            psum_2 = 4 * nsum(fdata[2::2])
            rval = self._h / 3.0 * ( fdata[0] + fdata[-1] + psum_1 + psum_2 )
        elif self._n == 3:
            psum_1 = 2 * nsum(fdata[3::3])
            psum_2 = 3 * nsum(fdata[1::3])
            psum_3 = 3 * nsum(fdata[2::3])
            rval = fdata[0] + fdata[-1] + psum_1 + psum_2 + psum_3
            rval *= 3 * self._h / 8
        elif self._n == 4:
            psum_1 = 32 * nsum(fdata[1::4])
            psum_2 = 32 * nsum(fdata[3::4])
            psum_3 = 12 * nsum(fdata[2::4])
            rval = 7 * fdata[0] + 7 * fdata[-1] + psum_1 + psum_2 + psum_3
            rval *= 2 * self._h / 45

        return rval
