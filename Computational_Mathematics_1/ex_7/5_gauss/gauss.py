"""
author        : Moritz Mossböck             | 11820925  | moritz.mossboeck@student.tugraz.at
file          : gauss.py                    | UTF-8
target version: python 3.10.9               | 64-bit
course        : Computational Mathematics 1 | MAT.208UB
pylint-score  : 9.33 / 10 (too few public methods for `GaussLegendre`)

Provide class for computing Gauss-Legendre quadrature
"""

# imports
from typing import Callable, Optional
from numpy import array, sqrt, isclose, sum as nsum, multiply


class GaussLegendre:
    """
    Class for computing Gauss-Legendre quadrature

    Attributes:
    -----------
    __nodes : list[list[float]], static
        list of nodes used for GL-quadrature
    __weigts : list[list[float]], static
        list of weights for GL-quadrature
    _order : int
        order of GL quadrature
    _nodes : ndarray
        nodes used for chosen order
    _weights : ndarray
        weights used for chosen order
    """

    __nodes   = [[0.0], [-sqrt(3)/3, sqrt(3)/3], [-sqrt(3/5), 0, sqrt(3/5)],]
    __weights = [[2], [1,1], [5/9, 8/9, 5/9], ]


    def __init__(self, order : Optional[int] = 2) -> None:
        self._order = order
        self._nodes = array(GaussLegendre.__nodes[order-1])
        self._weights = array(GaussLegendre.__weights[order-1])


    def integrate(self, func : Callable[[float], float], start : Optional[float] = -1.0,
                  stop : Optional[float] = 1.0) -> float:
        """
        Compute GL-quadrature of `func`

        Parameters:
        -----------
        func : Callable[[float], float]
            function to approximate integral of
        start : Optional[float], defaul = -1.0
            start of integration interval
        stop : Optional[float], default = 1.0
            stop of integration interval

        Returns:
        --------
        float
            GL-quadrature approximation of integral over [`start`,`stop`] of `func`
        """
        if isclose(start, stop):
            raise ValueError('integration interval is too small')
        transform =  (stop - start) / 2 * self._nodes + (start + stop) / 2
        f_values = func(transform)

        return (stop - start) /2 *  nsum(multiply(f_values, self._weights))
