"""
author        : Moritz Mossböck | 11280925 | moritz.mossboeck@student.tugraz.at 
file          : Polynomial.py   | UTF-8
target version: Python 3.10.8   | 64-bit
creation date : 03.12.2022
course        : Computational Mathematics 1
"""

# ----------------------------------------------------------------------------------------------------------------------
#                                                   IMPORT SECTION
# ----------------------------------------------------------------------------------------------------------------------
from __future__ import annotations  # just for return-type hinting in some methods
from numpy import ndarray, pad, flip, trim_zeros, float64, polyval, zeros, array
from typing import Iterator, Optional, Tuple

# ----------------------------------------------------------------------------------------------------------------------
#                                                       CLASS
# ----------------------------------------------------------------------------------------------------------------------


class Polynomial:
    def __init__(self, coefficients: ndarray) -> None:
        self.coefficients = trim_zeros(array(coefficients).astype(float64), 'b')

    # special methods
    def __call__(self, x: Optional[float | ndarray] = 0.0) -> float | ndarray:
        return polyval(flip(self.coefficients), x)

    def __iter__(self) -> Iterator[float]:
        for a in self.coefficients:
            yield a

    def __str__(self) -> str:
        return polystring(self.coefficients)

    # properties

    @property
    def degree(self) -> int:
        return len(self.coefficients)

    @property
    def latex_string(self) -> str:
        rval = str(self).replace(
            '*', '').replace('  +  ', '+').replace('  -  ', '-')
        return rf'${rval}$'

    # arithmetic

    def __add__(self, other: Polynomial) -> Polynomial:
        c1, c2 = adapt_coefficients(self.coefficients, other.coefficients)
        return Polynomial(c1 + c2)

    def __sub__(self, other: Polynomial) -> Polynomial:
        c1, c2 = adapt_coefficients(self.coefficients, other.coefficients)
        return Polynomial(c1 - c2)

    def __mul__(self, other: Polynomial) -> Polynomial:
        m = len(self.coefficients) + len(other.coefficients) - 1
        ck = zeros(m)
        for i, ak in enumerate(self.coefficients):
            for j, bk in enumerate(other.coefficients):
                ck[i + j] += ak * bk
        return Polynomial(ck)

    def __neg__(self) -> Polynomial:
        return Polynomial(-self.coefficients)

    def __eq__(self, other: Polynomial) -> bool:
        return self.coefficients == other.coefficients


# ----------------------------------------------------------------------------------------------------------------------
#                                                  HELPER FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------

def adapt_coefficients(c1: ndarray, c2: ndarray) -> Tuple[ndarray, ndarray]:
    n = len(c1)
    m = len(c2)
    return pad(c1, (0, abs(n - max(n, m))), 'constant'), pad(c2, (0, abs(m - max(n, m))), 'constant')


def get_sign(a: float) -> str:
    if a > 0:
        return '+'
    elif a < 0:
        return '-'


def polystring(coeffs: ndarray) -> str:
    rval = ''
    sign = '+'
    for i, a in enumerate(coeffs):
        if a != 0:
            sign = get_sign(a)
            a = abs(a)
            if i == 0:
                if sign == '-':
                    rval += f'{sign}{a} '
                else:
                    rval += f'{a} '
                continue
            if i == 1:
                if a == 1:
                    rval += f'{sign} x '
                else:
                    rval += f'{sign} {a}x '
            if i == len(coeffs) - 1:
                if a == 1:
                    rval += f'{sign} x^{i}'
                else:
                    rval += f'{sign} {a}x^{i}'
            elif i > 1 and i < len(coeffs) - 1:
                if a == 1:
                    rval += f'{sign} x^{i} '
                else:
                    rval += f'{sign} {a}x^{i} '
    return rval
