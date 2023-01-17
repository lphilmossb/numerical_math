"""
author             : Moritz Mossböck
file               : power_series.py
development-version: python 3.10.9 (64-bit)


Provides the `PowerSeries` class
"""

from typing import Callable, Optional, Iterator, Generator
from itertools import islice
from numpy import ndarray, array, zeros_like, complex64, eye
from util import get_axis_I

class PowerSeries:
    __gen_max = 100

    def __init__(self, sequence : Generator[complex, None, None], 
                 devpoint : Optional[complex], **kwargs) -> None:
        self._seq = sequence
        self._dev = devpoint
        self._precomp = kwargs.get('precomp', 0)

        if self._precomp > 0:
            self._seqvals = array(islice(self._seq, self._precomp))
        
    
    def __call__(self, axis : ndarray, summands : Optional[int] = 20) -> complex:
        rval = zeros_like(axis, dtype=complex64)
        i_matrix = get_axis_I(axis)

        if self._precomp > 0:
            for k, ak in enumerate(self._seqvals):
                rval += ak * (axis - self._devpoint * i_matrix) ** k
        
        for k in range(self._precomp, summands):
            rval += next(self._seq()) * (axis - self._devpoint * i_matrix) ** k
            if k == PowerSeries.__gen_max:
                break

        return rval