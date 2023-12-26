"""
    author: Laura Mossböck | 11820925 @ TUGraz | laura.mossboeck@gmail.com
    file  : interpolation.py
    date  : 26.12.23

    description:
        Provides various "interatpolation" or "sampling" methods to generate gradients.
"""
from typing import Optional
from numpy import linspace, zeros, array, ndarray, cos, pi, floor, concatenate
from numpy.random import uniform, normal
from Colour import Colour

import constants
import colours
#-----------------------------------------------------------------------------------------------------------------------
# helper functions
#-----------------------------------------------------------------------------------------------------------------------
def colour_array(red : ndarray, green : ndarray, blue : ndarray, steps : Optional[int] = constants.STEPS) -> ndarray:
    cols = [colours.BLACK] * steps

    for i, (r, g, b) in enumerate(zip(red, green, blue)):
        cols[i] = Colour(r, g, b)
    
    return array(cols)


#-----------------------------------------------------------------------------------------------------------------------
# linear
#-----------------------------------------------------------------------------------------------------------------------
def linear(start : Colour, end : Colour, steps : Optional[int] = constants.STEPS) -> ndarray:
    r0, g0, b0 = start
    r1, g1, b1 = end

    red   = linspace(r0, r1, steps, dtype=int)
    green = linspace(g0, g1, steps, dtype=int)
    blue  = linspace(b0, b1, steps, dtype=int)

    return colour_array(red, green, blue, steps)

#-----------------------------------------------------------------------------------------------------------------------
# chebyshev
#-----------------------------------------------------------------------------------------------------------------------
def chebyshev_nodes(start : int, end : int, steps : Optional[int] = constants.STEPS) -> ndarray:
    values = zeros(steps, dtype=int)

    for k in range(steps):
        values[k] = int(0.5*(start + end + (end - start) * cos((2*k-1)/steps * pi)))
    
    return values

def chebyshev(start : Colour, end : Colour, steps : Optional[int] = constants.STEPS) -> ndarray:
    r0, g0, b0 = start
    r1, g1, b1 = end

    red   = chebyshev_nodes(r0, r1, steps)
    green = chebyshev_nodes(g0, g1, steps)
    blue  = chebyshev_nodes(b0, b1, steps)

    return colour_array(red, green, blue, steps)


#-----------------------------------------------------------------------------------------------------------------------
# offset
#-----------------------------------------------------------------------------------------------------------------------
def offset_gradient(start : Colour, end : Colour, steps : Optional[int] = constants.STEPS,
                    offset : Optional[float] = 0.25) -> ndarray:
    
    center_col = start.average(end)
    center_index = int(floor(steps * offset))
    
    part1 = linear(start, center_col, center_index)
    part2 = linear(center_col, end, steps - center_index + 1)[1:]
    return concatenate((part1, part2))


def offset_left(start : Colour, end : Colour, steps : Optional[int] = constants.STEPS):
    return offset_gradient(start, end, steps, 0.25)