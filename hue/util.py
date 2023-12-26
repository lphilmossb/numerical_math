"""
    author: Laura Mossböck | 11820925 @ TUGraz | laura.mossboeck@gmail.com
    file  : util.py
    date  : 26.12.2023

    description:
        Provides some basic utility functions used in the (extended) color-grid generator.
        See the individual functions for a more detailed explanation.
"""
from typing import Optional, Tuple

#-----------------------------------------------------------------------------------------------------------------------
# numerics
#-----------------------------------------------------------------------------------------------------------------------
def base_convert(value : int | float | str, base : Optional[int | str] = 16) -> int:
    """
        Converts a given string / number to int. This is basically just a wrapper to avoid the `isinstance` call where
        this function is used.

        Parameters:
        -----------
        value: int | float | str
            value/string to convert
        base: int | str, optional
            if `input` is a string, this is the base used to generate the srting
        
        Returns:
        ________
        int
            input converted to an integer or None
    """
    rval = value
    
    if isinstance(value, str):
        try:
            rval = int(value, base)
        except ValueError:
            rval = None
    return rval

def cap(value : int, bounds : Tuple[int, int]) -> int:
    """
        Caps `value` between `min` and `max`.

        Parameters:
        -----------
        value: int
            value to cap
        bounds: 
            allowed region for value to be in (i.e. min an max)
        
        Returns:
        --------
        int
            the capped value if it exceeds the bounds, else `value`
    """
    rval = value
    if value > bounds[1]:
        rval = bounds[1]
    if value < bounds[0]:
        rval = bounds[0]
    return rval


def to_uint8(value : int) -> int:
    """
        A wrapper call to cap(value, 0, 255).
    """
    return cap(value, (0, 255))

