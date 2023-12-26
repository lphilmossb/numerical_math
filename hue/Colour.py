"""
    author: Laura Mossböck | 11820925 @ TUGraz | laura.mossboeck@gmail.com
    file  : Colour.py
    date  : 26.12.23

    description:
        Provides the `Colour` classe used for the hue generating project.
"""
from typing import Tuple, Optional, Iterable
from util import to_uint8, base_convert
from numpy import ndarray, array
import constants

#-----------------------------------------------------------------------------------------------------------------------
# helper functions
#-----------------------------------------------------------------------------------------------------------------------
def extract_rgb(value : int) -> Tuple[int, int, int]:
    """
        Assumes an integer representing the value of an RGB-hex code (e.g. '0xF5A9B8') and extracts the red, 
        green and blue components.

        Parameters:
        -----------
        value : int
            value to extract colour information from

        Return:
        Tuple[int,int,int]
            tuple of integers representing red, green and blue
    """
    red = value >> 16
    green = (value - (red << 16)) >> 8
    blue = value - (red << 16) - (green << 8)
    return red, green, blue


#-----------------------------------------------------------------------------------------------------------------------
# class definition
#-----------------------------------------------------------------------------------------------------------------------
class Colour:
    """
        Basically a fancy array with some shortcutes specific fot RGB-colours.

        Attributes:
        -----------
        _red : int
            red channel
        _green : int
            green channel
        _blue : int
            blue channel
    """
    def __init__(self, red : Optional[int] = 0, green : Optional[int] = 0, blue : Optional[int] = 0) -> None:
        self._red   = to_uint8(red)
        self._green = to_uint8(green)
        self._blue  = to_uint8(blue)

    @classmethod
    def fromhex(cls, colour):
        """
            Construct an instance from a hex-literal, such as '0xF5A9B8'.

            Parameters:
            -----------
            colour : int
                hex literal representing the desired colour

            Returns:
            --------
            Colour
                the colour with the extracted red, green and blue values from `colour`.
        """
        r, g, b = extract_rgb(colour)
        return cls(r, g, b)
    
    @classmethod
    def fromstring(cls, string, base : Optional[str] = constants.BASE):
        """
            Construct an instance from a string of a number in the given base

            Parameters:
            -----------
                string of literal in `base`

            Returns:
            --------
            Colour
                the colour with the specified red, green and blue values
        """
        return Colour.fromhex(base_convert(string))

    def __str__(self) -> str:
        """
            Gives a string representation of the associated hex-literal.
        """
        return hex(self.as_number())
    
    def __iter__(self) -> Iterable[int]:
        """
            Shortcut to allow for tuple unpacking, which is used for the interpolation functions.

            Returns:
            --------
            Iterable[ing]:
                iterable of the tuple containing the red, green and blue values
        """
        return iter((self._red, self._green, self._blue))
    
    def as_number(self) -> int:
        """
            Returns the hex-representation of the number
        """
        return (self._red << 16) + (self._green << 8) + self._blue
    
    def as_list(self) -> ndarray:
        """
            Returns an array containting the red, green and blue values
        """
        return array([self._red, self._green, self._blue])
