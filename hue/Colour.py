"""
    author: Laura Mossböck | 11820925 @ TUGraz | laura.mossboeck@gmail.com
    file  : Colour.py
    date  : 26.12.23

    description:
        Provides the `Colour` class used for the hue generating project.
"""
from typing import Tuple, Optional, Iterable
from util import to_uint8, base_convert
from numpy import ndarray, array, sqrt
import constants

#-----------------------------------------------------------------------------------------------------------------------
# helper functions
#-----------------------------------------------------------------------------------------------------------------------
def extract_rgb(value : int) -> Tuple[int, int, int]:
    """
        Assumes an integer representing the value of an RGB-hex code (e.g. '0xF5A9B8') and extracts the red, 
        green and blue intensities.

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
        Basically a fancy array with some shortcuts specific to RGB-colours.

        Attributes:
        -----------
        _red : int
            red intensity
        _green : int
            green intensity
        _blue : int
            blue intensity
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
        return Colour.fromhex(base_convert(string, base))

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
            Iterable[int]:
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


    def average(self, col2):
        """
            Computes the true average of two colours given in RGB format.

            Parameters:
            -----------
            col2 : Colour
                colour to average with
            
            Returns:
            --------
            Colour
                average colour of `self` and `col2`
        """
        red   = sqrt((self._red**2 + col2._red**2) * 0.5)
        green = sqrt((self._green**2 + col2._green**2) * 0.5)
        blue  = sqrt((self._blue**2 + col2._blue**2) * 0.5)
        return Colour(red, green, blue)

