"""
    author: Laura Mossböck | 11820925 @ TUGraz | laura.mossboeck@gmail.com
    file  : ColourgridGenerator.py
    date  : 26.12.23

    description:
        Provides the class `ColourgridGenerator` which implements a method to generate a color gradient in grid form 
        given four edge-colours.
"""
from typing import Callable, Optional
from numpy import array, zeros, ndarray
from Colour import Colour

import constants
import colours


Sampling = Callable[[Colour, Colour, Optional[int]], ndarray]


#-----------------------------------------------------------------------------------------------------------------------
# class definition
#-----------------------------------------------------------------------------------------------------------------------
class ColourgridGenerator:
    """
        Class to repeatedly generate colour-grids given the four edge colours.

        Attributes:
        -----------
        _top_left : Colour
            colour for the top left corner
        _top_right : Colour
            colour for the top right corner
        _bottom_right : Colour
            colour for the bottom right corner
        _bottom_left : Colour
            colour for the bottom left corner
        _ncols : int
            number of columns
        _nrows : int
            number of rows
    """

    def __init__(self, **kwargs):
        base = int(kwargs.get('base', constants.BASE))
        self._top_left     = Colour.fromstring(kwargs.get('top_left'    , 0xF5A9B8), base)
        self._top_right    = Colour.fromstring(kwargs.get('top_right'   , 0x5BCEFA), base)
        self._bottom_right = Colour.fromstring(kwargs.get('bottom_right', 0x373f47), base)
        self._bottom_left  = Colour.fromstring(kwargs.get('bottom_left' , 0xDBFE87), base)
        self._ncols        = int(kwargs.get('num_cols', constants.COLUMNS))
        self._nrows        = int(kwargs.get('num_rows', constants.ROWS))

    def generate(self, sample : Sampling) -> ndarray:
        """
            Generate the colour grid, where the intermediate colours are generated according to `sample`. 
            This method can be split into two phases:

            Phase 1: Generate all intermediate colours in their abstract form (using the `Colour` class)
            Phase 2: reduce this array of colours into a 3-tensor containing RGB-data of each colour.
            
            Parameters:
            -----------
            sample : Callable[[Colour, Colour, Optional[int]], ndarray]
                function to generate inbetween colours
            
            Returns:
            --------
            ndarray
                3-tensor contining RGB values of each grid-point

            Notes:
            ------
            The term „3-tensor“ is fancy mathematical jargon and is quite overkill in this case. A better description 
            would be 3-dimensional array, as we “layer“ the grid into its three different channels, corresponding to
            red, green and blue. This is already very tailored towards the `imshow` method from matplotlib. 
        """
        # phase 1
        cols = array([colours.BLACK] * (self._ncols * self._nrows)).reshape((self._ncols, self._nrows)).T

        cols[0]  = sample(self._top_left, self._top_right, self._ncols) # top row
        cols[-1] = sample(self._bottom_left, self._bottom_right, self._ncols) # bottom row
        
        for col in range(self._ncols): # interpolate vertically
            cols[:,col] = sample(cols[0, col], cols[-1, col], self._nrows)

        # phase 2
        cols = cols.T # I don't want to figure out the correct indexing below, sooooo
        grid = zeros((self._ncols, self._nrows, 3), dtype=int)

        for col in range(self._ncols):
            for row in range(self._nrows):
                grid[col,row] = cols[col,row].as_list()
    
        return grid
