from typing import Callable, Optional
from numpy import array, zeros, ndarray
from Colour import Colour

import constants
import colours


class ColorgridGenerator:

    def __init__(self, **kwargs):
        base = int(kwargs.get('base', constants.BASE))
        self._top_left     = Colour.fromstring(kwargs.get('top_left'    , 0xF5A9B8), base)
        self._top_right    = Colour.fromstring(kwargs.get('top_right'   , 0x5BCEFA), base)
        self._bottom_right = Colour.fromstring(kwargs.get('bottom_right', 0x373f47), base)
        self._bottom_left  = Colour.fromstring(kwargs.get('bottom_left' , 0xDBFE87), base)
        self._ncols        = int(kwargs.get('num_cols', constants.COLUMNS))
        self._nrows        = int(kwargs.get('num_rows', constants.ROWS))

    def generate(self, sample : Callable[[Colour, Colour, Optional[int]], ndarray]) -> ndarray:
        cols    = array([colours.BLACK] * (self._ncols*self._nrows)).reshape((self._ncols, self._nrows)).T

        cols[0] = sample(self._top_left, self._top_right, self._ncols)
        cols[-1] = sample(self._bottom_left, self._bottom_right, self._ncols)
        
        for col in range(self._ncols):
            cols[:,col] = sample(cols[0, col], cols[-1, col], self._nrows)

        cols = cols.T
        grid = zeros((self._ncols, self._nrows, 3), dtype=int)

        for col in range(self._ncols):
            for row in range(self._nrows):
                grid[col,row] = cols[col,row].as_list()
    
        return grid
