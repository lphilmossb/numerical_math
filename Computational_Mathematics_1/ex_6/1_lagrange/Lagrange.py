"""
author        : Moritz Mossböck             | 11820925  | moritz.mossboeck@student.tugraz.at
file          : Lagrange.py                 | UTF-8
target version: python 3.10.9               | 64-bit
course        : Computational Mathematics 1 | MAT.208UB
"""

# imports
from numpy import ndarray, delete, subtract, divide, prod, sum, min, max, linspace, abs
from typing import Optional, List
from matplotlib.axes import Axes
from matplotlib.lines import Line2D


class LagrangeBasis:
    """
    Forms a basis of the polynomial vector space R_n[x] using the lagrange polynomials

    Attributes:
    -----------
    x_data : ndarray
        sorted (ascending order) x-values for the y-samples
    x : ndarray
        x-axis where the interpolant for given y-samples is evaluated on
    basis : list
        list of the numerical values of the lagrange basis polynomials

    Methods:
    --------
    ljn(j : int) -> ndarray
        return the j-th basis polynomial (values over self.x)
    interpolate(y_data : ndarray) -> ndarray
        interpolate the function whose samples at the passed x-nodes (self.x_data) are stored in 
        y_data and return the interpolation polynomial (evaluated over x)

    Constructor:
    ------------
        The constructor of `LagrangeBasis` takes in a multitude of arguments to set-up the basis as generally as 
        possible. Based on the nodes and the offsets, the axis the interpolant is evaluated on can be specified.
        Furthermore the degree of the interpolant may also be set.
        Parameters:
        -----------
        x_data : ndarray
            x-coordinates of the samples
        n : int, default = -1
            degree of the interpolating polynomial (and thus the basis polynomials)
        resolution : int, default = 500
            number of x-values to evaluate 
        left_xoffset : float, default = 0.0
            offset of min(x_data) for evaluation
        right_xoffset : float, default = 0.0
            offset of max(x_data) for evaluation

        Raises:
        -------
        ValueError
            if `x_data` is not sorted in ascending order
    
    """

    def __init__(self, x_data: ndarray, n: Optional[int] = -1, resolution: Optional[int] = 500,
                 left_xoffset : Optional[float] = 0.0, right_xoffset : Optional[float] = 0.0) -> None:

        for i, x in enumerate(x_data[:-1]):
            if x > x_data[i+1]:
                raise ValueError('x-nodes must be sorted in increasing order')
        self.x_data = x_data
        self.x = linspace(min(x_data) - left_xoffset, max(x_data) + right_xoffset, resolution)

        if n == -1:
            self.n = len(x_data)
        else:
            self.n = n

        self.basis = [None] * self.n

    def ljn(self, j: int) -> ndarray:
        """
        Compute the j-th lagrange polynomial for the given nodes and degree

        In order to avoid redundant computation, the instance-attribute `basis` is used to check whether 
        the j-th lagrange polynomial has already been calculated or not. If that is the case, i.e. 
        `instance.basis[j]` is not `None`, then the method simply returns `instance.basis[j]`.

        Parameter:
        ----------
        j : int
            specifies which lagrange polynomial should be computed

        Returns:
        ________
        ndarray
            evaluated lagrange polynomial over `instance.x`
        """
        if self.basis[j] is None:
            x_tmp = delete(self.x_data, j)[:self.n]
            xj = self.x_data[j]
            factors = divide(subtract.outer(self.x, x_tmp), xj - x_tmp)
            self.basis[j] = prod(factors, 1)
        return self.basis[j]

    def interpolate(self, y_data : ndarray) -> ndarray:
        """
        Interpolate the given samples from y_data

        This method assumes the passed y_samples are measured on the corresponding x-nodes, which the instance 
        stores in `x_data`. Then the interpolant is computed on the instance-axis `x` and returned.

        Parameters:
        -----------
        y_data : ndarray
            samples of the unknown function

        Returns:
        --------
        ndarray
            evaluated interpolant over `instance.x`

        Raises:
        -------
        ValueError
            `y_data` has less than `instance.n` entries
        
        """
        if len(y_data) < self.n:
            raise ValueError(f'missing samples, at least {self.n} required')
        return sum([y_data[j] * self.ljn(j) for j in range(self.n)], 0)


def LagrangeInterpPoly(x_data: ndarray, y_data: ndarray, ax: Axes, n: Optional[int] = -1, **kwargs) -> List[Line2D]:
    """
    Produce the lagrange interpolation of the passed y-samples and plot it

    Uses `interp` of order `n` to produce the interpolation values with the method 
    `interp.interpolate()`. Then the evaluated interpolant is plotted on `ax`, and afterwards the 
    varying lagrange polynomials. If the kwarg `scatter` is set to `True`, the original samples 
    are plotted as a scatter-plot on `ax`.

    Parameters:
    -----------
    x_data : ndarray
        the x-nodes for the y-samples
    y_data : ndarray
        the y-samples of the unknown function
    ax : Axes
        matplotlib-axes object for plotting
    n : int, defailt = -1
        degree of interpolating polynomial
    **kwargs: dict, optional
        scatter : bool, default = False
            decide wether the samples are plotted on `ax` as a scatter plot

    Returns:
    --------
    List[Line2D]
        list of all generated plots (interpolant, lagrange polynomials, and samples)
    """
    interp = LagrangeBasis(x_data, n)
    pn = interp.interpolate(y_data)

    lines = [0] * (interp.n + 1)

    lines[0] = ax.plot(interp.x, pn, label=r'$p_n$')

    for j in range(interp.n):
        lines[j+1] = ax.plot(interp.x, interp.ljn(j), label=r'$\ell_{' + f'{j}{interp.n}' + r'}$')
    
    if kwargs.get('scatter', True):
        lines.append(ax.scatter(interp.x_data, y_data, color='red'))

    return lines
    
