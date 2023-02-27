"""
author        : Moritz Mossböck | 11820925 | moritz.mossboeck@student.tugraz.at
file          : Fitter.py       | UTF-8
target version: Python 3.10.8   | 64-bit
creation date : 02.12.2022
course        : Computational Mathematics 1
"""

# ----------------------------------------------------------------------------------------------------------------------
#                                                   IMPORT SECTION
# ----------------------------------------------------------------------------------------------------------------------
from __future__ import annotations
from numpy import zeros, float64, ndarray, sum, sqrt, linspace, min, max, array, dot
from numpy.linalg import norm
from typing import Optional, List, Tuple
from matplotlib.pyplot import Axes
from matplotlib.lines import Line2D

# project imports
from QR import QR
from gauss import GaussElim
from common import backsubs
from Polynomial import Polynomial

# ----------------------------------------------------------------------------------------------------------------------
#                                                       CLASS
# ----------------------------------------------------------------------------------------------------------------------


class Fitter:
    def __init__(self, x_data: ndarray, m: Optional[int] = 5, single: Optional[bool] = False) -> None:
        self.n = len(x_data)
        self.m = m + 1
        self.x = array(x_data).astype(float64)
        self.single = single
        self.f = Polynomial([1.0])

        if not single:
            self.Q, self.R = QR(compute_normal_matrix(x_data, self.m ))

    # special methods
    def __call__(self, x: float | ndarray) -> float | ndarray:
        return self.f(x)

    def __str__(self) -> str:
        return str(self.f)

    # fitting
    def PolyFit(self, y_data: ndarray) -> ndarray:
        """
        Fit a polynomial of degree m to the supplied y-values. If the instance is in single-mode, then gaussian 
        elimination is used for solving Ac=b. Else the constructor has already perfomred a QR-decomposition
        of A and c is solved via backward-substitution
        """
        if self.n != len(y_data):
            raise ValueError(
                f'invalid y-values passed, need {self.n} != {len(y)}')

        b = compute_normal_vector(self.x, y_data, self.m )
        c = zeros(self.m, dtype=float64)

        if self.single:
            c = GaussElim(compute_normal_matrix(self.x, self.m), b)
        else:
            y = self.Q.T @ b
            c = backsubs(self.R, y)

        self.f = Polynomial(c)
        return c

    # misc
    def residues(self, y_data: ndarray) -> ndarray:
        """
        Compute the residues of the fitted function and the supplied y-values
        """
        return y_data - self(self.x)

    def StdDev(self, y_data) -> float:
        """
        Compute the standard deviation of the current fitted function to the given y_data
        """
        r = self.residues(y_data)
        S = dot(r, r)
        return sqrt(S / (self.n - self.m + 1))

    def PlotPoly(self, ax: Axes, y_data: Optional[ndarray] = None) -> Line2D:
        """
        Plot the currently fitted function on the supplied axes

        If y_data is not None, the a scatter plot with the dataset is made as well
        """
        x = linspace(min(self.x)-0.1, max(self.x) + 0.1, 500)
        y = self.f(x)
        if y_data is None:
            return ax.plot(x, y)[0]
        else:
            return ax.plot(x, y)[0], ax.scatter(self.x, y_data, color='r')

    # static methods

    @staticmethod
    def find_best(x_data: ndarray, y_data: ndarray, aux: Optional[bool] = False) -> Fitter | Tuple[Fitter, List[Fitter]]:
        """
        Find the Fitter-instance with minimum standard deviation and return it

        Inputs:
            | name   | type            | size | optional | description                                    |
            |--------|-----------------|------|----------|------------------------------------------------|
            | x_data | ndarray[float64]| n    | false    | x-values for the supplied y-values             |
            | y_data | ndarray[float64]| n    | false    | y-values of the function at the given x-values |
            | aux    | boolean         |      | true     | return all generated fitting polynomials       |

        Outputs:
            | name         | type         | description                                    |
            |--------------|--------------|------------------------------------------------|
            | min_instance | Fitter       | orthogonal component of QR decomposition       |
            | instances    | list[Fitter] | upper triangular component of QR decomposition |

        A very brute froce approach, which simply computes the fitting polynomial for m = 1,...,n and checks wether
        or not the standard deviation for the given y-values is less than the current minimum. If that is the 
        case the old minimum instance is deleted and the current instance is set as the new minimum. Note that 
        if aux is True, then the instance is copied to the instances array first.
        """
        n = len(x_data)

        min_instance = Fitter(x_data, 0)
        min_instance.PolyFit(y_data)
        min_sigma = min_instance.StdDev(y_data)
        if aux:
            instances = [min_instance] * (n-1)
            instances[0] = min_instance

        for i in range(1,n-1):
            instance = Fitter(x_data, i, True)
            instance.PolyFit(y_data)
            sigma = instance.StdDev(y_data)

            if aux:
                instances[i] = instance

            if sigma < min_sigma:
                min_sigma = sigma
                del min_instance # optional (for small datasets probably)
                min_instance = instance

        min_instance.Q, min_instance.R = QR(compute_normal_matrix(x_data, min_instance.m))
        min_instance.single = False

        if aux:
            return min_instance, instances
        return min_instance


# ----------------------------------------------------------------------------------------------------------------------
#                                                  HELPER FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------

def compute_normal_matrix(x_data: ndarray, m: Optional[int] = 5) -> ndarray:
    """
    Computes the matrix for normal equations when using polynomials for fitting
    """
    A = zeros((m, m), dtype=float64)
    for i in range(m):
        for k in range(m):
            A[i, k] = sum(x_data**(i+k))
    return A


def compute_normal_vector(x_data: ndarray, y_data: ndarray, m: Optional[int] = 5) -> ndarray:
    """
    Computes the b-vector for the normal equations given x_data and y_data
    """
    b = zeros(m, dtype=float64)
    n = len(x_data)
    for i in range(m):
        b[i] = sum([(x_data[j]**i)*y_data[j] for j in range(n)])
    return b
