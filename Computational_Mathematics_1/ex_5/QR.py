"""
author        : Moritz Mossböck | 11820925 | moritz.mossboeck@student.tugraz.at
file          : QR.py           | URF-8
target version: python 3.10.8   | 64-bit
creation date : 8.12.2022
course        : Computational Mathematics 1
"""

# ----------------------------------------------------------------------------------------------------------------------
#                                                   IMPORT SECTION
# ----------------------------------------------------------------------------------------------------------------------
from numpy import ndarray, eye, outer, zeros, sign, float64, concatenate as concat
from numpy.linalg import norm
from typing import Union, Optional

# project imports
from common import backsubs


# ----------------------------------------------------------------------------------------------------------------------
#                                              FUNCTION DECLARATIONS
# ----------------------------------------------------------------------------------------------------------------------
def householder(u : ndarray) -> ndarray:
    """
    Return the matrix performing the housholder rotation around u
    """
    n = len(u)
    return eye(n) - 2 * outer(u, u.T) / norm(u)**2

def e1(n : int) -> ndarray:
    """
    Return the first basis vector of dimension n
    """
    e = zeros(n, dtype=float64)
    e[0] = 1.0
    return e

def QR(A : ndarray, b : Optional[ndarray] = None, **kwargs) -> Union[ndarray, ndarray] | ndarray:
    """
    Perform the QR-decomposition on A
    
    Inputs:
        | name | type             | size | optional | description                                    |
        |------|------------------|------|----------|------------------------------------------------|
        | A    | ndarray[float64] | mxn  | false    | the matrix to be decomposed                    |
        | b    | ndarray[float64] | n    | true     | Ax = b for a specific solution x               |
        | mode | string           |      | true     | switches operation between solve and full mode |

    Outputs:
        | name | type             | size | description                                    |
        |------|------------------|------|------------------------------------------------|
        | Q    | ndarray[float64] | mxm  | orthogonal component of QR decomposition       |
        | R    | ndarray[float64] | mxn  | upper triangular component of QR decomposition |
        | x    | ndarray[float64] | m    | solves Ax = b                                  |

    Operation Modes:

        full: decomposes A (mxn) into the product QR, i.e. A = QR, where Q is a mxm orthogonal matrix and 
              R is a mxn upper triangular matrix
        solve: find the least square solution x to Ax = b
    """
    mode = kwargs.get('mode', 'full')
    debug = kwargs.get('debug', False)
    R = A.astype(float64)
    m,n = R.shape

    if mode == 'full':
        Q = eye(m)
    elif mode == 'solve':
        c = b.astype(float64)


    for k in range(n):
        ak = R[:,k][k:]
        uk = concat((zeros(k, dtype=float64), ak + sign(R[k,k])* norm(ak) * e1(m-k)))
        R = R - 2 / norm(uk)**2 * outer(uk, R.T @ uk)

        if mode == 'full':
            Q = Q @ householder(uk)
            if debug:
                print(f'Q:\n{Q}')
                print(f'R:\n{R}')
        if mode == 'solve':
            c = c - 2 / norm(uk)**2 * (uk.T @ c) * uk


    if mode == 'full':
        return Q, R
    elif mode == 'solve':
        return backsubs(R, c)
