"""
author        : Moritz Mossböck | 11820925 | moritz.mossboeck@student.tugraz.at
file          : gauss.py | UTF-8
target version: python 3.10.8 | 64-bit
creation date :
course        : Computational Mathematics 1
"""

# ----------------------------------------------------------------------------------------------------------------------
#                                                   IMPORT SECTION
# ----------------------------------------------------------------------------------------------------------------------
from numpy import ndarray, ones, isin, column_stack as colstack, float64


# ----------------------------------------------------------------------------------------------------------------------
#                                               FUNCTION DECLARATION
# ----------------------------------------------------------------------------------------------------------------------


def GaussElim(A: ndarray, b: ndarray) -> ndarray:
    """
    Perform Gaussian Elimination on [A,b] to solve Ax = b

    Given a square-matrix ``A`` and a vector ``b`` this functions performs Gaussian Elimination by searching 
    the k-th column for a non-zero value, swaps it with the current iteration row (if they are different) and 
    normalizes the row. Then the k-th column is eliminated except for the k-th row. Since we apply 
    all operations on [A,b], the return value of [A,b][:,n] is the same as A^{-1}b, if ``A`` is regular. 

    Since each column is eliminated indivdually, this function may produce a zero-column in any iteration, thus
    showing that A is singular and Ax = b  having no (unique) solution. If this occurs, the iteration is aborted and 
    a ValueError is raised.
    """

    m,n = A.shape
    
    if m != n:
        raise ValueError(f'invalid matrix format ({m} x {n}), only square matrices may be passed')

    pivot_indices = (n+1)*ones(n) # make sure no possible k value already exists
    Ab = colstack((A.astype(float64),b.astype(float64)))

    for i in range(n):
        pivot = 0
        found_pivot = False
        for k in range(n):
            if isin(pivot_indices, k, assume_unique=True).any(): # check if row already has pivot-element
                continue
            else:
                if Ab[k,i] != 0:
                    pivot = Ab[k,i]
                    pivot_indices[i] = i
                    found_pivot = True
                    if i != k:
                        Ab[[k,i]] = Ab[[i,k]] # a very dirty row-swap (most likely pointer based?)
                    break
        if not found_pivot:
            raise ValueError(f'could not find a valid pivot-element in column {i}, aborting')
        
        Ab[i,:] = 1.0 / pivot * Ab[i,:] # we found a valid pivot-element, normalize the corresponding row
        
        # eliminate i-th column except i-th row
        for j in range(n):
            if j == i:
                continue
            else:
                Ab[j,:] = Ab[j,:] - Ab[j,i] * Ab[i,:]
    return Ab[:,n]
