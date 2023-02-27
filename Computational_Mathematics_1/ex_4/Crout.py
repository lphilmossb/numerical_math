from numpy import ndarray, float64, eye, tril, zeros
from typing import Tuple
from common import crout_forwsubs, crout_backsubs


def LUCrout(A : ndarray) -> Tuple[ndarray, ndarray]:
    """
    Perform the Crout-Decomposition for a tridiagonal matrix
    
    Given a tridiagonal matrix ``A`` (i.e. a band-matrix with bandwidth 3), this functions computes 
    matrices L and U of the following form:
    
        L = diag(l_jj) + diag(l_kl,-1) for j = 1,...n      , k = 2,...,n and l = 1,...,n-1
        U = I_n + diag(u_kl,1)         for k = 1,...,n-1 and l = 2,...,n

    Such that A = LU. If during the algorithm one diagonal element of L becomes 0, the 
    iteration is aborted and a ValueError is raised, stating that there does not exist a crout
    decomposition of the passed matrix.
    
    Additionally, if the passed matrix is non-square, a ValueError is raised informing of the 
    dimension mismatch. Note however, that no check is performed wether or not ``A`` is 
    actually a tridiagonal matrix. 
    """
    
    m,n = A.shape
    
    if m != n:
        raise ValueError('non square matrix passed')
    
    L = zeros((n,n),dtype=float64)
    U = eye(n,dtype=float64)
    L += tril(A,-1)
    L[0,0] = A[0,0]
    
    for k in range(1,n):
        if L[k-1,k-1] == 0.0:
            raise ValueError('No crout decomposition exists')
        L[k,k] = A[k,k] - A[k-1,k] * L[k,k-1] / L[k-1,k-1]
        U[k-1,k] = A[k-1,k] / L[k-1,k-1]

    return L,U

def LUCSolver(A : ndarray, b : ndarray) -> ndarray:
    """
    Solves Ax = b, where A is tridiagonal
    
    1. Decompose A with ```LUCrout`` into L and U
    2. Solve Ly = b with forward substitution
    3. Solve Ux = y with backward substitution
    """
    L, U = LUCrout(A)
    y = crout_forwsubs(L, b)
    return crout_backsubs(U, y)