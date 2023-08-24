from numpy import ndarray, zeros, sqrt, sum, square, multiply, float64
from common import forwsubs, backsubs

def CholeskyDecom(A : ndarray) -> ndarray:
    """
    Perform the Cholesky-decomposition for a hermitian matrix ``A``


    If ``A`` is non-square, the functions raises an error.
    If at any point L[k,k] = 0, the algorithm aborts and a ValueError is raised
    """
    
    m,n = A.shape
    
    if m != n:
        raise ValueError('passed non square matrix')
    
    L = zeros((n,n),dtype=float64)
    
    for k in range(n):
        L[k,k] = sqrt(A[k,k] - sum(square(L[k,:][:k])))
        if L[k,k] == 0.0:
            raise ValueError(f'produced 0 in diagonal element {k},{k}')
        for i in range(k+1,n):
            L[i,k] = (A[i,k]  - sum(multiply( L[i,:][:k], L[k,:][:k] )) ) / L[k,k]
    return L
    
def CholeskySolver(A : ndarray, b : ndarray) -> ndarray:
    L = CholeskyDecom(A)
    y = forwsubs(L,b)
    return backsubs(L.T,y)