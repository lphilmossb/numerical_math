from numpy import ndarray, float64, eye, argmax, absolute, tril, triu
from typing import Tuple
from common import forwsubs, backsubs

def LUP(A : ndarray) -> Tuple[ndarray, ndarray, ndarray]:
    """
    Decompose ``A`` into L,U,P such that PA = LU
    
    Applies the LU-decomposition with partial pivoting on ``A``. Note this runs in O(n^3), thus large matrices 
    should be avoided.
    """
    n,m = A.shape
    if n != m:
        raise ValueError('passed matrix is non-square')
    
    B = A.astype(float64)
    P = eye(n, dtype=float64)

    for j in range(n-1):
        s = argmax(absolute(B[:,j][j:]))
        if s != 0:
            B[[s+j,j]] = B[[j,s+j]]
            P[[s+j,j]] = P[[j,s+j]]
        
        for i in range(j+1,n):
            if B[j,j] == 0.0:
                raise ValueError(f'encountered 0 on diagonal ({j},{j}')
            B[i,j] = B[i,j] / B[j,j]
            for k in range(j+1,n):
                B[i,k] = B[i,k] - B[i,j] * B[j,k]
    
    return eye(n) + tril(B,k=-1), triu(B), P

def LUPSolver(A : ndarray, b : ndarray) -> ndarray:
    L,U,P = LUP(A)
    y = forwsubs(L,P@b)
    return backsubs(U,y)