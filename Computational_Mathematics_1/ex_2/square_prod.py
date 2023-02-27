import numpy as np
from numpy import ndarray as Matrix

def matprod(A : Matrix, B : Matrix) -> Matrix:
    # verify matrix product can be computed
    ma = A.shape[0]
    mb = B.shape[0]

    if ma != mb:
        raise ValueError('dimension mismatch between A and B')
    
    C = np.zeros((ma,mb))
    
    for i in range(ma):
        for j in range(ma):
            for k in range(ma):
                C[i][j] += A[i][k] * B[k][j]
    return C

def matprod_fast(A : Matrix, B : Matrix) -> Matrix:
    # verify matrix product can be computed
    ma = A.shape[0]
    mb = B.shape[0]

    if ma != mb:
        raise ValueError('dimension mismatch between A and B')
    
    C = np.zeros((ma,mb))
    
    for i in range(ma):
        for j in range(ma):
            C[i][j] = np.dot(A[i,...].ravel(),B[...,j].ravel())
    return C

A = np.array([[-2, 5, 1], [0, 8, -7], [9, -4, -3]])
B = np.array([[3, -4, 6,], [-5, 2, -1,], [8, -9, 0,]])

print(A@B)
print(matprod(A,B))
print(matprod_fast(A,B))