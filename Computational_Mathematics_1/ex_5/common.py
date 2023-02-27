from numpy import ndarray, zeros, dot

def backsubs(U : ndarray, b : ndarray) -> ndarray:
    """
    Apply back-substition for solving a linear equation with upper triangular matrix
    """
    m,n = U.shape
    x = zeros(n)
    for j in reversed(range(n)):
        x[j] = (b[j] - dot(U[j,:][j:],x[j:])) / U[j,j]
    return x

def forwsubs(L : ndarray, b : ndarray) -> ndarray:
    """
    Apply forward-substitution for solving a linear equation with lower triangular matrix
    """
    m, n = L.shape
    x = zeros(n)
    for j in range(n):
        x[j] = (b[j] - dot(L[j,:][:j],x[:j])) / L[j,j]
    return x