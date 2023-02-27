from numpy import array, ndarray, sqrt, cos, pi as PI, sin, ones, zeros, multiply, sum, diag
from typing import Optional

def toeplitz_eigvals(n : int,
                     a : Optional[float] = 2,
                     b : Optional[float] = -1,
                     c : Optional[float] = -1) -> ndarray:
    """
    Compute the eigenvalues of a Töplitz-Tridiagonal matrix
    
    A Töplitz tridiagonal matrix ``A`` has constant diagonals with bandwidth 3. 
    The parameter ``a`` denotes the entries of the main diagonal, ``b`` first upper diagonal and 
    ``c`` the first lower diagonal. The formula for the eigenvalues can be found here:
    
        https://de.wikipedia.org/wiki/Tridiagonal-Toeplitz-Matrix

    Note that there is no english wikipedia article available, at the point of writing.
    """
    return array( [a + 2 * sqrt(b*c) * cos(k*PI/(n+1)) for k in range(1,n+1)] )

def toeplitz(n : int,
             a : Optional[float] = 2,
             b : Optional[float] = -1,
             c : Optional[float] = -1) -> ndarray:
    """
    Construct a Töplitz tridiagonal matrix
    
    The constructed matrix has ``a`` on the main diagonal, ``b`` on the top diagonal and ``c`` on the bottom 
    diagonal
    """
    return diag([a]*n) + diag([b]*(n-1),1) + diag([c]*(n-1),-1)

def toeplitz_eigvec(n : int,
                    k : int,
                    b : Optional[float] = -1,
                    c : Optional[float] = -1) -> ndarray:
    """
    Return the k-th eigenvector of ``toeplitz(n,a,b,c)`` for an arbitrary a
    """
    return array([ ((b/c)**(l/2)) * sin( l * k * PI / (n+1)) for l in range(1,n+1)])

def An(n : int) -> ndarray:
    return (n+1)**2 * toeplitz(n,2,-1,-1)

def An_eigvals(n : int) -> ndarray:
    return (n+1)**2 * toeplitz_eigvals(n, 2, -1, -1)

def debug(msg : str, show : Optional[bool] = False) -> None:
    """
    Print debug-message to stdout, if a debug flag is set
    """
    if show:
        print(msg)

def Vandermonde(v : ndarray) -> ndarray:
    """
    Construct a vandermonde matrix from ``v``
    """
    n = v.shape[0]
    V = ones((n, n))
    for i,x in enumerate(v):
        for j in range(n):
            V[i,j] = x**j
    return V

def backsubs(U : ndarray, b : ndarray) -> ndarray:
    """
    Apply back-substition for solving a linear equation with upper triangular matrix
    """
    n = len(b)
    x = zeros(n)
    for j in reversed(range(n)):
        x[j] = (b[j] - sum(multiply(U[j,:][j:],x[j:]))) / U[j,j]
    return x

def forwsubs(L : ndarray, b : ndarray) -> ndarray:
    """
    Apply forward-substitution for solving a linear equation with lower triangular matrix
    """
    n = len(b)
    x = zeros(n)
    for j in range(n):
        x[j] = (b[j] - sum(multiply(L[j,:][:j],x[:j]))) / L[j,j]
    return x

def crout_backsubs(U : ndarray, b : ndarray) -> ndarray:
    """
    Same as ``backsubs(U,b)``, however exploits properties of tridiagonal matrices
    
    Given a tridiagonal matrix A and it's LU decomposition based on Crout's method, the linear equation 
    Ux = b produces a linear recurrence relation for the entries of x, which can be solved in reverse order
    (from n to 1).
    
    Note that U has to be a unit upper triangular matrix, i.e. U[k,k] = 1 for k in range(n).
    """
    n = len(b)
    x = zeros(n)
    x[-1] = b[-1] # setup recursion
    
    for l in reversed(range(n-1)):
        x[l] = b[l] - U[l,l+1] * x[l+1]
    return x    
    
def crout_forwsubs(L : ndarray, b : ndarray) -> ndarray:
    """
    Same as ``forwsubs(U,b)``, however exploits properties of tridiagonal matrices
    
    Given a tridiagonal matrix A and it's LU decomposition based on Crout's method, the linear equation 
    Lx = b produces a linear recurrence relation for the entries of x, which can be solved in regular order
    (from 1 to n).
    """
    n = len(b)
    x = zeros(n)
    x[0] = b[0] / L[0,0] # setup recursion
    
    for l in range(1,n):
        x[l] = (b[l] - L[l,l-1] * x[l-1]) / L[l,l]
    return x    