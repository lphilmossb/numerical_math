from numpy import ndarray
from sys import stderr
from typing import Optional, Callable

from gauss import GaussElim
from LUP import LUPSolver
from Cholesky import CholeskySolver
from Crout import LUCSolver

class DirectSolver:
    
    def __init__(self) -> None:
        self.methods = {
            'gauss': GaussElim,
            'LUP': LUPSolver,
            'cholesky': CholeskySolver,
            'crout': LUCSolver,
        }
        
    def solve(self, A      : ndarray,
                    b      : ndarray,  
                    method : Optional[str | Callable[[ndarray, ndarray], ndarray]] = 'gauss') -> ndarray:
        if isinstance(method, str):
            if method in self.methods.keys():
                return self.methods[method](A, b)
            else:
                print(f'invalid direct solver specified: {method}\n', file=stderr)
        elif isinstance(method, Callable[[ndarray, ndarray], ndarray]):
            return method(A, b)
        else:
            print(f'supplied solver does not match required signature', file=stderr)