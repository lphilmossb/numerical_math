from timeit import default_timer as timer
from typing import Callable, Optional
from numpy import ndarray, zeros, pi, min, max, average
from common import An, toeplitz_eigvec as bn

from gauss    import GaussElim
from LUP      import LUPSolver
from Cholesky import CholeskySolver
from Crout    import LUCSolver

def test_solver(f         :  Callable[[ndarray,ndarray],ndarray],
                dataset   : ndarray, 
                iterations: Optional[int] = 50) -> ndarray:
    times = zeros(iterations)
    for i in range(iterations):
        start = timer()
        for p in dataset:
            f(p[0],p[1])
        end = timer()
        times[i] = end - start
    return times
        
def gen_dataset(n_values : ndarray) -> ndarray:
    dataset = [0] * len(n_values)
    for i,n in enumerate(n_values):
        dataset[i] = [An(n), pi** 2 * bn(n,1)]
    return dataset

solvers = [['Gauss',GaussElim,20], ['LU',LUPSolver,2], ['Cholesky',CholeskySolver,50], ['Crout',LUCSolver,1000]]
timedata = {}
dataset = gen_dataset([10,100,1000])

for i,f in enumerate(solvers):
    print(f'Using {f[0]} for {f[2]} iterations')
    timedata[f[0]] = test_solver(f[1],dataset,f[2])

for d in timedata.keys():
    print(f'Results for {d}:')
    print(f'\tminimum time: {min(timedata[d])}')
    print(f'\taverage time: {average(timedata[d])}')
    print(f'\tmaximum time: {max(timedata[d])}')