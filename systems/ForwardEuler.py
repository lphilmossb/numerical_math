from typing import Callable, Optional, Tuple
from numpy import copy, zeros, ndarray, ceil

class ForwardEuler1D:

    def __init__(self, dy : Callable[[float, float], float],
                       y0 : Optional[float] = 0.0,
                       t0 : Optional[float] = 0.0,
                       t1 : Optional[float] = 1.0,
                       h : Optional[float] = 0.1) -> None:
        self._dy = dy

        self._y0 = y0
        self._yk = y0

        self._t0 = t0
        self._tk = t0
        self._t1 = t1
        self._h = h

    def step(self) -> float:
        self._yk += self._h * self._dy(self._yk, self._tk)
        self._tk += self._h
        return self._yk
    
    def run(self) -> Tuple[ndarray, ndarray]:
        N = int(ceil((self._t1 - self._t0) / self._h))
        y = zeros(N, dtype = float)
        t = zeros(N, dtype = float)

        for k in range(N):
            y[k] = self._yk
            t[k] = self._tk
            self.step()
        
        return y, t
    

class ForwardEulernD:
    
    def __init__(self, dy : Callable[[ndarray, float], ndarray],
                       y0 : ndarray,
                       t0 : Optional[float] = 0.0,
                       h : Optional[float] = 0.1) -> None:
        self._dy = dy
        self._y0 = copy(y0)
        self._yk = copy(y0)
        self._t0 = t0
        self._tk = t0
        self._h = h
        self._dim = y0.shape[0]

    def step(self) -> ndarray:
        self._yk = self._yk + self._h * self._dy(self._yk, self._tk)
        self._tk += self._h
        return self._yk
    
    def reset(self) -> None:
        self._yk = copy(self._y0)
        self._tk = self._t0

    def run(self, N : Optional[int] = 100) -> Tuple[ndarray, ndarray]:
        self.reset()
        y = zeros((self._dim, N), dtype = float)
        t = zeros(N, dtype=float)

        for k in range(N):
            y[:,k] = copy(self._yk)
            t[k] = self._tk
            self.step()
        
        return y.T, t