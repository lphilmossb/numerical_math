from numpy import arange, sum as nsum, maximum, minimum


class RiemannSum:

    def __init__(self, t0, t1, N) -> None:
        self._t0 = t0
        self._t1 = t1
        self._N = N
        self._dt = (t1 - t0) / N
        self._sampletimes = arange(t0, t1 + self._dt, self._dt)

    def lsum(self, f):
        y = f(self._sampletimes[:-1])
        return nsum(y * self._dt)
        
    def rsum(self, f):
        y = f(self._sampletimes[1:])
        return nsum(y * self._dt)
    
    def midsum(self, f):
        y = f(self._sampletimes[:-1] + 0.5 * self._dt)
        return nsum(y * self._dt)

    def topsum(self, f):
        y1 = f(self._sampletimes[:-1])
        y2 = f(self._sampletimes[1:])
        y = maximum(y1, y2)
        return nsum(y * self._dt)
    
    def botsum(self, f):
        y1 = f(self._sampletimes[:-1])
        y2 = f(self._sampletimes[1:])
        y = minimum(y1, y2)
        return nsum(y * self._dt)