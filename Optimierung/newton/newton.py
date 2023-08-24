"""
file         : newton.py | UTF-8
author       : Moritz Mossböck, 11820925 @ TUGraz | moritz.mossboeck@student.tugraz.at
testing os   : Arch Linux (x86, 64bit) | kernel-version: 6.4.11-arch2-1
python target: 3.11.4

python package versions:
numpy 1.25.1
scipy 1.11.1

file description:
This file provides a class to run various versions of newton's method for finding minima of twice differentiable
functions.
"""
from typing import Optional, Tuple, Callable
from numpy import zeros_like, dot, zeros, copy
from numpy import ndarray as vector
from numpy.linalg import norm, LinAlgError, solve
from scipy.ndimage import shift

matrix = vector # only for type annotations

class NewtonMethod:
    """
    Acts as a runtime-context for performing a newton-method.

    Attributes:
    -----------
    _f : Callable[[ndarray], float]
        callable that computes the function value
    _nabla : Callable[[ndarray], ndarray]
        callable that computes the gradient of f
    _hess : Callable[[ndarray], ndarray]
        callable that computes the hessian of f
    _epsilon : float
        smallest possible value of norm of gradient
    _M : float
        biggest possible value for norm of `_xk`
    _N : int
        maximum number of iterations
    _rho : float
        used for a condition of the current direction
    _sigma : float
        used for armijo-step
    _p : float
        used for a condition of the current condition
    _beta : float
        factor for decreasing the step size per armijo-iteration
    _m : float
        maximum number of samples to consider when using non-monotone armijo
    _xk : ndarray
        current iteration point
    
    Methods:
    --------
    checkparams() -> None
        checks that all attributes are in correct ranges and not None
    armijo(dk : ndarray) -> float
        computes the armijo-step based
    modified_armijo(dk : ndarray, Rk : float) -> float
        computes the non-monotonous armijo-step
    local(x0 : ndarray) -> Tuple[ndarray, int]
        performs the local newton method with starting point `x0`
    globalized(x0 : ndarray) -> Tuple[ndarray, int]
        performs the globalized newton method with starting point `x0`
    mod_globalized(x0 : ndarray) -> Tuple[ndarray, int]
        performs the globalized newton method using non-monotonous armijo-steps starting at `xo`
    run(x0 : ndarray, vers : int) -> Tuple[ndarray, int]
        interface function to allow modular testing setups

    Constructor:
    ------------
        The constructor of `NewtonMethod` only has the callables as positional arguments, as these are strictly 
        required. All other parameters are passed as keyword-arguments to avoid huge function signatures and
        make mis-assignments less likely. After assigning each parameter, their validity is checked with
        `checkparams()`

        Parameters:
        -----------
        f : Callable[[ndarray], float]
            callable that computes f
        nabla_f : Callable[[ndarray], ndarray]
            callable that computes the gradient of f
        hess_f : Callable[[ndarray], ndarray]
            callable that computes the hessian of f
        **kwargs : dict, optional
            additional parameters, see the class attributes for a list of possible parameters
    """

    __EPSILON__ = 1e-9
    __M__       = 1e9
    __N__       = int(1e4)
    __RHO__     = 1.0
    __P__       = 2.5
    __BETA__    = 0.5
    __SIGMA__   = 0.25
    __m__       = 10
    __VERS__    = 1

    def __init__(self, f : Callable[[vector], vector], nabla_f : Callable[[vector], vector], 
                 hess_f : Callable[[vector],matrix], **kwargs) -> None:
        self._f = f
        self._nabla = nabla_f
        self._hess = hess_f
        self._epsilon = kwargs.get('epsilon', NewtonMethod.__EPSILON__)
        self._M = kwargs.get('M', NewtonMethod.__M__)
        self._N = kwargs.get('N', NewtonMethod.__N__)
        self._rho = kwargs.get('rho', NewtonMethod.__RHO__)
        self._sigma = kwargs.get('sigma', NewtonMethod.__SIGMA__)
        self._p = kwargs.get('p', NewtonMethod.__P__)
        self._beta = kwargs.get('beta', NewtonMethod.__BETA__)
        self._m = kwargs.get('m', NewtonMethod.__m__)
        self._xk = zeros(3, dtype=float)
        self.checkparams()

    def checkparams(self):
        """
        Verifies that all attributes are in the correct range

        Raises:
        -------
        ValueError
            if `_f` is None
        ValueError
            if `_nabla` is None
        ValueError
            if `_hess` is None
        ValueError
            if `_epsilon` is negative
        ValueError
            if `_M` is negative
        ValueError
            if `_N` is negative
        ValueError
            if `_rho` is non-positive
        ValueError
            if `_sigma` is outside (0,0.5)
        ValueError
            if `_p` is less or equal to 2
        ValueError
            if `_beta` is outside (0,1)
        ValueError
            if `_m` is non-positive
        """
        if self._f is None:
            raise ValueError('invalid f, cannot be of None-type')
        if self._nabla is None:
            raise ValueError('invalid nabla_f, cannot be of None-type')
        if self._hess is None:
            raise ValueError('invalid hess_f, cannot be of None-type')
        if self._epsilon < 0.0:
            raise ValueError('invalid epsilon, must be positive')
        if self._M < 0.0:
            raise ValueError('invalid M, must be positive')
        if self._N < 0:
            raise ValueError('invalied N, must be positive')
        if self._rho <= 0.0:
            raise ValueError('invalid rho, must be strictly positive')
        if self._sigma <= 0.0 and self._sigma >= 0.5:
            raise ValueError('invalid sigma, must be in (0,0.5)')
        if self._p <= 2:
            raise ValueError('invalid p, must be strictly greater 2')
        if self._beta <= 0.0 or self._beta >= 1.0:
            raise ValueError('invalid beta, must be in (0,1)')
        if self._m <= 0:
            raise ValueError('invalid m, must be greater or equal to 1')

    def armijo(self, dk : vector) -> float:
        """
        Computes the armijo-step distance based on the passed direction

        Parameters:
        ----------
        dk : ndarray
            the current direction
        
        Returns:
        --------
        float
            the armijo-step distance
        """
        tk = 1.0
        fxk = self._f(self._xk)
        nfxk = self._nabla(self._xk)
        
        while self._f(self._xk + self._beta*tk*dk) >= fxk + self._beta*self._sigma * tk * dot(nfxk, dk) and tk >= 1e-12:
            tk *= self._beta
        return tk
    
    def modified_armijo(self, dk : vector, Rk : float) -> float:
        """
        Computes the non-monotonous armijo-step distance based on the passed direction and comparison value

        Parameters:
        -----------
        dk : ndarray
            the current direction
        Rk : float
            the comparison value for non-monotonicity

        Returns:
        --------
        float
            the non-monotonous armijo-step distance
        """
        tk = 1.0
        nfxk = self._nabla(self._xk)
        
        while self._f(self._xk + self._beta*tk*dk) >= Rk + self._beta*self._sigma * tk * dot(nfxk, dk) and tk >= 1e-12:
            tk *= self._beta
        return tk
    

    def local(self, x0 : vector) -> Tuple[vector, int]:
        """
        Performs the local newton method for finding a minimum of f

        Parameters:
        -----------
        x0 : ndarray
            the starting point
        
        Returns:
        --------
        Tuple[ndarray, int]
            the point where a termination condition is satisfied and which condition
        """
        self._xk = copy(x0)
        term_case = 1

        for _ in range(self._N):
            if norm(self._nabla(self._xk)) <= self._epsilon:
                term_case = 0
                break
            if norm(self._xk) >= self._M:
                term_case = 2
                break

            try:
                dk = -solve(self._hess(self._xk), self._nabla(self._xk))
                self._xk = self._xk + dk
            except LinAlgError:
                term_case = 3
                break
        return self._xk, term_case
    
    def globalized(self, x0 : vector) -> Tuple[vector, int]:
        """
        Performs the globalized newton method for finding a minimum of f

        Parameters:
        -----------
        x0 : ndarray
            starting point

        Returns:
        --------
        Tuple[ndarray, int]
            the point where a termination condition is met and which condition
        """
        self._xk = copy(x0)
        term_case = 1
        dk = zeros_like(x0, dtype=float)

        for _ in range(self._N):
            try:
                dk = -solve(self._hess(self._xk), self._nabla(self._xk))
                if dot(self._nabla(self._xk), dk) >= self._rho * norm(dk)**self._p:
                    dk = -self._nabla(self._xk)
            except LinAlgError:
                dk = -self._nabla(self._xk)
            
            tk = self.armijo(dk)
            self._xk = self._xk + tk * dk

            if norm(self._nabla(self._xk)) <= self._epsilon:
                term_case = 0
                break
            if norm(self._xk) >= self._M:
                term_case = 2
                break
        return self._xk, term_case
    
    def mod_globalized(self, x0 : vector) -> Tuple[vector, int]:
        """
        Performs the globalized newton method with the non-monotonous armijo step

        Parameters:
        -----------
        x0 : ndarray
            starting point

        Returns:
        --------
        Tuple[ndarray, int]
            the point where a termination condition is met and which condition
        
        """
        self._xk = copy(x0)
        term_case = 1
        dk = zeros_like(x0, dtype=float)
        R = zeros(self._m, dtype=float)
        mk = 0

        for _ in range(self._N):
            R = shift(R, -1, cval = 0.0)
            R[-1] = self._f(self._xk)
            mk = min(mk + 1, self._m)
            Rk = max(R[self._m - mk:])
             
            try:
                dk = -solve(self._hess(self._xk), self._nabla(self._xk))
            except LinAlgError:
                dk = -self._nabla(self._xk)

            if dot(self._nabla(self._xk), dk) >= -self._rho * norm(dk)**self._p:
                dk = -self._nabla(self._xk)
                mk = 0
            
            tk = self.modified_armijo(dk, Rk)
            self._xk = self._xk + tk*dk

            if norm(self._nabla(self._xk)) <= self._epsilon:
                term_case = 0
                break
            if norm(self._xk) >= self._M:
                term_case = 2
                break
        return self._xk, term_case
    
    def run(self, x0 : vector, vers : Optional[int] = 1) -> Tuple[vector, int]:
        """
        interface function to choose one of the newton methods without externally keeping track of them

        Parameters:
        -----------
        x0 : ndarray
            the starting point
        vers : int, opional = 1
            which method to run

        Returns:
        --------
        Tuple[ndarray, int]
            the point where a termination condition is met and which condition
        """
        if vers == 0:
            return self.local(x0)
        if vers == 1:
            return self.globalized(x0)
        if vers == 2:
            return self.mod_globalized(x0)