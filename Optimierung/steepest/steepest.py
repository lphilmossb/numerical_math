"""
file         : steepest.py | UTF-8
author       : Moritz Mossböck, 11820925 @ TUGraz | moritz.mossboeck@student.tugraz.at
testing os   : Arch Linux (x86, 64bit) | kernel-version: 6.4.11-arch2-1
python target: 3.11.4

python package versions:
numpy 1.25.1

file description:
This file provides a class to run a gradient descent for finding minima of differentiable functions
"""
from typing import Callable, Optional, Tuple
from numpy import ndarray as vector, dot, zeros, copy, array
from numpy.linalg import norm

class GradientDescent:
    """
    Acts as a runtime-context for performing a gradient descent

    Attributes:
    -----------
    _f : Callable[[ndarray], float]
        callable that computes the function value
    _nabla : Callable[[ndarray], ndarray]
        callable that computes the gradient of f
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
    _beta : float
        factor for decreasing the step size per armijo-iteration
    _eta : float
        used for finding initial search interval for the mixed-step size
    _xk : ndarray
        current iteration point
    _steps : List[callable]
        list of the step-functions to allow modularity
    _record : List
        if you want to record all steps, this stores the intermediate steps

    Methods:
    --------
    checkparams() -> None
        validates all attributes are in correct ranges and not None
    armijo() -> float
        computes the armijo-step distance
    wolfe_powell() -> float
        placeholder for future implementation
    mixed_step() -> float
        computes a step distance using a mix of armijo and wolfe-powell conditions
    dk() -> ndarray
        property, returns the direction based on the current `_xk`
    record() -> ndarray
        property, returns the recorded steps as a matrix
    run(x0 : ndarray, step : Optional[int]) -> Tuple[vector, int]
        performs the gradient descent using step-method specifed by `step`

    Constructor:
    ------------
        The constructor of `GradientDescent` only has the callables as positional arguments, as these are stricly
        required. All other parameters are passed as keyword arguments to avoid huge function signatures. After 
        assigning each parameter, their validity is verified with `checkparams()`.

        Parameters:
        -----------
        f : Callable[[ndarray], float]
            callable that computes f
        nabla_f : Callable[[ndarray], ndarray]
            callable that computes the gradient of f
        **kwargs : dict, optional
            additional parameters, see the class attributes for a list of possible parameters
    """

    __SIGMA__ = 0.5
    __BETA__ = 0.75
    __ETA__ = 1.5
    __EPSILON__ = 1e-9
    __RHO__ = 0.75
    __N__ = int(1e4)
    __M__ = 1e9

    def __init__(self, f : Callable[[vector], vector], nabla_f : Callable[[vector], vector], **kwargs) -> None:
        self._f = f
        self._nabla_f = nabla_f
        self._sigma = kwargs.get('sigma', GradientDescent.__SIGMA__)
        self._beta = kwargs.get('beta', GradientDescent.__BETA__)
        self._eta = kwargs.get('eta', GradientDescent.__ETA__)
        self._epsilon = kwargs.get('epsilon', GradientDescent.__EPSILON__)
        self._rho = kwargs.get('rho', GradientDescent.__RHO__)
        self._M = kwargs.get('M', GradientDescent.__M__)
        self._N = int(kwargs.get('N', GradientDescent.__N__))
        self._xk = zeros(3, dtype=float)
        self._steps = [self.armijo, self.wolfe_powell, self.mixed_step]
        self._record = []
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
            if `_epsilon` is negative
        ValueError
            if `_eta` less or equal to 1
        ValueError
            if `_M` is negative
        ValueError
            if `_N` is negative
        ValueError
            if `_rho` is non-positive
        ValueError
            if `_sigma` is outside (0,0.5)
        ValueError
            if `_beta` is outside (0,1)
        """
        if self._f is None:
            raise ValueError('invalid f, is of None-type')
        if self._nabla_f is None:
            raise ValueError('invalid nabla_f, is of None-type')
        if self._sigma <= 0.0 or self._sigma >= 1:
            raise ValueError('invalid sigma, must be in (0,1)')
        if self._eta <= 1.0:
            raise ValueError('invalid eta, must be strictly greater 1')
        if self._epsilon <= 0.0:
            raise ValueError('invalid epsilon, must be strictly positive')
        if self._M <= 0.0:
            raise ValueError('invalid M, must be strictly positive')
        if self._N < 0:
            raise ValueError('invalid N, must be positive')
        if self._rho <= 0 or self._rho >= 1.0:
            raise ValueError('invalid rho, must be in (0,1)')

    def armijo(self) -> float:
        """
        Computes the armijo-step distance based on the current direction
        
        Returns:
        --------
        float
            the armijo-step distance
        """
        tk = 1.0
        fxk = self._f(self._xk)
        dk = self.dk
        while self._f(self._xk + self._beta*tk*dk) >= fxk - self._beta * self._sigma*tk*dot(dk,dk) and tk >= 1e-12:
            tk *= self._beta
        return tk

    def wolfe_powell(self) -> float:
        """
        Placeholder for wolfe-powell step size

        Returns:
        --------
        float
            1.0
        """
        return 1.0

    def mixed_step(self) -> float:
        """
        Computes a step-size using a mix of armijo and wolfe-powell conditions

        Returns:
        --------
        float
            the corresponding step-size based on the current direction
        """
        k = 1
        tau = 1.0
        dk = self.dk
        cond_A = lambda : self._f(self._xk + tau * dk) <= self._f(self._xk) - self._sigma * tau * dot(dk, dk)
        cond_P = lambda : dot(self._nabla_f(self._xk + tau * dk), dk) >= -self._rho * dot(dk, dk)


        while self._f(self._xk +self._eta**k * dk) <= self._f(self._xk) - self._sigma * self._eta**k * dot(dk,dk) and k <= 50:
            k += 1
        
        tau0 = 0.0
        tau1 = self._eta**k
        tau = (tau0 + tau1) * 0.5

        for _ in range(self._N):
            if tau1 - tau0 <= 1e-9:
                break
            if cond_A() and cond_P():
                return tau
            elif cond_A() and not cond_P():
                tau0 = tau
            else:
                tau1 = tau
            tau = (tau0 + tau1) * 0.5
        return tau
    
    @property
    def dk(self) -> vector:
        """
        Property for quickly accessing the current direction
        """
        return -self._nabla_f(self._xk)

    @property
    def record(self) -> vector:
        """
        Property for accessing recorded steps
        """
        return array(self._steps)

    def run(self, x0 : vector, step : Optional[int] = 0, **kwargs) -> Tuple[vector, int]:
        """
        Method to perform a gradient descent using the step size specified with `step`.

        Parameters:
        -----------
        x0 : ndarray
            the starting point
        step : int, optinal = 1
            which step to use
        
        Returns:
        --------
        Tuple[ndarray, int]
            the point satisfying a termination condition and which condition was met
        """
        self._xk = copy(x0)
        term_cond = 1
        self._record = []
        record = kwargs.get('record', False)

        for _ in range(self._N):
            if norm(-self.dk) <= self._epsilon:
                term_cond = 0
                break
            if norm(self._xk) >= self._M:
                term_cond = 2
                break
            tk = self._steps[step]()
            self._xk = self._xk + tk * self.dk
            if record:
                self._record.append(self._xk)
        
        return self._xk, term_cond

