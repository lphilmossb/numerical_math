"""
file  : functions.py
author: Moritz Mossböck, 11820925 @ TUGraz | moritz.mossboeck@student.tugraz.at

"""
from numpy import array, exp, outer, eye, dot, ones_like, zeros, sin, cos
from numpy import ndarray as vector

# only for type hinting
matrix = vector



#-----------------------------------------------------------------------------------------------------------------------
# Rosenbrock-Function
#-----------------------------------------------------------------------------------------------------------------------
def r(x : vector) -> float:
    return 100*(x[1] - x[0]**2)**2 + (1-x[0])**2

def nabla_r(x : vector) -> vector:
    x1 = x[0]
    x2 = x[1]
    dx = -400 * (x2 - x1**2)*x1 - 2*(1-x1)
    dy = 200 * (x2 - x1**2)
    return array([dx,dy])

def hess_r(x : vector) -> matrix:
    x1 = x[0]
    x2 = x[1]
    ddx = 800*x1**2 - 400*(x2 - x1**2) + 2
    ddy = 200
    dxdy = -400 * x1
    return array([[ddx,dxdy],[dxdy,ddy]])

#-----------------------------------------------------------------------------------------------------------------------
# f
#-----------------------------------------------------------------------------------------------------------------------
def f(x : vector) -> float:
    return -exp(-dot(x,x))

def nabla_f(x : vector) -> vector:
    return 2*exp(-dot(x,x)) * x

def hess_f(x : vector) -> matrix:
    A = eye(len(x)) - 2*outer(x,x)
    return 2*exp(-dot(x,x)) * A

#-----------------------------------------------------------------------------------------------------------------------
# g
#-----------------------------------------------------------------------------------------------------------------------
def g(x : vector) -> float:
    x1 = x[0]
    x2 = x[1]
    return 2*x1**2 + x2**2 - 2*x1*x2 + 2*x1**3 + x1**4

def nabla_g(x : vector) -> vector:
    x1 = x[0]
    x2 = x[1]
    return 2*array([2*x1-x2+3*x1**2+2*x1**3,x2-x1])

def hess_g(x : vector) -> matrix:
    x1 = x[0]
    return 2*array([[2+6*x1+6*x1**2, -1],[-1,1]])

#-----------------------------------------------------------------------------------------------------------------------
# h
#-----------------------------------------------------------------------------------------------------------------------
def div(x : vector) -> float:
    return sum(x)

def nabla_div(x : vector) -> vector:
    return ones_like(x)

def hess_div(x : vector) -> matrix:
    return zeros((len(x),len(x)))

#-----------------------------------------------------------------------------------------------------------------------
# d
#-----------------------------------------------------------------------------------------------------------------------
def d(x : vector) -> float:
    x1 = x[0]
    x2 = x[1]
    return sin(x1*x2)

def nabla_d(x : vector) -> vector:
    x1 = x[0]
    x2 = x[1]
    return cos(x1*x2) * array([x2,x1])

def hess_d(x : vector) -> matrix:
    x1 = x[0]
    x2 = x[1]
    return array([[-x2**2 * sin(x1*x2), cos(x1*x2) - x1*x2*sin(x1*x2)],[cos(x1*x2) - x1*x2*sin(x1*x2), -x1**2*sin(x1*x2)]])

#-----------------------------------------------------------------------------------------------------------------------
# quadratic
#-----------------------------------------------------------------------------------------------------------------------
Q = array([[14, 9, -1], [9, 18, 6], [-1, 6, 5]])
c = array([0.5, 1.2, 3.14])
def q(x):
    return 0.5 * dot(x, Q @ x) + dot(x, c)

def nabla_q(x):
    return Q @ x + c
