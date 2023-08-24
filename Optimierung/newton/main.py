from numpy import array
from newton import NewtonMethod

def f(x):
    return 100*(x[1] - x[0]**2)**2 + (1-x[0])**2

def nabla_f(x):
    x1 = x[0]
    x2 = x[1]
    dx = -400 * (x2 - x1**2)*x1 - 2*(1-x1)
    dy = 200 * (x2 - x1**2)
    return array([dx,dy])

def hess_f(x):
    x1 = x[0]
    x2 = x[1]
    ddx = 800*x1**2 - 400*(x2 - x1**2) + 2
    ddy = 200
    dxdy = -400 * x1
    return array([[ddx,dxdy],[dxdy,ddy]])

xinit = array([0, 0])

nmeth = NewtonMethod(f, nabla_f, hess_f)

print(nmeth.run(xinit, 0))
print(nmeth.run(xinit, 1))
print(nmeth.run(xinit, 2))
 