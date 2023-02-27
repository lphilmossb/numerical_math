from Lagrange import LagrangeBasis
from Hermite import Hermite, Hermiteval
import matplotlib.pyplot as plt
import numpy as np
from numpy import abs, max


plt.rcParams.update({
    'axes.grid': True,
})

f = lambda x : 1.0 / (1 + 8 * (x**2))
fp = lambda x : - 16.0 * x /((1+8.0*(x**2))**2)
x = np.linspace(-1,1,1000)
y = f(x)
yp = fp(x)


def interpolate(n : int):
    xn = np.linspace(-1,1, n, endpoint=True)
    yn = f(xn)
    ypn = fp(xn)

    inner1 = np.where(x >= xn[1])
    inner2 = np.where(x <= xn[-2])
    inner = np.intersect1d(inner1, inner2)

    lag = LagrangeBasis(xn, -1, 1000)
    ylag = lag.interpolate(yn)
    c, xcol = Hermite(xn, yn, ypn)
    yher = Hermiteval(c, xcol, x)
    h_error = abs(yher - y)
    l_error = abs(ylag - y)

    ltrunc = abs(ylag[inner] - y[inner])
    htrunc = abs(yher[inner] - y[inner])

    return np.average(l_error), max(l_error), np.average(ltrunc), np.average(h_error), max(h_error), np.average(htrunc)


N = 100
offset = 3
l_errors = np.zeros(N-offset)
l_errors_max = np.zeros(N-offset)
ltruncated = np.zeros(N-offset)
h_errors = np.zeros(N-offset)
h_errors_max = np.zeros(N-offset)
htruncated = np.zeros(N-offset)

for i in range(offset,N):
    l, lmax, ltr, h, hmax, htr = interpolate(i)
    l_errors[i-offset] = l
    l_errors_max[i-offset] = lmax
    ltruncated[i-offset] = ltr
    h_errors[i-offset] = h
    h_errors_max[i-offset] = hmax
    h_errors_max[i-offset] = htr

plt.plot(list(range(offset,N)), l_errors, label='lagrange (avg')
plt.plot(list(range(offset,N)), l_errors_max, label='lagrange (max)')
# plt.plot(list(range(offset,N)), ltruncated, label='lagrange (truncated)')
plt.plot(list(range(offset,N)), h_errors, label='hermite (avg)')
plt.plot(list(range(offset,N)), h_errors_max, label='hermite (max)')
plt.legend()
plt.show()

# data = np.column_stack((list(range(offset,N)), l_errors, l_errors_max, h_errors, h_errors_max))
# np.savetxt('runge_error.csv', data, header='n,lag,lagmax,her,hermax', comments='', delimiter=',')

data_tr = np.column_stack((list(range(offset,N)), ltruncated, htruncated))
np.savetxt('runge_truncated.csv', data_tr, header='n,lag,her', comments='', delimiter=',')