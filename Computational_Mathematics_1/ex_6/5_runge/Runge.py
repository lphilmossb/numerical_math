


from Lagrange import LagrangeBasis
from Newton import compute_coeffs, horner
from Neville import Neville
import matplotlib.pyplot as plt
import numpy as np
from numpy import abs
from Hermite import Hermite, Hermiteval


plt.rcParams.update({
    'axes.grid': True,
})

f = lambda x : 1.0 / (1 + 8 * (x**2))
fp = lambda x : - 16.0 * x /((1+8.0*(x**2))**2)
x = np.linspace(-1,1,1000)
y = f(x)

x1 = np.arange(-1.0,1.5,0.5)
x2 = np.arange(-1.0,1.25,0.25)
x3 = np.arange(-1.0,1.2,0.2)
y1 = f(x1)
y2 = f(x2)
y3 = f(x3)
yp1 = fp(x1)
yp2 = fp(x2)
yp3 = fp(x3)

xdata = [x1,x2,x3]
ydata = [y1,y2,y3]

lag1 = LagrangeBasis(x1,-1, 1000)
lag2 = LagrangeBasis(x2,-1, 1000)
lag3 = LagrangeBasis(x3,-1, 1000)
ylag1 = lag1.interpolate(y1)
ylag2 = lag2.interpolate(y2)
ylag3 = lag3.interpolate(y3)


newt1 = compute_coeffs(y1, x1)
newt2 = compute_coeffs(y2, x2)
newt3 = compute_coeffs(y3, x3)
ynewt1 = horner(newt1, x, x1)
ynewt2 = horner(newt2, x, x2)
ynewt3 = horner(newt3, x, x3)

fig1, ax1 = plt.subplots(2, 3)


data = [[ylag1, ylag2, ylag3], [ynewt1, ynewt2, ynewt3]]
errors = [[abs(data[k][i] - y) for i in range(3)] for k in range(2)]

def save():
    interp_data = np.column_stack((x, y, ylag1, ylag2, ylag3, ynewt1, ynewt2, ynewt3))
    samples1 = np.column_stack((x1, y1))
    samples2 = np.column_stack((x2, y2))
    samples3 = np.column_stack((x3, y3))

    np.savetxt('interpolation.csv', interp_data, header='x,f,l1,l2,l3,n1,n2,n3', comments='', delimiter=',')
    np.savetxt('samples1.csv', samples1, header='x,y', comments='', delimiter=',')
    np.savetxt('samples2.csv', samples2, header='x,y', comments='', delimiter=',')
    np.savetxt('samples3.csv', samples3, header='x,y', comments='', delimiter=',')



def plot():
    for i in range(6):
        if i >= 3:
            ax1[0, i % 3].plot(x, y, label=r'$f$')
            ax1[0, i % 3].scatter(xdata[i % 3], ydata[i % 3], marker='x', color='r', alpha=0.75, label='samples')
            ax1[0, i % 3].plot(x, data[0][i % 3], label=r'$p_{L,' + f'{len(xdata[i % 3]) -1 }' + r'}$')
            ax1[0, i % 3].legend()
            ax1[0, i % 3].set_title(f'Neville Interpolation of degree {len(xdata[i % 3]) -1 }')
        else:
            ax1[1, i % 3].plot(x, y, label=r'$f$')
            ax1[1, i % 3].scatter(xdata[i % 3], ydata[i % 3], marker='x', color='r', alpha=0.75, label='samples')
            ax1[1, i % 3].plot(x, data[1][i % 3], label=r'$p_{N,' + f'{len(xdata[i % 3]) -1 }' + r'}$')
            ax1[1, i % 3].legend()
            ax1[1, i % 3].set_title(f'Newton Interpolation of degree {len(xdata[i % 3]) -1 }')


    fig1.set_size_inches(20,12)


def herplot():
    her1, xcol1 = Hermite(x1, y1, yp1)
    her2, xcol2 = Hermite(x2, y2, yp2)
    her3, xcol3 = Hermite(x3, y3, yp3)
    yher1 = Hermiteval(her1, xcol1, x)
    yher2 = Hermiteval(her2, xcol2, x)
    yher3 = Hermiteval(her3, xcol3, x)
    fig2, ax2 = plt.subplots(1, 3)

    ax2[0].plot(x, y, label=r'$f$')
    ax2[0].plot(x, yher1, label=r'$p_{H,4}$')
    ax2[1].plot(x, y, label=r'$f$')
    ax2[1].plot(x, yher2, label=r'$p_{H,8}$')
    ax2[2].plot(x, y, label=r'$f$')
    ax2[2].plot(x, yher3, label=r'$p_{H,10}$')

    data = np.column_stack((x, y, yher1, yher2, yher3))
    np.savetxt('hermite.csv', data, header='x,y,h1,h2,h3', comments='', delimiter=',')

plot()
#herplot()
#save()

plt.show()