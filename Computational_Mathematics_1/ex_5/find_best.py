import numpy as np
import matplotlib.pyplot as plt

from Fitter import Fitter

plt.rcParams.update({
    'text.usetex': True,
    'grid.linestyle': '--',
    'font.size': 18,
})

x_data = np.array([-0.04, 0.93, 1.95, 2.90, 3.83, 5.00, 5.98, 7.05, 8.21, 9.08, 10.09])
y_data = np.array([-8.66, -6.44, -4.36, -3.27, -0.88, 0.87, 3.31, 4.63, 6.19, 7.40, 8.85])
fitter, aux = Fitter.find_best(x_data, y_data, True)



fig, ax = plt.subplots(1,2)
fitter.PlotPoly(ax[0], y_data)

ax[0].grid()
ax[1].grid()
ax[0].set_xlabel(r'$x$')
ax[1].set_xlabel(r'$x$')
ax[0].set_ylabel(r'$f(x)$')
ax[0].set_title(rf'Polynomial Fitting of degree ${fitter.m-1}$ with $\sigma={fitter.StdDev(y_data):.4f}$')
fig.set_size_inches(24, 12)

x = np.linspace(np.min(x_data)-0.1, np.max(x_data)+0.1, 500)

for f in aux:
    if f.m == fitter.m:
        continue
    ax[1].plot(x, f(x), linewidth=0.75, label=rf'deg = {f.m-1}, $\sigma=${f.StdDev(y_data):.4f}')

ax[1].set_title('Other tested polynomials')
ax[1].legend()
fig.tight_layout()
plt.show()
# fig.savefig('ex_5_3.png', dpi=300)