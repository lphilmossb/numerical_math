from numpy import *
import matplotlib.pyplot as plt
from plot_util import *


c = 1.0
x = linspace(-4, 4, 1000)
y = sqrt(c + x**2)

mpl_setscheme()
fig, ax = get_figure()

set_limits(ax, column_stack((concatenate([x,x]), concatenate([y,-y]))))
fillplot(ax, x, y, -y, 'cyan', 'cyan')
# segplot(ax, column_stack((x, y)), 'cyan')
segplot(ax, column_stack((x, -y)), 'cyan')

plt.show()