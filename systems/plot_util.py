from matplotlib.pyplot import subplots, rcParams
from matplotlib.collections import LineCollection
from numpy import array, concatenate, column_stack

SCHEMA = {
    'report':
    {
        'font.family': 'serif',
        'font.serif': ['Latin Modern Roman'] + rcParams['font.serif'],
        'text.color': 'black',
        'axes.labelcolor': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
        'axes.facecolor': 'white',
        'figure.facecolor': 'white'
    },
    'show':
    {
        'font.family': 'serif',
        'font.serif': ['Latin Modern Roman'] + rcParams['font.serif'],
        'text.color': 'white',
        'axes.labelcolor': 'white',
        'xtick.color': 'white',
        'ytick.color': 'white',
        'axes.facecolor': 'black',
        'figure.facecolor': 'black',
        'grid.color': 'white',
        'grid.alpha': 0.3,
        'grid.linewidth': 0.4
    }
}

def mpl_setscheme(scheme = 'show'):

    for key, val in SCHEMA[scheme].items():
        rcParams[key] = val

def get_figure(nrows = 1, ncols = 1, **kwargs):
    fig, ax = subplots(nrows, ncols, **kwargs)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(zorder=0)
    return fig, ax

def get_3dfiugre(nrows=1, ncols = 1, **kwargs):
    fig, ax = subplots(nrows, ncols, subplot_kw={'projection': '3d'} **kwargs)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.grid(zorder=0)
    return fig, ax

def get_segments(data):
    points = array([data[:,0], data[:,1]]).T.reshape(-1, 1, 2)
    return concatenate([points[:-1], points[1:]], axis=1)

def segplot(ax, data, colours):
    segments = get_segments(data)
    lc = LineCollection(segments)
    lc.set_colors(colours)
    lc.set_capstyle('round')
    ax.add_collection(lc)
    return lc

def fillplot(ax, x, y, y0, colours, fcolour):
    fill = ax.fill_between(x, y0, y, color=fcolour, alpha=0.2)
    return segplot(ax, column_stack((x, y)), colours), fill

def set_limits(axis, data, offset=0.5):
    axis.set_xlim(min(data[:,0]) - offset, max(data[:,0]) + offset)
    axis.set_ylim(min(data[:,1]) - offset, max(data[:,1]) + offset)