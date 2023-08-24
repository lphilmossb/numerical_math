"""
author        : Moritz Mossböck             | 11820925  | moritz.mossboeck@student.tugraz.at
file          : trig_inter.py               | UTF-8
target version: python 3.10.9               | 64-bit
course        : Computational Mathematics 1 | MAT.208UB
pylint-score  : 10 / 10

Provide functions for computing the real trigonometric interpolant of a signal
"""

# imports
from numpy import ndarray, sum as nsum, cos, pi, sin
from scipy.fft import fftfreq
from fft import fft

def trig_inter(signal : ndarray, axis : ndarray, **kwargs) -> ndarray:
    """
    Compute the real trigonometric inperolant of the function defined in `x`

    Parameters:
    -----------
    x : ndarray
        function values to interpolate
    axis : ndarray
        x-axis to interpolate over
    dt : Optional float, default = 1.0
        sample time spacing

    Returns:
    --------
    ndarray
        trigonometric interpolant evaluated over `axis`
    """
    dft = fft(signal)
    order = len(dft) // 2

    faxis = fftfreq(len(dft), kwargs.get('dt'))


    def arg(k):
        return (axis - axis[0]) * faxis[k] * 2 * pi

    interp = dft[0].real + 2 * nsum( [dft[k].real * cos(arg(k)) for k in range(1,order+1) ], axis=0)
    interp = interp - 2 * nsum( [dft[k].imag * sin(arg(k)) for k in range(1,order+1) ], axis=0)
    return interp / len(dft)
