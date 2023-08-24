"""
author        : Moritz Mossböck             | 11820925  | moritz.mossboeck@student.tugraz.at
file          : fft.py                      | UTF-8
target version: python 3.10.9               | 64-bit
course        : Computational Mathematics 1 | MAT.208UB
pylint-score  : 10 / 10

Provide functions for computing the DFT and iDFT using the FFT algorithm.
"""

# imports
from numpy import ndarray, complex64, zeros, pad, exp, pi, conjugate, log2


def shift_bit_length(num: int):
    """
    Get next power of 2 greater N

    Parameters:
    -----------
    N : int
        number to get the next power of 2 of

    Returns:
    --------
    int
        next greater power of `num`
    """
    return 1 << ((num - 1).bit_length())


def bit_reverse(num: int, log2n: int) -> int:
    """
    Compute bit-reversal and return it in base 10

    Parameters:
    -----------
    num : int
        number to bit-reverse
    log2n : int
        number of required bits

    Returns:
    --------
    int
        bit reversed value of `num`
    """
    rval = 0
    for _ in range(log2n):
        rval <<= 1
        rval |= (num & 1)
        num >>= 1
    return rval


def fft(signal_values: ndarray) -> ndarray:
    """
    Compute the DFT of `signal_values` using recursive FFT

    Parameters:
    -----------
    signal_values : ndarray
        values of the signal

    Returns:
    --------
    ndarray
        DFT of `signal`
    """
    length = len(signal_values)
    signal = signal_values.astype(complex64)

    dft = zeros(length, dtype=complex64)

    if length == 1:
        dft[0] = signal[0]
    else:
        even_dft = fft(signal[0::2])
        odd_dft = fft(signal[1::2])

        def phase(k):
            return exp(-2j * pi * k / length)

        shift = length // 2

        for index, (even, odd) in enumerate(zip(even_dft, odd_dft)):
            dft[index] = even + phase(index) * odd
            dft[index + shift] = even - phase(index) * odd
    return dft


def fft_iterative(signal_values: ndarray):
    """
    Compute the DFT of `signal_values` using iterative FFT

    Parameters:
    -----------
    signal_values : ndarray
        values of the signal

    Returns:
    --------
    ndarray
        DFT of `signal`
    """

    length = shift_bit_length(len(signal_values))
    log2n = int(log2(length))
    signal = pad(signal_values, (0, length-len(signal_values)), 'constant')
    dft = zeros(length, dtype=complex)

    # bit reverse indices of f
    for i in range(length):
        dft[i] = signal[bit_reverse(i, log2n)]

    for layer in range(1, log2n+1):
        m_1 = 1 << layer
        m_2 = m_1 >> 1
        butterfly_factor = 1.0
        unit_root = exp(1j * pi / m_2)
        for j in range(m_2):
            for k in range(j, length, m_1):
                top = dft[k + m_2]
                bot = butterfly_factor * dft[k]

                dft[k] = bot + top
                dft[k+m_2] = bot - top
            butterfly_factor *= unit_root
    return dft


def ifft(dft: ndarray) -> ndarray:
    """
    Compute the inverse DFT of `dft`

    Parameters:
    -----------
    dft : ndarray
        dft of the original signal

    Returns:
    --------
    ndarray
        original signal
    """
    return 1.0 / len(dft) * conjugate(fft(conjugate(dft)))
