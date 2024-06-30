# Project : Bachelor thesis Mathematics & Computer science
# Author  : Bokke v.d. Bergh
# Contents:
# This file contains a borrowed efficient circular cross
# correlation function.
# The original naive function has been removed, as it was
# far too slow for practical use.
# It also contains the PCE and SPCE functions, which calculate
# the (signed) peak-to-correlation energy between two images.
# It is likely that these functions contain some errors or are lacking
# some constants.

import numpy as np
from numpy.fft import fft, ifft, fft2, ifft2, ifftshift

def circular_cross_correlation_1d(k, a, b):
    """Determine the circular cross-correlation between
    two vectors in R^n"""
    n, nc = len(a), len(b)
    assert n == nc
    return np.inner(a, np.roll(b, k)) / n

def cross_correlate_2d(x, h):
    """2d circular cross correlation using the fast fourier transform.
    Code sample from https://dsp.stackexchange.com/questions/85917/2d-cross-correlation-using-1d-fft
    """
    h = ifftshift(ifftshift(h, axes=0), axes=1)
    return ifft2(fft2(x) * np.conj(fft2(h))).real

def circular_cross_correlation_2d(W, R, c):
    """Determine the circular cross-correlation between
    two 2d arrays containing 3 color channels."""
    return cross_correlate_2d(W[:,:,c], R[:,:,c])

    # return np.sum(np.multiply(W, np.roll(R, (x, y), axis=(0,1)))) / np.prod(n)


def peak_correlation_energy(W, R, peak_size, c):
    n, nc = W.shape, R.shape
    assert n == nc
    ccc = circular_cross_correlation_2d(W, R, c)
    top = ccc[0, 0]
    total = 0
    rem = 0
    for y in range(n[0]):
        for x in range(n[1]):
            tx, ty = min(x, n[1] - x), min(y, n[0] - y)
            if tx*tx + ty*ty > peak_size*peak_size:
                total += ccc[y, x]
            else:
                rem += 1
    bottom = total / (np.prod(n) - rem)
    return top * top / bottom

def signed_peak_correlation_energy(W, R, peak_size, c):
    n, nc = W.shape, R.shape
    assert n == nc
    ccc = circular_cross_correlation_2d(W, R, c)
    top = ccc[0, 0]
    total = 0
    rem = 0
    for y in range(n[0]):
        for x in range(n[1]):
            tx, ty = min(x, n[1] - x), min(y, n[0] - y)
            if tx*tx + ty*ty > peak_size*peak_size:
                total += ccc[y, x]
            else:
                rem += 1
    bottom = total / (np.prod(n) - rem)
    return np.abs(top) * top / bottom
