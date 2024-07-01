# Project : Bachelor thesis Mathematics & Computer science
# Author  : Bokke v.d. Bergh
# Contents:
# Contains a number of helper function used for the
# intersection of confidence intervals technique.

import numpy as np

from globals import tau, sigma

def construct_window(x, h, data):
    """Sets up a simple window around x, not exceeding the given bounds."""
    return np.arange(max(x-h, 0), min(x+h+1, len(data)))

def confidence_interval(x, h, data):
    """Constructs a confidence interval using the given data
    and estimated distribution values."""
    window = construct_window(x, h, data)
    Nh = len(window)
    center_xh = np.sum(data[window])/Nh
    deviation_xh = sigma / np.sqrt(Nh)
    return (center_xh - tau * deviation_xh, center_xh + tau * deviation_xh)

def intersection_of_intervals(x, data):
    """Apply the intersection of confidence intervals method described
    in the thesis."""
    hmax = 0
    (L, U) = confidence_interval(x,hmax, data)
    h = 1
    while h <= max(x, x - len(data)):
        (Ln, Un) = confidence_interval(x,h, data)
        if max(Ln, L) < min(Un, U):
            break
        L, U = max(Ln, L), min(Un, U)
        hmax = h
        h += 1
    return hmax
