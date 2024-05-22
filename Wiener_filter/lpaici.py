"""
A module containing 
"""

import numpy as np

tau = 0
sigma = 0

def construct_window(x, h, data):
    return np.arange(max(x-h, 0), min(x+h+1, len(data)))

def Confidence_interval(x, h, data):
    window = construct_window(x, h, data)
    Nh = len(window)
    center_xh = np.sum(data[window])/Nh
    deviation_xh = sigma / np.sqrt(Nh)
    return (center_xh - tau * deviation_xh, center_xh + tau * deviation_xh)

def Intersection_of_Intervals(x, data):
    hmax = 0
    (L, U) = Confidence_interval(x,hmax, data)
    h = 1
    while h <= max(x, x - len(data)):
        (Ln, Un) = Confidence_interval(x,h, data)
        if max(Ln, L) < min(Un, U):
            break
        L, U = max(Ln, L), min(Un, U)
        hmax = h
        h += 1
    return hmax
